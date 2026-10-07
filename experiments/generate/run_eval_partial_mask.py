#!/usr/bin/env python3
"""Evaluate ONE partial-mask configuration in memory — no folder written.

    python experiments/generate/run_eval_partial_mask.py <circuit folder> \\
        --base-size 288 --augment canonical:50 --closure --prune causal:50 \\
        --shots 200000 --seed 0 --workers 48 --tc 40 --tl 0 --th 60 [--iso-fh 0.001] \\
        [--stages] [--diag] [--save DIR]

The mask flags are exactly those of algorithms/gen_partial_mask_dem.py (base,
ordered --augment/--closure/--prune ops, essentials files, contraction), so
the configuration you settle on is the command line you then give the
generator (with --out <folder> --contract).

Shots: postselect-passing attempts replayed from the shared trace
(out/traces/<folder>/, seed + worker count fix the shot set), complete-decoded
once per run; the partial decoder of the configuration is run on the SAME
shots. Metrics:
  A structure   |M|, fraction, contracted vertices, contracted-DEM edge/fault
                counts, detector-degree and fault-node-arity distributions,
                S(M) crossing mass, measured straddles per shot
  B gap fidelity P(gp=gc), P(gp<gc), P(gp>gc), mean|gp-gc|, mean(gp-gc),
                E[gc-gp | gp<gc], E[gp-gc | gp>gc], quantiles, Spearman
  C gating      three tiers at (tc, tl, th): gp<tl reject, tl<=gp<=th defer,
                gp>th accept; tier shares; early-reject error P(gc>=tc|gp<tl)
                and joint rate; false-high P(gc<tc|gp>th) and joint rate;
                with --iso-fh the smallest th meeting the false-high budget
  D diagnostics (--diag) Q coverage (all/exp), canonical coverage,
                causal-score stats of selected vs hidden detectors
--stages evaluates the mask after the base and after every op as well.
--save DIR writes stats.json, mask_bool.npy, mask(.stage).svg, gap_scatter.png,
gating_vs_th.png. No preparation-time / latency model here (see the estimator).
"""
import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702

import sinter
from scipy.stats import spearmanr

import gen_partial_mask_dem as gen
from attempt_stream import AttemptTrace, ShotCursor
from circuit_folder import load_circuit_folder, compile_samplers
from dem_stats import dem_stats
from dem_structure import DemStructure
from partial_desaturation_sampler import PartialDesaturationSampler
from partial_mask_builder import PartialMaskBuilder

_G: dict = {}          # fork-shared state for worker processes


# ----------------------------------------------------------------------------
# shots: replay + complete decode (once per run), partial decode per mask
# ----------------------------------------------------------------------------
def _complete_worker(k):
    cf, W, per, seed, trace_dir = (_G[x] for x in ("cf", "W", "per", "seed", "trace_dir"))
    comp, _ = compile_samplers(load_circuit_folder(cf.folder, None))
    nd = cf.circuit.num_detectors
    ps = np.zeros(nd, bool); ps[sorted(cf.postselected)] = True
    tr = AttemptTrace.create_or_open(trace_dir, cf.gap_circuit, seed)
    cur = ShotCursor(tr, k, W)
    rows, gc, err = [], [], []
    while len(rows) < per:
        if cur.next_index >= tr.meta.num_attempts:
            tr.ensure(cur.next_index + 1, quiet=True)
        row, obs, _ = cur.next()
        fired = np.flatnonzero(np.unpackbits(row, bitorder="little")[:nd])
        if ps[fired].any():
            continue
        oc, g, _ = comp.decode_det_set_with_time({int(d) for d in fired}, "parallel")
        rows.append(row); gc.append(int(g)); err.append(int(bool(oc) != obs))
    return np.array(rows), np.array(gc, np.int64), np.array(err, np.uint8)


def complete_shots(cf, out_root, *, shots, seed, workers):
    trace_dir = cf.trace_dir(out_root)
    AttemptTrace.create_or_open(trace_dir, cf.gap_circuit, seed).ensure(2 * shots, quiet=True)
    _G.update(cf=cf, W=workers, per=-(-shots // workers), seed=seed, trace_dir=trace_dir)
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as ex:
        parts = list(ex.map(_complete_worker, range(workers)))
    cat = lambda j: np.concatenate([p[j] for p in parts])[:shots]
    return {"rows": cat(0), "gc": cat(1), "err": cat(2)}


def _partial_worker(k):
    dec, rows, W, nd = (_G[x] for x in ("dec", "rows", "W", "nd"))
    out = []
    for r in range(k, len(rows), W):
        fired = np.flatnonzero(np.unpackbits(rows[r], bitorder="little")[:nd])
        out.append(dec.decode_det_set_with_time({int(d) for d in fired}, "parallel")[1])
    return k, np.array(out)


def partial_gaps(dec, rows, nd, workers):
    _G.update(dec=dec, rows=rows, W=workers, nd=nd)
    gp = np.empty(len(rows))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as ex:
        for k, arr in ex.map(_partial_worker, range(workers)):
            gp[k::workers] = arr
    return gp


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def straddles_per_shot(st: DemStructure, mask, rows, nd):
    """Visible fired detectors having a hidden fired component-neighbour."""
    bits = np.unpackbits(rows, axis=1, bitorder="little")[:, :nd].astype(bool)
    hidden_fired = bits & ~mask[None, :]
    out = np.zeros(len(rows), np.int64)
    for d in np.flatnonzero(mask):
        nb = np.array(sorted(n for n in st.nbrs[d] if not mask[n]), np.int64)
        if len(nb):
            out += bits[:, d] & hidden_fired[:, nb].any(axis=1)
    return out


def structure_metrics(mask, dec, st, rows, nd):
    cs = dem_stats(dec.gap_dem)
    return {
        "num_detectors": int(nd), "selected": int(mask.sum()), "selected_fraction": float(mask.sum() / nd),
        "contracted_vertices": len(dec.kept),
        "contracted_dem": {k: cs[k] for k in ("num_fault_nodes", "num_components", "n_boundary_edges",
                                              "n_edges", "n_hyperedges", "n_obs_flipping")},
        "detector_degree": cs["detector_degree"], "fault_node_degree": cs["fault_node_degree"],
        "S_crossing_mass": st.S(mask),
        "straddles_per_shot": None if rows is None else {
            "mean": float(straddles_per_shot(st, mask, rows, nd).mean()),
            "p_any": float((straddles_per_shot(st, mask, rows, nd) > 0).mean())},
    }


def gap_metrics(gc, gp):
    d = gp - gc
    lo, hi = d < 0, d > 0
    return {
        "p_equal": float(np.mean(d == 0)), "p_below": float(lo.mean()), "p_above": float(hi.mean()),
        "mean_abs_diff": float(np.abs(d).mean()), "mean_diff": float(d.mean()),
        "mean_deficit_given_below": float(-d[lo].mean()) if lo.any() else 0.0,
        "mean_excess_given_above": float(d[hi].mean()) if hi.any() else 0.0,
        "diff_quantiles": {q: float(np.percentile(d, q)) for q in (1, 5, 25, 50, 75, 95, 99)},
        "spearman": float(spearmanr(gc, gp)[0]),
    }


def gating_metrics(gc, gp, err, tc, tl, th):
    t1, t3 = gp < tl, gp > th
    t2 = ~t1 & ~t3
    acc = t3 | (t2 & (gc >= tc))
    return {
        "tc": tc, "tl": tl, "th": th,
        "tier_share": {"T1_reject": float(t1.mean()), "T2_defer": float(t2.mean()), "T3_accept": float(t3.mean())},
        "early_reject_error": float(np.mean(gc[t1] >= tc)) if t1.any() else 0.0,
        "early_reject_joint": float(np.mean(t1 & (gc >= tc))),
        "false_high": float(np.mean(gc[t3] < tc)) if t3.any() else 0.0,
        "false_high_joint": float(np.mean(t3 & (gc < tc))),
        "accept_rate": float(acc.mean()), "complete_only_accept_rate": float(np.mean(gc >= tc)),
        "ler_accepted": float(err[acc].mean()) if acc.any() else 0.0, "errors_accepted": int(err[acc].sum()),
        "ler_complete_only": float(err[gc >= tc].mean()), "errors_complete_only": int(err[gc >= tc].sum()),
    }


def iso_th(gc, gp, tc, cap):
    """Smallest th >= tc at which the early-accept tier is non-empty and its
    false-high rate is within the budget; None if no such th (nothing certifiable)."""
    for th in range(int(tc), 70):
        t3 = gp > th
        if t3.any() and np.mean(gc[t3] < tc) <= cap:
            return th
    return None


def diag_metrics(builder, mask):
    out = {}
    if builder.logical_ambiguity_essentials_meta is not None:
        out["q_coverage"] = {q: builder.logical_ambiguity_coverage(mask, q_type=q) for q in ("all", "exp")}
    if builder.canonical_essentials_meta is not None:
        c = np.maximum(np.asarray(builder.canonical_essentials_meta["score_values"]), 0.0)
        out["canonical_coverage"] = float(c[mask].sum() / max(c.sum(), 1e-300))
    if builder.causal_essentials_meta is not None:
        a = np.asarray(builder.causal_essentials_meta["score_values"])
        out["causal_score"] = {"selected_median": float(np.median(a[mask])), "selected_mean": float(a[mask].mean()),
                               "hidden_median": float(np.median(a[~mask])) if (~mask).any() else 0.0}
    return out


# ----------------------------------------------------------------------------
def evaluate(label, mask, cf, shots, st, args, builder):
    task = sinter.Task(circuit=cf.circuit, detector_error_model=cf.circuit.detector_error_model())
    dec = PartialDesaturationSampler().compiled_sampler_for_task(
        task, partial_mask_bool=mask, weight_cutoff=args.weight_cutoff, connectivity_fix_type=args.connectivity_fix)
    gp = partial_gaps(dec, shots["rows"], cf.circuit.num_detectors, args.workers)
    gc, err = shots["gc"], shots["err"]
    r = {"label": label,
         "structure": structure_metrics(mask, dec, st, shots["rows"], cf.circuit.num_detectors),
         "gap": gap_metrics(gc, gp),
         "gating": gating_metrics(gc, gp, err, args.tc, args.tl, args.th)}
    if args.iso_fh is not None:
        th = iso_th(gc, gp, args.tc, args.iso_fh)
        r["iso"] = None if th is None else {"false_high_cap": args.iso_fh,
                                            **gating_metrics(gc, gp, err, args.tc, args.tl, th)}
    if args.diag:
        r["diag"] = diag_metrics(builder, mask)
    r["_gp"] = gp
    return r


def print_result(r, *, full=True):
    s, g, t = r["structure"], r["gap"], r["gating"]
    print(f"\n=== {r['label']} ===")
    print(f"A structure : |M| {s['selected']}/{s['num_detectors']} ({s['selected_fraction']:.3f})  contracted vertices {s['contracted_vertices']}  "
          f"S(M) {s['S_crossing_mass']:.3f}" + (f"  straddles/shot {s['straddles_per_shot']['mean']:.2f} (P>0 {s['straddles_per_shot']['p_any']:.2f})" if s['straddles_per_shot'] else ""))
    if full:
        cd = s["contracted_dem"]
        print(f"              contracted DEM: faults {cd['num_fault_nodes']} components {cd['num_components']} "
              f"(boundary {cd['n_boundary_edges']}, edges {cd['n_edges']}, hyper {cd['n_hyperedges']}, obs-flipping {cd['n_obs_flipping']})")
        for name in ("detector_degree", "fault_node_degree"):
            d = s[name]
            print(f"              {name:<18} min {d['min']:.0f} mean {d['mean']:.2f} median {d['median']:.0f} max {d['max']:.0f} p95 {d['p95']:.0f} p99 {d['p99']:.0f}")
    print(f"B gap       : P(=) {g['p_equal']:.3f}  P(<) {g['p_below']:.3f}  P(>) {g['p_above']:.3f}  mean|d| {g['mean_abs_diff']:.2f}  mean d {g['mean_diff']:+.2f}  "
          f"E[gc-gp|<] {g['mean_deficit_given_below']:.1f}  E[gp-gc|>] {g['mean_excess_given_above']:.1f}  spearman {g['spearman']:.3f}")
    ts = t["tier_share"]
    print(f"C gating    : tc {t['tc']:g} tl {t['tl']:g} th {t['th']:g} | T1 {ts['T1_reject']:.3f} T2 {ts['T2_defer']:.3f} T3 {ts['T3_accept']:.3f} | "
          f"early-reject err {t['early_reject_error']:.4f} (joint {t['early_reject_joint']:.4f}) | false-high {t['false_high']:.4f} (joint {t['false_high_joint']:.5f}) | "
          f"accept {t['accept_rate']:.3f} (complete-only {t['complete_only_accept_rate']:.3f}) LER {t['ler_accepted']:.1e} ({t['errors_accepted']}) vs {t['ler_complete_only']:.1e} ({t['errors_complete_only']})")
    if r.get("iso") is not None:
        i = r["iso"]
        print(f"   iso-FH<= {i['false_high_cap']:g}: th* {i['th']}  T3 {i['tier_share']['T3_accept']:.3f}  false-high {i['false_high']:.4f}  accept {i['accept_rate']:.3f}")
    elif "iso" in r:
        print("   iso-FH: no th meets the budget")
    if r.get("diag"):
        print(f"D diag      : {json.dumps(r['diag'])}")


def save_bundle(save_dir, results, builder, cf, shots, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    save_dir.mkdir(parents=True, exist_ok=True)
    final = results[-1]
    np.save(save_dir / "mask_bool.npy", final["_mask"])
    for r in results:
        tag = "" if r is final else f".{r['label'].replace(' ', '_').replace(':', '-')}"
        _, _, meta = builder.build_mask_from_indices(np.flatnonzero(r["_mask"]))
        builder.visualize_partial_region_mask_svg(save_dir / f"mask{tag}.svg", mask_bool=r["_mask"], metadata=meta)
    gc, gp = shots["gc"], final["_gp"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(gc, gp, gridsize=40, bins="log"); ax.plot([0, 70], [0, 70], "r--", lw=0.8)
    ax.set_xlabel("complete gap g_c [dB]"); ax.set_ylabel("partial gap g_p [dB]"); ax.set_title(final["label"])
    fig.tight_layout(); fig.savefig(save_dir / "gap_scatter.png", dpi=130); plt.close(fig)
    ths = np.arange(int(args.tc), 70)
    fh = [np.mean(gc[gp > th] < args.tc) if (gp > th).any() else 0 for th in ths]
    t3 = [np.mean(gp > th) for th in ths]
    fig, ax = plt.subplots(figsize=(6, 4)); ax2 = ax.twinx()
    ax.semilogy(ths, np.maximum(fh, 1e-6), "r-", label="false-high P(gc<tc | gp>th)")
    ax2.plot(ths, t3, "b-", label="T3 share P(gp>th)")
    ax.set_xlabel("th"); ax.set_ylabel("false-high", color="r"); ax2.set_ylabel("T3 share", color="b")
    fig.tight_layout(); fig.savefig(save_dir / "gating_vs_th.png", dpi=130); plt.close(fig)
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    (save_dir / "stats.json").write_text(json.dumps(
        {"folder": cf.folder.name, "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
         "shots": {"n": int(len(gc)), "seed": args.seed, "workers": args.workers}, "results": clean}, indent=1))
    print(f"saved {save_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path, help="circuit data folder (read only)")
    gen.add_mask_arguments(ap)
    ev = ap.add_argument_group("evaluation")
    ev.add_argument("--shots", type=int, default=200_000, help="postselect-passing shots to evaluate on")
    ev.add_argument("--seed", type=int, default=0, help="trace seed (shot identity)")
    ev.add_argument("--workers", type=int, default=48)
    ev.add_argument("--tc", type=float, default=40.0, help="complete decision threshold")
    ev.add_argument("--tl", type=float, default=0.0, help="early-reject threshold (0 = fast-accept only)")
    ev.add_argument("--th", type=float, default=60.0, help="early-accept threshold")
    ev.add_argument("--iso-fh", type=float, default=None, help="also report the smallest th with false-high <= this")
    ev.add_argument("--stages", action="store_true", help="evaluate after the base and after every op")
    ev.add_argument("--diag", action="store_true", help="Q / canonical coverage, causal-score statistics")
    ev.add_argument("--out-root", type=pathlib.Path, default=paths.OUT, help="traces root")
    ev.add_argument("--save", type=pathlib.Path, default=None, help="write stats.json, mask, SVGs, plots here")
    args = ap.parse_args()
    gen.validate_mask_args(ap, args)
    if not (args.tl <= args.th):
        ap.error("--tl must be <= --th")

    t0 = time.time()
    cf = load_circuit_folder(args.folder, None)
    builder = PartialMaskBuilder(circuit=cf.circuit)
    st = DemStructure(cf.circuit.detector_error_model())
    ess_dir = args.essentials_dir or args.folder
    shots = complete_shots(cf, args.out_root, shots=args.shots, seed=args.seed, workers=args.workers)
    print(f"{cf.folder.name}: {len(shots['gc'])} postselect-passing shots complete-decoded ({time.time() - t0:.0f}s); "
          f"complete-only at tc {args.tc:g}: accept {np.mean(shots['gc'] >= args.tc):.3f}, LER {shots['err'][shots['gc'] >= args.tc].mean():.1e}")

    results = []
    stage_masks = []
    mask, _, base_spec, op_log = gen.build_mask(
        builder, args, ess_dir, on_stage=(lambda label, m: stage_masks.append((label, m.copy()))) if args.stages else None)
    if args.stages:
        for label, m in stage_masks[:-1]:
            r = evaluate(label, m, cf, shots, st, args, builder); r["_mask"] = m
            results.append(r); print_result(r, full=False)
    r = evaluate("final: " + " ".join(sys.argv[2:]), mask, cf, shots, st, args, builder); r["_mask"] = mask
    r["base"] = base_spec; r["ops"] = op_log
    results.append(r); print_result(r)
    print(f"\n({time.time() - t0:.0f}s total)")
    if args.save is not None:
        save_bundle(args.save, results, builder, cf, shots, args)


if __name__ == "__main__":
    main()
