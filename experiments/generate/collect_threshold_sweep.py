#!/usr/bin/env python3
"""Threshold-sweep data collection (times from the estimator, LERs from the
gap table) — stored, so the figures can be redrawn without recomputing.

For every t_c in --tc-grid:
    single   complete decoder only, gap threshold t_c
    pcp      two-stage: partial fast-accept with t_l = 0 and t_h = t_c (or --th),
             complete rescue at t_c                              (needs --partial)
Each point's preparation-time breakdown (gate / control / gap decode, per
accepted magic state) comes from the RuntimeEstimator at --epochs (means are
converged at 1M); its LER / accept rate come from the per-shot gap table
(collect_gap_table.py: 10^7-10^8 post-selected shots), evaluated with the
estimator's gating semantics. Existing rows are reused (resumable).

Output: experiments/result/runtime_estimation/<config>/threshold_sweep_<variant>_fb<ns>[_cs<tag>].json
    {"circuit", "partial", "feedback_ns", "control_latency", "epochs", "gap_table": {...},
     "single": [{tc, gate_us, control_us, decode_us, total_us, accept, ler, errors, n_accepted}, ...],
     "pcp":    [{tc, tl, th, ...}, ...]}

    python experiments/generate/collect_threshold_sweep.py experiments/data/circuits/<config> \
        --partial partial_<tag> --tc-grid 30 35 40 45 50 55 60 65 67 \
        --feedback-ns 3100 [--control-latency PROFILE.json] [--epochs 1000000 --workers 32]
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
REPO = paths.REPO
from estimator_run import build, gating, shard                             # noqa: E402
from gap_table import rule_stats, GAP_TABLE_DIR as GAP_OUT                 # noqa: E402

RESULT = paths.RESULT / "runtime_estimation"


def estimate(args, mode, tc, tl=None, th=None):
    """One estimator run (forked shards); returns the ready-time breakdown."""
    a = argparse.Namespace(**vars(args)); a.mode = mode; a.tc = float(tc); a.tl = tl; a.th = th; a.tp = None
    if mode == "single":
        a.partial = None
    est, _ = build(a, a.epochs, start=0, step=1)      # build() applies --control-latency in every shard
    K = a.workers
    if K > 1:
        shares = [a.epochs // K + (1 if k < a.epochs % K else 0) for k in range(K)]
        with ProcessPoolExecutor(max_workers=K, mp_context=get_context("fork")) as ex:
            est.merge_raw(list(ex.map(shard, [a] * K, range(K), [K] * K, shares)))
    else:
        est.estimate_runtime(decoder_time_measure_mode=a.measure, **gating(a))
    est.collect_statistics()
    s = est.statistics; r = est.stage_names.index("ready"); us = s["records_until_success"]
    return dict(gate_us=us["gate"]["mean"][r] / 1000, control_us=us["feedback"]["mean"][r] / 1000,
                decode_us=us["gap_decode"]["mean"][r] / 1000, total_us=us["total"]["mean"][r] / 1000,
                accept_estimator=s["accept_rate"], ler_estimator=s["logical_error_rate"], epochs=s["epoch"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", default=None, help="partial subfolder tag (enables the pcp series)")
    ap.add_argument("--tc-grid", nargs="+", default=[30, 35, 40, 45, 50, 55, 60, 65, 67],
                    help="complete-gap thresholds; the token 'cap' expands to the table's complete-gap cap")
    ap.add_argument("--th", type=float, default=None, help="pcp fast-accept threshold (default: = t_c)")
    ap.add_argument("--th-grid", nargs="+", default=None,
                    help="pcp: sweep t_h over this grid for every t_c (pairs with t_h >= t_c), for the "
                         "two-stage Pareto envelope; overrides --th. The token 'cap' expands to the gap "
                         "table's cap and cap-1 (the cap depends on p: 69 at 1e-3, 71 at 9e-4, ...); "
                         "t_h = cap means no fast accept, i.e. the complete-only limit")
    ap.add_argument("--modes", nargs="+", default=["single", "pcp"], choices=["single", "pcp"])
    ap.add_argument("--measure", choices=["serial", "parallel"], default="parallel")
    ap.add_argument("--feedback-ns", type=float, default=3100.0)
    ap.add_argument("--control-latency", type=pathlib.Path, default=None)
    ap.add_argument("--wait-rounds", type=int, default=2)
    ap.add_argument("--latency", choices=["mb", "constant"], default="mb")
    ap.add_argument("--mb-seed", type=int, default=0)
    ap.add_argument("--mb-shots", type=int, default=1000)
    ap.add_argument("--mb-freq", type=float, default=43e6)
    ap.add_argument("--decoder-ns", type=float, default=10_000.0)
    ap.add_argument("--partial-decoder-ns", type=float, default=500.0)
    ap.add_argument("--trace-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--gap-seed", type=int, default=0, help="seed of the gap table to read LERs from")
    ap.add_argument("--refresh-ler", action="store_true",
                    help="recompute the LER / accept columns of every stored row from the current gap table "
                         "(use after growing the table); the estimator times are kept")
    args = ap.parse_args()
    if "pcp" in args.modes and args.partial is None:
        sys.exit("error: pcp needs --partial")

    name = args.folder.resolve().name
    variant = args.partial or "complete"
    cs = ""
    if args.control_latency:
        p = json.loads(args.control_latency.read_text()); cs = "_cs" + p.get("tag", f"q{p['q']}f{p['fanout']}")
    out = RESULT / name / f"threshold_sweep_{variant}_fb{args.feedback_ns:g}{cs}.json"
    data = json.loads(out.read_text()) if out.exists() else {
        "circuit": name, "partial": args.partial, "feedback_ns": args.feedback_ns,
        "control_latency": str(args.control_latency) if args.control_latency else None,
        "epochs": args.epochs, "single": [], "pcp": []}

    # gap table (LER source)
    table = GAP_OUT / name / f"{variant}_s{args.gap_seed}.npz"
    z = np.load(table); gc, gp, err = z["gc"], z["gp"], z["err"]
    meta = json.loads(table.with_suffix(".meta.json").read_text())
    data["gap_table"] = {"file": paths.rel(table), "shots": int(len(gc)),
                         "postselect_acceptance": meta["postselect_acceptance"]}
    cap = int(gp.max()) if args.partial else int(gc.max())
    data["gap_table"]["cap"] = cap
    data["gap_table"]["complete_cap"] = int(gc.max())
    print(f"gap table: {len(gc)} post-selected shots, gap cap {cap} (complete {gc.max()})")
    args.tc_grid = sorted({float(gc.max()) if t == "cap" else float(t) for t in args.tc_grid})
    if args.th_grid:
        grid = set()
        for t in args.th_grid:
            grid.update({cap - 1, cap} if t == "cap" else {float(t)})
        args.th_grid = sorted(grid)
    if args.refresh_ler:
        # full-table LER plus a held-out split: `select` (first half) is what a frontier / operating
        # point may be chosen on, `eval` (second half) is what gets reported
        h = len(gc) // 2
        for mode in ("single", "pcp"):
            for row in data[mode]:
                kw = dict(tc=row["tc"], tl=row.get("tl"), th=row.get("th"))
                n, e, ler = rule_stats(gc, gp, err, **kw)
                nA, eA, lA = rule_stats(gc[:h], gp[:h], err[:h], **kw)
                nB, eB, lB = rule_stats(gc[h:], gp[h:], err[h:], **kw)
                row.update(n_accepted=n, errors=e, ler=ler, accept_of_postselected=n / len(gc),
                           ler_select=lA, errors_select=eA, n_select=nA, ler_eval=lB, errors_eval=eB, n_eval=nB)
        data["gap_table"]["split"] = "select = first half of the table, eval = second half"
        out.write_text(json.dumps(data, indent=1))
        print(f"refreshed LERs (full + select/eval halves) of {len(data['single'])} single + {len(data['pcp'])} pcp rows from {len(gc)} shots")

    points = []
    for mode in args.modes:
        for tc in args.tc_grid:
            if mode == "single":
                points.append((mode, dict(tc=tc)))
            elif args.th_grid:
                points += [(mode, dict(tc=tc, tl=0.0, th=th)) for th in args.th_grid if th >= tc]
            else:
                points.append((mode, dict(tc=tc, tl=0.0, th=(args.th if args.th is not None else tc))))
    for mode, key in points:
            tc, th = key["tc"], key.get("th")
            if any(all(abs(row.get(k, -1) - v) < 1e-9 for k, v in key.items()) for row in data[mode]):
                continue
            n, e, ler = rule_stats(gc, gp, err, tc=tc, tl=key.get("tl"), th=key.get("th"))
            h = len(gc) // 2
            nA, eA, lA = rule_stats(gc[:h], gp[:h], err[:h], tc=tc, tl=key.get("tl"), th=key.get("th"))
            nB, eB, lB = rule_stats(gc[h:], gp[h:], err[h:], tc=tc, tl=key.get("tl"), th=key.get("th"))
            row = dict(key, **estimate(args, mode, tc, key.get("tl"), key.get("th")),
                       n_accepted=n, errors=e, ler=ler, accept_of_postselected=n / len(gc),
                       ler_select=lA, errors_select=eA, n_select=nA, ler_eval=lB, errors_eval=eB, n_eval=nB)
            data[mode].append(row)
            data[mode].sort(key=lambda r: (r["tc"], r.get("th", 0)))
            out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(data, indent=1))
            print(f"{mode:>6} tc {tc:g}{'' if mode == 'single' else f' th {th:g}'}: prep {row['total_us']:6.2f} us "
                  f"(gate {row['gate_us']:.2f} control {row['control_us']:.2f} decode {row['decode_us']:.2f}) "
                  f"LER {ler:.2e} ({e} errs of {n})")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
