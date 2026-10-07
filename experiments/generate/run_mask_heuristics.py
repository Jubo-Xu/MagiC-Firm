#!/usr/bin/env python3
"""Mask-ablation step 1: four mask heuristics at (about) the same number of
detectors, compared by partial-gap accuracy on the complete DEM
(collect_mask_ablation.py: P(g_m = g_c), P(<), P(>); contracted numbers stored too).

Per circuit (size target N = the campaign's Q_exp+closure mask):
    qexp_cl      Q_exp(K_Q) + closure           (the campaign mask; --ref tag)
    qexp         Q_exp(N)                       (no closure)
    gap_impact   top-N detectors of the gap-impact ranking: detectors sorted by the mean
                 complete gap of the accepted shots in which they fired, ascending
                 (run_gap_sensitivity.py output), among detectors that fired >= --min-count times
    handmade     rectangular space-time region in the escape stage (--base region), geometry
                 searched for the mask size closest to N
All four are contracted identically (wc 15) so the mask is the only variable.

    python experiments/generate/run_mask_heuristics.py experiments/data/circuits/<config> --ref partial_qexp250_cl_wc15 \
        [--shots 200000 --workers 32]
    -> experiments/result/mask_ablation/heuristics_<config>.json (+ masks under the circuit folder)
"""
import argparse
import glob
import json
import pathlib
import subprocess
import sys

import numpy as np
import stim

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from partial_mask_builder import PartialMaskBuilder                        # noqa: E402
from campaign import essentials_flags, WC, PY                              # noqa: E402

RESULT = paths.RESULT / "mask_ablation"


def run(cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.call([str(c) for c in cmd])


def gap_impact_ids(folder, n, min_count):
    files = sorted(glob.glob(str(paths.RESULT / "gap_sensitivity" / folder.name / "overall_gap_sens_s*_n*.json")),
                   key=lambda f: int(f.split("_n")[-1].split(".")[0]))
    if not files:
        sys.exit(f"no gap-sensitivity file for {folder.name}: run experiments/generate/run_gap_sensitivity.py first")
    j = json.loads(pathlib.Path(files[-1]).read_text())["per_detector"]
    count = np.asarray(j["count"]); mg = np.asarray(j["mean_gap_when_active"], dtype=float)
    ok = count >= min_count
    order = np.argsort(np.where(ok, mg, np.inf))
    ids = order[:n]
    assert ok[ids].all(), "fewer eligible detectors than the size target"
    return sorted(int(i) for i in ids), pathlib.Path(files[-1]).name


def closure_size_search(circuit, folder, n, k_lo=40, k_hi=None):
    """K_Q whose Q_exp(K_Q)+closure mask has the size closest to n (closure only adds, and the
    closed size is monotone in K_Q up to noise -> bisection on the closed size)."""
    from gen_partial_mask_dem import discover_essentials
    b = PartialMaskBuilder(circuit=circuit)
    b.load_essentials_from_json(discover_essentials("logical_ambiguity", folder, None), type="logical_ambiguity")
    def closed(K):
        m, _, _ = b.logical_ambiguity_base_mask(K, q_type="exp")
        m, _, _, _ = b.closure_mask(m, tau=0.0, return_info=True)
        return int(m.sum())
    k_hi = k_hi or n
    best = None
    while k_hi - k_lo > 1:
        K = (k_lo + k_hi) // 2; s = closed(K)
        if best is None or abs(s - n) < abs(best[1] - n): best = (K, s)
        if s < n: k_lo = K
        else: k_hi = K
    for K in (k_lo, k_hi):
        s = closed(K)
        if abs(s - n) < abs(best[1] - n): best = (K, s)
    return best


def region_search(circuit, n, tmax=14, lmax=24):
    """Rectangle (left/top ranks from 1, t from 1) whose mask size is closest to n:
    squares over (side, t_end) first, then the aspect ratio refined around the best."""
    b = PartialMaskBuilder(circuit=circuit)
    def size(l_end, p_end, t_end):
        try:
            m, _, _ = b.build_partial_region_mask(left_len_start=1, left_len_end=l_end, top_len_start=1,
                                                  top_len_end=p_end, t_start=1, t_end=t_end)
            return int(m.sum())
        except Exception:
            return None
    cands = {}
    for t_end in range(2, tmax + 1):
        for side in range(2, lmax + 1):
            s = size(side, side, t_end)
            if s is not None:
                cands[(side, side, t_end)] = s
    best = min(cands, key=lambda k: (abs(cands[k] - n), k))
    l0, p0, t0 = best
    for l_end in range(max(2, l0 - 3), l0 + 4):
        for p_end in range(max(2, p0 - 3), p0 + 4):
            if (l_end, p_end, t0) not in cands:
                s = size(l_end, p_end, t0)
                if s is not None:
                    cands[(l_end, p_end, t0)] = s
    best = min(cands, key=lambda k: (abs(cands[k] - n), k))
    return (cands[best],) + best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--ref", required=True, help="campaign mask tag, e.g. partial_qexp250_cl_wc15 (sets the size target)")
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--min-count", type=int, default=50, help="gap_impact: minimum firings for eligibility")
    ap.add_argument("--size", type=int, default=None,
                    help="fixed size target N (default: the size of --ref). When it differs from --ref's size, the "
                         "Q_exp+closure variant is rebuilt by searching K_Q so that the CLOSED mask lands on N")
    ap.add_argument("--kinds", nargs="+", default=["qexp_cl", "qexp", "gap_impact", "handmade"],
                    help="variants to build/evaluate (gap_impact needs run_gap_sensitivity.py output)")
    args = ap.parse_args()
    folder = args.folder.resolve()
    ref_stats = json.loads((folder / args.ref / "stats.json").read_text())
    n = args.size or int(ref_stats["mask"]["mask_sum"])
    circuit = stim.Circuit.from_file(next(folder.glob("*.stim")))
    print(f"{folder.name}: size target N = {n} ({'fixed' if args.size else 'from ' + args.ref})")

    variants = {}
    if "qexp_cl" in args.kinds:
        if n == int(ref_stats["mask"]["mask_sum"]):
            variants["qexp_cl"] = args.ref
        else:                                   # search K_Q: closed size closest to N
            K, size = closure_size_search(circuit, folder, n)
            print(f"closure search: K_Q {K} -> closed mask {size} detectors (target {n})")
            tag = f"partial_qexp{K}_cl_wc{WC:g}"
            if not (folder / tag / "contracted.dem").exists():
                run([PY, paths.ALGORITHMS / "gen_partial_mask_dem.py", "--out", folder, "--base-size", K, "--q-type", "exp",
                     "--closure", "--contract", "--weight-cutoff", WC, *essentials_flags(folder)])
            variants["qexp_cl"] = tag
    out = {"circuit": folder.name, "size_target": n, "ref": args.ref, "shots": args.shots, "variants": {}}
    if "qexp" in args.kinds:                    # Q_exp alone at N
        if not (folder / f"partial_qexp{n}_wc{WC:g}" / "contracted.dem").exists():
            run([PY, paths.ALGORITHMS / "gen_partial_mask_dem.py", "--out", folder, "--base-size", n, "--q-type", "exp",
                 "--contract", "--weight-cutoff", WC, *essentials_flags(folder)])
        variants["qexp"] = f"partial_qexp{n}_wc{WC:g}"
    if "gap_impact" in args.kinds:              # gap-impact list
        ids, sens_file = gap_impact_ids(folder, n, args.min_count)
        lst = folder / f"gap_impact_top{n}.json"; lst.write_text(json.dumps(ids))
        before = set(p.name for p in folder.glob("partial_list*"))
        run([PY, paths.ALGORITHMS / "gen_partial_mask_dem.py", "--out", folder, "--base", "list", "--detectors", lst,
             "--contract", "--weight-cutoff", WC])
        new = [p.name for p in folder.glob(f"partial_list{n}-*_wc{WC:g}")]
        variants["gap_impact"] = sorted(new, key=lambda t: (t in before, t))[0]
        out["gap_impact"] = {"sensitivity_file": sens_file, "min_count": args.min_count}
    if "handmade" in args.kinds:                # handmade region
        s, l_end, p_end, t_end = region_search(circuit, n)
        print(f"region search: left 1-{l_end} top 1-{p_end} t 1-{t_end} -> {s} detectors (target {n})")
        run([PY, paths.ALGORITHMS / "gen_partial_mask_dem.py", "--out", folder, "--base", "region",
             "--left-len", 1, l_end, "--top-len", 1, p_end, "--t", 1, t_end, "--mode", "rectangle",
             "--contract", "--weight-cutoff", WC])
        variants["handmade"] = f"partial_l1-{l_end}_t1-{p_end}_ts1-{t_end}_wc{WC:g}"
        out["handmade"] = {"left_len": [1, l_end], "top_len": [1, p_end], "t": [1, t_end]}
    for kind, tag in variants.items():
        assert (folder / tag / "contracted.dem").exists(), f"{tag} missing"
        rc = run([PY, paths.EXPERIMENTS / "generate" / "collect_mask_ablation.py", folder, "--partial", tag,
                  "--shots", args.shots, "--workers", args.workers])
        assert rc == 0
        meta = json.loads((paths.RESULT / "mask_ablation" / "accuracy" / folder.name / f"{tag}_s0.json").read_text())
        st = json.loads((folder / tag / "stats.json").read_text())["mask"]
        meta["S_crossing_mass"] = st["S_crossing_mass"]; meta["mask_spec"] = st["base"]; meta["ops"] = st.get("ops", [])
        meta["profiling_shots"] = st["base"].get("essentials_shots")
        out["variants"][kind] = meta
        m = meta["masked_on_complete_dem_vs_complete"]
        print(f"  {kind:10s} {tag:45s} mask {meta['mask_detectors']:4d}  S {meta['S_crossing_mass']:.2f}  P(=) {m['p_equal']:.4f}  "
              f"P(<) {m['p_below']:.4f}  P(>) {m['p_above']:.4f}")
    RESULT.mkdir(parents=True, exist_ok=True)
    suffix = f"_N{n}" if args.size else ""
    (RESULT / f"heuristics_{folder.name}{suffix}.json").write_text(json.dumps(out, indent=1))
    print(f"saved {RESULT / f'heuristics_{folder.name}{suffix}.json'}")


if __name__ == "__main__":
    main()
