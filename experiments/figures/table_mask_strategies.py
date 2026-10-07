#!/usr/bin/env python3
"""Paper table for the mask-heuristic ablation: one row per (circuit, mask type) with the
mask size, the DEM crossing mass S(M), and the partial-gap accuracy on the complete DEM
P(g_p = g_c), P(g_p > g_c), P(g_p < g_c) — from run_mask_heuristics.py results.

    python experiments/figures/table_mask_strategies.py experiments/result/mask_ablation/heuristics_*_N336.json [--pair masked|contracted]
    -> prints Markdown (and writes <fig-dir>/mask_heuristics_table.md / .csv)
"""
import argparse
import csv
import json
import pathlib
import re

import sys
sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
LABEL = {"handmade": "Handmade region", "gap_impact": "Gap-impact list", "qexp": "Q_exp", "qexp_cl": "Q_exp + closure"}
ORDER = ["handmade", "gap_impact", "qexp", "qexp_cl"]
PAIR = {"masked": "masked_on_complete_dem_vs_complete", "contracted": "contracted_vs_complete"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=pathlib.Path, nargs="+")
    ap.add_argument("--pair", choices=list(PAIR), default="masked")
    ap.add_argument("--out-dir", type=pathlib.Path, default=paths.RESULT / "mask_ablation" / "figures")
    ap.add_argument("--name", default="mask_heuristics_table")
    args = ap.parse_args()
    rows = []
    for f in sorted(args.results, key=lambda p: int(re.search(r"d2=(\d+)", p.name)[1])):
        d = json.loads(f.read_text()); m = re.match(r"end2end_d1=(\d+)_d2=(\d+)_.*_p=([0-9.e-]+)_", d["circuit"])
        for kind in ORDER:
            if kind not in d["variants"]:
                continue
            v = d["variants"][kind]; s = v[PAIR[args.pair]]
            rows.append(dict(d1=int(m[1]), d2=int(m[2]), p=float(m[3]), mask=LABEL[kind], detectors=v["mask_detectors"],
                             S=v["S_crossing_mass"], p_equal=s["p_equal"], p_above=s["p_above"], p_below=s["p_below"],
                             shots=v["shots"], profiling_shots=v.get("profiling_shots"), tag=v["partial"]))
    lines = ["| d1 | d2 | p | mask | detectors | S(M) | P(g_p = g_c) | P(g_p > g_c) | P(g_p < g_c) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['d1']} | {r['d2']} | {r['p']:g} | {r['mask']} | {r['detectors']} | {r['S']:.2f} | "
                     f"{r['p_equal']:.3f} | {r['p_above']:.3f} | {r['p_below']:.3f} |")
    md = "\n".join(lines) + "\n"
    print(md)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"{args.name}.md").write_text(md)
    with open(args.out_dir / f"{args.name}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"saved {args.out_dir / args.name}.{{md,csv}}  (evaluation shots per mask: {sorted({r['shots'] for r in rows})}, "
          f"Q profiling shots: {sorted({r['profiling_shots'] for r in rows if r['profiling_shots']})})")


if __name__ == "__main__":
    main()
