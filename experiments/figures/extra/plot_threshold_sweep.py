#!/usr/bin/env python3
"""Preparation-time breakdown vs the complete-gap threshold t_c, from a stored
threshold sweep (collect_threshold_sweep.py): one stacked Gate / Control /
Gap-decode bar per t_c with the accepted-shot LER written above. Plot only —
no statistics are recomputed.

    python experiments/figures/extra/plot_threshold_sweep.py <threshold_sweep_*.json> [--mode single|pcp] [--size W H]
    -> experiments/result/runtime_estimation/figures/tc_sweep_<config>_<mode>_fb<ns>[_cs..].{pdf,png,csv}
"""
import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.patches import Patch                                       # noqa: E402

from plot_style import STACK, BAR_W, INK, INK2, GRID, PAPER_RC, PAPER      # noqa: E402
from naming import fmt_p, short_name, circuit_label                        # noqa: E402
from sweep import draw_tc_sweep, KEYS                                      # noqa: E402



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", type=pathlib.Path, help="threshold_sweep_*.json from collect_threshold_sweep.py")
    ap.add_argument("--mode", choices=["single", "pcp"], default="single")
    ap.add_argument("--fig-dir", type=pathlib.Path, default=paths.RESULT / "runtime_estimation" / "figures")
    ap.add_argument("--size", type=float, nargs=2, default=(3.5, 2.7), help="figure size (in)")
    ap.add_argument("--tc-max", type=float, default=1e9, help="drop bars with t_c above this")
    ap.add_argument("--suffix", default="", help="appended to the figure file name")
    args = ap.parse_args()
    data = json.loads(args.sweep.read_text())
    rows = sorted((r for r in data[args.mode] if r["tc"] <= args.tc_max), key=lambda r: r["tc"])
    if not rows:
        sys.exit(f"no {args.mode} rows in {args.sweep}")

    plt.rcParams.update(PAPER_RC)
    fig, ax = plt.subplots(figsize=tuple(args.size))
    draw_tc_sweep(ax, rows)
    fig.tight_layout(pad=0.4)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"tc_sweep_{short_name(data['circuit'])}_{args.mode}_{args.sweep.stem.split('_', 2)[-1]}{args.suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    with open(args.fig_dir / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([f"{v:.4g}" if isinstance(v, float) else v for v in r.values()])
    print(f"saved {args.fig_dir / stem}.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
