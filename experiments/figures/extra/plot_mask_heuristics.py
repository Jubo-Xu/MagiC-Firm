#!/usr/bin/env python3
"""Mask-ablation step 1 figure: P(g_m = g_c) of four mask heuristics at equal size,
one group per circuit, from run_mask_heuristics.py outputs. Plot only.

    python experiments/figures/extra/plot_mask_heuristics.py experiments/result/mask_ablation/heuristics_<cfgA>.json \
        experiments/result/mask_ablation/heuristics_<cfgB>.json [--pair masked|contracted]
    -> experiments/result/mask_ablation/figures/mask_heuristics[_contracted].{pdf,png,csv}
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

from plot_style import INK, INK2, GRID                                     # noqa: E402
from naming import fmt_p                                                   # noqa: E402

KINDS = [("handmade", "Handmade region", "#9aa3ad"), ("gap_impact", "Gap-impact list", "#eb6834"),
         ("qexp", "$Q_{exp}$", "#2a78d6"), ("qexp_cl", "$Q_{exp}$ + closure", "#1baf7a")]
PAIR = {"masked": "masked_on_complete_dem_vs_complete", "contracted": "contracted_vs_complete"}


def group_label(name):
    m = re.match(r"end2end_d1=(\d+)_d2=(\d+)_.*_p=([0-9.e-]+)_", name)
    return f"$d_1$={m[1]}, $d_2$={m[2]}\n$p$={fmt_p(float(m[3]))}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=pathlib.Path, nargs="+")
    ap.add_argument("--pair", choices=list(PAIR), default="masked",
                    help="masked: mask on the complete DEM vs complete (default); contracted: contracted decoder vs complete")
    ap.add_argument("--fig-dir", type=pathlib.Path, default=paths.RESULT / "mask_ablation" / "figures")
    ap.add_argument("--size", type=float, nargs=2, default=(3.5, 2.4))
    ap.add_argument("--kinds", nargs="+", default=["handmade", "gap_impact", "qexp", "qexp_cl"],
                    help="which masks to draw, in order (default: all four)")
    args = ap.parse_args()
    datas = [json.loads(p.read_text()) for p in args.results]
    kinds = [k for k in KINDS if k[0] in args.kinds]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
                         "legend.fontsize": 7.0, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=tuple(args.size))
    w = 0.75 / len(kinds); rows = []
    for g, d in enumerate(datas):
        for i, (kind, lab, col) in enumerate(kinds):
            v = d["variants"][kind]; s = v[PAIR[args.pair]]
            x = g + (i - (len(kinds) - 1) / 2) * w
            ax.bar(x, s["p_equal"], w * 0.92, color=col, edgecolor="white", linewidth=0.6, zorder=2,
                   label=lab if g == 0 else None)
            ax.text(x, s["p_equal"] + 0.012, f"{s['p_equal']:.2f}", ha="center", va="bottom", fontsize=5.8, color=INK)
            rows.append(dict(circuit=d["circuit"], kind=kind, tag=v["partial"], mask_detectors=v["mask_detectors"],
                             contracted_detectors=v["contracted_detectors"], shots=v["shots"], **s))
    ax.set_xticks(range(len(datas)), [group_label(d["circuit"]) for d in datas])
    ax.set_ylabel("$P(g_p = g_c)$")
    ax.set_ylim(0, 1.0)
    ax.text(0.99, 0.98, f"masks of equal size: {', '.join(str(d['size_target']) for d in datas)} detectors",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=INK2)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, length=2.5)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=len(kinds), handlelength=1.2,
              borderaxespad=0.0, columnspacing=0.8, handletextpad=0.4)
    fig.tight_layout(pad=0.4)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = "mask_heuristics" + ("" if args.pair == "masked" else "_contracted")
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    with open(args.fig_dir / f"{stem}.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"saved {args.fig_dir / stem}.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
