#!/usr/bin/env python3
"""Mask-ablation step 2 figure: partial-gap accuracy vs mask coverage, one panel per d_cultiv,
shared y axis, from the stored mask-size table (mask_size_table.csv; collect_mask_ablation.py
results, masked syndrome on the complete DEM, 200k post-selected shots per mask).

    x  = retained detectors |M| / |D| in %   (the CLOSED mask size, not K_Q)
    y  = P(g_p = g_c), P(g_p < g_c), P(g_p > g_c)
The two campaign masks (d_cultiv=3: K_Q=250, d_cultiv=5: K_Q=450) get a black outer ring.

    python experiments/figures/fig_mask_size_sensitivity.py [--csv .../mask_size_table.csv] [--size 3.35 3.4]
    -> experiments/result/mask_ablation/figures/mask_size.{pdf,png}
"""
import argparse
import csv
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402

from plot_style import INK, INK2, GRID                                     # noqa: E402  (applies the base rcParams)

# same validated palette as the stacked-bar figures (blue / orange / green), distinct markers
SERIES = [("p_equal", "$P(g_p = g_c)$", "#2a78d6", "o", "-"),
          ("p_below", "$P(g_p < g_c)$", "#eb6834", "s", "--"),
          ("p_above", "$P(g_p > g_c)$", "#1baf7a", "^", ":")]
CAMPAIGN = {3: 250, 5: 450}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=pathlib.Path, default=paths.RESULT / "mask_ablation" / "mask_size_table.csv")
    ap.add_argument("--fig-dir", type=pathlib.Path, default=paths.RESULT / "mask_ablation" / "figures")
    ap.add_argument("--size", type=float, nargs=2, default=(3.35, 1.55), help="figure size (in); 3.35 = one ACM column")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.csv)))
    data = {}
    for r in rows:
        d1 = int(r["d1"]); cov = 100.0 * int(r["detectors"]) / int(r["total"])
        data.setdefault(d1, []).append(dict(K=int(r["K_Q"]), cov=cov, **{k: float(r[k]) for k in ("p_equal", "p_below", "p_above")}))
    for d1 in data:
        data[d1].sort(key=lambda x: x["cov"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 7.5,
                         "legend.fontsize": 6.4, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.size), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, d1, letter in zip(axes, (3, 5), ("a", "b")):
        pts = data[d1]
        x = [p["cov"] for p in pts]
        for key, lab, col, mk, ls in SERIES:
            ax.plot(x, [p[key] for p in pts], color=col, ls=ls, lw=1.1, marker=mk, ms=3.2,
                    markeredgewidth=0.0, zorder=3, label=lab)
        camp = next(p for p in pts if p["K"] == CAMPAIGN[d1])
        for key, *_ in SERIES:                          # thin black ring on the campaign mask's points
            ax.plot([camp["cov"]], [camp[key]], ls="none", marker="o", ms=6.5, markerfacecolor="none",
                    markeredgecolor=INK, markeredgewidth=0.7, zorder=4)
        if d1 == 3:
            ax.annotate(f"$K_Q$={CAMPAIGN[d1]}", (camp["cov"], camp["p_equal"]), xytext=(-5, -8), textcoords="offset points",
                        ha="left", va="top", fontsize=6.2, color=INK)
        else:
            ax.annotate(f"$K_Q$={CAMPAIGN[d1]}", (camp["cov"], camp["p_equal"]), xytext=(4, -7), textcoords="offset points",
                        ha="left", va="top", fontsize=6.2, color=INK)
        ax.text(0.03, 0.96, f"({letter}) $d_{{\\mathrm{{cultiv}}}}={d1}$", transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=INK)
        ax.set_ylim(-0.04, 1.04); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(20, 100); ax.set_xticks([20, 40, 60, 80] if d1 == 3 else [40, 60, 80, 100])   # no colliding 100 | 20
        ax.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, length=2, pad=1.5)
        ax.set_xlabel("Retained detectors (%)", labelpad=1.5)
    axes[0].set_ylabel("Shot fraction", labelpad=2)
    axes[1].tick_params(axis="y", length=0)
    handles = [Line2D([], [], color=c, ls=ls, lw=1.1, marker=mk, ms=3.2, markeredgewidth=0, label=lab) for _, lab, c, mk, ls in SERIES]
    axes[0].legend(handles=handles, loc="center", bbox_to_anchor=(0.66, 0.47), frameon=True, fancybox=False,
                   edgecolor=GRID, framealpha=0.95, handlelength=2.0, labelspacing=0.3, borderpad=0.35, handletextpad=0.5)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.2, top=0.98, wspace=0.08)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"mask_size.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {args.fig_dir / 'mask_size'}.{{pdf,png}}")
    for d1, pts in data.items():
        print(f"  d_cultiv={d1}: " + "  ".join(f"K_Q {p['K']}: {p['cov']:.1f}%" for p in pts))


if __name__ == "__main__":
    main()
