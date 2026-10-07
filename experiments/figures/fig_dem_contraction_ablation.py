#!/usr/bin/env python3
"""DEM-contraction ablation as ONE double-column figure: four 3D bar panels in a row,
(a) d_cultiv=3 MB cycles, (b) d_cultiv=5 MB cycles, (c) d_cultiv=3 pymatching µs,
(d) d_cultiv=5 pymatching µs; three bars per (p, d_escape) cell (complete / mask on complete
DEM / contracted DEM), bar colour = P(g_p = g_c). One shared legend and one shared colour
bar. Data loading is shared with plot_contraction_3d.py (stored results only).

    python experiments/figures/fig_dem_contraction_ablation.py [--size 7.0 2.1]
    -> experiments/result/mask_ablation/figures/contraction_grid.{pdf,png}
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
from matplotlib import cm, colors                                          # noqa: E402
from matplotlib.patches import Patch                                       # noqa: E402
from mpl_toolkits.mplot3d import proj3d                                    # noqa: E402

from contraction_data import load_cell, KINDS                              # noqa: E402
from plot_style import INK                                                 # noqa: E402
from naming import fmt_p, parse_name                                       # noqa: E402

HATCH = {"complete": "", "masked": "///", "contracted": "..."}
HATCH_PAIR = {"complete": "", "masked": "/" * 14, "contracted": "." * 14}   # bars are ~0.045 in wide: a hatch needs >~60 lines/in to show on a face
PANELS = [(3, "mb"), (5, "mb"), (3, "pymatching"), (5, "pymatching")]


def draw_panel(ax, cells, *, norm, cmap, bar, gap, elev, azim, box=(1.25, 1.55, 0.65), zoom=0.98, ax_fs=1.0, p_pos=(0.10, 0.04),
               hatch=HATCH, edge_lw=0.25, edge_col=(0.15, 0.15, 0.15, 0.9), rasterize=False):
    ps = sorted({c["p"] for c in cells}); d2s = sorted({c["d2"] for c in cells})
    XP = 5.0
    xr = (len(ps) - 0.1) * XP; yr = len(d2s)
    ux, uy = box[0] / xr, box[1] / yr
    W, DY = bar / ux, bar / uy; ST = W * (1.0 + gap)
    boxes = []
    for c in cells:
        xi, yi = ps.index(c["p"]), d2s.index(c["d2"])
        for k, (kind, _) in enumerate(KINDS):
            x0, y0 = xi * XP + (k - 1) * ST - W / 2, yi - DY / 2
            bars = ax.bar3d(x0, y0, 0, W, DY, c["lat"][kind], color=cmap(norm(c["peq"][kind])),
                            edgecolor=edge_col, linewidth=edge_lw, shade=True)
            bars.set_hatch(hatch[kind])
            if rasterize:                      # dense hatches make a vector PDF enormous (one pattern per face colour)
                bars.set_rasterized(True)
            boxes.append((x0, x0 + W, y0, y0 + DY, bars))
    for i in range(len(boxes)):                    # footprints must be disjoint
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            assert not (a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]), "footprints overlap"
    ax.set_xticks([i * XP for i in range(len(ps))]); ax.set_xticklabels([fmt_p(p) for p in ps], fontsize=5.2 * ax_fs)
    ax.set_yticks(range(len(d2s))); ax.set_yticklabels([str(d) for d in d2s], fontsize=5.4 * ax_fs)
    ax.tick_params(axis="z", labelsize=5.4 * ax_fs, pad=0.5); ax.tick_params(axis="x", pad=-3); ax.tick_params(axis="y", pad=-2)
    ax.set_xlabel(""); ax.set_ylabel("$d_{\\mathrm{escape}}$", labelpad=-3, fontsize=6.5 * ax_fs)
    ax.text2D(*p_pos, "$p$", transform=ax.transAxes, ha="center", va="center", fontsize=6.5 * ax_fs)   # 3D xlabel is clipped; place it by hand
    ax.set_zlabel("")
    from matplotlib.ticker import MaxNLocator
    ax.zaxis.set_major_locator(MaxNLocator(4))
    ax.set_xlim(-XP * 0.45, (len(ps) - 0.55) * XP); ax.set_ylim(-0.5, len(d2s) - 0.5); ax.set_zlim(0, None)
    ax.set_box_aspect(box, zoom=zoom)
    ax.view_init(elev=elev, azim=azim)
    M = ax.get_proj()
    depth = lambda x, y: proj3d.proj_transform(x, y, 0.0, M)[2]
    sign = 1.0 if depth(0.0, len(d2s) - 1) > depth(0.0, 0.0) else -1.0
    for x0, x1, y0, y1, bars in boxes:             # global back-to-front painter order under this camera
        d = sign * depth((x0 + x1) / 2, (y0 + y1) / 2)
        bars.do_3d_projection = (lambda orig=bars.do_3d_projection, d=d: (orig(), d)[1])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False; axis.pane.set_edgecolor((0.85, 0.85, 0.85, 1)); axis._axinfo["grid"]["linewidth"] = 0.3


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", type=pathlib.Path, default=paths.RESULT / "runtime_estimation" / "summary_csq14f29.json")
    ap.add_argument("--fig-dir", type=pathlib.Path, default=paths.RESULT / "mask_ablation" / "figures")
    ap.add_argument("--size", type=float, nargs=2, default=(7.0, 1.9))
    ap.add_argument("--elev", type=float, default=30); ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--cmin", type=float, default=0.2)
    ap.add_argument("--p-max", type=float, default=1.2e-3)
    ap.add_argument("--bar", type=float, default=0.045); ap.add_argument("--gap", type=float, default=0.6)
    ap.add_argument("--d1", type=int, nargs="+", default=[3, 5], help="cultivation distances to include (one -> single-column pair)")
    ap.add_argument("--name", default=None, help="output stem (default contraction_grid, or contraction_pair_d1-<d1> for one d1)")
    args = ap.parse_args()
    panels = [(d1, src) for src in ("mb", "pymatching") for d1 in args.d1]
    if len(args.d1) == 2:                                          # keep the documented order for the full grid
        panels = [(3, "mb"), (5, "mb"), (3, "pymatching"), (5, "pymatching")]
    n = len(panels)
    if n == 2 and tuple(args.size) == (7.0, 1.9):                          # single-column default
        args.size = [3.13, 1.5]
    stem = args.name or ("contraction_grid" if n == 4 else f"contraction_pair_d1-{args.d1[0]}")
    summ = json.loads(args.summary.read_text())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.linewidth": 0.5 if n == 4 else 0.3})
    norm = colors.Normalize(vmin=args.cmin, vmax=1.0); cmap = plt.get_cmap("plasma")
    fig = plt.figure(figsize=tuple(args.size))
    gs = fig.add_gridspec(1, n, left=0.0, right=0.925 if n == 4 else 0.80, bottom=0.07, top=0.99 if n == 4 else 1.0, wspace=0.30 if n == 4 else 0.32)
    for i, (d1, source) in enumerate(panels):
        cells = [load_cell(n, v["partial_tag"], source) for n, v in summ.items()
                 if parse_name(n)[0] == d1 and parse_name(n)[2] <= args.p_max and v.get("partial_tag")]
        cells = [c for c in cells if c]
        ax = fig.add_subplot(gs[0, i], projection="3d")
        draw_panel(ax, cells, norm=norm, cmap=cmap, bar=args.bar, gap=args.gap, elev=args.elev, azim=args.azim,
                   zoom=0.98 if n == 4 else 1.10, ax_fs=1.0 if n == 4 else 0.95,
                   p_pos=(0.10, 0.04) if n == 4 else (0.05, -0.02), hatch=HATCH if n == 4 else HATCH_PAIR,
                   edge_lw=0.25, edge_col=(0.15, 0.15, 0.15, 0.9), rasterize=n == 2)
        unit = "MB cycles" if source == "mb" else "pymatching μs"
        if n == 4:
            ax.text2D(0.5, 0.94, f"({'abcd'[i]}) $d_{{\\mathrm{{cultiv}}}}={d1}$ — {unit}", transform=ax.transAxes,
                      ha="center", va="top", fontsize=6.8, color=INK)
        else:   # pair: title in the empty upper-left corner of the 3D box (the tall bars sit at the right)
            ax.text2D(0.02, 0.92, f"({'ab'[i]}) {unit}", transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=INK)
    # shared legend (top) and colour bar (right), in the style of the bar figures
    hat = HATCH if n == 4 else HATCH_PAIR
    handles = [Patch(facecolor="#e8e8e8", edgecolor=(0.15, 0.15, 0.15), linewidth=0.6, hatch=hat[k] * (2 if n == 4 else 1), label=lab) for k, lab in KINDS]
    fig.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.46 if n == 4 else 0.42, 0.955 if n == 4 else 0.97),
               frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, fontsize=6.5 if n == 4 else 5.8,
               handlelength=1.6, handleheight=1.0, columnspacing=1.2 if n == 4 else 0.7, handletextpad=0.5, borderpad=0.35,
               borderaxespad=0.0)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.972 if n == 4 else 0.905, 0.27, 0.010 if n == 4 else 0.016, 0.42])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0.2, 0.4, 0.6, 0.8, 1.0]); cb.set_label("$P(g_p = g_c)$", fontsize=6.5, labelpad=2)
    cb.ax.tick_params(labelsize=5.6, length=2); cb.outline.set_linewidth(0.6)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):      # rasterized bars are embedded at 900 dpi in the PDF; axes, text and legend stay vector
        fig.savefig(args.fig_dir / f"{stem}.{ext}", dpi=900 if ext == "pdf" else 300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {args.fig_dir / stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
