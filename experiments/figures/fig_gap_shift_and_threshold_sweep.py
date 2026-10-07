#!/usr/bin/env python3
"""Paper figure: (a) gap-sensitivity deviation maps of three code slices (ticks) stacked
vertically with one shared colour bar, (b) the preparation-latency breakdown vs t_c
(plot_threshold_sweep.draw_tc_sweep). Same style for both panels. Plot only, from stored
results (run_gap_sensitivity.py JSON, collect_threshold_sweep.py JSON).

    python experiments/figures/fig_gap_shift_and_threshold_sweep.py \
        --sens experiments/result/gap_sensitivity/<cfg>/overall_gap_sens_s0_n2000000.json \
        --sweep experiments/result/runtime_estimation/<cfg>/threshold_sweep_<tag>_fb3100.json \
        [--ticks 48 58 68] [--clip-quantile 0.98] [--cmap RdBu] [--size 7.0 2.7]
    -> experiments/result/runtime_estimation/figures/paper_gap_panels_<config>.{pdf,png}
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import gen                                                                  # noqa: E402
import stim                                                                 # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib import colors                                               # noqa: E402
from matplotlib.patches import Polygon                                      # noqa: E402

from sweep import draw_tc_sweep                                            # noqa: E402
from plot_style import PAPER_RC                                            # noqa: E402
from naming import short_name                                              # noqa: E402


def tile_polygon(tile):
    """Vertices (x, y) of a tile: the data qubits ordered around their centroid; a 2-qubit
    boundary tile becomes a half-disc bulging towards the tile's measure qubit / centre."""
    pts = np.array([[q.real, q.imag] for q in tile.data_qubits], dtype=float)
    if len(pts) >= 3:
        c = pts.mean(0); ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
        return pts[np.argsort(ang)]
    a, b = pts; m = (a + b) / 2; r = np.linalg.norm(b - a) / 2
    n = np.array([-(b - a)[1], (b - a)[0]]) / (2 * r)           # unit normal
    c = tile.center() if callable(tile.center) else tile.center
    ref = c if c is not None else (tile.measure_qubit if tile.measure_qubit is not None else None)
    if ref is not None and np.dot(np.array([ref.real, ref.imag]) - m, n) < 0:
        n = -n
    arc = [m + r * (np.cos(t) * (b - a) / (2 * r) + np.sin(t) * n) for t in np.linspace(0, np.pi, 12)]
    return np.array([a] + arc[1:-1] + [b])


def pipeline_image(svg_path, *, width_in, dpi, font_scale=1.0):
    """draw.io SVG -> RGBA array (cairosvg). The circuit labels are rewritten as bold |Ψ⟩_L, |T⟩_L
    and MSC (subscript via tspan), draw.io's 'Text is not SVG' fallback is removed, and the font is
    one the rasterizer has (DejaVu Sans, the figure font)."""
    import io, re
    import cairosvg
    from PIL import Image
    txt = pathlib.Path(svg_path).read_text()
    txt = re.sub(r"<text[^>]*>Text is not SVG - cannot display</text>", "", txt)
    txt = re.sub(r'font-family="[^"]*"', 'font-family="DejaVu Sans" font-weight="bold"', txt)   # bold everything
    if font_scale != 1.0:
        txt = re.sub(r'font-size="([0-9.]+)px"', lambda m: f'font-size="{float(m.group(1)) * font_scale:.1f}px"', txt)
    def ket(m, sym):
        head = m.group(1)
        return f'{head}|{sym}⟩<tspan dx="-1.5" dy="3.5" font-size="8px">L</tspan></text>'
    txt = re.sub(r'(<text[^>]*>)\|𝜓⟩L</text>', lambda m: ket(m, "Ψ"), txt)
    txt = re.sub(r'(<text[^>]*>)\|𝑇⟩L</text>', lambda m: ket(m, "T"), txt)
    txt = re.sub(r'(<text[^>]*>)Prepare\.\.\.</text>',
                 lambda m: m.group(1) + "MSC</text>", txt)
    png = cairosvg.svg2png(bytestring=txt.encode(), output_width=int(width_in * dpi), background_color="white")
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def draw_slice(ax, code, values, counts, norm, cmap, *, grey=(0.55, 0.55, 0.55)):
    for tile in code.tiles:
        d = int(next(iter(tile.flags))) if tile.flags else -1
        if d < 0 or d >= len(values) or counts[d] <= 0:
            col = grey
        else:
            col = cmap(norm(values[d]))
        ax.add_patch(Polygon(tile_polygon(tile), closed=True, facecolor=col, edgecolor="black", linewidth=0.35))
    xs = [q.real for t in code.tiles for q in t.data_qubits]; ys = [q.imag for t in code.tiles for q in t.data_qubits]
    ax.set_xlim(min(xs) - 0.6, max(xs) + 0.6); ax.set_ylim(max(ys) + 0.6, min(ys) - 0.6)     # y down, as in the SVGs
    ax.set_aspect("equal"); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sens", type=pathlib.Path, required=True)
    ap.add_argument("--sweep", type=pathlib.Path, required=True)
    ap.add_argument("--ticks", type=int, nargs="+", default=[48, 58, 68])
    ap.add_argument("--clip-quantile", type=float, default=0.98)
    ap.add_argument("--cmap", default="RdBu")
    ap.add_argument("--size", type=float, nargs=2, default=(7.0, 3.7))
    ap.add_argument("--fig-dir", type=pathlib.Path, default=paths.RESULT / "runtime_estimation" / "figures")
    ap.add_argument("--pipeline", type=pathlib.Path, default=None,
                    help="draw.io SVG of the consumption pipeline, added as panel (c) below (a)/(b), full width, framed")
    ap.add_argument("--pipeline-title", default="Logical Magic State Consumption Pipeline")
    ap.add_argument("--pipeline-dpi", type=int, default=600, help="rasterization resolution of the SVG")
    ap.add_argument("--pipeline-scale", type=float, default=0.985, help="content scale of the pipeline inside its box")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="multiply every font size (for a single-column placement: the figure is printed at ~0.47 of "
                         "its 7 in width, so 2.1 gives 6-7 pt in print)")
    ap.add_argument("--svg-font-scale", type=float, default=1.0, help="multiply the font sizes inside the pipeline SVG")
    ap.add_argument("--suffix", default="", help="appended to the output file name")
    args = ap.parse_args()
    FS = args.font_scale

    doc = json.loads(args.sens.read_text()); pd = doc["per_detector"]
    counts = np.asarray(pd["count"], float); mg = np.asarray(pd["mean_gap_when_active"], float)
    mean = doc["summary"]["shots"]["gap"]["mean"]
    active = counts > 0; dev = np.where(active, mg - mean, 0.0)
    L = float(np.quantile(np.abs(dev[active]), args.clip_quantile))
    norm = colors.Normalize(-L, L, clip=True); cmap = plt.get_cmap(args.cmap)
    circuit = stim.Circuit.from_file(next((paths.CIRCUITS / doc["folder"]).glob("*.stim")))
    codes = {t: c.with_transformed_coords(lambda e: e * (1 + 1j)) for t, c in gen.circuit_to_cycle_code_slices(circuit).items()}   # axis-aligned squares, as in the SVGs
    sweep = json.loads(args.sweep.read_text())
    rows = sorted(sweep["single"], key=lambda r: r["tc"])

    plt.rcParams.update({k: (v * FS if isinstance(v, (int, float)) and 'size' in k else v) for k, v in PAPER_RC.items()})
    pipe_img = None
    if args.pipeline:
        pipe_img = pipeline_image(args.pipeline, width_in=args.size[0], dpi=args.pipeline_dpi, font_scale=args.svg_font_scale)
    W_in, H_in = args.size
    if pipe_img is not None:                                   # extra row for (c): image aspect preserved
        ratio = pipe_img.shape[0] / pipe_img.shape[1]
        SHRINK = args.pipeline_scale                           # content scale inside the (c) box
        pipe_h_in = 0.975 * W_in * ratio * SHRINK * 1.10 + 0.36   # box height + letter/gap room
        H_in += pipe_h_in
    fig = plt.figure(figsize=(W_in, H_in))
    top_frac = 1.0 if pipe_img is None else 1.0 - pipe_h_in / H_in   # fraction of the height used by (a)/(b)
    n = len(args.ticks)
    # layout columns: maps | colour bar (ticks on its left) | annotations | spacer (bar y-label) | bars
    # columns: maps | tick-label gap | colour bar | annotations | spacer (bar y-label) | bars
    gs = fig.add_gridspec(n, 6, width_ratios=[0.60, 0.21, 0.07, 0.47, 0.22, 2.25], wspace=0.0, hspace=0.03,
                          left=0.006, right=0.995, top=1 - 0.02 * (1 - top_frac + 1), bottom=1 - top_frac * (1 - 0.175))
    axb = fig.add_subplot(gs[:, 5])
    draw_tc_sweep(axb, rows, label_fs=7.4 * FS, fs=FS)
    pb = axb.get_position()                      # panel (a) is framed to exactly this height
    map_axes = []
    for i, t in enumerate(args.ticks):
        ax = fig.add_subplot(gs[i, 0]); map_axes.append(ax)
        pm = ax.get_position()                   # inset the maps from the frame (top/bottom rows touch it otherwise)
        ax.set_position([pm.x0 + 0.002, pm.y0 + 0.006, pm.width, pm.height - 0.012])
        draw_slice(ax, codes[t], dev, counts, norm, cmap)
    # colour bar: shorter than the frame so its title and end labels stay inside the frame
    pc = fig.add_subplot(gs[:, 2]).get_position(); fig.delaxes(fig.axes[-1])
    cax = fig.add_axes([pc.x0, pb.y0 + 0.055, pc.width, pb.height - 0.15])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = fig.colorbar(sm, cax=cax)
    ticks = [-L, -L / 2, 0, L / 2, L]
    cb.set_ticks(ticks); cb.set_ticklabels([f"≤{-L:.0f}", f"{-L/2:.0f}", "0", f"+{L/2:.0f}", f"≥+{L:.0f}"])
    cb.ax.yaxis.set_ticks_position("left"); cb.ax.tick_params(labelsize=7.2 * FS, length=2.5, width=0.7, pad=1.5)
    cb.outline.set_linewidth(0.8)
    cb.ax.set_title("Δgap (dB)", fontsize=7.4 * FS, pad=3, loc="center")
    tr = cax.get_yaxis_transform()
    cb.ax.annotate(f"mean gap\n{mean:.1f} dB", xy=(1.0, 0.0), xycoords=("axes fraction", "data"),
                   xytext=(1.45, 0.0), textcoords=("axes fraction", "data"), fontsize=7.2 * FS, ha="left", va="center",
                   arrowprops=dict(arrowstyle="-|>", color="black", lw=0.8, shrinkA=0, shrinkB=0.5))
    cb.ax.text(1.45, L * 0.62, "fired →\ngap above\nmean", transform=tr, fontsize=7.0 * FS, ha="left", va="center", color="#1f4e9c", linespacing=1.05)
    cb.ax.text(1.45, -L * 0.62, "fired →\ngap below\nmean", transform=tr, fontsize=7.0 * FS, ha="left", va="center", color="#8b1a1a", linespacing=1.05)
    # frame around panel (a): same height as the bar axes, same line weight
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    arts = list(cax.texts) + [c for c in cax.get_children() if isinstance(c, matplotlib.text.Annotation)]
    xr = max(inv.transform(t.get_window_extent())[1][0] for t in arts)
    x0 = map_axes[0].get_position().x0 - 0.004; x1 = xr + 0.010
    from matplotlib.patches import Rectangle
    fig.add_artist(Rectangle((x0, pb.y0), x1 - x0, pb.height, transform=fig.transFigure, fill=False,
                             edgecolor="black", linewidth=0.8, zorder=0))
    # panel (a) caption line (fills the gap under the frame, level with the bar x-label) and letters
    fig.text(0.5 * (x0 + x1), pb.y0 - 0.028, "Spatiotemporal Distribution of\nDetector-Conditioned Gap Shifts",
             ha="center", va="top", fontsize=9 * FS, linespacing=1.05)
    yl = pb.y0 - 0.108 * top_frac                # letters: tight under the x-label / caption
    fig.text(0.5 * (x0 + x1), yl, "(a)", ha="center", va="top", fontsize=11 * FS, fontweight="bold")
    fig.text(0.5 * (pb.x0 + pb.x1), yl, "(b)", ha="center", va="top", fontsize=11 * FS, fontweight="bold")
    if pipe_img is not None:                     # (c): full width from the (a) frame to the (b) axes, framed, titled
        letter_h = 0.18 / H_in                   # room for the letter under the (c) frame (figure fraction)
        y_top = yl - 0.058 * top_frac
        w_c = pb.x1 - x0
        h_c = w_c * W_in * ratio * SHRINK * 1.10 / H_in        # box: scaled image height + 10 % breathing room
        y_bot = y_top - h_c
        axc = fig.add_axes([x0, y_bot, w_c, h_c])
        # image centred horizontally at SHRINK of the box width, aspect preserved (axes fraction units)
        iw = SHRINK; ih = 1.0 / 1.10                      # displayed size = SHRINK * box width x its true aspect
        axc.imshow(pipe_img, aspect="auto", interpolation="lanczos",
                   extent=(0.5 - iw / 2, 0.5 + iw / 2, 0.5 - ih / 2 - 0.02, 0.5 + ih / 2 - 0.02))
        axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.set_xticks([]); axc.set_yticks([])
        for sp in axc.spines.values():
            sp.set_color("black"); sp.set_linewidth(0.8)
        # title level with the top pipeline row (top row ~ 0.86 of the image height)
        axc.text(0.010, 0.5 - ih / 2 - 0.02 + ih * 0.925, args.pipeline_title, transform=axc.transAxes,
                 ha="left", va="center", fontsize=8.6 * FS)
        fig.text(0.5 * (x0 + pb.x1), y_bot - 0.01 / H_in, "(c)", ha="center", va="top", fontsize=11 * FS, fontweight="bold")
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"paper_gap_panels_{short_name(doc['folder'])}{args.suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {args.fig_dir / stem}.{{pdf,png}}  (L = {L:.1f} dB, mean gap {mean:.1f} dB)")


if __name__ == "__main__":
    main()
