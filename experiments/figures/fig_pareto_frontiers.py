#!/usr/bin/env python3
"""Paper figure (one column): LER vs preparation latency from the stored threshold sweeps
(collect_threshold_sweep.py, control-latency profile), two panels.

  (a) d_escape=15, p=1e-3: Complete curve + two-stage frontier for d_cultiv=3 and d_cultiv=5.
      Where both cultivation distances reach the same LER, the slower one is drawn faded; the LER
      range in which the d_cultiv=3 two-stage scheme is the fastest option is shaded as its
      operating region.
  (b) d_cultiv=3, d_escape=15: Complete + two-stage frontiers for two masks at p=1e-3 and for
      the campaign mask at p=5e-4.

Complete = single-decoder curve over t_c; two-stage = Pareto frontier over (t_c, t_h) chosen on
the selection half of the gap table and reported on the evaluation half (plot_ler_latency rules).

    python experiments/figures/fig_pareto_frontiers.py [--size 3.35 2.0] [--min-gain 0.01]
    -> experiments/result/runtime_estimation/figures/paper_frontiers_d2-15_csq14f29.{pdf,png,csv}
"""
import argparse
import csv
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import matplotlib.ticker                                                   # noqa: E402
from matplotlib.lines import Line2D                                        # noqa: E402
from matplotlib.patches import Patch                                       # noqa: E402

from sweep import ler_eff, errs, frontier                                  # noqa: E402
from plot_style import PAPER_RC                                            # noqa: E402

RESULT = paths.RESULT / "runtime_estimation"
CS = "_csq14f29"


def sweep(d1, d2, p, kq):
    name = f"end2end_d1={d1}_d2={d2}_r1={d1}_r2=0_p={p:g}_inj=unitary_b=Y"
    f = RESULT / name / f"threshold_sweep_partial_qexp{kq}_cl_wc15_fb3100{CS}.json"
    return json.loads(f.read_text())


def curves(data, min_gain, tc_max):
    single = sorted((r for r in data["single"] if errs(r) > 0 and r["tc"] <= tc_max), key=ler_eff)
    fr = frontier([r for r in data["pcp"] if r["tc"] <= tc_max], min_gain)
    return single, fr


def xy(rows):
    return np.array([ler_eff(r) for r in rows]), np.array([r["total_us"] for r in rows])


def time_at(rows, ler):
    """Latency of a curve at a given LER (log-log interpolation); NaN outside its LER range."""
    x, y = xy(rows)
    o = np.argsort(x)
    lx, ly = np.log(x[o]), np.log(y[o])
    if not (lx[0] <= np.log(ler) <= lx[-1]):
        return np.nan
    return float(np.exp(np.interp(np.log(ler), lx, ly)))


def style_axes(ax, xlabel, ylabel, yticks):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
    ax.set_yticks(yticks); ax.set_yticklabels([str(t) for t in yticks])
    ax.set_xlabel(xlabel, fontsize=7.0, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.6, labelpad=2)
    ax.grid(True, which="major", color="#999999", lw=0.4, ls="--", zorder=0); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.6)
    ax.tick_params(colors="black", length=2.5, width=0.6, labelsize=6.8, which="major")
    ax.tick_params(length=1.5, width=0.4, which="minor")
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def top_legend(ax, ncol):
    """Framed legend sitting on the panel's top edge, flush with its left/right boundaries."""
    lg = ax.legend(ncol=ncol, loc="lower left", bbox_to_anchor=(0.0, 1.01, 1.0, 0.1), mode="expand", frameon=True,
                   fancybox=False, edgecolor="black", framealpha=1.0, fontsize=5.4, handlelength=1.7, borderpad=0.35,
                   labelspacing=0.25, handletextpad=0.35, columnspacing=0.6, borderaxespad=0.0)
    lg.get_frame().set_linewidth(0.5)


def plot_curve(ax, rows, color, ls, marker, *, fade=None, lw=1.1, ms=3.0, zorder=3, label=None, split_at=None):
    """fade: boolean mask per point (True = draw faded). With `split_at` (an LER), the curve is cut
    exactly there (log-log interpolated point, no marker) and the part at LER > split_at is faded."""
    x, y = xy(rows)
    kw = dict(color=color, ls=ls, marker=marker, lw=lw, ms=ms, markerfacecolor="white", markeredgewidth=0.9)
    if split_at is not None:
        o = np.argsort(x); x, y = x[o], y[o]
        if x[0] < split_at < x[-1]:
            ys = float(np.exp(np.interp(np.log(split_at), np.log(x), np.log(y))))
            lkw = dict(kw, marker="none"); mkw = dict(kw, ls="none")
            for sel, alpha in ((x <= split_at, 1.0), (x >= split_at, 0.28)):
                xs = np.concatenate([x[sel], [split_at]]) if alpha == 1.0 else np.concatenate([[split_at], x[sel]])
                yy = np.concatenate([y[sel], [ys]]) if alpha == 1.0 else np.concatenate([[ys], y[sel]])
                ax.plot(xs, yy, zorder=zorder - (alpha < 1), alpha=alpha, **lkw)
                ax.plot(x[sel], y[sel], zorder=zorder - (alpha < 1), alpha=alpha, **mkw)
            ax.plot([], [], label=label, **kw)
            return
        fade = x > split_at
    if fade is None or not fade.any():
        ax.plot(x, y, zorder=zorder, label=label, **kw)
        return
    ax.plot([], [], label=label, **kw)
    # split into runs of equal fade state; the segment joining two runs is drawn in the faded state
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and fade[j + 1] == fade[i]:
            j += 1
        sl = slice(max(i - 1, 0), j + 2) if fade[i] else slice(i, j + 1)
        ax.plot(x[sl], y[sl], zorder=zorder - (1 if fade[i] else 0), alpha=0.28 if fade[i] else 1.0, **kw)
        i = j + 1


def fill_saving(ax, single, fr, color, alpha=0.32):
    """Shade the latency saved by the two-stage frontier under the Complete curve (common LER grid)."""
    xs, ys = xy(single); xf, yf = xy(fr)
    lo, hi = max(xs.min(), xf.min()), min(xs.max(), xf.max())
    g = np.logspace(np.log10(lo), np.log10(hi), 200)
    yc = np.array([time_at(single, l) for l in g]); yt = np.array([time_at(fr, l) for l in g])
    ax.fill_between(g, yt, yc, where=yt < yc, color=color, alpha=alpha, lw=0, zorder=1)


def panel_a_broken(fig, A, COL, other_faster, lo, hi, *, top_lim=(136, 262), bot_lim=(18, 63), min_saving=0.02):
    """Panel (a), paper layout: two header strips (preferred cultivation distance; partial-mask
    beneficial region) above a broken latency axis (bottom: d_cultiv=3, top: d_cultiv=5). The
    beneficial region (two-stage saves > min_saving on d_cultiv=3) is washed light blue through
    both axes; the legend sits in the empty upper-left of the top axis and a zoom of the small
    d_cultiv=5 saving in the empty lower-right of the bottom axis."""
    FS = 5.8                                                        # one annotation/legend size
    LB, LR, LG = "#dbe7f7", "#f7dede", "#eeeeee"                      # header fills
    gs = fig.add_gridspec(4, 1, height_ratios=[0.24, 0.24, 1.7, 1.8], hspace=0.06, left=0.15, right=0.985, bottom=0.17, top=0.985)
    axh1 = fig.add_subplot(gs[0]); axh2 = fig.add_subplot(gs[1]); axt = fig.add_subplot(gs[2]); axb = fig.add_subplot(gs[3])
    for ax in (axt, axb):
        for d1 in (3, 5):
            single, fr = A[d1]
            # d_cultiv=5 is low-lighted where d_cultiv=3 is the preferred choice (left of the crossover), cut exactly there
            sp = lo if d1 == 5 else None
            plot_curve(ax, single, COL[d1], "-", "o", lw=0.9, ms=2.6, split_at=sp, label=f"Complete, $d_{{\\mathrm{{cultiv}}}}$={d1}")
            plot_curve(ax, fr, COL[d1], "--", "s", lw=1.15, ms=2.3, split_at=sp, label=f"Two-stage, $d_{{\\mathrm{{cultiv}}}}$={d1}")
        ax.set_xscale("log")
        ax.grid(True, which="major", color="#ececec", lw=0.35, ls="-", zorder=0); ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color("black"); sp.set_linewidth(0.6)
        ax.tick_params(colors="black", length=2, width=0.5, labelsize=6.3, which="major", pad=1.5)
        ax.tick_params(length=1.2, width=0.4, which="minor")
    axt.invert_xaxis(); xlim = (1.0e-4, axt.get_xlim()[1])           # relaxed side starts at LER 1e-4; stringent side as autoscaled
    for ax in (axb, axh1, axh2):
        ax.set_xscale("log"); ax.set_xlim(xlim)
    axt.set_ylim(*top_lim); axt.set_yticks([150, 200, 250])
    axb.set_ylim(*bot_lim); axb.set_yticks([20, 30, 40, 50, 60])
    axt.spines["bottom"].set_visible(False); axb.spines["top"].set_visible(False)
    axt.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    d = 0.35                                                    # axis-break marks (diagonal ticks)
    kw = dict(marker=[(-1, -d), (1, d)], markersize=5, linestyle="none", color="black", mec="black", mew=0.6, clip_on=False)
    axt.plot([0, 1], [0, 0], transform=axt.transAxes, **kw); axb.plot([0, 1], [1, 1], transform=axb.transAxes, **kw)
    # --- decision LERs
    x_cross = lo                                                # d_cultiv=3 runs out here
    single3, fr3 = A[3]
    g = np.logspace(np.log10(max(xy(single3)[0].min(), xy(fr3)[0].min())), np.log10(min(xy(single3)[0].max(), xy(fr3)[0].max())), 400)
    sav = np.array([1 - time_at(fr3, l) / time_at(single3, l) for l in g])
    x_ben = float(g[np.argmax(sav > min_saving)]) if (sav > min_saving).any() else np.nan
    # --- header strips
    for ax in (axh1, axh2):
        ax.set_yticks([]); ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for sp in ax.spines.values():
            sp.set_color("black"); sp.set_linewidth(0.6)
    axh1.axvspan(xlim[0], x_cross, color=LB, lw=0); axh1.axvspan(x_cross, xlim[1], color=LR, lw=0)
    axh1.text(np.sqrt(xlim[0] * x_cross), 0.5, "$d_{\\mathrm{cultiv}}=3$", ha="center", va="center", fontsize=FS, transform=axh1.get_xaxis_transform())
    axh1.text(np.sqrt(x_cross * xlim[1]), 0.5, "$d_{\\mathrm{cultiv}}=5$", ha="center", va="center", fontsize=FS, transform=axh1.get_xaxis_transform())
    axh2.axvspan(xlim[0], x_ben, color=LB, lw=0); axh2.axvspan(x_ben, xlim[1], color=LG, lw=0)
    axh2.text(np.sqrt(xlim[0] * x_ben), 0.5, "Beneficial", ha="center", va="center", fontsize=FS, color=COL[3],
              transform=axh2.get_xaxis_transform())
    axh2.text(np.sqrt(x_ben * xlim[1]), 0.5, "Limited benefit", ha="center", va="center", fontsize=FS, color="#555555",
              transform=axh2.get_xaxis_transform())
    axh1.text(-0.015, 0.5, "Cultivation", transform=axh1.transAxes, ha="right", va="center", fontsize=FS - 0.4)
    axh2.text(-0.015, 0.5, "Partial mask", transform=axh2.transAxes, ha="right", va="center", fontsize=FS - 0.4)
    for ax in (axh1, axt, axb):
        ax.axvline(x_cross, color="#444444", ls="--", lw=1.1, zorder=1)
    for ax in (axh2, axt, axb):
        ax.axvspan(xlim[0], x_ben, color=COL[3], alpha=0.07, lw=0, zorder=0)
        ax.axvline(x_ben, color=COL[3], ls="--", lw=0.65, alpha=0.6, zorder=1)
    # --- legend: empty upper-left of the top axis
    lg = axt.legend(loc="upper left", bbox_to_anchor=(0.012, 0.975), frameon=True, fancybox=False, edgecolor="#444444", framealpha=1.0,
                    fontsize=FS - 0.2, handlelength=1.8, borderpad=0.3, labelspacing=0.22, handletextpad=0.4, borderaxespad=0.0)
    lg.get_frame().set_linewidth(0.4)
    # --- d_cultiv=5 saving: fill + zoom inset in the empty lower-right of the bottom axis
    single5, fr5 = A[5]
    zx = (1e-6, max(xy(fr5)[0].max(), xy(single5)[0].max()))          # LER window: 1e-6 .. highest LER
    zy = (min(xy(fr5)[1].min(), xy(single5)[1].min()) - 2, max(time_at(single5, zx[0]), time_at(fr5, zx[0])) + 2)
    axins = axb.inset_axes([0.625, 0.22, 0.345, 0.64])
    axins.set_xscale("log")
    plot_curve(axins, single5, COL[5], "-", "o", ms=2.2, lw=0.8)
    plot_curve(axins, fr5, COL[5], "--", "s", ms=2.0, lw=1.0)
    axins.set_xticks([1e-5, 1e-6]); axins.set_xlim(zx[1], zx[0] * 0.72); axins.set_ylim(*zy)   # x inverted; margin keeps the 1e-6 label inside
    axins.tick_params(labelsize=4.6, length=1.2, width=0.4, pad=1)
    axins.xaxis.set_major_formatter(matplotlib.ticker.LogFormatterMathtext()); axins.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axins.set_yticks([140, 150, 160]); axins.grid(True, color="#ececec", lw=0.3, ls="-", zorder=0); axins.set_axisbelow(True)
    ZC = "#333333"
    for sp in axins.spines.values():
        sp.set_color(ZC); sp.set_linewidth(0.7); sp.set_linestyle("-")
    axins.set_facecolor("#fbfbfb")
    axins.text(0.04, 0.95, "Zoom, $d_{\\mathrm{cultiv}}=5$", transform=axins.transAxes, ha="left", va="top", fontsize=FS - 0.6, color=ZC)
    from matplotlib.patches import Rectangle
    axt.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], facecolor="#000000", alpha=0.035, edgecolor="none", zorder=0.5))
    axt.add_patch(Rectangle((zx[0], zy[0]), zx[1] - zx[0], zy[1] - zy[0], fill=False, edgecolor="#666666", lw=0.5, zorder=6))
    for ax in (axh1, axh2, axt, axb):                          # fills/patches must not re-autoscale the LER axis
        ax.set_xlim(xlim)
    axb.set_xlabel("Target LER of accepted states   [$d_{\\mathrm{escape}}=15$, $p=10^{-3}$]", fontsize=6.8, labelpad=2)
    fig.text(0.02, (axb.get_position().y0 + axt.get_position().y1) / 2, "Avg. preparation time (μs)", rotation=90,
             ha="left", va="center", fontsize=6.8)
    print(f"crossover LER {x_cross:.2e}; two-stage saving > {min_saving:.0%} for LER >= {x_ben:.2e}")
    return axt, axb


def beneficial_range(single, fr, min_saving=0.02, n=400):
    """LER range [stringent, relaxed] over which the two-stage frontier saves > min_saving of the
    Complete latency (log-log interpolation on the overlap of the two curves); measured, not placed."""
    xs, xf = xy(single)[0], xy(fr)[0]
    g = np.logspace(np.log10(max(xs.min(), xf.min())), np.log10(min(xs.max(), xf.max())), n)
    sav = np.array([1 - time_at(fr, l) / time_at(single, l) for l in g])
    ok = np.where(sav > min_saving)[0]
    return (float(g[ok[0]]), float(g[ok[-1]])) if len(ok) else (np.nan, np.nan)


def sparse_markers(x, xlim, n=10):
    """Indices of ~n points spread uniformly in log-LER over the plotted window (line uses all points)."""
    inside = np.where((x <= max(xlim)) & (x >= min(xlim)))[0]
    if len(inside) <= n:
        return inside
    targets = np.linspace(np.log10(x[inside].max()), np.log10(x[inside].min()), n)
    idx = sorted({int(inside[np.argmin(np.abs(np.log10(x[inside]) - t))]) for t in targets})
    return np.array(idx)


def panel_b_paper(fig, args, rows_out):
    """Panel (b), paper layout, single column, two square sub-panels side by side.
    Left: impact of p. Complete / two-stage (campaign mask) for p=1e-3 (blue) and p=5e-4 (orange)
    over the common measured LER window; strip above with the measured two-stage beneficial range
    (bars from the relaxed boundary to the crossover, dot + p label, dashed guide into the plot).
    Right: impact of mask size at p=1e-3: Complete, K_Q=<k1> (campaign), K_Q=<k2>, over the window
    where the two masks differ. Same quantities, same styling; no zoom/inset elements."""
    FS = 5.8
    BLUE, ORANGE = "#4c72b0", "#dd8452"
    GRID = dict(color="0.88", alpha=0.4, lw=0.4, ls="-")
    GUIDE = dict(ls="--", lw=0.9, alpha=0.75)
    k1, k2 = args.masks_b
    base = dict(markerfacecolor="white", markeredgewidth=0.95)
    DASH, DOT = (0, (2.6, 1.3)), (0, (1.0, 1.2))
    LW, MS = 1.3, 3.3

    def style(ax):
        ax.grid(False, which="both"); ax.set_axisbelow(True)              # no grid
        for sp in ax.spines.values():
            sp.set_color("black"); sp.set_linewidth(0.6)
        ax.tick_params(colors="black", length=2, width=0.5, labelsize=6.3, which="major", pad=1.5)
        ax.tick_params(length=1.2, width=0.4, which="minor")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # ---- layout: strip over the left panel, two square panels below
    gs = fig.add_gridspec(2, 2, height_ratios=[0.27, 2.0], width_ratios=[1, 1], hspace=0.06, wspace=0.20,
                          left=0.13, right=0.985, bottom=0.17, top=0.94)
    # mask-size panel on the left (a), p panel on the right (b)
    axs = fig.add_subplot(gs[0, 1]); axl = fig.add_subplot(gs[1, 1]); axr = fig.add_subplot(gs[1, 0]); axs2 = fig.add_subplot(gs[0, 0])

    def strip(axst, xlim_, rows, title):
        """rows: (row, colour, label, single, frontier). Bars from the relaxed boundary to the measured
        crossover (two-stage saves > 2 % over Complete), dot + label after the endpoint, guide into the plot."""
        axst.set_xscale("log"); axst.set_xlim(xlim_); axst.set_ylim(-1.3, 2.3)
        axst.set_yticks([]); axst.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        for sp in axst.spines.values():
            sp.set_color("black"); sp.set_linewidth(0.6)
        axst.set_title(title, fontsize=FS - (1.0 if axst is axs2 else 0.6), pad=1.0, color="black")
        out = []
        for row, col, lab, single_, fr_ in rows:
            lo_, hi_ = beneficial_range(single_, fr_)
            axst.plot([xlim_[0], lo_], [row, row], color=col, lw=2.4, solid_capstyle="butt", zorder=3)      # butt cap at the boundary
            axst.plot([lo_], [row], marker="o", ms=4.4, color=col, markeredgewidth=0, zorder=4)              # measured endpoint
            axst.plot([lo_, lo_], [row, -1.3], color=col, zorder=2, clip_on=False, **GUIDE)
            import matplotlib.patheffects as pe                                   # label fused into the bar, ending at the circle
            axst.annotate(lab, xy=(lo_, row), xytext=(-4.0, 0), textcoords="offset points", ha="right", va="center", fontsize=FS - 1.5,
                          color="white", zorder=5, path_effects=[pe.withStroke(linewidth=1.2, foreground=col)])
            out.append((lo_, col))
            print(f"beneficial range [{lab}]: LER {hi_:.2e} .. {lo_:.2e} (saving > 2%)")
        return out

    # ---- left: impact of p
    kl = args.mask_b_lowp
    sets = [(1e-3, BLUE, k1, "$p=10^{-3}$"), (5e-4, ORANGE, kl, "$p=5\\times10^{-4}$")]
    keep, lo_all, hi_all = {}, [], []
    for p_, col, kq, lab in sets:
        single, fr = curves(sweep(3, 15, p_, kq), args.min_gain, args.tc_max)
        keep[p_] = (single, fr, col, lab)
        for rows in (single, fr):
            x = xy(rows)[0]; lo_all.append(x.min()); hi_all.append(x.max())
        rows_out += [dict(panel="b", series=f"complete_p={p_:g}", ler_plotted=ler_eff(r), **r) for r in single]
        rows_out += [dict(panel="b", series=f"two-stage_K{kq}_p={p_:g}", ler_plotted=ler_eff(r), **r) for r in fr]
    L_lo, L_hi = min(lo_all), min(hi_all)                            # relaxed side: common support; stringent side: to the last measured point
    xlim = (L_hi * 10 ** 0.16, L_lo / 10 ** 0.10)
    ys = []
    for p_, (single, fr, col, lab) in keep.items():
        for rows, ls, mk in ((single, "-", "o"), (fr, DASH, "s")):
            x, y = xy(rows)
            axl.plot(x, y, color=col, lw=LW, ls=ls, zorder=4)
            mi = sparse_markers(x, xlim, n=6)
            axl.plot(x[mi], y[mi], color=col, ls="none", marker=mk, ms=MS, zorder=5, **base)
            ys += list(y[(x <= xlim[0]) & (x >= xlim[1])])
    axl.set_xscale("log"); axl.set_xlim(xlim)
    axl.set_xticks([1e-5, 1e-6]); axl.set_xticklabels(["$10^{-5}$", "$10^{-6}$"])
    axl.set_xlim(xlim)                                                   # set_xticks may widen the view; keep the window
    y0, y1 = min(ys), max(ys); pad = 0.08 * (y1 - y0); axl.set_ylim(y0 - pad, y1 + pad)
    style(axl)
    axl.set_xlabel("(b) Target LER", fontsize=6.8, labelpad=2)
    kw = dict(color="black", lw=LW, ms=MS, markerfacecolor="white", markeredgewidth=0.95)
    lg = axl.legend(handles=[Line2D([], [], ls="-", marker="o", label="Complete", **kw), Line2D([], [], ls=DASH, marker="s", label="Two-stage", **kw)],
                    loc="upper left", bbox_to_anchor=(0.012, 0.975), frameon=True, fancybox=False, edgecolor="#444444", framealpha=1.0,
                    fontsize=FS - 0.4, handlelength=2.0, borderpad=0.28, labelspacing=0.2, handletextpad=0.35, borderaxespad=0.0)
    lg.get_frame().set_linewidth(0.4); lg.set_zorder(7)
    # strip (a): measured beneficial range per p
    for lo_, col in strip(axs, xlim, [(1, BLUE, "$p=10^{-3}$", keep[1e-3][0], keep[1e-3][1]),
                                      (0, ORANGE, "$p=5\\times10^{-4}$", keep[5e-4][0], keep[5e-4][1])],
                          f"Variant $p$, Mask Size={k1}" if kl == k1 else f"Variant $p$, $K_Q$={k1}/{kl}"):
        axl.axvline(lo_, color=col, zorder=1, **GUIDE)

    # ---- right: impact of mask size at p=1e-3
    single, fr = keep[1e-3][:2]
    _, fr2 = curves(sweep(3, 15, 1e-3, k2), args.min_gain, args.tc_max)
    rows_out += [dict(panel="b", series=f"two-stage_K{k2}_p=0.001", ler_plotted=ler_eff(r), **r) for r in fr2]
    x1, x2 = xy(fr)[0], xy(fr2)[0]
    rise = min(r["total_us"] for r in fr if r["tc"] >= 60)              # shared vertical rise at the LER cap: not a mask effect
    w_lo = max(min(ler_eff(r) for r in fr if r["total_us"] < rise), min(ler_eff(r) for r in fr2 if r["total_us"] < rise))
    g = np.logspace(np.log10(w_lo), np.log10(min(x1.max(), x2.max())), 400)
    gain = np.array([1 - time_at(fr2, l) / time_at(fr, l) for l in g])
    w_hi = float(g[np.where(gain >= 0)[0].max()])                         # where the larger mask starts to pay off
    rx = (w_hi * 10 ** 0.16, w_lo / 10 ** 0.12)
    rys = []
    RS = ((single, "-", "o", "Complete", 1.15, 3.2, 1.0), (fr, (0, (0.8, 1.0)), "s", f"$K_Q$={k1}", 1.0, 2.3, 1.0),
          (fr2, DASH, "^", f"$K_Q$={k2}", 1.5, 4.2, 1.0))
    for rows, ls, mk, lab, lw_, ms_, al in RS:
        x, y = xy(rows)
        axr.plot(x, y, color=BLUE, lw=lw_, ls=ls, alpha=al, zorder=4)
        mi = sparse_markers(x, rx, n=6)
        axr.plot(x[mi], y[mi], color=BLUE, ls="none", marker=mk, ms=ms_, alpha=al, zorder=5, **base)
        axr.plot([], [], color=BLUE, lw=lw_, ls=ls, marker=mk, ms=ms_, alpha=al, label=lab, **base)
        rys += list(y[(x <= rx[0]) & (x >= rx[1])])
    axr.set_xscale("log"); axr.set_xlim(rx)
    axr.set_xticks([2e-5, 5e-6]); axr.set_xticklabels(["$2{\\times}10^{-5}$", "$5{\\times}10^{-6}$"])
    axr.set_xlim(rx)
    ry0, ry1 = min(rys), max(rys); rpad = 0.08 * (ry1 - ry0); axr.set_ylim(ry0 - rpad, ry1 + rpad)
    style(axr)
    axr.set_xlabel("(a) Target LER", fontsize=6.8, labelpad=2)
    axr.set_ylabel("Avg. preparation time (μs)", fontsize=6.8, labelpad=2)
    BLUE2 = "#8fb0dc"                                                  # lighter blue for the second mask's bar
    for lo_, col in strip(axs2, rx, [(1, BLUE2, f"$K_Q$={k1}", single, fr), (0, BLUE, f"$K_Q$={k2}", single, fr2)],
                          "Variant Mask Size, $p=10^{-3}$"):
        axr.axvline(lo_, color=col, zorder=1, **GUIDE)
    lgi = axr.legend(loc="upper left", bbox_to_anchor=(0.012, 0.975), frameon=True, fancybox=False, edgecolor="#999999", framealpha=1.0,
                     fontsize=FS - 0.8, handlelength=2.0, borderpad=0.28, labelspacing=0.2, handletextpad=0.35, borderaxespad=0.0)
    lgi.get_frame().set_linewidth(0.3); lgi.set_zorder(7)
    print(f"left window LER {L_hi:.2e} .. {L_lo:.2e}; right window LER {w_hi:.2e} .. {w_lo:.2e}")
    # ---- make both panels square (equal axes width and height in inches)
    fig.canvas.draw()
    for ax in (axl, axr):
        pos = ax.get_position(); W_, H_ = fig.get_size_inches()
        w_in, h_in = pos.width * W_, pos.height * H_
        side = min(w_in, h_in)
        ax.set_position([pos.x0, pos.y0 + (h_in - side) / H_ / 2 if h_in > side else pos.y0, side / W_, side / H_])
    for axst, axm in ((axs, axl), (axs2, axr)):                          # strips share their panel's x-position, width and x-range
        pm, ps = axm.get_position(), axst.get_position()
        axst.set_position([pm.x0, ps.y0, pm.width, ps.height]); axst.set_xlim(axm.get_xlim())
    # ---- overlap audit
    fig.canvas.draw(); rd = fig.canvas.get_renderer()
    checks = [("left legend", lg, axl, [v[0] for v in keep.values()] + [v[1] for v in keep.values()]),
              ("right legend", lgi, axr, [single, fr, fr2])]
    for name, l, ax, rows_list in checks:
        bb = l.get_window_extent(rd)
        for rows in rows_list:
            pts = ax.transData.transform(np.column_stack(xy(rows))); mids = (pts[1:] + pts[:-1]) / 2
            if any(bb.contains(px, py) for px, py in np.vstack([pts, mids])):
                print(f"OVERLAP: {name} touches a curve"); break
    print("overlap audit done")
    return axl


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", choices=["a", "b", "both"], default="both", help="one panel per figure, or both stacked")
    ap.add_argument("--size", type=float, nargs=2, default=None, help="figure size (in); default 3.35x1.9 per panel (a: 3.35x2.3 broken axis), 3.35x3.4 stacked")
    ap.add_argument("--min-gain", type=float, default=0.01)
    ap.add_argument("--tc-max", type=float, default=1e9)
    ap.add_argument("--masks-b", type=int, nargs=2, default=(250, 350), help="two K_Q masks for panel (b) at p=1e-3")
    ap.add_argument("--mask-b-lowp", type=int, default=250, help="K_Q mask for the p=5e-4 curves of panel (b) (campaign mask: 100)")
    ap.add_argument("--out", type=pathlib.Path, default=RESULT / "figures")
    ap.add_argument("--no-break", action="store_true", help="panel a alone: plain log axis instead of the broken linear axis")
    args = ap.parse_args()

    plt.rcParams.update(PAPER_RC)
    both = args.panel == "both"
    size = tuple(args.size) if args.size else ((3.35, 3.4) if both else (3.35, 2.6) if (args.panel == "a" and not args.no_break) else (3.35, 2.05) if (args.panel == "b" and not args.no_break) else (3.35, 1.9))
    if both:
        fig, (axa, axb) = plt.subplots(2, 1, figsize=size, gridspec_kw=dict(hspace=0.55))   # stacked, each panel flat
    elif args.panel == "a" and not args.no_break:
        fig = plt.figure(figsize=size); axa = axb = None            # built after the data is loaded
    elif args.panel == "b" and not args.no_break:
        fig = plt.figure(figsize=size); axa = axb = None
    else:
        fig, ax1 = plt.subplots(figsize=size); axa = axb = ax1
    rows_out = []
    k1, k2 = args.masks_b

    # ---- (a) d_cultiv = 3 vs 5 at d_escape=15, p=1e-3 -------------------------------------------
    COL = {3: "#4c72b0", 5: "#c44e52"}
    lo = hi = float("nan")
    if args.panel in ("a", "both"):
      A = {3: curves(sweep(3, 15, 1e-3, 250), args.min_gain, args.tc_max),
           5: curves(sweep(5, 15, 1e-3, 450), args.min_gain, args.tc_max)}
      # fastest option at each LER: fade a curve where the OTHER d_cultiv offers a faster point
      def other_faster(d1, rows):
          o = 5 if d1 == 3 else 3
          out = []
          for r in rows:
              l, t = ler_eff(r), r["total_us"]
              cand = [v for v in (time_at(A[o][0], l), time_at(A[o][1], l)) if not np.isnan(v)]
              out.append(bool(cand and min(cand) < t))
          return np.array(out)
      for d1 in (3, 5):
          single, fr = A[d1]
          rows_out += [dict(panel="a", series=f"complete_d1={d1}", ler_plotted=ler_eff(r), **r) for r in single]
          rows_out += [dict(panel="a", series=f"two-stage_d1={d1}", ler_plotted=ler_eff(r), **r) for r in fr]
      # operating region of the d_cultiv=3 two-stage scheme: LER range where it is the fastest option overall
      fr3 = A[3][1]
      ok = [ler_eff(r) for r in fr3 if not other_faster(3, [r])[0]]
      lo, hi = min(ok), max(ok)
      if axa is None:                                    # single figure, broken axis
          axt, axa = panel_a_broken(fig, A, COL, other_faster, lo, hi); axb = axa
      else:
        for d1 in (3, 5):
          single, fr = A[d1]
          plot_curve(axa, single, COL[d1], "-", "o", fade=other_faster(d1, single))
          plot_curve(axa, fr, COL[d1], "--", "s", fade=other_faster(d1, fr), ms=2.6)
        axa.axvspan(lo, hi, color=COL[3], alpha=0.10, lw=0, zorder=0)
        axa.text(np.sqrt(lo * hi), 0.96, "two-stage operating region ($d_{\\mathrm{cultiv}}$=3)", transform=axa.get_xaxis_transform(),
                 ha="center", va="top", fontsize=5.6, color=COL[3])
        style_axes(axa, "(a) LER of accepted states   [$d_{\\mathrm{escape}}$=15, $p$=1×10$^{-3}$]", "Wall-Clock Avg\nPrep. Time (μs)",
                   [20, 50, 100, 200, 400])

    # ---- (b) masks and p at d_cultiv=3, d_escape=15 -----------------------------------------------
    if args.panel == "b" and not args.no_break:
        axb = axa = panel_b_paper(fig, args, rows_out); handles_off = True
    elif args.panel in ("b", "both"):
      B = [(1e-3, k1, "#4c72b0", "--", "s", f"$K_Q$={k1} ($p$=10$^{{-3}}$)"),
           (1e-3, k2, "#55a868", ":", "^", f"$K_Q$={k2} ($p$=10$^{{-3}}$)"),
           (5e-4, 100, "#dd8452", "--", "s", f"$K_Q$=100 ($p$=5×10$^{{-4}}$)")]
      done_complete = set()
      for p, kq, col, ls, mk, lab in B:
          single, fr = curves(sweep(3, 15, p, kq), args.min_gain, args.tc_max)
          if p not in done_complete:
              ccol = "#4c72b0" if p == 1e-3 else "#dd8452"
              plot_curve(axb, single, ccol, "-", "o", label=f"Complete ($p$={'10$^{-3}$' if p == 1e-3 else '5×10$^{-4}$'})")
              rows_out += [dict(panel="b", series=f"complete_p={p:g}", ler_plotted=ler_eff(r), **r) for r in single]
              done_complete.add(p)
          plot_curve(axb, fr, col, ls, mk, label=lab, ms=2.6)
          rows_out += [dict(panel="b", series=f"two-stage_K{kq}_p={p:g}", ler_plotted=ler_eff(r), **r) for r in fr]
      style_axes(axb, "(b) LER of accepted states   [$d_{\\mathrm{cultiv}}$=3, $d_{\\mathrm{escape}}$=15]", "Wall-Clock Avg\nPrep. Time (μs)",
                 [15, 20, 30, 40, 60])

    if both:
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.10, top=0.86)
    elif args.no_break:
        fig.subplots_adjust(left=0.16, right=0.985, bottom=0.21, top=0.83 if args.panel == "a" else 0.78)
    # one shared legend: line style = decoder scheme, colour = circuit (blue is d_cultiv=3, p=1e-3 in both panels)
    kw = dict(lw=1.1, ms=3.0, markerfacecolor="white", markeredgewidth=0.9, color="black")
    handles = [Line2D([], [], ls="-", marker="o", label="Complete", **kw),                       # column 1: schemes
               Line2D([], [], ls="--", marker="s", label="Two-stage (campaign mask)", **kw),
               Line2D([], [], ls=":", marker="^", label=f"Two-stage ($K_Q$={k2} mask)", **dict(kw, color="#55a868")),
               Patch(facecolor="#4c72b0", label="$d_{\\mathrm{cultiv}}$=3, $p$=10$^{-3}$"),   # column 2: circuits
               Patch(facecolor="#c44e52", label="$d_{\\mathrm{cultiv}}$=5, $p$=10$^{-3}$"),
               Patch(facecolor="#dd8452", label="$d_{\\mathrm{cultiv}}$=3, $p$=5×10$^{-4}$")]
    if args.panel in ("a", "b") and not args.no_break:
        handles = []                                              # paper layouts draw their own legend
    elif args.panel == "a":
        handles = [handles[0], handles[1], handles[3], handles[4]]
    elif args.panel == "b":
        handles = [handles[0], handles[1], handles[2], handles[3], handles[5]]
    axtop = axt if (args.panel == "a" and not args.no_break) else axa
    x0, x1 = axtop.get_position().x0, axb.get_position().x1
    ytop = axtop.get_position().y1 + 0.012
    if handles:
      lg = fig.legend(handles=handles, ncol=2, loc="lower left", bbox_to_anchor=(x0, ytop, x1 - x0, 0.1), mode="expand",
                    frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, fontsize=5.6, handlelength=1.8,
                    handleheight=0.8, borderpad=0.35, labelspacing=0.3, handletextpad=0.4, columnspacing=0.8, borderaxespad=0.0)
      lg.get_frame().set_linewidth(0.5)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"paper_frontiers_d2-15{'' if both else '_' + args.panel}{CS}"
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    with open(args.out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows_out for k in r}))
        w.writeheader(); w.writerows(rows_out)
    print(f"saved {args.out / stem}.{{pdf,png,csv}}" + (f"; d_cultiv=3 two-stage operating region LER [{lo:.2e}, {hi:.2e}]" if lo == lo else ""))


if __name__ == "__main__":
    main()
