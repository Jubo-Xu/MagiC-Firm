#!/usr/bin/env python3
"""Single-column paper figure: the two iso-LER scaling panels side by side, in the
architecture-paper bar style of plot_threshold_sweep.draw_tc_sweep (saturated fills, black
edges, dashed grid, framed legend).

  (a) d_escape sweep at fixed d_cultiv, p      (b) p sweep at fixed d_cultiv, d_escape
Per x position two separated stacked bars: Complete (LER-matched t_c, plain) and
Two-stage partial (hatched), each = gate + control + gap decode; reduction above the pair.
Numbers come from the campaign summary (tc_matched = iso-LER baseline), nothing recomputed.

    python experiments/figures/fig_iso_ler_scaling.py [--summary summary_csq14f29.json] [--d1 3 --p 0.001 --d2 15]
    -> experiments/result/runtime_estimation/figures/paper_scaling_d1-<d1>_<summary tag>.{pdf,png,csv}
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
from matplotlib.lines import Line2D                                        # noqa: E402
from matplotlib.legend_handler import HandlerTuple                         # noqa: E402

from sweep import select, iso_ler_view                                     # noqa: E402
from plot_style import PAPER_RC, PAPER                                     # noqa: E402
from naming import fmt_p                                                   # noqa: E402

RESULT = paths.RESULT / "runtime_estimation"

STACK = [("gate_us", "Gate", PAPER["gate_us"]), ("feedback_us", "Control", PAPER["control_us"]),
         ("gap_decode_us", "Gap decode", PAPER["decode_us"])]
SIDES = [("baseline", "Complete", ""), ("partial", "Two-stage", "///")]
LINE = {"baseline": dict(color="black", ls="-", marker="o"), "partial": dict(color="#52514e", ls="--", marker="s")}
LINE_KW = dict(lw=1.0, ms=3.0, markerfacecolor="white", markeredgewidth=0.9)
W, OFF = 0.30, 0.20          # bar width / half-distance between the two bars of an entry (gap = 2*OFF - W)


def draw(ax, recs, xlabels, xlabel, sub, *, ylabel, fs=1.0, wscale=1.0):
    W_, OFF_ = W * wscale, OFF * wscale          # wscale keeps the physical bar width equal across panels
    tops = {key: [] for key, _, _ in SIDES}
    for i, r in enumerate(recs):
        for side, (key, _, hatch) in zip((-1, +1), SIDES):
            s, bottom = r[key], 0.0
            for k, _, c in STACK:
                ax.bar(i + side * OFF_, s[k], W_, bottom=bottom, color=c, edgecolor="black", linewidth=0.35,
                       hatch=hatch, zorder=3)
                bottom += s[k]
            tops[key].append((i + side * OFF_, bottom))
    for key, pts in tops.items():                # total-time trend line through the bar tops, as in the original figures
        ax.plot([q[0] for q in pts], [q[1] for q in pts], zorder=4, **LINE[key], **LINE_KW)
    ymax = max(r["baseline"]["prep_us"] for r in recs)
    ys = [q[1] for q in tops["baseline"]]
    hw = 0.55 * wscale                            # half text width of "−39%" in data units (6 pt, ~1.4 in panel)
    for i, r in enumerate(recs):
        # label above the Complete marker, lifted over whatever the neighbouring line segments rise within
        # the text's own width, so no line crosses the digits
        y = ys[i]
        rise = 0.0
        if i + 1 < len(ys):
            rise = max(rise, (ys[i + 1] - y) * hw)
        if i > 0:
            rise = max(rise, (ys[i - 1] - y) * hw)
        ax.text(i - OFF_, y + rise + ymax * 0.03, f"−{r['reduction']:.0%}", ha="center", va="bottom",
                fontsize=6.0 * fs, color="black", zorder=6)
    ax.set_xticks(range(len(recs)), xlabels, fontsize=6.8 * fs)
    ax.set_xlabel(f"{xlabel}\n{sub}", fontsize=7.6 * fs, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=7.6 * fs, labelpad=2)
    ax.set_xlim(-0.75 * wscale, len(recs) - 1 + 0.6 * wscale)      # room for the first label at the left
    ax.yaxis.grid(True, color="#999999", lw=0.4, ls="--", zorder=0); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.6)
    ax.tick_params(colors="black", length=2.5, width=0.6, labelsize=6.8 * fs)
    return ymax


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default="summary_csq14f29.json", help="file under experiments/result/runtime_estimation/")
    ap.add_argument("--d1", type=int, default=3)
    ap.add_argument("--p", type=float, default=1e-3, help="fixed p of the d_escape sweep")
    ap.add_argument("--d2", type=int, default=15, help="fixed d_escape of the p sweep")
    ap.add_argument("--p-max", type=float, default=1e9, help="drop p above this from the p sweep")
    ap.add_argument("--size", type=float, nargs=2, default=(3.35, 1.85), help="figure size (in); 3.35 = one ACM column")
    ap.add_argument("--out", type=pathlib.Path, default=RESULT / "figures")
    args = ap.parse_args()
    summary = json.loads((RESULT / args.summary).read_text())
    summary = {k: iso_ler_view(r) for k, r in summary.items() if r.get("tc_matched")}
    recs_a = select(summary, d1=args.d1, p=args.p)
    recs_b = [r for r in select(summary, d1=args.d1, d2=args.d2) if r["p"] <= args.p_max]
    assert recs_a and recs_b, "no iso-LER entries for the requested sweeps"

    plt.rcParams.update(PAPER_RC); plt.rcParams["hatch.linewidth"] = 0.4
    fig, axes = plt.subplots(1, 2, figsize=tuple(args.size), sharey=True, gridspec_kw=dict(wspace=0.06))   # equal panels
    ws = len(recs_b) / len(recs_a)                                    # same physical bar width in both panels
    ym = max(
        draw(axes[0], recs_a, [str(r["d2"]) for r in recs_a], "(a) $d_{\\mathrm{escape}}$",
             f"$d_{{\\mathrm{{cultiv}}}}$={args.d1}, $p$={fmt_p(args.p)}", ylabel="Wall-Clock Avg\nPreparation Time (μs)"),
        draw(axes[1], recs_b, [f"{r['p'] * 1e4:g}" for r in recs_b], "(b) $p$  (×10$^{-4}$)",
             f"$d_{{\\mathrm{{cultiv}}}}$={args.d1}, $d_{{\\mathrm{{escape}}}}$={args.d2}", ylabel=None, wscale=ws))
    axes[0].set_ylim(0, ym * 1.24)
    axes[1].tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.26, top=0.985, wspace=0.06)
    handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.35, label=l) for _, l, c in STACK]
    handles += [(Patch(facecolor="white", edgecolor="black", linewidth=0.35, hatch=h * 2 if h else ""), Line2D([], [], **LINE[k], **LINE_KW))
                for k, _, h in SIDES]
    labels = [l for _, l, _ in STACK] + [l for _, l, _ in SIDES]
    x0, x1 = axes[0].get_position().x0, axes[1].get_position().x1     # legend frame flush with the two panels' outer edges
    lg = fig.legend(handles=handles, labels=labels, ncol=5, loc="lower left", bbox_to_anchor=(x0, 0.99, x1 - x0, 0.05),
                    mode="expand", frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, fontsize=5.8,
                    handlelength=1.7, handleheight=0.9, columnspacing=0.7, handletextpad=0.3, borderpad=0.35,
                    borderaxespad=0.0, handler_map={tuple: HandlerTuple(ndivide=2, pad=0.5)})
    lg.get_frame().set_linewidth(0.6)
    args.out.mkdir(parents=True, exist_ok=True)
    tag = pathlib.Path(args.summary).stem.replace("summary_", "").replace("summary", "")
    stem = f"paper_scaling_d1-{args.d1}" + (f"_{tag}" if tag else "")
    for ext in ("pdf", "png"):
        fig.savefig(args.out / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    with open(args.out / f"{stem}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["panel", "circuit", "d1", "d2", "p", "variant", "th", "tc", "gate_us", "control_us", "gap_decode_us",
                    "total_us", "LER", "reduction"])
        for panel, recs in (("a", recs_a), ("b", recs_b)):
            for r in recs:
                for key, name, _ in SIDES:
                    s = r[key]
                    w.writerow([panel, r["circuit"], r["d1"], r["d2"], r["p"], name, r["th_star"] if key == "partial" else "",
                                f"{s.get('tc', r['tc']):.2f}", f"{s['gate_us']:.3f}", f"{s['feedback_us']:.3f}",
                                f"{s['gap_decode_us']:.3f}", f"{s['prep_us']:.3f}", f"{s['ler']:.2e}", f"{r['reduction']:.4f}"])
    print(f"saved {args.out / stem}.{{pdf,png,csv}}  ({len(recs_a)} + {len(recs_b)} entries)")


if __name__ == "__main__":
    main()
