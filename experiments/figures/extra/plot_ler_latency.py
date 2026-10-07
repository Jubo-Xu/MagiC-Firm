#!/usr/bin/env python3
"""LER vs preparation latency, from a stored threshold sweep
(collect_threshold_sweep.py): one curve for the complete decoder alone
(single(t_c) over the t_c grid) and one for the two-stage scheme
(pcp(t_c, t_l=0, t_h) over the same grid), points labelled with t_c. The
two-stage curve lying below the complete-only curve means the fast-accept path
is faster at every logical error rate. Plot only — nothing is recomputed.

    python experiments/figures/extra/plot_ler_latency.py <threshold_sweep_*.json> [--size W H] [--no-labels]
    -> experiments/result/runtime_estimation/figures/ler_latency_<config>_fb<ns>[_cs..].{pdf,png,csv}
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
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

from plot_style import INK, INK2, GRID, LINE                               # noqa: E402
from naming import circuit_label, short_name                               # noqa: E402
from sweep import ler_eff, errs, thin                                      # noqa: E402

SERIES = [("single", "Complete", LINE["Complete"]), ("pcp", "Partial (two-stage)", LINE["Partial"])]


def draw_bounds(ax, rows, style, labels):
    """Zero-error points: detached open right-pointing triangles at the 95 % upper bound
    (the LER axis is inverted, so 'right' is 'lower LER', where the true value lies).
    They are not joined to the curve — the bound loosens as fewer shots are accepted, so a
    line through them would bend the wrong way."""
    b = [r for r in rows if errs(r) == 0]
    if b:
        ax.plot([ler_eff(r) for r in b], [r["total_us"] for r in b], ls="none", marker=">", ms=4.5,
                markerfacecolor="white", markeredgewidth=1.0, color=style["color"], zorder=5)
        if labels:
            for r in b:
                ax.annotate(f"{r['tc']:g}", (ler_eff(r), r["total_us"]), xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=6, color=INK2)
    return bool(b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep", type=pathlib.Path)
    ap.add_argument("--fig-dir", type=pathlib.Path, default=paths.RESULT / "runtime_estimation" / "figures")
    ap.add_argument("--size", type=float, nargs=2, default=(3.5, 2.7))
    ap.add_argument("--no-labels", action="store_true", help="omit the t_c labels next to the points")
    ap.add_argument("--tc-max", type=float, default=1e9, help="drop points with t_c above this")
    ap.add_argument("--logy", action="store_true", default=None, help="log latency axis (default: automatic when the range exceeds 4x)")
    ap.add_argument("--linear", action="store_false", dest="logy", help="force a linear latency axis")
    ap.add_argument("--suffix", default="", help="appended to the figure file name")
    ap.add_argument("--min-gain", type=float, default=0.01,
                    help="a two-stage point joins the frontier only if it is faster than every lower-LER "
                         "point by this fraction (default 1%%)")
    args = ap.parse_args()
    data = json.loads(args.sweep.read_text())

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
                         "legend.fontsize": 7.2, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=tuple(args.size))
    rows_out = []
    bounds = False
    tmax = tmin = None
    # label spacing (factor in LER) grows with the LER span so labels stay legible
    lers = [ler_eff(r) for k in ("single", "pcp") for r in data.get(k, []) if errs(r) > 0 and r["tc"] <= args.tc_max]
    span = max(lers) / min(lers) if lers else 1.0
    f_single, f_pcp = (1.3, 1.5) if span < 300 else (2.0, 2.5)
    # complete-only: one knob, its curve is its frontier (resolved points only; zero-error
    # points are shown as detached upper bounds)
    single = sorted((r for r in data.get("single", []) if r["tc"] <= args.tc_max), key=ler_eff)
    if single:
        res = [r for r in single if errs(r) > 0]
        ax.plot([ler_eff(r) for r in res], [r["total_us"] for r in res], lw=1.4, ms=4, zorder=3,
                markerfacecolor="white", markeredgewidth=1.2, label="Complete", **LINE["Complete"])
        bounds |= draw_bounds(ax, single, LINE["Complete"], not args.no_labels)
        if not args.no_labels:
            for r in thin(res, f_single):
                ax.annotate(f"{r['tc']:g}", (ler_eff(r), r["total_us"]), xytext=(0, 5), textcoords="offset points",
                            ha="center", fontsize=6, color=INK2)
        rows_out += [dict(series="single", frontier=True, ler_plotted=ler_eff(r), **r) for r in single]
        tmax, tmin = max(r["total_us"] for r in single), min(r["total_us"] for r in single)
    # two-stage: every (t_c, t_h) point is an operating point; draw all faintly and the
    # Pareto frontier (no other point has both lower LER and lower time) as the curve
    pcp = [r for r in data.get("pcp", []) if r["tc"] <= args.tc_max]
    if pcp:
        res = [r for r in pcp if errs(r) > 0]
        # Pareto frontier CHOSEN on the selection half (ler_select) and REPORTED on the evaluation
        # half, so the choice cannot ride a downward fluctuation of the value that is drawn
        sel = "select" if all("ler_select" in r for r in res) else "eval"
        pts = sorted(res, key=lambda r: (ler_eff(r, sel), r["total_us"]))    # ascending LER
        frontier, best_t = [], float("inf")
        for r in pts:                     # walking from the lowest LER up, keep only time improvements
            # ... of at least --min-gain: points that are only marginally faster than a
            # lower-LER point (e.g. t_h one below the cap: 1 us faster, higher LER) are
            # strictly Pareto-optimal but would draw a step above the complete curve
            if r["total_us"] < best_t * (1.0 - args.min_gain):
                frontier.append(r); best_t = r["total_us"]
        frontier.sort(key=ler_eff)
        multi = len({(r["tc"], r["th"]) for r in pcp}) > len({r["tc"] for r in pcp})
        if multi:
            ax.scatter([ler_eff(r) for r in res], [r["total_us"] for r in res], s=5, color=LINE["Partial"]["color"],
                       alpha=0.22, zorder=2, linewidths=0, label="all ($t_c$, $t_h$)")
        ax.plot([ler_eff(r) for r in frontier], [r["total_us"] for r in frontier], lw=1.4, ms=3 if multi else 4,
                zorder=4, markerfacecolor="white", markeredgewidth=1.0,
                label="Two-stage" + (" frontier" if multi else ""), **LINE["Partial"])
        if not args.no_labels:
            for r in thin(frontier, f_pcp):
                lab = f"{r['tc']:g}/{r['th']:g}" if multi else f"{r['tc']:g}"
                ax.annotate(lab, (ler_eff(r), r["total_us"]), xytext=(0, -9), textcoords="offset points",
                            ha="center", fontsize=5.6, color=INK2)
        fr = {(r["tc"], r["th"]) for r in frontier}
        rows_out += [dict(series="pcp", frontier=(r["tc"], r["th"]) in fr, ler_plotted=ler_eff(r), **r) for r in pcp]
        tmax = max([tmax or 0] + [r["total_us"] for r in res]); tmin = min([tmin or 1e9] + [r["total_us"] for r in res])
    if bounds:
        ax.plot([], [], ls="none", marker=">", ms=4.5, markerfacecolor="white", markeredgewidth=1.0,
                color=INK2, label="0 errors: 95% upper bound")
    if args.logy or (args.logy is None and tmax and tmax / tmin > 4):
        ax.set_yscale("log")
        nice = [20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000]
        ax.set_yticks([t for t in nice if tmin / 1.1 <= t <= tmax * 1.15])
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xscale("log")
    ax.invert_xaxis()                      # lower LER (stricter threshold) to the right
    ax.set_xlabel(f"logical error rate of accepted states\n[{circuit_label(data['circuit'])}; labels: $t_c$ / $t_h$ (dB)]")
    ax.set_ylabel("Avg. preparation latency\nper accepted magic state (μs)")
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=INK2, length=2.5)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=4 if bounds else 3,
              handlelength=1.6, borderaxespad=0.0, columnspacing=0.8, handletextpad=0.5,
              fontsize=6.0 if bounds else 6.6)
    fig.tight_layout(pad=0.4)
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ler_latency_{short_name(data['circuit'])}_{args.sweep.stem.split('_', 2)[-1]}{args.suffix}"
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight", pad_inches=0.02)
    if rows_out:
        with open(args.fig_dir / f"{stem}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows_out for k in r}))
            w.writeheader(); w.writerows(rows_out)
    print(f"saved {args.fig_dir / stem}.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
