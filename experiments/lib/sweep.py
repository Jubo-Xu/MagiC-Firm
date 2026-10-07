# sweep.py — readers of stored threshold sweeps and campaign summaries shared
# by the figure scripts: effective LER, Pareto frontier, iso-LER view, and the
# stacked preparation-time bars versus t_c.

from matplotlib.patches import Patch

from plot_style import PAPER

KEYS = [("gate_us", "Gate", PAPER["gate_us"]), ("control_us", "Control", PAPER["control_us"]),
        ("decode_us", "Gap decode", PAPER["decode_us"])]


def ler_eff(r, key="eval"):
    """Plotted LER of a sweep row: the held-out evaluation-half value when the
    sweep stores the split (key='select' gives the selection half), else the
    full-table value; a zero-error point gives its 95% upper bound (3/n)."""
    if f"ler_{key}" in r:
        e, n, l = r[f"errors_{key}"], r[f"n_{key}"], r[f"ler_{key}"]
    else:
        e, n, l = r["errors"], r["n_accepted"], r["ler"]
    return l if e > 0 else 3.0 / n


def errs(r, key="eval"):
    return r[f"errors_{key}"] if f"errors_{key}" in r else r["errors"]


def thin(rows, factor):
    """Rows to label: walking down in LER, keep those at least `factor` apart."""
    keep, last = [], None
    for r in sorted(rows, key=lambda r: -ler_eff(r)):
        if last is None or ler_eff(r) <= last / factor:
            keep.append(r); last = ler_eff(r)
    return keep


def frontier(rows, min_gain):
    """Pareto frontier of (LER, preparation time): chosen on the selection half
    when the sweep stores one, keeping points that improve time by min_gain."""
    rows = [r for r in rows if errs(r) > 0]
    sel = "select" if all("ler_select" in r for r in rows) else "eval"
    pts = sorted(rows, key=lambda r: (ler_eff(r, sel), r["total_us"]))
    out, best = [], float("inf")
    for r in pts:
        if r["total_us"] < best * (1.0 - min_gain):
            out.append(r); best = r["total_us"]
    return sorted(out, key=ler_eff)


def select(summary, **fixed):
    """Campaign-summary entries with an operating point, matching the fixed
    parameters (d1=, d2=, p=), sorted by (d2, p)."""
    return sorted((r for r in summary.values() if r.get("th_star") is not None
                   and all(abs(r[k] - v) < 1e-12 for k, v in fixed.items())),
                  key=lambda r: (r["d2"], r["p"]))


def iso_ler_view(r):
    """Copy of a summary entry whose Complete side is the LER-matched baseline."""
    v = dict(r)
    v["baseline"] = dict(r["tc_matched"]["baseline"])
    v["reduction"] = r["tc_matched"]["reduction"]
    return v


def draw_tc_sweep(ax, rows, *, legend=True, note=True, label_fs=6.2, ylabel="Wall-Clock Avg Preparation Time (μs)", fs=1.0):
    """Stacked gate / control / gap-decode bars of the preparation time versus
    t_c, with the accepted-state LER above each bar. Returns the y maximum."""
    W = 0.55
    for i, row in enumerate(rows):
        bottom = 0.0
        for k, _, c in KEYS:
            ax.bar(i, row[k], W, bottom=bottom, color=c, edgecolor="black", linewidth=0.6, zorder=3)
            bottom += row[k]
    ymax = max(r["total_us"] for r in rows)
    def compact(x):                      # 5.8e-05 -> 5.8e-5
        m, e = f"{x:.1e}".split("e")
        return f"{m}e{int(e)}"
    for i, row in enumerate(rows):
        lab = compact(row["ler"]) if row["errors"] > 0 else "<" + compact(3.0 / row["n_accepted"])   # 0 errors: 95 % bound
        ax.text(i, row["total_us"] + ymax * 0.02, lab, ha="center", va="bottom",
                fontsize=label_fs, color="black", rotation=90 if len(rows) > 7 else 0, zorder=5)
    ax.set_xticks(range(len(rows)), [f"{r['tc']:g}" for r in rows],
                  fontsize=7.5 * fs if len(rows) > 10 else None, rotation=90 if len(rows) > 12 else 0)
    ax.set_xlabel("complementary-gap threshold $t_c$ (dB)", fontsize=9.5 * fs)
    ax.set_ylabel(ylabel, fontsize=9.5 * fs)
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.yaxis.grid(True, color="#999999", lw=0.5, ls="--", zorder=0); ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_color("black"); sp.set_linewidth(0.8)
    ax.tick_params(colors="black", length=3, width=0.7)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, edgecolor="black", linewidth=0.6, label=l) for _, l, c in KEYS], ncol=1,
                  frameon=True, fancybox=False, edgecolor="black", framealpha=1.0, loc="upper left",
                  bbox_to_anchor=(0.012, 0.985), borderaxespad=0.0, handlelength=1.2, handleheight=0.9,
                  labelspacing=0.35, handletextpad=0.5, borderpad=0.4, fontsize=6.8 * fs)
    if note:
        ax.text(0.012, 0.66, "labels: LER of\naccepted states", transform=ax.transAxes, ha="left", va="top",
                fontsize=label_fs, color="black")
    ax.set_ylim(0, ymax * 1.36)
    return ymax
