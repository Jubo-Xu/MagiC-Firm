# plot_style.py — colours and matplotlib settings shared by the figure scripts.
#
# Importing this module applies the base rcParams below (the figure scripts
# rely on that). Paper figures then override with PAPER_RC.

import matplotlib.pyplot as plt

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

# stacked preparation-time bars (evaluation figures): bottom to top
STACK = [("gate_us", "Gate", "#2a78d6"),
         ("feedback_us", "Feedback", "#eb6834"),
         ("gap_decode_us", "Gap decode", "#1baf7a")]
BAR_W, PAIR_GAP = 0.34, 0.05
LINE = {"Complete": dict(color=INK, ls="-", marker="o"),
        "Partial": dict(color=INK2, ls="--", marker="s")}

# paper figures
PAPER_RC = {"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 6.8,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans"}
PAPER = {"gate_us": "#c44e52", "control_us": "#4c72b0", "decode_us": "#dd8452"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8.5,
                     "axes.titlesize": 9, "legend.fontsize": 7.2, "xtick.labelsize": 7.8,
                     "ytick.labelsize": 7.8, "pdf.fonttype": 42, "ps.fonttype": 42})
