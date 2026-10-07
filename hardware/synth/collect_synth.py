#!/usr/bin/env python3
"""collect_synth.py — gather the per-board Vivado results of gen_synth.py bundles into the
paper table: per bundle (circuit) and board ROLE (leaf / router / root), the maximum resource
usage and the minimum Fmax over the synthesized boards of that role, plus which board set each.

Fmax per board = 1 / (period - WNS) from summary.json. If WNS >= 0 the board MET the target,
so that value is only a lower bound (the tools stop optimizing once timing closes); such rows
are flagged 'met' and should be re-run at a tighter --period-ns for a true number.

Usage: collect_synth.py <bundle root> [--result-dir experiments/result/synthesis]
       (writes results.json, table.md and boards.csv there; --csv/--md/--json override single files)
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re

UTIL_ROWS = {"CLB LUTs": "lut", "LUT as Logic": "lut_logic", "LUT as Memory": "lut_mem",
             "CLB Registers": "ff", "Block RAM Tile": "bram", "DSPs": "dsp"}
_ROW = re.compile(r"^\|\s*(" + "|".join(re.escape(k) for k in UTIL_ROWS) + r")\s*\|\s*([\d.]+)\s*\|")


def parse_util(path):
    """Used-count column of the report_utilization summary tables (first occurrence of each row)."""
    out = {}
    with open(path) as f:
        for ln in f:
            m = _ROW.match(ln)
            if m and UTIL_ROWS[m.group(1)] not in out:
                v = float(m.group(2))
                out[UTIL_ROWS[m.group(1)]] = int(v) if v == int(v) else v
    return out


def collect(root):
    """One row per selected board of every bundle under root (status: ok / met / missing)."""
    rows = []
    for bj in sorted(glob.glob(os.path.join(root, "*", "boards.json"))):
        bdir = os.path.dirname(bj)
        info = json.load(open(bj))
        name = os.path.basename(bdir)
        for b, p in info["boards"].items():
            run = os.path.join(bdir, "runs", f"board{b}")
            sj = os.path.join(run, "summary.json")
            row = {"bundle": name, "board": int(b), **p}
            if not os.path.exists(sj):
                rows.append({**row, "status": "missing"})
                continue
            s = json.load(open(sj))
            u = parse_util(os.path.join(run, "util_route.rpt"))
            rows.append({**row, "status": "met" if s["wns_ns"] >= 0 else "ok", **u,
                         "period_ns": s["period_ns"], "wns_ns": s["wns_ns"], "fmax_mhz": s["fmax_mhz"],
                         "crit_start": s["crit_start"], "crit_end": s["crit_end"]})
    return rows


def summarize(rows):
    """Per (bundle, role): max of each resource and min Fmax, each with the board that set it."""
    out = {}
    for r in rows:
        if r["status"] == "missing":
            continue
        g = out.setdefault((r["bundle"], r["role"]), {"bundle": r["bundle"], "role": r["role"], "n": 0})
        g["n"] += 1
        for k in ("lut", "ff", "bram", "dsp"):
            if k in r and (k not in g or r[k] > g[k]):
                g[k], g[k + "_board"] = r[k], r["board"]
        if "fmax_mhz" not in g or r["fmax_mhz"] < g["fmax_mhz"]:
            g["fmax_mhz"], g["fmax_board"], g["fmax_status"] = r["fmax_mhz"], r["board"], r["status"]
    return [out[k] for k in sorted(out)]


def markdown(summary):
    lines = ["| circuit | role | boards | LUT (board) | FF (board) | BRAM (board) | Fmax MHz (board) |",
             "|---|---|---|---|---|---|---|"]
    for g in summary:
        fm = f"{g['fmax_mhz']:.0f} ({g['fmax_board']})" + (" met, lower bound" if g["fmax_status"] == "met" else "")
        lines.append(f"| {g['bundle']} | {g['role']} | {g['n']} | {g['lut']} ({g['lut_board']}) | "
                     f"{g['ff']} ({g['ff_board']}) | {g['bram']} ({g['bram_board']}) | {fm} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="bundle root (gen_synth.py --out-dir)")
    ap.add_argument("--csv", default=None, help="write every board's row")
    ap.add_argument("--md", default=None, help="write the per-role summary table")
    ap.add_argument("--json", default=None, help="write rows + summary")
    ap.add_argument("--result-dir", default=None,
                    help="write results.json, table.md and boards.csv here (default: experiments/result/synthesis)")
    args = ap.parse_args()
    if not (args.csv or args.md or args.json):
        d = args.result_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "experiments", "result", "synthesis")
        os.makedirs(d, exist_ok=True)
        args.csv, args.md, args.json = (os.path.join(d, n) for n in ("boards.csv", "table.md", "results.json"))

    rows = collect(args.root)
    summary = summarize(rows)
    missing = [f"{r['bundle']}/board{r['board']}" for r in rows if r["status"] == "missing"]
    if args.csv:
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("bundle", "board", "role", "status"), k))
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    if args.md:
        with open(args.md, "w") as f:
            f.write(markdown(summary))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"boards": rows, "summary": summary}, f, indent=1)
    print(markdown(summary))
    if missing:
        print(f"missing results: {', '.join(missing)}")


if __name__ == "__main__":
    main()
