#!/usr/bin/env python3
"""FPGA resource table: per board role, the maximum post-route resource usage
and the minimum Fmax over the synthesized configurations, from the stored
synthesis summary (hardware/synth collect_synth.py output).

    python experiments/figures/table_fpga_resources.py [--results experiments/result/synthesis/results.json]
    -> experiments/result/figures/fpga_resources_table.{md,csv}
"""
import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702

ROLES = ["leaf", "router", "root"]
CAPACITY = {"lut": 4_085_760, "ff": 8_171_520, "bram": 2_160, "dsp": 3_840}   # AMD Virtex UltraScale+ VU19P


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=pathlib.Path, default=paths.RESULT / "synthesis" / "results.json")
    ap.add_argument("--target-mhz", type=float, default=500.0,
                    help="clock target of the runs; a role whose boards all met it reports Fmax >= target")
    ap.add_argument("--out-dir", type=pathlib.Path, default=paths.RESULT / "figures")
    args = ap.parse_args()
    summary = json.loads(args.results.read_text())["summary"]
    rows = []
    for role in ROLES:
        s = [e for e in summary if e["role"] == role]
        if not s:
            continue
        met = all(e["fmax_status"] == "met" for e in s)
        rows.append({"role": role, **{k: max(e[k] for e in s) for k in ("lut", "ff", "bram", "dsp")},
                     "fmax_mhz": f">= {args.target_mhz:g}" if met else f"{min(e['fmax_mhz'] for e in s):.0f}",
                     "configs": len(s)})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = ["| Role | LUT (/{:,}) | FF (/{:,}) | BRAM (/{:,}) | DSP (/{:,}) | Fmax (MHz) |".format(
          CAPACITY["lut"], CAPACITY["ff"], CAPACITY["bram"], CAPACITY["dsp"]), "|---|---|---|---|---|---|"]
    md += [f"| {r['role'].capitalize()} | {r['lut']:,} | {r['ff']:,} | {r['bram']:g} | {r['dsp']} | {r['fmax_mhz']} |" for r in rows]
    (args.out_dir / "fpga_resources_table.md").write_text("\n".join(md) + "\n")
    with open(args.out_dir / "fpga_resources_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
