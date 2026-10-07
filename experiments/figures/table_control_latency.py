#!/usr/bin/env python3
"""Control-system latency table: compiled tree and worst-case latencies per
circuit, recomputed from the stored link files (gen_control_config output)
with the link model of algorithms/control_latency.py.

    python experiments/figures/table_control_latency.py [--q 14 --fanout 29] [--circuits ...]
    -> experiments/result/figures/control_latency_table.{md,csv}
"""
import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import stim                                                                # noqa: E402
from control_latency import profile                                        # noqa: E402
from naming import parse_name                                              # noqa: E402

DEFAULT = [f"end2end_d1={d1}_d2={d2}_r1={d1}_r2=0_p=0.001_inj=unitary_b=Y"
           for d1 in (3, 5) for d2 in (13, 15, 17)]


def row(folder: pathlib.Path, q: int, fanout: int, **link):
    d1, d2, _, _ = parse_name(folder.name)
    links = json.loads((folder / "control_system" / f"links_q{q}_f{fanout}.json").read_text())
    p = profile(links, **link)
    circuit = stim.Circuit.from_file(next(folder.glob("end2end*.stim")))
    leaves, routers, root = p["boards_per_layer"]
    return {"d_cultiv": d1, "d_escape": d2, "qubits": circuit.num_qubits,
            "tree": f"{leaves}/{routers}/{root}", "deliver_ns": round(p["deliver_ns"], 1),
            "gap_feedback_ns": round(p["gap_feedback_ns"], 1),
            "ps_feedback_max_ns": round(max(p["ps_feedback_ns"].values()), 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--circuits", nargs="*", default=DEFAULT)
    ap.add_argument("--q", type=int, default=14); ap.add_argument("--fanout", type=int, default=29)
    ap.add_argument("--gbps", type=float, default=10.0); ap.add_argument("--l-fixed-ns", type=float, default=157.0)
    ap.add_argument("--f-board-mhz", type=float, default=100.0)
    ap.add_argument("--out-dir", type=pathlib.Path, default=paths.RESULT / "figures")
    args = ap.parse_args()
    rows = [row(paths.CIRCUITS / c, args.q, args.fanout, gbps=args.gbps, l_fixed_ns=args.l_fixed_ns,
                f_board_mhz=args.f_board_mhz) for c in args.circuits]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    md = ["| d_cultiv/d_escape | Qubits | Tree (L/R/Rt) | L_del (ns) | L_gap (ns) | L_ps max (ns) |", "|---|---|---|---|---|---|"]
    md += [f"| {r['d_cultiv']}/{r['d_escape']} | {r['qubits']} | {r['tree']} | {r['deliver_ns']} | "
           f"{r['gap_feedback_ns']} | {r['ps_feedback_max_ns']} |" for r in rows]
    (args.out_dir / "control_latency_table.md").write_text("\n".join(md) + "\n")
    with open(args.out_dir / "control_latency_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n".join(md))


if __name__ == "__main__":
    main()
