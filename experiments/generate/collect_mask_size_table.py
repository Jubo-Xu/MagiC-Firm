#!/usr/bin/env python3
"""Mask-size table for the mask-size sensitivity figure: one row per
Q_exp + closure mask of each circuit, from stored results only.

    detectors, total           mask size and detector count   (accuracy JSON)
    contracted/complete_vertices   decoder graph sizes         (partial stats.json)
    S                          crossing mass S(M)              (partial stats.json)
    p_equal/p_above/p_below    masked syndrome on the complete DEM vs complete gap
    p_equal_contracted         contracted DEM vs complete gap  (accuracy JSON)
    mb_us                      micro-blossom parallel mean latency, if characterized
    shots                      post-selected shots of the accuracy run

    python experiments/generate/collect_mask_size_table.py [<circuit folder> ...] [--out CSV]
    -> experiments/result/mask_ablation/mask_size_table.csv
"""
import argparse
import csv
import json
import pathlib
import re
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from naming import parse_name, freq_tag, FREQ_BY_D2                        # noqa: E402

ACC = paths.RESULT / "mask_ablation" / "accuracy"
MB = paths.RESULT / "mb_characterization"
DEFAULT_CIRCUITS = ["end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y",
                    "end2end_d1=5_d2=15_r1=5_r2=0_p=0.001_inj=unitary_b=Y"]
COLS = ["d1", "K_Q", "detectors", "total", "contracted_vertices", "complete_vertices", "S",
        "p_equal", "p_above", "p_below", "p_equal_contracted", "mb_us", "shots"]


def rows_for(folder: pathlib.Path):
    d1, d2, p, _ = parse_name(folder.name)
    out = []
    for pdir in sorted(folder.glob("partial_qexp*_cl_wc15")):
        m = re.fullmatch(r"partial_qexp(\d+)_cl_wc15", pdir.name)
        acc_path = ACC / folder.name / f"{pdir.name}_s0.json"
        if not m or not acc_path.exists():
            continue
        acc = json.loads(acc_path.read_text())
        st = json.loads((pdir / "stats.json").read_text())
        c = st["contraction"]
        mb_path = MB / f"{folder.name}__{pdir.name}__s0_n1000_{freq_tag(FREQ_BY_D2[d2])}_realsched.json"
        mb_us = json.loads(mb_path.read_text())["latency"]["parallel"]["mean_us"] if mb_path.exists() else ""
        a, b = acc["masked_on_complete_dem_vs_complete"], acc["contracted_vs_complete"]
        out.append({"d1": d1, "K_Q": int(m.group(1)), "detectors": acc["mask_detectors"],
                    "total": acc["num_detectors"], "contracted_vertices": c["n_kept"],
                    "complete_vertices": c["n_kept"] + c["n_clipped"], "S": st["mask"]["S_crossing_mass"],
                    "p_equal": a["p_equal"], "p_above": a["p_above"], "p_below": a["p_below"],
                    "p_equal_contracted": b["p_equal"], "mb_us": mb_us, "shots": acc["shots"]})
    return sorted(out, key=lambda r: r["K_Q"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="*", type=pathlib.Path, default=[paths.CIRCUITS / c for c in DEFAULT_CIRCUITS])
    ap.add_argument("--out", type=pathlib.Path, default=paths.RESULT / "mask_ablation" / "mask_size_table.csv")
    args = ap.parse_args()
    rows = [r for f in args.folders for r in rows_for(f.resolve())]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
