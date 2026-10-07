#!/usr/bin/env python3
"""Mask slices: one time slice of the logical-ambiguity base mask (top K_Q
detectors) and of the same mask after structural closure, drawn on the code
patch. Red cells are retained detectors. Built in memory from the circuit's
stored logical-ambiguity essentials; no mask folder is needed.

    python experiments/figures/fig_mask_slices.py [<circuit folder>] [--kq 200] [--tick T]
    -> experiments/result/figures/mask_slice_{base,closure}.svg
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import stim                                                                # noqa: E402
from partial_mask_builder import PartialMaskBuilder                        # noqa: E402
from campaign import pick_essentials                                       # noqa: E402

DEFAULT = paths.CIRCUITS / "end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", nargs="?", type=pathlib.Path, default=DEFAULT)
    ap.add_argument("--kq", type=int, default=200, help="base-mask size K_Q")
    ap.add_argument("--tick", type=int, default=None,
                    help="cycle-slice tick to draw (default: the slice holding most base-mask detectors)")
    ap.add_argument("--out-dir", type=pathlib.Path, default=paths.RESULT / "figures")
    args = ap.parse_args()
    folder = args.folder.resolve()
    circuit = stim.Circuit.from_file(next(folder.glob("end2end*.stim")))
    b = PartialMaskBuilder(circuit=circuit)
    b.load_essentials_from_json(pick_essentials(folder, "logical_ambiguity"), type="logical_ambiguity")
    base, _, _ = b.logical_ambiguity_base_mask(args.kq, q_type="exp")
    closed, _, added = b.closure_mask(base)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _, _, meta = b.build_mask_from_indices(np.flatnonzero(base))
    ticks, codes = b._cycle_slices()
    if args.tick is None:
        in_slice = {t: int(base[b._detector_ids_for_tick(codes[t])].sum()) for t in meta["selected_slice_ticks"]}
        args.tick = max(in_slice, key=in_slice.get)
    for name, mask in (("base", base), ("closure", closed)):
        _, _, meta = b.build_mask_from_indices(np.flatnonzero(mask))
        meta["selected_slice_ticks"] = [args.tick]
        out = args.out_dir / f"mask_slice_{name}.svg"
        b.visualize_partial_region_mask_svg(out, mask_bool=mask, metadata=meta)
        n = int(mask[b._detector_ids_for_tick(codes[args.tick])].sum())
        print(f"wrote {out}  (mask {int(mask.sum())} detectors, {n} in slice tick {args.tick})")
    print(f"closure added {added} detectors")


if __name__ == "__main__":
    main()
