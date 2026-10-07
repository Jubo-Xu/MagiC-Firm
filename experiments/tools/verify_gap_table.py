#!/usr/bin/env python3
"""Verify a gap table against the CURRENT mask/DEM by regenerating its first chunk
(chunk sampling is deterministic: seed*1_000_003 + chunk) and comparing the stored
per-shot (g_c, g_p, err) bit for bit. On a match the table's meta is stamped with the
current fingerprint (collect_gap_table.mask_fingerprint); on a mismatch it is left
unstamped and reported, so every consumer refuses it.

    python experiments/tools/verify_gap_table.py experiments/data/circuits/<config> --partial partial_<tag> [--seed 0]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from gap_table import decode_chunk, mask_fingerprint, GAP_TABLE_DIR as OUT  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=50_000, help="must equal the batch the table was sampled with")
    args = ap.parse_args()
    name = args.folder.resolve().name; variant = args.partial or "complete"
    npz = OUT / name / f"{variant}_s{args.seed}.npz"; meta_path = npz.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()); z = np.load(npz)
    want = int(meta["chunk"])
    gc, gp, err, _, _ = decode_chunk(args.folder, args.partial, args.seed, 0, want, args.batch)
    same = (np.array_equal(gc, z["gc"][:want]) and np.array_equal(gp, z["gp"][:want]) and np.array_equal(err, z["err"][:want]))
    fp = mask_fingerprint(args.folder, args.partial)
    if same:
        meta["mask_sha"] = fp; meta["fingerprint_source"] = f"verified: chunk 0 ({want} shots) regenerated and identical"
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"OK   {npz.name}: chunk 0 identical under the current mask/DEM -> stamped {fp}")
    else:
        d = int((gc != z["gc"][:want]).sum()); dp = int((gp != z["gp"][:want]).sum())
        print(f"FAIL {npz.name}: chunk 0 differs (g_c {d}, g_p {dp} of {want} shots) -> NOT stamped; delete and resample")
        sys.exit(1)


if __name__ == "__main__":
    main()
