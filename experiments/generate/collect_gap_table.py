#!/usr/bin/env python3
"""Per-shot gap table for LER-vs-threshold statistics — decoupled from the
runtime estimator (which only needs ~1M shots for converged *times*, while
LERs at high thresholds need 10^7-10^8 post-selected shots).

Samples shots of the circuit in worker processes, drops the post-selection
failures, decodes every surviving shot ONCE with the complete decoder (and,
with --partial, once with the partial decoder), and stores per shot:
    gc   complete gap (dB, ceil — the estimator's convention)
    gp   partial gap  (dB, ceil; -1 without --partial)
    err  complete-decoder prediction != actual observable
Any gating rule's LER / accept rate then follows from the table without
re-decoding (see `rule_stats`): single(tc): accept iff gc >= tc;
pcp(tc, tl, th): reject iff gp < tl, accept iff gp > th, else iff gc >= tc —
the correction is the complete decoder's in every tier, as in the estimator.

Storage (append-only growth): out/gap_tables/<config>/<variant>_s<seed>.npz +
.meta.json; re-running with a larger --shots adds only the missing shots
(chunk seeds derive from --seed and the chunk index, so growth is
reproducible and never resamples a stored shot).

    python experiments/generate/collect_gap_table.py experiments/data/circuits/<config> \
        [--partial partial_<tag>] --shots 100000000 [--workers 32] [--seed 0] [--batch 50000]
"""
import argparse
import json
import math
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from gap_table import rule_stats, mask_fingerprint, result_fingerprint, decode_chunk, GAP_TABLE_DIR  # noqa: E402

OUT = GAP_TABLE_DIR


# --------------------------------------------------------------------------- rules
# --------------------------------------------------------------------------- workers
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", default=None)
    ap.add_argument("--shots", type=int, required=True, help="post-selected shots wanted in the table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--batch", type=int, default=50_000, help="stim sampling batch per worker step")
    ap.add_argument("--chunk", type=int, default=250_000, help="post-selected shots per work chunk")
    args = ap.parse_args()

    name = args.folder.resolve().name
    variant = args.partial or "complete"
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    npz, meta_path = d / f"{variant}_s{args.seed}.npz", d / f"{variant}_s{args.seed}.meta.json"
    mask_sha = mask_fingerprint(args.folder, args.partial)
    if npz.exists():
        z = np.load(npz); meta = json.loads(meta_path.read_text())
        gc, gp, err = z["gc"], z["gp"], z["err"]
        attempts, passed, next_chunk = meta["attempts"], meta["passed"], meta["next_chunk"]
        assert meta["chunk"] == args.chunk, "chunk size must match the stored table"
        if meta.get("mask_sha") != mask_sha:                 # missing or different fingerprint: never mix
            sys.exit(f"error: {npz.name} has no valid fingerprint for the current mask/DEM "
                     f"(stored {meta.get('mask_sha')}, current {mask_sha}). A legacy table can be verified and "
                     f"stamped with experiments/tools/verify_gap_table.py; otherwise delete it to resample.")
    else:
        gc = np.zeros(0, np.int16); gp = np.zeros(0, np.int16); err = np.zeros(0, bool)
        attempts, passed, next_chunk = 0, 0, 0
    missing = args.shots - len(gc)
    if missing <= 0:
        print(f"{npz.name}: already {len(gc)} shots (>= {args.shots})")
    else:
        n_chunks = math.ceil(missing / args.chunk)
        wants = [args.chunk] * (n_chunks - 1) + [missing - args.chunk * (n_chunks - 1)]
        chunks = list(range(next_chunk, next_chunk + n_chunks))
        print(f"{name} {variant}: {len(gc)} stored, sampling {missing} more in {n_chunks} chunks on {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("fork")) as ex:
            parts = list(ex.map(decode_chunk, [args.folder] * n_chunks, [args.partial] * n_chunks,
                                [args.seed] * n_chunks, chunks, wants, [args.batch] * n_chunks))
        gc = np.concatenate([gc] + [p[0] for p in parts]); gp = np.concatenate([gp] + [p[1] for p in parts])
        err = np.concatenate([err] + [p[2] for p in parts])
        attempts += sum(p[3] for p in parts); passed += sum(p[4] for p in parts)
        next_chunk += n_chunks
        np.savez_compressed(npz, gc=gc, gp=gp, err=err)
        meta_path.write_text(json.dumps(dict(circuit=name, partial=args.partial, seed=args.seed, chunk=args.chunk,
                                             shots=int(len(gc)), attempts=int(attempts), passed=int(passed),
                                             next_chunk=next_chunk, postselect_acceptance=passed / attempts,
                                             mask_sha=mask_sha), indent=1))
    n, e, ler = rule_stats(gc, gp, err, tc=35)
    print(f"table: {len(gc)} post-selected shots (post-selection acceptance {passed/max(attempts,1):.4f}); "
          f"single tc35: accept {n/len(gc):.4f}, LER {ler:.2e} ({e} errs)")
    if args.partial:
        for th in (40, 47, 60):
            n, e, ler = rule_stats(gc, gp, err, tc=35, tl=0, th=th)
            print(f"  pcp tc35 tl0 th{th}: accept {n/len(gc):.4f}, LER {ler:.2e} ({e} errs)")
    print(f"saved {npz}")


if __name__ == "__main__":
    main()
