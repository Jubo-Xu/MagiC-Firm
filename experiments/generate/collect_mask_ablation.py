#!/usr/bin/env python3
"""Partial-mask accuracy table: for one circuit and one mask folder, sample
post-selected shots and decode each shot three ways
    g_c   complete decoder, full syndrome                    (reference)
    g_m   complete decoder (complete DEM), MASKED syndrome    (unmasked detectors forced to 0)
    g_p   contracted decoder (partial_<tag>/contracted.dem)   (the deployed partial decoder)
and store the per-shot gaps -> out/mask_ablation/<config>/<tag>_s<seed>.npz plus a
summary JSON with, for each pair, P(g_x = g_ref), P(g_x < g_ref), P(g_x > g_ref) and
mean |d| (equality = |difference| < --tol dB, default 0.5, i.e. same integer dB).

Steps 1/2 of the mask ablation read (g_m vs g_c): the mask alone, on the reference
decoder. Step 3 reads (g_p vs g_c) and (g_p vs g_m): what DEM contraction costs.

    python experiments/generate/collect_mask_ablation.py experiments/data/circuits/<config> \
        --partial partial_<tag> --shots 200000 [--workers 32 --seed 0]
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from circuit_folder import load_circuit_folder, compile_samplers           # noqa: E402
from gap_table import mask_fingerprint                                     # noqa: E402

OUT = paths.OUT / "mask_ablation"                           # per-shot gaps (.npz)
ACC = paths.RESULT / "mask_ablation" / "accuracy"           # summary JSON (read by the figures)


def pair_stats(gx, gref, tol):
    d = gx - gref
    return {"p_equal": float(np.mean(np.abs(d) < tol)), "p_below": float(np.mean(d <= -tol)),
            "p_above": float(np.mean(d >= tol)), "mean_abs_diff": float(np.mean(np.abs(d))),
            "mean_diff": float(np.mean(d)), "pearson": float(np.corrcoef(gx, gref)[0, 1])}


def decode_chunk(folder, partial, seed, chunk, want, batch):
    cf = load_circuit_folder(folder, partial)
    complete, part = compile_samplers(cf)
    # mask in the gap-circuit detector layout: original detectors masked, padding +
    # obs detector untouched (the obs byte is overwritten by the gap decode anyway)
    gm = np.ones(cf.gap_circuit.num_detectors, dtype=np.bool_)
    gm[:cf.circuit.num_detectors] = cf.mask_bool
    mask_packed = np.packbits(gm, bitorder="little")
    sampler = cf.gap_circuit.compile_detector_sampler(seed=seed * 1_000_003 + chunk)
    gcs, gms, gps, errs = [], [], [], []
    attempts = passed = got = 0
    while got < want:
        dets, obs = sampler.sample(batch, separate_observables=True, bit_packed=True)
        keep = ~np.any(dets & complete._discard_mask, axis=1)
        attempts += batch; passed += int(keep.sum())
        dets, obs = dets[keep], obs[keep][:, 0]
        if not len(dets):
            continue
        take = min(len(dets), want - got)
        dets, obs = dets[:take], obs[:take]
        _, gp = part._decode_batch_overwrite_last_byte(dets)                 # repacks, non-mutating
        _, gmk = complete._decode_batch_overwrite_last_byte(dets & mask_packed)   # copy: masked syndrome
        pred, gc = complete._decode_batch_overwrite_last_byte(dets)          # mutates dets (last)
        gcs.append(gc.astype(np.float32)); gms.append(gmk.astype(np.float32)); gps.append(gp.astype(np.float32))
        errs.append(pred.astype(bool) != obs.astype(bool))
        got += take
    return (np.concatenate(gcs), np.concatenate(gms), np.concatenate(gps), np.concatenate(errs),
            attempts, passed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--shots", type=int, default=200_000, help="post-selected shots")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--batch", type=int, default=20_000)
    ap.add_argument("--tol", type=float, default=0.5, help="|d| < tol dB counts as equal")
    args = ap.parse_args()

    name = args.folder.resolve().name
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    npz = out / f"{args.partial}_s{args.seed}.npz"
    acc = ACC / name; acc.mkdir(parents=True, exist_ok=True)
    meta_path = acc / f"{args.partial}_s{args.seed}.json"
    mask_sha = mask_fingerprint(args.folder, args.partial)
    if npz.exists():
        z = np.load(npz); gc, gm, gp, err = z["gc"], z["gm"], z["gp"], z["err"]
        meta = json.loads(meta_path.read_text())
        if meta.get("mask_sha") != mask_sha:
            sys.exit(f"error: {npz.name} has no valid fingerprint for the current mask/DEM "
                     f"(stored {meta.get('mask_sha')}, current {mask_sha}); delete it to recollect")
        print(f"{npz.name}: {len(gc)} shots already stored")
    else:
        K = args.workers
        shares = [args.shots // K + (1 if k < args.shots % K else 0) for k in range(K)]
        with ProcessPoolExecutor(max_workers=K, mp_context=get_context("fork")) as ex:
            parts = list(ex.map(decode_chunk, [args.folder] * K, [args.partial] * K, [args.seed] * K,
                                range(K), shares, [args.batch] * K))
        gc = np.concatenate([p[0] for p in parts]); gm = np.concatenate([p[1] for p in parts])
        gp = np.concatenate([p[2] for p in parts]); err = np.concatenate([p[3] for p in parts])
        attempts = sum(p[4] for p in parts); passed = sum(p[5] for p in parts)
        np.savez_compressed(npz, gc=gc, gm=gm, gp=gp, err=err)
        st = json.loads((args.folder / args.partial / "stats.json").read_text())
        meta = {"circuit": name, "partial": args.partial, "seed": args.seed, "shots": int(len(gc)),
                "postselect_acceptance": passed / attempts, "tol_db": args.tol,
                "mask_detectors": st["mask"]["mask_sum"], "num_detectors": st["mask"]["num_detectors"],
                "contracted_detectors": st.get("contracted", {}).get("num_detectors"),
                "complete_ler": float(err.mean()), "gap_cap_complete": float(gc.max()), "mask_sha": mask_sha}
    meta["masked_on_complete_dem_vs_complete"] = pair_stats(gm, gc, args.tol)
    meta["contracted_vs_complete"] = pair_stats(gp, gc, args.tol)
    meta["contracted_vs_masked_on_complete_dem"] = pair_stats(gp, gm, args.tol)
    meta_path.write_text(json.dumps(meta, indent=1))
    for k in ("masked_on_complete_dem_vs_complete", "contracted_vs_complete", "contracted_vs_masked_on_complete_dem"):
        s = meta[k]
        print(f"  {k:40s} P(=) {s['p_equal']:.4f}  P(<) {s['p_below']:.4f}  P(>) {s['p_above']:.4f}  "
              f"mean|d| {s['mean_abs_diff']:.2f}  r {s['pearson']:.4f}")
    print(f"saved {npz} and {meta_path}  mask {meta['mask_detectors']}/{meta['num_detectors']} "
          f"contracted {meta['contracted_detectors']}")


if __name__ == "__main__":
    main()
