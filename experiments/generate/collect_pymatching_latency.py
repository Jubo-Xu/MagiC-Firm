#!/usr/bin/env python3
"""Software (pymatching sparse-blossom) decode latency for the DEM-contraction
ablation: per accepted shot, time the gap decode (obs-in + obs-out, serial) with
    complete      complete DEM, full syndrome
    masked        complete DEM, masked syndrome (unmasked detectors forced to 0)
    contracted    contracted DEM (partial_<tag>)
one shot per decode_batch call so the number is a per-shot latency; the obs-in and
obs-out decodes are timed separately: serial = on + off, parallel = max(on, off)
(the MB convention; the charts use parallel). `--repeats` passes with the per-shot
minimum kept (removes scheduler noise). Also records the defect count per shot.

    python experiments/generate/collect_pymatching_latency.py experiments/data/circuits/<config> \
        --partial partial_<tag> [--shots 2000 --repeats 3 --seed 0]
    -> experiments/result/mask_ablation/pymatching/<config>__<tag>_s<seed>.json (+ .npz per-shot)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from circuit_folder import load_circuit_folder, compile_samplers           # noqa: E402

OUT = paths.RESULT / "mask_ablation" / "pymatching"


def time_shots(dec, dets, repeats):
    """Per-shot decode times in us: obs-in ('on') and obs-out ('off') timed separately
    (same two pymatching calls as _decode_batch_overwrite_last_byte), min over repeats.
    serial = on + off, parallel = max(on, off) — the same two conventions as the MB results."""
    n = len(dets); on = np.full(n, np.inf); off = np.full(n, np.inf)
    for _ in range(repeats):
        for i in range(n):
            target = dec._repack_to_decoder_layout(dets[i:i + 1].copy())
            target[:, -1] |= dec._obs_det_byte
            t0 = time.perf_counter_ns()
            dec.gap_decoder.decode_batch(target, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
            t1 = time.perf_counter_ns()
            target[:, -1] ^= dec._obs_det_byte
            t2 = time.perf_counter_ns()
            dec.gap_decoder.decode_batch(target, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
            t3 = time.perf_counter_ns()
            on[i] = min(on[i], (t1 - t0) / 1000.0); off[i] = min(off[i], (t3 - t2) / 1000.0)
    return on, off


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", required=True)
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cf = load_circuit_folder(args.folder, args.partial)
    complete, part = compile_samplers(cf)
    gm = np.ones(cf.gap_circuit.num_detectors, dtype=np.bool_); gm[:cf.circuit.num_detectors] = cf.mask_bool
    mask_packed = np.packbits(gm, bitorder="little")
    sampler = cf.gap_circuit.compile_detector_sampler(seed=args.seed * 7919 + 1)
    rows = []
    while sum(len(r) for r in rows) < args.shots:
        dets, _ = sampler.sample(20_000, separate_observables=True, bit_packed=True)
        rows.append(dets[~np.any(dets & complete._discard_mask, axis=1)])
    dets = np.concatenate(rows)[:args.shots]
    masked = dets & mask_packed
    nbits = cf.gap_circuit.num_detectors
    def ndef(d):        # fired detectors per shot (full layout; obs/padding bits are 0 in samples)
        return np.unpackbits(d, axis=1, bitorder="little")[:, :nbits].sum(1)
    # warm-up
    for dec, d in ((complete, dets[:50]), (complete, masked[:50]), (part, dets[:50])):
        dec._decode_batch_overwrite_last_byte(d.copy())
    res = {}
    for name, dec, d in (("complete", complete, dets), ("masked", complete, masked), ("contracted", part, dets)):
        on, off = time_shots(dec, d, args.repeats)
        ser, par = on + off, np.maximum(on, off)
        defects = ndef(d) if name != "contracted" else ndef(masked)   # contracted sees the masked detectors
        def st_(a):
            return {"mean_us": float(a.mean()), "median_us": float(np.median(a)), "p99_us": float(np.percentile(a, 99))}
        res[name] = {"serial": st_(ser), "parallel": st_(par), "on": st_(on), "off": st_(off),
                     "defects_mean": float(defects.mean()), "on_us": on, "off_us": off, "defects": defects}
        print(f"  {name:10s} parallel mean {par.mean():7.1f} us  median {np.median(par):7.1f}  p99 {np.percentile(par, 99):7.1f}"
              f"  | serial median {np.median(ser):7.1f}  defects {defects.mean():.1f}")
    st = json.loads((args.folder / args.partial / "stats.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    stem = str(OUT / f"{args.folder.resolve().name}__{args.partial}_s{args.seed}")   # str: circuit names contain dots
    np.savez_compressed(stem + ".npz", **{f"{k}_on_us": v["on_us"] for k, v in res.items()},
                        **{f"{k}_off_us": v["off_us"] for k, v in res.items()},
                        **{f"{k}_defects": v["defects"] for k, v in res.items()})
    doc = {"circuit": args.folder.resolve().name, "partial": args.partial, "shots": int(len(dets)), "repeats": args.repeats,
           "graph": {"complete_vertices": st["desaturated_with_obs"]["num_detectors"], "complete_edges": st["desaturated_with_obs"]["n_edges"],
                     "contracted_vertices": st["contracted"]["num_detectors"], "contracted_edges": st["contracted"]["n_edges"],
                     "mask_detectors": st["mask"]["mask_sum"], "num_detectors": st["mask"]["num_detectors"]},
           "latency": {k: {kk: vv for kk, vv in v.items() if kk not in ("on_us", "off_us", "defects")} for k, v in res.items()}}
    pathlib.Path(stem + ".json").write_text(json.dumps(doc, indent=1))
    print(f"saved {stem}.json")


if __name__ == "__main__":
    main()
