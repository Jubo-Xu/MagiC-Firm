"""validate_serialized.py — round-trip check: reconstruct detectors from the
SERIALIZED regfiles alone (no compiler objects) and confirm they match stim.

Proves the emitted artifact (per-board JSON regfiles + measurement_map) is
self-sufficient: the emulator/FPGA can rely on the files only.
"""

from __future__ import annotations

import json
import os

import numpy as np
import stim


def _load(path):
    with open(path) as f:
        return json.load(f)


def validate(example_dir, shots=1000, seed=1):
    manifest = _load(os.path.join(example_dir, "manifest.json"))
    mrr = _load(os.path.join(example_dir, "measurement_map.json"))     # [[qubit|null, meas_time], ...]
    ndet, T = manifest["detectors"], manifest["meas_times"]
    ndt = manifest["det_times"]    # real det-times; a copy-last wait row (if any) sits at nt==ndt
    circuit = stim.Circuit.from_file(os.path.join(example_dir, manifest["circuit"]))

    # ground truth: detectors derived from the sampled measurements
    M = circuit.compile_sampler(seed=seed).sample(shots)
    D_stim = circuit.compile_m2d_converter().convert(measurements=M, separate_observables=False)

    # value_at[meas_time][qubit] = column over shots (from the measurement map)
    value_at = {}
    for i, (q, t) in enumerate(mrr):
        if q is None:
            continue
        value_at.setdefault(t, {})[q] = M[:, i]

    constructed = [None] * ndet
    for bd in manifest["boards"]:
        bid = bd["board_id"]
        jdir = os.path.join(example_dir, f"board{bid}", "json")
        if "sync" not in bd["regfiles"]:                    # no kernels on this board
            continue
        channel_qubit = _load(os.path.join(jdir, "sync.json"))["channel_qubit"]
        for rf in bd["regfiles"]:
            if not rf.endswith("_core"):
                continue
            prefix = rf[:-len("_core")]                     # e.g. "k3"
            core = _load(os.path.join(jdir, f"{rf}.json"))
            if not core["detectors"]:                       # idle kernel (physical layout)
                continue
            sel = _load(os.path.join(jdir, f"{prefix}_selector.json"))
            # select-bit -> qubit (sentinel selector lines are padding, never referenced)
            support = [channel_qubit[line] if line < len(channel_qubit) else None
                       for line in sel["lines"]]
            h, steps, dets = core["h"], core["steps"], core["detectors"]

            acc = [np.zeros(shots, dtype=bool) for _ in range(h)]
            emit_idx = 0
            for t in range(T):
                for c in range(h):
                    step = steps.get(str(c), {}).get(str(t))
                    if step is None:
                        continue
                    for si in step["select"]:
                        acc[c] ^= value_at[t][support[si]]
                    if step["emit"]:
                        constructed[dets[emit_idx]] = acc[c].copy()
                        acc[c][:] = False
                        emit_idx += 1
            assert emit_idx == len(dets)

    missing = [d for d in range(ndet) if constructed[d] is None]
    assert not missing, f"{len(missing)} detectors never constructed, e.g. {missing[:8]}"
    C = np.stack(constructed, axis=1)
    mismatch = C != D_stim

    # also confirm the root global_index maps to correct stim indices
    root_bid = next(bd["board_id"] for bd in manifest["boards"] if "global_index" in bd["regfiles"])
    gi = _load(os.path.join(example_dir, f"board{root_bid}", "json", "global_index.json"))
    osync = _load(os.path.join(example_dir, f"board{root_bid}", "json", "output_sync.json"))
    out_bad = 0
    for nt, active in enumerate(osync["entries"]):
        if nt >= ndt:          # appended copy-last wait row: no stim ground truth, skip
            continue
        for ln in active:
            d = gi["rows"][nt][ln]
            if not np.array_equal(C[:, d], D_stim[:, d]):
                out_bad += 1

    bad = int(mismatch.any(axis=0).sum())
    print(f"  {os.path.basename(example_dir)}: dets={ndet} shots={shots}  "
          f"mismatched_detectors={bad}  root_output_bad_slots={out_bad}  "
          f"{'PASS' if bad == 0 and out_bad == 0 else 'FAIL'}")
    return bad == 0 and out_bad == 0


if __name__ == "__main__":
    import sys
    dirs = sys.argv[1:] or [os.path.join("results", d) for d in sorted(os.listdir("results"))
                            if os.path.isdir(os.path.join("results", d))]
    ok = all(validate(d) for d in dirs)
    print("ALL PASS" if ok else "SOME FAILED")
