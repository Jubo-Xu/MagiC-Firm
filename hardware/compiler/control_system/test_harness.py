"""test_harness.py — validate the compiled masks against stim.

Drives the compiled per-board programs with stim-sampled measurements and checks
the constructed detectors match stim's detectors by global index (det_idx == stim
index). Runs the ACTUAL selector + core (select/emit) masks, so it validates the
compiler, not just the parser's adjacency.
"""

from __future__ import annotations

import numpy as np
import stim

from parser import DetectorConstructionScanner
from compiler import Compiler


def run_test(circuit: stim.Circuit, comp: Compiler, shots: int = 2000, seed: int = 1):
    s = comp.s
    ndet = len(s.detector_coord)

    # --- Step 2: ground truth (detectors derived from the sampled measurements) ---
    sampler = circuit.compile_sampler(seed=seed)
    M = sampler.sample(shots)                                   # (shots, num_measurements) bool
    D_stim = circuit.compile_m2d_converter().convert(measurements=M, separate_observables=False)

    # --- Step 3: value_at[t][qubit] = column vector over shots (from mrr_flat) ---
    mrr_flat = s._measurement_rec_raw_flat
    value_at: dict = {}                                         # meas_time -> {qubit: np.ndarray(shots)}
    seen = {}
    for i, (q, t) in enumerate(mrr_flat):
        if q is None:                                          # observable placeholder slot
            continue
        key = (q, t)
        assert key not in seen, f"records {seen[key]} and {i} share (qubit={q}, meas_time={t})"
        seen[key] = i
        value_at.setdefault(t, {})[q] = M[:, i]

    def meas(t, q):
        col = value_at.get(t, {}).get(q)
        assert col is not None, f"no measurement for qubit {q} at meas_time {t}"
        return col

    # --- Step 4: emulate construction with the compiled masks ---
    constructed = [None] * ndet
    for prog in comp.programs.values():
        for kp in prog.kernels:
            acc = [np.zeros(shots, dtype=bool) for _ in range(kp.h)]
            emit_idx = 0
            T = len(s.measurement_rec)
            for t in range(T):
                for core in range(kp.h):
                    step = kp.core_steps[core].get(t)
                    if step is None:
                        continue
                    sel, emit = step
                    for si in sel:
                        acc[core] ^= meas(t, kp.support[si])
                    if emit:
                        d = kp.detectors[emit_idx]
                        assert core == emit_idx % kp.h, \
                            f"counter mismatch: det {d} emit_idx {emit_idx} on core {core}"
                        constructed[d] = acc[core].copy()
                        acc[core][:] = False                  # emit = send + clear
                        emit_idx += 1
            assert emit_idx == len(kp.detectors)

    # --- Step 5: compare all detectors ---
    missing = [d for d in range(ndet) if constructed[d] is None]
    assert not missing, f"{len(missing)} detectors never constructed, e.g. {missing[:8]}"
    C = np.stack(constructed, axis=1)                          # (shots, ndet)
    mismatch = C != D_stim
    bad_dets = np.where(mismatch.any(axis=0))[0]
    print(f"  detectors: {ndet}, shots: {shots}")
    print(f"  detector-value mismatches: cells={int(mismatch.sum())}  "
          f"detectors_with_any_mismatch={len(bad_dets)}")
    if len(bad_dets):
        d = int(bad_dets[0])
        print(f"    e.g. detector {d} (coord {s.detector_coord[d][0]}, new_t {s.detector_coord[d][2]}): "
              f"mismatched in {int(mismatch[:, d].sum())} shots")

    # --- Step 6: root output contract ---
    o = comp.outputs[comp.cfg.root_id]
    out_bad = 0
    for nt, lines in o.sync_det_mask.items():
        for ln in lines:
            d = o.global_det_index[nt][ln]
            if not np.array_equal(C[:, d], D_stim[:, d]):
                out_bad += 1
    print(f"  root output: {len(o.out_lines)} lines, "
          f"{'OK' if out_bad == 0 else f'{out_bad} bad slots'}")

    # --- Step 7: postselect (OR of a stage's postselect detectors matches stim) ---
    ps_ok = True
    for bid, out in comp.outputs.items():
        for si in out.postselect_mask:
            idxs = _postselect_indices(comp, si)
            emu = np.zeros(shots, dtype=bool)
            ref = np.zeros(shots, dtype=bool)
            for d in idxs:
                emu |= C[:, d]
                ref |= D_stim[:, d]
            ok_s = np.array_equal(emu, ref)
            ps_ok &= ok_s
            print(f"  postselect board {bid} stage {si}: {len(idxs)} detectors  "
                  f"{'OK' if ok_s else 'MISMATCH'}")

    ok = len(bad_dets) == 0 and out_bad == 0 and ps_ok
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def _postselect_indices(comp, stage_idx):
    s = comp.s
    return [d for d in s.stages[stage_idx]["detectors"]
            if d in comp.cfg.postselect_detectors]


def _load_postselect_list(json_path, circuit: stim.Circuit) -> list:
    """Explicit postselect detector list from a decoder-workflow
    postselected_detectors.json, sanity-checked against the circuit."""
    import json as _json
    doc = _json.loads(pathlib.Path(json_path).read_text())
    dets = doc["postselected_detectors"] if isinstance(doc, dict) else doc
    if isinstance(doc, dict) and "n_circuit_dets" in doc:
        assert int(doc["n_circuit_dets"]) == circuit.num_detectors, (
            f"{json_path}: n_circuit_dets={doc['n_circuit_dets']} but circuit "
            f"has {circuit.num_detectors} detectors — wrong file for this circuit")
    return list(dets)


if __name__ == "__main__":
    import pathlib
    import sys
    sys.path.insert(0, ".")
    from placement import make_partition_config

    # (circuit, postselect json from the decoder workflow — explicit detector
    # lists; the legacy 'cultivation' preset was removed from the compiler)
    default_cases = [
        ("../../../magic_state_cultivation/circuits_dump/"
         "c=end2end-inplace-distillation,p=0.001,noise=uniform,g=css,q=166,b=Y,r=11,r1=3,d1=3,r2=3,d2=9.stim",
         "data/cultivation_d1_3_d2_9/postselected_detectors.json"),
        ("../../../magic_state_cultivation/circuits_dump/"
         "c=end2end-inplace-distillation,p=0.001,noise=uniform,g=css,q=255,b=Y,r=16,r1=3,d1=5,r2=3,d2=11.stim",
         "data/cultivation_d1_5_d2_11/postselected_detectors.json"),
    ]
    cases = [(p, None) for p in sys.argv[1:]] or default_cases
    for path, ps_json in cases:
        print("=" * 70)
        print(path.split("/")[-1])
        circuit = stim.Circuit.from_file(path)
        postselect = "none" if ps_json is None else _load_postselect_list(ps_json, circuit)
        s = DetectorConstructionScanner.from_file(path)
        s.scan(); s.refine()
        comp = Compiler(make_partition_config(s, 4, postselect=postselect), s)
        run_test(circuit=circuit, comp=comp)
