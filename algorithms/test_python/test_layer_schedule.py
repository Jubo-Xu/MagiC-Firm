# Tests for algorithms/layer_schedule.py: timing cross-check against the
# compiler's independent implementation, detector/measurement mapping
# sanity, and real-folder schedule properties.

import json
import pathlib
import sys

import numpy as np
import pytest
import stim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import layer_schedule
from gate_times import GOOGLE_GATE_TIMES_NS

REPO = pathlib.Path(__file__).resolve().parents[2]
FOLDER = (REPO / "experiments" / "data" / "circuits"
          / "end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y")
OUT_MB = (REPO / "out").resolve() / "mb" if (REPO / "out").exists() else None


def small_circuit() -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=4,
        after_clifford_depolarization=0.01)


def test_cross_check_against_compiler_timing():
    sys.path.insert(0, str(REPO / "hardware" / "compiler" / "control_system"))
    import timing as compiler_timing

    for circuit in filter(None, [
        small_circuit(),
        stim.Circuit.from_file(FOLDER / "gap_circuit.stim") if FOLDER.exists() else None,
    ]):
        ours, _ = layer_schedule.circuit_timing(circuit, GOOGLE_GATE_TIMES_NS)
        theirs = compiler_timing.round_completion_ns(circuit, GOOGLE_GATE_TIMES_NS)
        assert ours == theirs, "algorithms vs compiler timing disagree"


def test_meas_round_ids_consistent():
    c = small_circuit()
    round_ns, meas_rounds = layer_schedule.circuit_timing(c)
    assert len(meas_rounds) == c.num_measurements
    assert meas_rounds[0] == 0 and meas_rounds[-1] == len(round_ns) - 1
    assert all(b >= a for a, b in zip(meas_rounds, meas_rounds[1:]))


def test_detector_last_measurements():
    c = small_circuit()
    last = layer_schedule.detector_last_measurements(c)
    assert len(last) == c.num_detectors
    assert all(0 <= m < c.num_measurements for m in last)
    # surface-code memory: later detectors never complete before earlier rounds
    _, meas_rounds = layer_schedule.circuit_timing(c)
    det_rounds = [meas_rounds[m] for m in last]
    assert all(b >= a for a, b in zip(det_rounds, det_rounds[1:]))


def test_uniform_rounds_give_uniform_offsets():
    # surface-code bulk rounds are structurally identical -> equal spacing
    c = small_circuit()
    round_ns, _ = layer_schedule.circuit_timing(c)
    deltas = np.diff(round_ns)
    assert len(set(np.round(deltas[1:], 6))) == 1, f"bulk rounds not uniform: {deltas}"


@pytest.mark.skipif(not FOLDER.exists() or OUT_MB is None
                    or not (OUT_MB / f"{FOLDER.name}__complete" / "graph.json").exists(),
                    reason="real decoder graphs not generated")
def test_real_folder_schedules():
    gap = stim.Circuit.from_file(FOLDER / "gap_circuit.stim")

    gc = json.loads((OUT_MB / f"{FOLDER.name}__complete" / "graph.json").read_text())
    complete = layer_schedule.fusion_layer_offsets_ns(gc, gap)
    assert len(complete) == gc["layer_fusion"]["num_layers"]
    assert complete[0] == 0
    assert all(b >= a for a, b in zip(complete, complete[1:]))
    # boundary policy: terminal MPP round priced like the previous round
    raw, _ = layer_schedule.circuit_timing(gap)
    assert complete[-1] - complete[-2] == int(round(raw[-2] - raw[-3]))

    partial_key = (f"{FOLDER.name}__partial_l1-14_t1-9_ts1-7_"
                   "aug-canonical50_pr-causal50_wc15")
    pdir = OUT_MB / partial_key
    if pdir.exists():
        kept = np.load(FOLDER / "partial_l1-14_t1-9_ts1-7_aug-canonical50_pr-causal50_wc15"
                       / "kept.npy")
        gp = json.loads((pdir / "graph.json").read_text())
        partial = layer_schedule.fusion_layer_offsets_ns(gp, gap, kept=kept)
        assert len(partial) == gp["layer_fusion"]["num_layers"]
        assert partial[0] == 0 and all(b >= a for a, b in zip(partial, partial[1:]))
        # the mask's time window ends no later than the full circuit
        assert partial[-1] <= complete[-1]
