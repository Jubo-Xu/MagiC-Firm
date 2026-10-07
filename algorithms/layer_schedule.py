# layer_schedule.py — when each decoder graph layer's syndrome data arrives,
# derived from a stim circuit and a gate-time table.
#
# A layer's detectors map to their last contributing measurement, then to that
# measurement's round. Detector coordinates are not used: they are not
# continuous in cultivation circuits.
# A TICK slice costs its max op duration and slices sum. Must agree with the
# compiler's timing.py (checked by test_layer_schedule).

import stim

from gate_times import GOOGLE_GATE_TIMES_NS

_ZERO = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"}


def _inst_duration_ns(inst: stim.CircuitInstruction, gt: dict) -> float:
    """Duration in ns. Noise channels cost 0; MR costs MEAS + RESET."""
    name = inst.name
    if name in _ZERO:
        return 0.0
    if name in gt:
        return float(gt[name])
    data = stim.GateData(name)
    if data.is_noisy_gate and not data.produces_measurements and not data.is_reset:
        return 0.0
    if data.produces_measurements and data.is_reset:
        return float(gt["MEAS"]) + float(gt["RESET"])
    if data.produces_measurements:
        return float(gt["MEAS"])
    if data.is_reset:
        return float(gt["RESET"])
    if data.is_two_qubit_gate:
        return float(gt["2Q"])
    if data.is_single_qubit_gate:
        return float(gt["1Q"])
    return float(gt["DEFAULT"])


def _num_records(inst: stim.CircuitInstruction) -> int:
    """Number of measurement records one measurement instruction produces."""
    n = getattr(inst, "num_measurements", None)
    if n is not None:
        return int(n)
    targets = inst.targets_copy()
    combiners = sum(1 for t in targets if t.is_combiner)
    return len(targets) - 2 * combiners if inst.name == "MPP" else len(targets)


def circuit_timing(circuit: stim.Circuit, gate_times: dict | None = None,
                   ) -> tuple[list[float], list[int]]:
    """Returns (round_completion_ns, meas_round_ids):
      round_completion_ns[r]  cumulative ns when round r's TICK slice closes
      meas_round_ids[m]       round id of the m-th measurement record
    A round starts at the first measurement and at any measurement after a
    TICK. A terminal MPP block is its own round.
    [used: round_schedule, needed_rounds]"""
    gt = gate_times or GOOGLE_GATE_TIMES_NS

    cum = 0.0
    slice_max = 0.0
    tick_since = False
    seen_meas = False
    meas_t = -1
    pending = None
    out: dict[int, float] = {}
    meas_round_ids: list[int] = []

    def close_slice():
        nonlocal cum, slice_max, pending
        cum += slice_max
        slice_max = 0.0
        if pending is not None:
            out[pending] = cum
            pending = None

    for inst in circuit.flattened():
        name = inst.name
        if name == "TICK":
            close_slice()
            tick_since = True
            continue
        if name in _ZERO:
            continue
        if name == "MPP":                       # terminal MPP: its own round
            close_slice()
            meas_t += 1
            tick_since = False
            seen_meas = True
            pending = meas_t
            meas_round_ids.extend([meas_t] * _num_records(inst))
            slice_max = max(slice_max, float(gt.get("MPP", gt["MEAS"])))
            continue
        if name[0] == "M":                      # normal measurement
            if tick_since or not seen_meas:
                meas_t += 1
                tick_since = False
                seen_meas = True
                pending = meas_t
            meas_round_ids.extend([meas_t] * _num_records(inst))
            slice_max = max(slice_max, _inst_duration_ns(inst, gt))
            continue
        slice_max = max(slice_max, _inst_duration_ns(inst, gt))

    close_slice()

    n = (max(out) + 1) if out else 0
    missing = [k for k in range(n) if k not in out]
    assert not missing, f"rounds {missing[:5]} missing completion times"
    return [out[k] for k in range(n)], meas_round_ids


def detector_last_measurements(circuit: stim.Circuit) -> list:
    """Per detector, the absolute index of its last contributing measurement.
    None for a DETECTOR without record targets (the gap circuit's always-zero
    virtual pair nodes), which imposes no arrival constraint.
    [used: fusion_layer_offsets_ns, RuntimeEstimator]"""
    meas_count = 0
    last: list = []
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            idxs = [meas_count + t.value for t in inst.targets_copy()
                    if t.is_measurement_record_target]
            last.append(max(idxs) if idxs else None)
            continue
        data = stim.GateData(inst.name)
        if data.produces_measurements:
            meas_count += _num_records(inst)
    return last


def needed_rounds(circuit: stim.Circuit, det_ids, obs_detector=None) -> dict:
    """First and last measurement round a detector set depends on.
    obs_detector is excluded: gap decoding forces its bit, so it has no data
    dependency. rounds_saved_at_end counts final rounds with no needed data.
    [used: run_mb_characterization, gen_partial_mask_dem]"""
    round_ns, meas_round_ids = circuit_timing(circuit)
    det_last = detector_last_measurements(circuit)
    rounds = [meas_round_ids[det_last[d]] for d in det_ids
              if d != obs_detector and det_last[d] is not None]
    total = len(round_ns)
    first, last = (min(rounds), max(rounds)) if rounds else (None, None)
    return {
        "num_circuit_rounds": total,
        "first_needed_round": first,
        "last_needed_round": last,
        "rounds_saved_at_end": (total - 1 - last) if last is not None else None,
        "obs_detector_excluded": obs_detector is not None,
    }


def has_terminal_mpp(circuit: stim.Circuit) -> bool:
    """True when the last measurement instruction is an MPP block, i.e. the
    idealized closing boundary from add_mpp_boundaries."""
    last = None
    for inst in circuit.flattened():
        if stim.GateData(inst.name).produces_measurements:
            last = inst.name
    return last == "MPP"


def round_schedule(circuit: stim.Circuit, gate_times: dict | None = None,
                   boundary_round_as_previous: bool = True,
                   ) -> tuple[list[float], list[int]]:
    """circuit_timing plus the boundary-round policy. The terminal MPP block
    stands in for the next full stabilizer round, so it is priced like the
    previous round instead of as one measurement slice. Applied only when the
    circuit has >= 3 rounds.
    [used: RuntimeEstimator.__init__, fusion_layer_offsets_ns]"""
    round_ns, meas_round_ids = circuit_timing(circuit, gate_times)
    if boundary_round_as_previous and len(round_ns) >= 3 and has_terminal_mpp(circuit):
        round_ns = list(round_ns)
        round_ns[-1] = round_ns[-2] + (round_ns[-2] - round_ns[-3])
    return round_ns, meas_round_ids


def fusion_layer_offsets_ns(graph_json: dict, gap_circuit: stim.Circuit,
                            kept=None, gate_times: dict | None = None,
                            boundary_round_as_previous: bool = True) -> list[int]:
    """Arrival offset per fusion layer (graph_json['layer_fusion']), for
    LAYER_SCHEDULE_NS. A layer arrives at the max round-completion time over
    its vertices' detectors. `kept` maps vertex -> detector for contracted
    graphs (identity when None); the virtual boundary vertex is skipped.
    Returns integer ns, non-decreasing, rebased so entry 0 is 0.
    [used: run_mb_characterization --schedule real]"""
    round_ns, meas_round_ids = round_schedule(
        gap_circuit, gate_times, boundary_round_as_previous=boundary_round_as_previous)
    det_last = detector_last_measurements(gap_circuit)
    det_arrival = [None if m is None else round_ns[meas_round_ids[m]]
                   for m in det_last]

    lf = graph_json["layer_fusion"]
    num_layers = int(lf["num_layers"])
    boundary = int(graph_json["vertex_num"]) - 1
    per_layer: list[float] = [float("-inf")] * num_layers
    for v_str, layer in lf["vertex_layer_id"].items():
        v = int(v_str)
        if v == boundary:
            continue
        det = int(kept[v]) if kept is not None else v
        a = det_arrival[det]
        if a is not None and a > per_layer[int(layer)]:
            per_layer[int(layer)] = a

    # Layers of only virtual pair nodes carry no data and are ready with the
    # previous layer: forward-fill, and back-fill a virtual prefix. Such
    # layers are why fusion layers can outnumber measurement rounds.
    assert any(a != float("-inf") for a in per_layer), "no physical layer at all"
    first_real = next(a for a in per_layer if a != float("-inf"))
    filled = []
    prev = first_real
    for a in per_layer:
        prev = a if a != float("-inf") else prev
        filled.append(prev)
    per_layer = filled

    offsets = [int(round(a - per_layer[0])) for a in per_layer]
    assert offsets[0] == 0
    assert all(b >= a for a, b in zip(offsets, offsets[1:])), \
        f"non-monotonic layer arrivals: {offsets}"
    return offsets
