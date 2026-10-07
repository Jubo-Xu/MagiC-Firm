"""timing.py — circuit -> per-measurement-round wall-clock (standalone).

Self-contained on purpose: it does NOT import the runtime estimator or any other
module beyond `stim`, so the compiler stays independent. The gate-time table and
layer model are reimplemented here (a copy, not a reference).

Timing model (matches the physical picture): operations WITHIN one TICK slice run
in parallel, so the slice costs the MAX single-op duration; consecutive TICK slices
are SEQUENTIAL, so their costs SUM. The time between two measurement rounds is thus
the sum of the max-durations of the TICK slices between them.

`round_completion_ns(circuit)` returns `t` with `t[k]` = cumulative time (ns) at
which measurement round `k` completes, indexed by the SAME meas_time convention as
`parser.DetectorConstructionScanner` (a round increments on a measurement that
follows a TICK, or the first measurement, or a terminal MPP), so the indices line
up with `measurement_map`.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import stim

# Default per-gate durations (ns): Google's superconducting-hardware numbers
# (25ns 1Q, 34ns 2Q, 500ns measurement, 160ns reset). A deliberate COPY of
# algorithms/gate_times.py — intentionally NOT imported, the compiler stays
# standalone. If the timing model changes, update both places.
# Note RX/RY are stim RESET-basis gates, priced as resets.
DEFAULT_GATE_TIMES_NS: Dict[str, float] = {
    "1Q": 25.0, "2Q": 34.0, "MEAS": 500.0, "RESET": 160.0, "DEFAULT": 25.0,
    "H": 25.0, "S": 25.0, "S_DAG": 25.0, "X": 25.0, "Y": 25.0, "Z": 25.0,
    "RX": 160.0, "RY": 160.0, "R": 160.0,
    "MX": 500.0, "MY": 500.0, "M": 500.0, "MPP": 500.0,
    "CX": 34.0, "CZ": 34.0,
}

# Annotations / control-flow markers with no physical duration.
_ZERO = {"QUBIT_COORDS", "DETECTOR", "OBSERVABLE_INCLUDE", "SHIFT_COORDS", "TICK"}


def _inst_duration_ns(inst: stim.CircuitInstruction, gt: Dict[str, float]) -> float:
    """Duration of one instruction, by explicit name then by gate class."""
    name = inst.name
    if name in _ZERO:
        return 0.0
    if name in gt:
        return float(gt[name])
    data = stim.GateData(name)
    # Pure noise channels (DEPOLARIZE*, X_ERROR, E, ...) are annotations with
    # no physical duration (is_noisy_gate alone is not the test: measurements
    # are also flagged noisy).
    if data.is_noisy_gate and not data.produces_measurements and not data.is_reset:
        return 0.0
    if data.produces_measurements and data.is_reset:    # MR: sequential meas+reset
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


def round_completion_ns(circuit: stim.Circuit,
                        gate_times: Optional[Dict[str, float]] = None) -> List[float]:
    """Cumulative circuit time (ns) at which each measurement round completes.

    Returns a dense list indexed by meas_time (0..T-1). A round's completion time
    is the cumulative time once the TICK slice carrying its measurement is closed.
    """
    gt = gate_times or DEFAULT_GATE_TIMES_NS

    cum = 0.0            # committed time up to the last closed TICK slice
    slice_max = 0.0      # max op duration seen in the currently-open slice
    tick_since = False   # a TICK has occurred since the last measurement round
    seen_meas = False    # any measurement round started yet
    meas_t = -1          # current round index (parser convention)
    pending: Optional[int] = None   # round whose measurement is in the open slice
    out: Dict[int, float] = {}

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
        if name == "MPP":                       # terminal MPP: its own round (parser)
            close_slice()
            meas_t += 1
            tick_since = False
            pending = meas_t
            slice_max = max(slice_max, float(gt.get("MPP", gt["MEAS"])))
            continue
        if name[0] == "M":                      # normal measurement
            if tick_since or not seen_meas:
                meas_t += 1
                tick_since = False
                seen_meas = True
                pending = meas_t
            slice_max = max(slice_max, _inst_duration_ns(inst, gt))
            continue
        slice_max = max(slice_max, _inst_duration_ns(inst, gt))

    close_slice()   # flush the final open slice

    n = (max(out) + 1) if out else 0
    missing = [k for k in range(n) if k not in out]
    assert not missing, (
        f"round_completion_ns: rounds {missing[:5]} have no completion time — "
        f"meas_time indexing bug; refusing to emit silent zeros")
    return [out[k] for k in range(n)]
