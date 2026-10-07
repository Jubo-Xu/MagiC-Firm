# gate_times.py — gate durations in ns, from Google's superconducting-hardware
# data. Used by runtime_estimator and layer_schedule.
#
# hardware/compiler/control_system/timing.py holds an independent copy of
# these numbers (the compiler stays standalone). Update both together.

GOOGLE_GATE_TIMES_NS: dict[str, float] = {
    "1Q": 25.0, "2Q": 34.0, "MEAS": 500.0, "RESET": 160.0, "DEFAULT": 25.0,
    "H": 25.0, "S": 25.0, "S_DAG": 25.0, "X": 25.0, "Y": 25.0, "Z": 25.0,
    "RX": 160.0, "RY": 160.0, "R": 160.0,
    "MX": 500.0, "MY": 500.0, "M": 500.0, "MPP": 500.0,
    "CX": 34.0, "CZ": 34.0,
}
