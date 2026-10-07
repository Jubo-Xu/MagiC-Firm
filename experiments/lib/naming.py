# naming.py — circuit-name parsing and labels shared by the experiment scripts.
#
# Circuit folders are named end2end_d1=<d1>_d2=<d2>_r1=<r1>_r2=<r2>_p=<p>_inj=<inj>_b=<basis>.

import re

import numpy as np

_NAME = re.compile(r"end2end_d1=(\d+)_d2=(\d+)_r1=\d+_r2=\d+_p=([0-9.e-]+)_inj=(\w+)_b=")

# Decoder clock per escape distance: micro-blossom's reported Fmax for d = 11,
# 13, 15; the d = 15 value is kept for larger d2 (no published value).
FREQ_BY_D2 = {11: 77e6, 13: 62e6, 15: 43e6, 17: 43e6, 19: 43e6, 21: 43e6}


def parse_name(name: str) -> tuple[int, int, float, str]:
    """(d1, d2, p, injection) of a circuit folder name."""
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"not a circuit folder name: {name!r}")
    return int(m.group(1)), int(m.group(2)), float(m.group(3)), m.group(4)


def short_name(name: str) -> str:
    """end2end_d1=3_d2=15_r1=3_r2=0_p=0.0009_inj=unitary_b=Y -> d1=3_d2=15_r1=3_r2=0_p=0.0009"""
    return name.replace("end2end_", "").split("_inj")[0]


def fmt_p(p: float) -> str:
    """Physical error rate as m×10^e for matplotlib text."""
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    return f"{m:g}×10$^{{{e}}}$"


def circuit_label(name: str) -> str:
    """Axis / title label: $d_1$=3, $d_2$=15, $p$=1×10^-3."""
    d1, d2, p, _ = parse_name(name)
    return f"$d_1$={d1}, $d_2$={d2}, $p$={fmt_p(p)}"


def freq_tag(frequency_hz: float) -> str:
    """43e6 -> '43MHz'; used in micro-blossom result and cache file names."""
    return f"{frequency_hz / 1e6:g}MHz"
