# contraction_data.py — stored results of the DEM-contraction ablation: decode
# latency (micro-blossom cycles or pymatching microseconds) and gap agreement
# for the three decoding setups of one circuit.

import json

import paths
from naming import parse_name, freq_tag, FREQ_BY_D2

MB = paths.RESULT / "mb_characterization"
PM = paths.RESULT / "mask_ablation" / "pymatching"
ACC = paths.RESULT / "mask_ablation" / "accuracy"

KINDS = [("complete", "Complete"), ("masked", "Mask on complete DEM"), ("contracted", "Contracted DEM")]


def load_cell(name, tag, source):
    """dict(circuit, tag, d1, d2, p, lat, peq) for one circuit and mask tag, or
    None if a file is missing. source 'mb': parallel mean cycles; 'pymatching':
    parallel median microseconds. peq is P(g_p = g_c) per kind (complete = 1)."""
    d1, d2, p, _ = parse_name(name)
    f = freq_tag(FREQ_BY_D2[d2])
    if source == "mb":
        files = {"complete": MB / f"{name}__complete__s0_n1000_{f}_realsched.json",
                 "masked": MB / f"{name}__complete__s0_n1000_{f}_realsched_mask-{tag}.json",
                 "contracted": MB / f"{name}__{tag}__s0_n1000_{f}_realsched.json"}
        if not all(x.exists() for x in files.values()):
            return None
        lat = {k: json.loads(x.read_text())["latency"]["parallel"]["mean_cycles"] for k, x in files.items()}
    else:
        x = PM / f"{name}__{tag}_s0.json"
        if not x.exists():
            return None
        j = json.loads(x.read_text())["latency"]
        lat = {k: j[k]["parallel"]["median_us"] for k in ("complete", "masked", "contracted")}
    a = ACC / name / f"{tag}_s0.json"
    if not a.exists():
        return None
    acc = json.loads(a.read_text())
    peq = {"complete": 1.0, "masked": acc["masked_on_complete_dem_vs_complete"]["p_equal"],
           "contracted": acc["contracted_vs_complete"]["p_equal"]}
    return dict(circuit=name, tag=tag, d1=d1, d2=d2, p=p, lat=lat, peq=peq)
