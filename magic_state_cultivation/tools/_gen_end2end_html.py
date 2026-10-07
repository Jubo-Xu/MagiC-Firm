#!/usr/bin/env python3
"""Generate per-tick HTML viewers for two end2end cultivation circuits."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import cultiv
import gen

BASIS = "Y"  # canonical T-state basis used throughout the cultivation paper

configs = [
    dict(dcolor=3, dsurface=9, r_growing=3, r_end=5),
    dict(dcolor=5, dsurface=11, r_growing=3, r_end=5),
]

for cfg in configs:
    label = f"end2end_d1={cfg['dcolor']}_d2={cfg['dsurface']}_r1={cfg['r_growing']}_r2={cfg['r_end']}_b={BASIS}"
    print(f"[+] building {label} ...", flush=True)
    circuit = cultiv.make_end2end_cultivation_circuit(
        basis=BASIS,
        inject_style="unitary",
        **cfg,
    )
    out_logical = ROOT / f"{label}_logical.html"
    gen.write_file(out_logical, gen.stim_circuit_html_viewer(circuit))
    print(f"    -> {out_logical}  (qubits={circuit.num_qubits}, ticks={circuit.num_ticks})")

    cz_circuit = gen.transpile_to_z_basis_interaction_circuit(circuit)
    out_cz = ROOT / f"{label}_cz.html"
    gen.write_file(out_cz, gen.stim_circuit_html_viewer(cz_circuit))
    print(f"    -> {out_cz}        (qubits={cz_circuit.num_qubits}, ticks={cz_circuit.num_ticks})")

print("done.")
