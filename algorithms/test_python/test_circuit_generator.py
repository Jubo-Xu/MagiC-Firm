# Equivalence test: magic_state_cultivation/circuit_generator.py must produce
# circuits identical to upstream tools/make_circuits for equivalent parameters.
# This is the guarantee that lets Python-side experiments and shell-side
# (make_circuits) workflows share circuit definitions interchangeably.

import pathlib
import subprocess
import sys
import tempfile

import pytest
import stim

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "magic_state_cultivation"))

from circuit_generator import CircuitGenerator

MAKE_CIRCUITS = REPO / "magic_state_cultivation" / "upstream" / "tools" / "make_circuits"


def run_make_circuits(**kw) -> stim.Circuit:
    """Run upstream tools/make_circuits for one parameter combo, return the circuit."""
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, str(MAKE_CIRCUITS), "--out_dir", td]
        for k, v in kw.items():
            cmd += [f"--{k}", str(v)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        paths = list(pathlib.Path(td).glob("*.stim"))
        assert len(paths) == 1, f"expected exactly one circuit, got {paths}"
        return stim.Circuit.from_file(paths[0])


CASES = [
    # (our from_input_params kwargs, make_circuits kwargs)
    pytest.param(
        dict(circuit_type="surface-code-memory", basis="X", gateset="css",
             noise_model="uniform-depolarizing", noise_strength=1e-3, d2=5, r_post_escape=3),
        dict(circuit_type="surface-code-memory", basis="X", gateset="css",
             noise_strength=1e-3, d2=5, r2=3),
        id="surface-code-memory-css",
    ),
    pytest.param(
        dict(circuit_type="inject+cultivate", injection_protocol="unitary", basis="Y",
             gateset="cz", noise_model="circuit-level-SI1000", noise_strength=1e-3, d1=3),
        dict(circuit_type="inject[unitary]+cultivate", basis="Y", gateset="cz",
             noise_strength=1e-3, d1=3),
        id="inject-unitary-cultivate-cz",
    ),
    pytest.param(
        dict(circuit_type="escape-to-big-matchable-code", basis="Y", gateset="css",
             noise_model="uniform-depolarizing", noise_strength=1e-3,
             d1=3, d2=9, r_in_escape=3, r_post_escape=3),
        dict(circuit_type="escape-to-big-matchable-code", basis="Y", gateset="css",
             noise_strength=1e-3, d1=3, d2=9, r1=3, r2=3),
        id="escape-to-big-matchable-css",
    ),
]


@pytest.mark.parametrize("ours,theirs", CASES)
def test_matches_make_circuits(ours, theirs):
    cg = CircuitGenerator.from_input_params(**ours)
    cg.generate()
    expected = run_make_circuits(**theirs)
    assert cg.noisy_circuit is not None
    assert stim.Circuit(str(cg.noisy_circuit)) == expected


def test_noiseless_mode():
    cg = CircuitGenerator.from_input_params(
        circuit_type="surface-code-memory", basis="X", gateset="css",
        noise_model="uniform-depolarizing", noise_strength=1e-3, d2=5, r_post_escape=3,
        HasNoise=False,
    )
    cg.generate()
    assert cg.noisy_circuit is None
    assert cg.ideal_circuit is not None
    assert cg.params.num_layers > 0


def test_rejects_bad_params():
    with pytest.raises(ValueError):
        CircuitGenerator.from_input_params(
            circuit_type="not-a-type", basis="X", gateset="css",
            noise_model="uniform-depolarizing", noise_strength=1e-3)
    with pytest.raises(ValueError):
        CircuitGenerator.from_input_params(
            circuit_type="inject+cultivate", injection_protocol="nope", basis="Y",
            gateset="cz", noise_model="circuit-level-SI1000", noise_strength=1e-3, d1=3)
    # cz gateset requires SI1000
    cg = CircuitGenerator.from_input_params(
        circuit_type="surface-code-memory", basis="X", gateset="cz",
        noise_model="uniform-depolarizing", noise_strength=1e-3, d2=5, r_post_escape=3)
    with pytest.raises(ValueError):
        cg.generate()
