"""Tests for dem_structure.DemStructure and PartialMaskBuilder.closure_mask.

Synthetic cases pin the component/crossing conventions (graph edge, hyperedge,
duplicate detector targets, separator-decomposed instructions); the flagship
regression reproduces the validated mask-lab closure result exactly.
"""
import json
import pathlib
import warnings

import numpy as np
import pytest
import stim

from dem_structure import DemStructure

REPO = pathlib.Path(__file__).resolve().parents[2]
FLAGSHIP = REPO / "experiments" / "data" / "circuits" / "end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y"
PARTIAL = "partial_l1-14_t1-9_ts1-7_aug-canonical50_pr-causal50_wc15"


def _synthetic():
    return DemStructure(stim.DetectorErrorModel(
        "error(0.1) D0 D1\n"            # graph edge
        "error(0.2) D0 D1 D2\n"         # hyperedge (full support counts)
        "error(0.3) D0 D0 D1\n"         # duplicate target -> XOR-merged to {D1}
        "error(0.4) D0 D1 L0 ^ D2 D3\n"  # separator -> two components, both p=0.4
        "error(0.5) D3\n"               # boundary component
    ))


def test_components_and_conventions():
    st = _synthetic()
    assert st.components == [[0, 1], [0, 1, 2], [1], [0, 1], [2, 3], [3]]
    assert np.allclose(st.p, [0.1, 0.2, 0.3, 0.4, 0.4, 0.5])
    assert st.nbrs[0] == {1, 2} and st.nbrs[3] == {2}


def test_crossing_mass_full_support():
    st = _synthetic()
    m = np.array([1, 0, 0, 0], bool)
    # {0,1}: crosses; {0,1,2}: crosses; {1}: no; {0,1}: crosses; {2,3}: no; {3}: no
    assert abs(st.S(m) - (0.1 + 0.2 + 0.4)) < 1e-12
    m = np.array([1, 1, 0, 0], bool)
    assert abs(st.S(m) - 0.2) < 1e-12          # only the hyperedge still crosses
    assert abs(st.delta(np.array([1, 0, 0, 0], bool), 1) - (0.2 - 0.7)) < 1e-12
    assert st.boundary(np.array([1, 1, 0, 0], bool)) == [0, 1]
    assert st.frontier(np.array([1, 1, 0, 0], bool)) == [2]


def test_closure_batch_semantics_and_info():
    st = _synthetic()
    m, info = st.closure(np.array([1, 0, 0, 0], bool), return_info=True)
    # iteration 1: frontier {1, 2}; delta(1) = -0.5 < 0 (fill), delta(2) = +0.4 (no)
    # iteration 2: frontier {2}; delta(2) = -0.2 + 0.4 = +0.2 (no) -> fixed point
    assert m.tolist() == [True, True, False, False]
    assert info["iterations"] == 1 and info["detectors_added"] == 1 and info["converged"]
    assert len(info["S_per_iteration"]) == info["iterations"] + 1
    assert info["S_per_iteration"][0] == info["S_before"] and info["S_per_iteration"][-1] == info["S_after"]
    assert all(b < a for a, b in zip(info["S_per_iteration"], info["S_per_iteration"][1:]))


def test_closure_max_iter_warns_and_reports():
    st = _synthetic()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        m, info = st.closure(np.array([1, 0, 0, 0], bool), max_iter=0, return_info=True)
    assert not info["converged"] and info["iterations"] == 0 and len(w) == 1
    assert m.tolist() == [True, False, False, False]


@pytest.mark.skipif(not (FLAGSHIP / PARTIAL / "mask_bool.npy").exists(), reason="flagship folder not present")
def test_flagship_regression_matches_validated_lab_closure():
    """Closure of the current flagship mask (geometric -> canonical augment ->
    causal prune): the mask-lab experiments recorded 338 -> 408 detectors and
    S(M) 1.741 -> 1.111 (lab_holdout JSON, row 'A+closure')."""
    from partial_mask_builder import PartialMaskBuilder
    ps = json.loads((FLAGSHIP / "postselected_detectors.json").read_text())
    circuit = stim.Circuit.from_file(FLAGSHIP / ps["circuit_file"])
    mask = np.load(FLAGSHIP / PARTIAL / "mask_bool.npy")
    b = PartialMaskBuilder(circuit=circuit)
    out, packed, n_added, info = b.closure_mask(mask, return_info=True)
    assert int(mask.sum()) == 338 and int(out.sum()) == 408 and n_added == 70
    assert abs(info["S_before"] - 1.741) < 5e-3 and abs(info["S_after"] - 1.111) < 5e-3
    assert info["converged"] and info["iterations"] == 8
    assert np.array_equal(packed, np.packbits(out, bitorder="little"))
    assert abs(b.mask_crossing_mass(out) - info["S_after"]) < 1e-12
    assert all(b <= a for a, b in zip(info["S_per_iteration"], info["S_per_iteration"][1:]))
