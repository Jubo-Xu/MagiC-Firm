# Tests for algorithms/mb_graph.py.
#
# test_golden_reference_graph freezes the output of the mb4msc reference
# implementation (get_syndromes_and_json_graph.py @ bc13405) as an inline
# expected value — validated identical against our implementation on a real
# 2225-vertex escape-circuit DEM (11208 edges) before mb4msc's retirement.
# No test here depends on mb4msc at runtime.

import json
import pathlib
import sys

import numpy as np
import pytest
import stim

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mb_graph

REPO = pathlib.Path(__file__).resolve().parents[2]
CIRCUITS = REPO / "experiments" / "data" / "circuits"
CIRCUIT_DIRS = sorted(CIRCUITS.glob("end2end_*")) if CIRCUITS.exists() else []


def test_probability_to_even_weight():
    # p=0.001 -> ln(999)=6.9 -> 7 -> 14;  p=0.1 -> ln(9)=2.2 -> 2 -> 4
    assert mb_graph.probability_to_even_weight(0.001) == 14
    assert mb_graph.probability_to_even_weight(0.1) == 4
    # clamps: never below 2, never above 2*max_half_weight
    assert mb_graph.probability_to_even_weight(0.49) == 2
    assert mb_graph.probability_to_even_weight(1e-12) == 14
    assert mb_graph.probability_to_even_weight(1e-12, max_half_weight=3) == 6


def test_golden_reference_graph():
    # Expected value produced by the mb4msc reference implementation
    # (get_initialiser, max_half_weight=7) on this exact DEM.
    dem = stim.DetectorErrorModel("""
        error(0.001) D0
        error(0.1) D0 D1
        error(0.2) D0 D1
        error(0.01) D1 D2 ^ D3
        error(0.3) D0 D1 D2
        error(0.05) D2
    """)
    expected = {
        "vertex_num": 5,
        "weighted_edges": [[0, 1, 2], [0, 4, 14], [1, 2, 10], [2, 4, 6], [3, 4, 10]],
        "virtual_vertices": [4],
    }
    graph, stats = mb_graph.dem_to_graph(dem)
    assert graph == expected
    assert stats["skipped_components"] == 1  # the 3-detector error(0.3) D0 D1 D2
    assert stats["isolated_vertices"] == 0


def test_graph_hash_stable_and_order_insensitive():
    g1 = {"vertex_num": 3, "weighted_edges": [[0, 1, 2]], "virtual_vertices": [2]}
    g2 = {"virtual_vertices": [2], "weighted_edges": [[0, 1, 2]], "vertex_num": 3}
    assert mb_graph.graph_hash(g1) == mb_graph.graph_hash(g2)
    g3 = {**g1, "weighted_edges": [[0, 1, 4]]}
    assert mb_graph.graph_hash(g1) != mb_graph.graph_hash(g3)


def test_sampling_seeded_and_postselected():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x", distance=3, rounds=3,
        after_clifford_depolarization=0.01)
    dets1, obs1, disc1 = mb_graph.sample_gap_circuit_shots(circuit, set(), 100, seed=7)
    dets2, obs2, disc2 = mb_graph.sample_gap_circuit_shots(circuit, set(), 100, seed=7)
    assert np.array_equal(dets1, dets2) and np.array_equal(obs1, obs2)
    assert disc1 == disc2 == 0
    assert dets1.shape[0] == 100

    # postselecting on every detector keeps only defect-free shots
    all_dets = set(range(circuit.num_detectors))
    kept, _, discards = mb_graph.sample_gap_circuit_shots(circuit, all_dets, 100, seed=7)
    defect_free = sum(1 for row in dets1 if not row.any())
    assert kept.shape[0] == defect_free
    assert discards == 100 - defect_free


def test_defect_list_round_trip():
    num_dets = 19
    rng = np.random.default_rng(3)
    dense = rng.random((8, num_dets)) < 0.2
    packed = np.packbits(dense, axis=1, bitorder="little")
    lists = mb_graph.bit_packed_to_defect_lists(packed, num_dets)
    for row, defects in zip(dense, lists):
        assert defects == [int(i) for i in np.nonzero(row)[0]]


def test_write_syndrome_file(tmp_path):
    graph = {"vertex_num": 3, "weighted_edges": [[0, 2, 2]], "virtual_vertices": [2]}
    positions = [{"i": 0.0, "j": 0.0, "t": 0.0}] * 2 + [{"i": -1.0, "j": -1.0, "t": -1.0}]
    path = tmp_path / "x.syndromes"
    mb_graph.write_syndrome_file(path, graph, positions, [[0], []])
    lines = path.read_text().splitlines()
    assert lines[0].startswith("Syndrome Pattern v1.0")
    assert json.loads(lines[1]) == graph
    assert json.loads(lines[3]) == {"defect_vertices": [0]}
    assert json.loads(lines[4]) == {"defect_vertices": []}


@pytest.mark.skipif(not CIRCUIT_DIRS, reason="no circuit folders generated yet")
def test_real_desaturated_dem_is_fully_matchable():
    dem = stim.DetectorErrorModel.from_file(CIRCUIT_DIRS[0] / "desaturated_with_obs.dem")
    graph, stats = mb_graph.dem_to_graph(dem)
    assert stats["skipped_components"] == 0
    assert graph["vertex_num"] == dem.num_detectors + 1
    assert graph["weighted_edges"], "graph has no edges"
    max_w = max(w for _, _, w in graph["weighted_edges"])
    assert max_w <= 2 * mb_graph.DEFAULT_MAX_HALF_WEIGHT
