# mb_graph.py — convert stim DEMs to micro-blossom's graph format and
# prepare defect inputs for the RTL decoder.
# Adapted from mb4msc's get_syndromes_and_json_graph.py (collaborator's
# integration repo).
# Vertex v is detector v (0-based); the boundary vertex is num_detectors.

import json
import hashlib
import math

import numpy as np
import stim

# Micro Blossom circuit-level convention for small bit width.
DEFAULT_MAX_HALF_WEIGHT = 7


def probability_to_even_weight(p: float, max_half_weight: int = DEFAULT_MAX_HALF_WEIGHT) -> int:
    """Map an error probability to micro-blossom's even integer weight:
    2 * clamp(round(ln((1-p)/p)), 1, max_half_weight)."""
    return 2 * max(1, min(max_half_weight, round(math.log((1.0 - p) / p))))


def dem_to_graph(dem: stim.DetectorErrorModel, *, max_half_weight: int = DEFAULT_MAX_HALF_WEIGHT) -> tuple[dict, int]:
    """Convert a DEM to micro-blossom's initializer dict. Returns
    (initializer, stats), stats keys "skipped_components", "isolated_vertices".

    A 1-detector component is a boundary edge, a 2-detector one an internal
    edge; larger ones are skipped and counted. Parallel edges keep the
    minimum weight. Isolated vertices get an inert max-weight boundary edge,
    because micro-blossom's RTL generator cannot elaborate an edgeless vertex.
    [used: run_mb_characterization]"""
    boundary = dem.num_detectors
    weights_map: dict[tuple[int, int], int] = {}
    num_skipped = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        for p, detectors in _split_error_into_components(instruction):
            if len(detectors) == 1:
                u, v = detectors[0], boundary
            elif len(detectors) == 2:
                u, v = detectors
            else:
                num_skipped += 1
                continue
            if u == v:
                continue
            if u > v:
                u, v = v, u
            w = probability_to_even_weight(p, max_half_weight)
            if (u, v) not in weights_map or w < weights_map[(u, v)]:
                weights_map[(u, v)] = w
    connected = {v for edge in weights_map for v in edge}
    isolated = [v for v in range(dem.num_detectors) if v not in connected]
    for v in isolated:
        weights_map[(v, boundary)] = 2 * max_half_weight

    initializer = {
        "vertex_num": dem.num_detectors + 1,
        "weighted_edges": [[u, v, w] for (u, v), w in sorted(weights_map.items())],
        "virtual_vertices": [boundary],
    }
    return initializer, {"skipped_components": num_skipped, "isolated_vertices": len(isolated)}


def _split_error_into_components(instruction: stim.DemInstruction) -> list[tuple[float, list[int]]]:
    """Split a ^-separated DEM error into (p, detectors) components.
    Observable targets are ignored: our DEMs carry the observable as a
    detector. Raises NotImplementedError on any other target type."""
    p = float(instruction.args_copy()[0])
    parts: list[tuple[float, list[int]]] = []
    detectors: list[int] = []
    for target in instruction.targets_copy():
        if target.is_separator():
            parts.append((p, detectors))
            detectors = []
        elif target.is_relative_detector_id():
            detectors.append(int(target.val))
        elif target.is_logical_observable_id():
            continue
        else:
            raise NotImplementedError(f"Unsupported DEM target in {instruction}")
    parts.append((p, detectors))
    return parts


def get_positions(dem: stim.DetectorErrorModel) -> list[dict]:
    """Per-vertex {i,j,t} positions (first 3 detector coords), plus the
    boundary vertex at (-1,-1,-1)."""
    coordinates = dem.get_detector_coordinates()
    positions = [
        dict(zip("ijt", map(float, coordinates.get(d, [0.0, 0.0, 0.0])[:3])))
        for d in range(dem.num_detectors)
    ]
    positions.append({"i": -1.0, "j": -1.0, "t": -1.0})
    return positions


def graph_hash(initializer: dict) -> str:
    """16-hex sha256 of the initializer's canonical JSON: vertex count,
    weighted edges and virtual vertices, not positions. Cache key for RTL
    builds and the check that shots are paired with the right decoder."""
    canonical = json.dumps(initializer, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def sample_gap_circuit_shots(
    gap_circuit: stim.Circuit,
    postselected_detectors: set[int],
    shots: int,
    *,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Sample the gap circuit and discard shots where a postselected detector
    fired, as PartialDesaturationSampler.sample() does. Returns (dets, obs,
    num_discards) for the kept shots: dets bit-packed, little bit order; obs
    is observable 0. Reproducible for a given seed."""
    num_dets = gap_circuit.num_detectors
    sampler = gap_circuit.compile_detector_sampler(seed=seed)
    dets, actual_obs = sampler.sample(shots, separate_observables=True, bit_packed=True)

    discard_mask = np.packbits(
        np.array([k in postselected_detectors for k in range(num_dets)], dtype=np.bool_),
        bitorder="little",
    )
    keep_mask = ~np.any(dets & discard_mask, axis=1)
    num_discards = int(shots - np.count_nonzero(keep_mask))
    return dets[keep_mask], actual_obs[keep_mask][:, 0], num_discards


def bit_packed_to_defect_lists(bit_packed_dets: np.ndarray, num_dets: int) -> list[list[int]]:
    """Unpack bit-packed detection events (little bit order) into per-shot
    lists of fired vertex indices, ascending."""
    unpacked = np.unpackbits(bit_packed_dets, axis=1, count=num_dets, bitorder="little")
    return [[int(v) for v in np.nonzero(row)[0]] for row in unpacked]


def write_syndrome_file(
    path,
    initializer: dict,
    positions: list[dict],
    defect_lists: list[list[int]] | None = None,
) -> None:
    """Write micro-blossom's 'Syndrome Pattern v1.0' file format."""
    with open(path, "w", encoding="utf8") as f:
        f.write("Syndrome Pattern v1.0   <initializer> <positions> <syndrome_pattern>*\n")
        f.write(json.dumps(initializer, separators=(",", ":")) + "\n")
        f.write(json.dumps(positions, separators=(",", ":")) + "\n")
        for defects in defect_lists or []:
            f.write(json.dumps({"defect_vertices": defects}, separators=(",", ":")) + "\n")
