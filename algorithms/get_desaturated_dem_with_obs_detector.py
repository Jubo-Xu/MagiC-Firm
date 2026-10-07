# Adapted from cultiv._decoding._desaturation_sampler (Apache 2.0):
#   https://github.com/quantumlib/magic-state-cultivation
#
# Turns a cultivation circuit into the gap DEM (matchable, clipped, with an
# obs detector), the aligned gap circuit and the postselected-detector set.
# Matchable means every error splits into parts of at most two detectors, as
# pymatching requires. The obs detector is flipped by every error that flips
# the observable, so decoding with it set and cleared gives the gap.

import collections
import dataclasses
import heapq
import math
import pathlib
import sys
from typing import Literal, cast, Any, AbstractSet, Optional

import stim

src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))

import gen
from cultiv._error_set import int_to_flipped_bits


# ---------------- DEM error class and matchable/clipped DEM builders ----------------
# (verbatim port from cultiv._decoding._desaturation_sampler)

@dataclasses.dataclass(frozen=True)
class _DemError:
    p: float
    det_set: frozenset[int]
    obs_mask: int

    @staticmethod
    def from_error_instruction(instruction: stim.DemInstruction) -> "_DemError":
        p = instruction.args_copy()[0]
        det_list: list[int] = []
        obs_mask = 0
        for target in instruction.targets_copy():
            if target.is_logical_observable_id():
                obs_mask ^= 1 << target.val
            elif target.is_relative_detector_id():
                det_list.append(target.val)
            elif target.is_separator():
                pass
            else:
                raise NotImplementedError(f"{instruction}")
        return _DemError(p=p, det_set=frozenset(gen.xor_sorted(det_list)), obs_mask=obs_mask)

    def to_instruction(self) -> stim.DemInstruction:
        targets = []
        for d in self.det_set:
            targets.append(stim.target_relative_detector_id(d))
        for d in int_to_flipped_bits(self.obs_mask)[::-1]:
            targets.append(stim.target_logical_observable_id(d))
        return stim.DemInstruction("error", [self.p], targets)

    @staticmethod
    def to_separated_instruction(parts: list["_DemError"]) -> stim.DemInstruction:
        assert len(parts) >= 1
        assert len(set(p.p for p in parts)) == 1
        targets = []
        for k in range(len(parts)):
            if k:
                targets.append(stim.target_separator())
            for d in parts[k].det_set:
                targets.append(stim.target_relative_detector_id(d))
            for d in int_to_flipped_bits(parts[k].obs_mask)[::-1]:
                targets.append(stim.target_logical_observable_id(d))
        return stim.DemInstruction("error", [parts[0].p], targets)


def _clipped_matchable_dem(
    flat_dem: stim.DetectorErrorModel, clip: AbstractSet[int],
) -> stim.DetectorErrorModel:
    neighbors: dict[int, dict[int, tuple[float, int]]] = collections.defaultdict(dict)

    heap: list[tuple[float, int, int]] = []
    boundaries: set[int] = set()
    for inst in flat_dem:
        if inst.type == "error":
            if any(t.is_separator() for t in inst.targets_copy()):
                continue
            err = _DemError.from_error_instruction(inst)
            w = -math.log(err.p / (1 - err.p))
            if len(err.det_set) == 1:
                a, = err.det_set
                heapq.heappush(heap, (w, a, err.obs_mask))
                boundaries.add(a)
            elif len(err.det_set) == 2:
                a, b = err.det_set
                neighbors[a][b] = (w, err.obs_mask)
                neighbors[b][a] = (w, err.obs_mask)

    classification: dict[int, tuple[int, float]] = {}
    while heap:
        cost, node, obs = heapq.heappop(heap)
        if node in classification:
            continue
        classification[node] = (obs, cost)
        for neighbor, (extra_cost, extra_obs) in neighbors[node].items():
            if neighbor not in classification:
                heapq.heappush(heap, (cost + extra_cost, neighbor, obs ^ extra_obs))

    new_dem = stim.DetectorErrorModel()
    for inst in flat_dem:
        if inst.type != "error" or sum(
            t.is_relative_detector_id() and t.val in clip for t in inst.targets_copy()
        ) < 2:
            new_dem.append(inst)
    for c in clip:
        if c not in boundaries and c in classification:
            obs, w = classification[c]
            p = math.exp(-w) / (math.exp(-w) + 1)
            targets = [stim.target_relative_detector_id(c)]
            for b in int_to_flipped_bits(obs):
                targets.append(stim.target_logical_observable_id(b))
            new_dem.append("error", [p], targets)
    return new_dem


def _dem_with_obs_detector(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    obs_det = stim.target_relative_detector_id(dem.num_detectors)
    new_dem = stim.DetectorErrorModel()
    new_dem.append("detector", [-10, -10, -10, -10, -10], [obs_det])
    for inst in dem:
        if inst.type == "error":
            targets = inst.targets_copy()
            new_targets = []
            for t in targets:
                if t.is_logical_observable_id():
                    new_targets.append(obs_det)
                new_targets.append(t)
            new_dem.append("error", inst.args_copy(), new_targets)
        else:
            new_dem.append(inst)
    return new_dem


# ---------------- public entry point ----------------

def get_desaturated_dem_with_obs_detector(
    circuit: stim.Circuit,
    *,
    dem: Optional[stim.DetectorErrorModel] = None,
) -> tuple[stim.DetectorErrorModel, stim.Circuit, frozenset[int]]:
    """Build the gap DEM of a cultivation circuit.

    Returns (gap_dem_complete, gap_circuit, postselected_detectors):
      gap_dem_complete: flattened DEM made matchable (virtual color-code pair
        detectors appended), clipped, with the obs detector at the last index.
      gap_circuit: `circuit` plus target-less DETECTORs (always 0) for the
        virtual pairs and the obs detector, so sampled syndromes align column
        for column with gap_dem_complete.
      postselected_detectors: hidden-from-matcher | visible-to-matcher.
    `dem` defaults to circuit.detector_error_model().
    """
    if dem is None:
        dem = circuit.detector_error_model()
    dem = dem.flattened()
    num_dets = dem.num_detectors
    gap_circuit = circuit.copy()

    det_coords = dem.get_detector_coordinates()
    det_bases: list[Literal["X", "Z", "!"]] = []
    det_colors: list[Literal["r", "g", "b", "_"]] = []
    postselected_detectors_hidden_from_matcher: set[int] = set()
    postselected_detectors_visible_to_matcher: set[int] = set()

    for d in range(num_dets):
        coords = det_coords[d]
        if len(coords) <= 4 or coords[4] == -9:
            postselected_detectors_hidden_from_matcher.add(d)
            det_bases.append("!")
            det_colors.append("_")
            continue

        coord_annotation = int(coords[4])
        basis = cast(Any, "XXXZZZXZ"[coord_annotation])
        color = cast(Any, "rgbrgb__"[coord_annotation])
        det_bases.append(basis)
        det_colors.append(color)
        if (color == "r" and basis == "X") or (color == "g" and basis == "Z"):
            # Postselect color-code detectors that can be ablated, leaving a matchable code.
            postselected_detectors_visible_to_matcher.add(d)

    errors: list[_DemError] = []
    for inst in dem:
        if inst.type != "error":
            continue
        errors.append(_DemError.from_error_instruction(inst))

    # Classify each single-basis error's obs flip, to help with decomposition.
    dets_to_obs: dict[frozenset[int], int] = {}
    for err in errors:
        bases = {det_bases[d] for d in err.det_set}
        if len(bases) == 1 and len(err.det_set) == 2 and err.obs_mask:
            a, b = err.det_set
            postselected_detectors_hidden_from_matcher.add(a)
            postselected_detectors_hidden_from_matcher.add(b)
        if len(bases) == 1:
            dets_to_obs[err.det_set] = err.obs_mask

    # Find single-basis RGB triplet errors that need to be simplified for the matcher.
    virtual_pair_nodes: set[frozenset[int]] = set()
    for err in errors:
        colors = {det_colors[d] for d in err.det_set}
        bases = {det_bases[d] for d in err.det_set}
        if len(err.det_set) == 3:
            if len(bases) == 1 and colors == {"r", "g", "b"}:
                a, b, c = err.det_set
                virtual_pair_nodes.add(frozenset([a, b]))
                virtual_pair_nodes.add(frozenset([a, c]))
                virtual_pair_nodes.add(frozenset([b, c]))
        elif (
            len(err.det_set) == 2
            and len(bases) == 1
            and (colors == {"r", "g"} or colors == {"r", "b"} or colors == {"b", "g"})
        ):
            a, b = err.det_set
            virtual_pair_nodes.add(frozenset([a, b]))

    pair2virtual: dict[frozenset[int], int] = {}
    for pair in sorted(virtual_pair_nodes, key=lambda e: tuple(sorted(e))):
        k = len(pair2virtual) + num_dets
        pair2virtual[pair] = k
        a, b = pair
        if a != -1 and b != -1:
            det_coords[k] = [(x + y) / 2 for x, y in list(zip(det_coords[a], det_coords[b]))[:3]]
        else:
            c = a if a != -1 else b
            det_coords[k] = det_coords[c][:3]
            det_coords[k][0] += 0.25
            det_coords[k][1] += 0.25
            det_coords[k][2] += 0.25
        gap_circuit.append("DETECTOR", [], det_coords[k])

    matchable_dem = stim.DetectorErrorModel()
    for k in range(num_dets + len(pair2virtual)):
        matchable_dem.append("detector", det_coords[k], [stim.target_relative_detector_id(k)])
    for err in errors:
        colors_c = collections.Counter(det_colors[d] for d in err.det_set)
        bases_c = collections.Counter(det_bases[d] for d in err.det_set)
        if len(err.det_set) == 2 and err.det_set in virtual_pair_nodes:
            # Boundary error at the side of the color code region.
            virtual_err = _DemError(
                p=err.p,
                det_set=frozenset([pair2virtual[err.det_set]]),
                obs_mask=err.obs_mask,
            )
            matchable_dem.append(err.to_instruction())
            matchable_dem.append(virtual_err.to_instruction())

        elif (
            len(err.det_set) == 3
            and len(bases_c) == 1
            and colors_c == collections.Counter("rgb")
        ):
            # Bulk error within the color code region.
            # Split into three node-to-virtual-node-pair errors.
            assert err.obs_mask == 0
            for solo in err.det_set:
                virtual_err = _DemError(
                    p=err.p,
                    obs_mask=err.obs_mask,
                    det_set=frozenset([solo, pair2virtual[err.det_set ^ frozenset([solo])]]),
                )
                matchable_dem.append(virtual_err.to_instruction())

        elif len(err.det_set) <= 2 and (bases_c.keys() == {"X"} or bases_c.keys() == {"Z"}):
            # Simple matchable error.
            matchable_dem.append(err.to_instruction())

        elif bases_c["X"] <= 2 and bases_c["Z"] <= 2:
            # Decomposable matchable error: split into X and Z parts.
            xs = frozenset([d for d in err.det_set if det_bases[d] == "X"])
            zs = frozenset([d for d in err.det_set if det_bases[d] == "Z"])
            if xs not in dets_to_obs or zs not in dets_to_obs:
                continue
            obs_x = dets_to_obs[xs]
            obs_z = dets_to_obs[zs]
            if obs_x ^ obs_z != err.obs_mask:
                # Decomposition failed (could be a distance-3 logical error).
                continue
            x_part = _DemError(p=err.p, det_set=xs, obs_mask=obs_x)
            z_part = _DemError(p=err.p, det_set=zs, obs_mask=obs_z)
            matchable_dem.append(_DemError.to_separated_instruction([x_part, z_part]))
        else:
            # Too complicated for the matcher — drop.
            pass

    clipped_dem = _clipped_matchable_dem(matchable_dem, postselected_detectors_hidden_from_matcher)
    clipped_dem_with_det_for_obs = _dem_with_obs_detector(clipped_dem)
    gap_circuit.append("DETECTOR", [], [-9, -9, -9])  # gap observable detector
    assert gap_circuit.num_detectors == clipped_dem_with_det_for_obs.num_detectors

    gap_dem_complete = clipped_dem_with_det_for_obs
    postselected_detectors = frozenset(
        postselected_detectors_hidden_from_matcher | postselected_detectors_visible_to_matcher
    )
    return gap_dem_complete, gap_circuit, postselected_detectors


# ---------------- CLI: bootstrap a circuit data folder ----------------

def main() -> None:
    """Write the circuit-level files of a circuit data folder and print
    dem_stats for both DEMs.

        <out>/<circuit filename>           copy of the input circuit
        <out>/original.dem                 DEM of the input circuit
        <out>/desaturated_with_obs.dem     gap DEM
        <out>/gap_circuit.stim             circuit aligned with the gap DEM
        <out>/postselected_detectors.json  postselected detectors and classes
    """
    import argparse
    import json
    import shutil

    from dem_stats import dem_stats, print_dem_stats

    parser = argparse.ArgumentParser(
        description="Generate the desaturated (matchable, clipped, "
                    "obs-detector-augmented) gap DEM and companion artifacts "
                    "for a cultivation circuit.",
    )
    parser.add_argument("--circuit", type=pathlib.Path, required=True,
                        help="Path to the noisy stim circuit file.")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Circuit data folder to create/populate.")
    args = parser.parse_args()

    if not args.circuit.is_file():
        parser.error(f"circuit file not found: {args.circuit}")
    args.out.mkdir(parents=True, exist_ok=True)

    circuit = stim.Circuit.from_file(args.circuit)
    dem = circuit.detector_error_model()
    print_dem_stats(dem_stats(dem), label="original")

    gap_dem, gap_circuit, postselected = get_desaturated_dem_with_obs_detector(
        circuit, dem=dem,
    )
    print_dem_stats(dem_stats(gap_dem), label="desaturated+obs")

    dest = args.out / args.circuit.name
    if dest.resolve() != args.circuit.resolve():
        shutil.copyfile(args.circuit, dest)
    dem.to_file(args.out / "original.dem")
    gap_dem.to_file(args.out / "desaturated_with_obs.dem")
    gap_circuit.to_file(args.out / "gap_circuit.stim")

    # Class of each postselected detector for the control-system compiler,
    # whose predicate tests remaining = coords[3:] (remaining[1] = stim coord 4).
    #   hidden_unannotated, hidden_marked   caught by preset 'cultivation'
    #   visible_color (0 red-X, 4 green-Z)  caught by 'cultivation+color'
    #   hidden_error_derived                found from the error model, so no
    #     coordinate rule matches. The compiler config must handle these
    #     explicitly for exact discard parity.
    det_coords_flat = dem.flattened().get_detector_coordinates()
    records = []
    uncaught: list[int] = []
    for d in sorted(int(x) for x in postselected):
        c = [float(v) for v in det_coords_flat.get(d, [])]
        remaining = c[3:]
        ann = remaining[1] if len(remaining) > 1 else None
        if ann is None:
            cls = "hidden_unannotated"
        elif ann == -9:
            cls = "hidden_marked"
        elif ann in (0.0, 4.0):
            cls = "visible_color"
        else:
            cls = "hidden_error_derived"
            uncaught.append(d)
        records.append({
            "detector": d,
            "coords": c,
            "remaining": remaining,
            "annotation": None if ann is None else int(ann),
            "class": cls,
        })

    (args.out / "postselected_detectors.json").write_text(json.dumps({
        "circuit_file": args.circuit.name,
        "n_circuit_dets": int(circuit.num_detectors),
        "n_gap_dets": int(gap_dem.num_detectors),
        "obs_detector_index": int(gap_dem.num_detectors - 1),
        "postselected_detectors": sorted(int(x) for x in postselected),
        "annotation_convention": {
            "annotation_coord_index": 4,
            "compiler_remaining_index": 1,
            "hidden_values": ["unannotated", -9],
            "visible_color_values": [0, 4],
            "matching_compiler_preset": "cultivation+color",
        },
        "compiler_preset_covers_all": not uncaught,
        "detectors_not_expressible_by_coords": uncaught,
        "postselect_annotations": records,
    }, indent=2))

    print(f"wrote circuit folder: {args.out}")
    for f in [args.circuit.name, "original.dem", "desaturated_with_obs.dem",
              "gap_circuit.stim", "postselected_detectors.json"]:
        print(f"  {f}")
    if uncaught:
        print(f"  WARNING: {len(uncaught)} postselect detector(s) not "
              f"expressible by any coordinate rule (error-derived): {uncaught}")


if __name__ == "__main__":
    main()
