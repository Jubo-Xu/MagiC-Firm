# Adapted from cultiv._decoding._desaturation_sampler (Apache 2.0):
#   https://github.com/quantumlib/magic-state-cultivation
#
# Complementary-gap sampler on the gap DEM built by
# get_desaturated_dem_with_obs_detector.py.
#
# Optional partial mode: given a 1-D bool mask over the circuit's detectors,
# the gap DEM is contracted by a chain-aware Dijkstra so unmasked detectors
# collapse into synthetic edges between kept terminals. Decoding then runs on
# the smaller matching graph.

import collections
import dataclasses
import heapq
import math
import time
from typing import Optional

import numpy as np
import pymatching
import sinter
import stim

from get_desaturated_dem_with_obs_detector import get_desaturated_dem_with_obs_detector


# ---------------- chain-aware DEM reduction (partial mode) ----------------

def _apply_connectivity_fix_per_detector(
    selective_kk: dict[tuple[int, int], tuple[float, int]],
    selective_bdy: dict[int, tuple[float, int]],
    chain_kb: dict[int, list[tuple[float, int]]],
    direct_kk_w: dict[tuple[int, int], float],
    direct_bdy_w: dict[int, float],
    n_kept: int,
) -> int:
    """Connectivity fix, one kept detector at a time in renumbered order. A
    detector that cannot reach a node with boundary access gets its cheapest
    chain_kb candidate added to selective_bdy, ignoring the cutoff.
    Mutates selective_bdy. Returns the number of edges added."""
    red_adj: dict[int, set[int]] = collections.defaultdict(set)
    for (a, b) in direct_kk_w.keys():
        red_adj[a].add(b); red_adj[b].add(a)
    for (a, b) in selective_kk.keys():
        red_adj[a].add(b); red_adj[b].add(a)

    has_boundary: set[int] = set(direct_bdy_w) | set(selective_bdy)
    fix_added = 0

    for d in range(n_kept):
        if d in has_boundary:
            continue
        seen = {d}
        stack = [d]
        reached = False
        while stack and not reached:
            cur = stack.pop()
            for nbr in red_adj[cur]:
                if nbr in seen:
                    continue
                if nbr in has_boundary:
                    reached = True
                    break
                seen.add(nbr)
                stack.append(nbr)
        if reached:
            continue
        cands = chain_kb.get(d, [])
        if not cands:
            continue   # truly isolated from boundary even in original DEM
        w_min, obs_min = min(cands, key=lambda x: x[0])
        selective_bdy[d] = (w_min, obs_min)
        has_boundary.add(d)
        fix_added += 1

    return fix_added


def _apply_connectivity_fix_union_find(
    selective_kk: dict[tuple[int, int], tuple[float, int]],
    selective_bdy: dict[int, tuple[float, int]],
    chain_kb: dict[int, list[tuple[float, int]]],
    direct_kk_w: dict[tuple[int, int], float],
    direct_bdy_w: dict[int, float],
    n_kept: int,
) -> int:
    """Connectivity fix per connected component. Each component without
    boundary access gets the cheapest chain_kb candidate over all of its
    detectors added to selective_bdy, ignoring the cutoff.
    Mutates selective_bdy. Returns the number of edges added."""
    parent = list(range(n_kept))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in direct_kk_w.keys():
        union(a, b)
    for (a, b) in selective_kk.keys():
        union(a, b)

    has_boundary_root: set[int] = set()
    for n in (set(direct_bdy_w) | set(selective_bdy)):
        has_boundary_root.add(find(n))

    comps_to_fix: dict[int, list[int]] = collections.defaultdict(list)
    for d in range(n_kept):
        r = find(d)
        if r in has_boundary_root:
            continue
        comps_to_fix[r].append(d)

    fix_added = 0
    for _r, members in comps_to_fix.items():
        best_d: Optional[int] = None
        best_w = float("inf")
        best_obs = 0
        for d in members:
            for (w, obs) in chain_kb.get(d, []):
                if w < best_w:
                    best_d, best_w, best_obs = d, w, obs
        if best_d is None:
            continue   # truly isolated component
        selective_bdy[best_d] = (best_w, best_obs)
        fix_added += 1

    return fix_added


def _apply_connectivity_fix(
    selective_kk: dict[tuple[int, int], tuple[float, int]],
    selective_bdy: dict[int, tuple[float, int]],
    chain_kb: dict[int, list[tuple[float, int]]],
    direct_kk_w: dict[tuple[int, int], float],
    direct_bdy_w: dict[int, float],
    n_kept: int,
    *,
    type: str = "per_detector",
) -> int:
    """Give each component of the reduced graph that lacks boundary access
    one boundary edge. A low `weight_cutoff` can strand kept detectors, and
    pymatching then fails on odd-parity syndromes ("No perfect matching could
    be found").
      "per_detector": cheapest chain of the first stranded detector found.
      "union_find":   cheapest chain over the whole stranded component.
    Both add the same number of edges. Mutates selective_bdy, returns the
    number of edges added. Raises ValueError on any other `type`.
    """
    if type == "per_detector":
        return _apply_connectivity_fix_per_detector(
            selective_kk, selective_bdy, chain_kb,
            direct_kk_w, direct_bdy_w, n_kept,
        )
    if type == "union_find":
        return _apply_connectivity_fix_union_find(
            selective_kk, selective_bdy, chain_kb,
            direct_kk_w, direct_bdy_w, n_kept,
        )
    raise ValueError(
        f"connectivity_fix_type must be 'per_detector' or 'union_find'; got {type!r}."
    )


def _reduce_dem_chain_aware_with_cutoff(
    dem: stim.DetectorErrorModel,
    clip_set: set[int],
    *,
    weight_cutoff: float = 15.0,
    connectivity_fix_type: str = "per_detector",
) -> tuple[stim.DetectorErrorModel, list[int], dict[int, int]]:
    """Contract the matching graph of `dem`: detectors in `clip_set` are
    absorbed into synthetic edges between kept detectors, or from a kept
    detector to the boundary, found by a Dijkstra from each kept detector.
    A synthetic edge is emitted only if its weight ln((1-p)/p) is below
    `weight_cutoff` and strictly below the direct edge it parallels.
    `connectivity_fix_type` is passed to _apply_connectivity_fix.

    Returns (new_dem, kept, old_to_new): kept[i] is the old index of new
    detector i, old_to_new is the inverse map.
    """
    n_orig = dem.num_detectors
    kept = sorted(d for d in range(n_orig) if d not in clip_set)
    old_to_new = {d_old: d_new for d_new, d_old in enumerate(kept)}
    kept_set = set(kept)

    # Parse the DEM into adjacency lists.
    adj: list[list[tuple[int, float, int]]] = [[] for _ in range(n_orig)]
    bdy_edges: list[list[tuple[float, int]]] = [[] for _ in range(n_orig)]
    direct_errors: list[tuple[list[int], int, float, float]] = []

    for inst in dem:
        if inst.type != "error":
            continue
        p = float(inst.args_copy()[0])
        if p <= 0 or p >= 1:
            continue
        w = -math.log(p / (1 - p))

        # Split at separators: `D_a D_b L0 ^ D_c D_d` is two independent edges
        # of probability p, as pymatching reads it. Merging them would emit a
        # 3+-detector error, which pymatching silently drops, and would hide
        # both edges from the Dijkstra.
        comp_dets: list[list[int]] = [[]]
        comp_obs: list[int] = [0]
        for t in inst.targets_copy():
            if t.is_separator():
                comp_dets.append([])
                comp_obs.append(0)
            elif t.is_relative_detector_id():
                comp_dets[-1].append(t.val)
            elif t.is_logical_observable_id():
                comp_obs[-1] ^= (1 << t.val)

        for dets_raw, obs_mask in zip(comp_dets, comp_obs):
            # XOR-merge duplicate detectors within the component.
            parity: dict[int, int] = {}
            for d in dets_raw:
                parity[d] = parity.get(d, 0) ^ 1
            det_targets = [d for d, par in parity.items() if par]

            if det_targets and all(d in old_to_new for d in det_targets):
                direct_errors.append((det_targets, obs_mask, p, w))

            if len(det_targets) == 1:
                d = det_targets[0]
                bdy_edges[d].append((w, obs_mask))
            elif len(det_targets) == 2:
                a, b = det_targets
                adj[a].append((b, w, obs_mask))
                adj[b].append((a, w, obs_mask))
            # 0- or 3+-det components: ignored for matching graph

    # Cheapest direct edge per kept pair and per kept-to-boundary.
    direct_kk_w: dict[tuple[int, int], float] = {}
    direct_bdy_w: dict[int, float] = {}
    for det_targets, _obs, _p, w in direct_errors:
        if len(det_targets) == 1:
            a = old_to_new[det_targets[0]]
            if a not in direct_bdy_w or direct_bdy_w[a] > w:
                direct_bdy_w[a] = w
        elif len(det_targets) == 2:
            a, b = sorted([old_to_new[d] for d in det_targets])
            if (a, b) not in direct_kk_w or direct_kk_w[(a, b)] > w:
                direct_kk_w[(a, b)] = w

    # Dijkstra from each kept detector through the clipped subgraph.
    chain_kk: dict[tuple[int, int], tuple[float, int]] = {}
    chain_kb: dict[int, list[tuple[float, int]]] = {}

    for src in kept:
        dist: dict[int, tuple[float, int]] = {}
        heap: list[tuple[float, int, int]] = []
        for (nbr, w, obs) in adj[src]:
            if nbr in clip_set:
                if nbr not in dist or dist[nbr][0] > w:
                    dist[nbr] = (w, obs)
                    heapq.heappush(heap, (w, nbr, obs))

        while heap:
            cur_w, cur, cur_obs = heapq.heappop(heap)
            if cur_w > dist.get(cur, (float("inf"),))[0]:
                continue

            # A boundary edge at this clipped node closes a src-to-boundary chain.
            for (w_b, obs_b) in bdy_edges[cur]:
                src_new = old_to_new[src]
                chain_kb.setdefault(src_new, []).append((cur_w + w_b, cur_obs ^ obs_b))

            for (nbr, w_e, obs_e) in adj[cur]:
                new_w = cur_w + w_e
                new_obs = cur_obs ^ obs_e
                if nbr == src:
                    continue
                if nbr in kept_set:
                    a, b = sorted([old_to_new[src], old_to_new[nbr]])
                    cur_best = chain_kk.get((a, b))
                    if cur_best is None or cur_best[0] > new_w:
                        chain_kk[(a, b)] = (new_w, new_obs)
                    continue   # don't relax through other kept nodes
                if nbr not in dist or dist[nbr][0] > new_w:
                    dist[nbr] = (new_w, new_obs)
                    heapq.heappush(heap, (new_w, nbr, new_obs))

    # Keep chains strictly cheaper than the direct edge and below the cutoff.
    selective_kk: dict[tuple[int, int], tuple[float, int]] = {}
    for (a, b), (w, obs) in chain_kk.items():
        direct_w = direct_kk_w.get((a, b), float("inf"))
        if w >= direct_w or w >= weight_cutoff:
            continue
        selective_kk[(a, b)] = (w, obs)

    selective_bdy: dict[int, tuple[float, int]] = {}
    for a, edges in chain_kb.items():
        min_chain_w, min_chain_obs = min(edges, key=lambda x: x[0])
        direct_w = direct_bdy_w.get(a, float("inf"))
        if min_chain_w >= direct_w or min_chain_w >= weight_cutoff:
            continue
        selective_bdy[a] = (min_chain_w, min_chain_obs)

    # Every kept detector needs a path to the boundary in the reduced graph.
    _apply_connectivity_fix(
        selective_kk, selective_bdy, chain_kb,
        direct_kk_w, direct_bdy_w,
        n_kept=len(kept),
        type=connectivity_fix_type,
    )

    # Assemble the reduced DEM.
    new_dem = stim.DetectorErrorModel()
    coords = dem.get_detector_coordinates()
    for d_old in kept:
        d_new = old_to_new[d_old]
        c = coords.get(d_old, [])
        new_dem.append("detector", list(c), [stim.target_relative_detector_id(d_new)])

    # Original errors on kept detectors only: renumbered, same probabilities.
    for det_targets, obs_mask, p, _w in direct_errors:
        new_targets = [stim.target_relative_detector_id(old_to_new[d]) for d in det_targets]
        for o in range(64):
            if (obs_mask >> o) & 1:
                new_targets.append(stim.target_logical_observable_id(o))
        new_dem.append("error", [p], new_targets)

    # Synthetic kept-kept edges.
    for (a, b), (w, obs_mask) in selective_kk.items():
        p = math.exp(-w) / (1 + math.exp(-w))
        if p <= 0 or p >= 1:
            continue
        new_targets = [
            stim.target_relative_detector_id(a),
            stim.target_relative_detector_id(b),
        ]
        for o in range(64):
            if (obs_mask >> o) & 1:
                new_targets.append(stim.target_logical_observable_id(o))
        new_dem.append("error", [p], new_targets)

    # Synthetic kept-to-boundary edges.
    for src_new, (w, obs_mask) in selective_bdy.items():
        p = math.exp(-w) / (1 + math.exp(-w))
        if p <= 0 or p >= 1:
            continue
        new_targets = [stim.target_relative_detector_id(src_new)]
        for o in range(64):
            if (obs_mask >> o) & 1:
                new_targets.append(stim.target_logical_observable_id(o))
        new_dem.append("error", [p], new_targets)

    return new_dem, kept, old_to_new


def _repack_syndromes_to_kept(
    packed_dets: np.ndarray,
    kept: list[int],
    n_orig_dets: int,
) -> np.ndarray:
    """Select the `kept` detector bits of bit-packed syndromes, in `kept`
    order: (n_shots, ceil(n_orig/8)) -> (n_shots, ceil(len(kept)/8)) uint8,
    little bit order."""
    unpacked = np.unpackbits(packed_dets, axis=1, bitorder="little")[:, :n_orig_dets]
    return np.packbits(unpacked[:, kept], axis=1, bitorder="little")


# ---------------- sinter.Sampler / sinter.CompiledSampler ----------------

class PartialDesaturationSampler(sinter.Sampler):
    """Drop-in replacement for cultiv.DesaturationSampler. Its
    compiled_sampler_for_task also accepts partial_mask_bool, weight_cutoff
    and connectivity_fix_type to enable the partial-DEM reduction."""

    def compiled_sampler_for_task(
        self,
        task: sinter.Task,
        *,
        partial_mask_bool: Optional[np.ndarray] = None,
        weight_cutoff: float = 15.0,
        connectivity_fix_type: str = "per_detector",
    ) -> "CompiledPartialDesaturationSampler":
        return CompiledPartialDesaturationSampler.from_task(
            task,
            partial_mask_bool=partial_mask_bool,
            weight_cutoff=weight_cutoff,
            connectivity_fix_type=connectivity_fix_type,
        )


@dataclasses.dataclass
class CompiledPartialDesaturationSampler(sinter.CompiledSampler):
    """Complementary-gap sampler with optional partial-DEM reduction.

    Attributes mirror cultiv's CompiledDesaturationSampler, plus:
      gap_dem: DEM decoded on: reduced in partial mode, else gap_dem_complete.
      gap_dem_complete: full gap DEM, kept in partial mode so a complete
        decoder can be built for the same shots.
      kept: kept[i] is the old index of new detector i. None in full mode.
      old_to_new: dict, old -> new detector index. None in full mode.
      partial_mask_bool: input mask, True = detector kept. None in full mode.
      weight_cutoff: synthetic edges at or above this weight are dropped.
        The default 15.0 is the empirical speed/quality knee.
      connectivity_fix_type: "per_detector" or "union_find".

    In partial mode sample() reports the raw error count of the reduced
    decoder, expected to be tens of percent. The partial gap is a cheap gating
    signal for a two-stage protocol, not a standalone decoder.
    """

    def __init__(
        self,
        task: sinter.Task,
        gap_dem: stim.DetectorErrorModel,
        gap_dem_complete: stim.DetectorErrorModel,
        postselected_detectors: frozenset[int],
        gap_circuit: stim.Circuit,
        *,
        kept: Optional[list[int]] = None,
        old_to_new: Optional[dict[int, int]] = None,
        partial_mask_bool: Optional[np.ndarray] = None,
        weight_cutoff: float = 15.0,
        connectivity_fix_type: str = "per_detector",
    ):
        self.task = task
        self.gap_dem = gap_dem
        self.gap_dem_complete = gap_dem_complete
        self.postselected_detectors = postselected_detectors
        self.gap_circuit = gap_circuit

        self.partial_mask_bool = partial_mask_bool
        self.weight_cutoff = float(weight_cutoff)
        self.connectivity_fix_type = str(connectivity_fix_type)
        self.kept = kept
        self.old_to_new = old_to_new

        self.num_dets = self.gap_circuit.num_detectors
        self.num_det_bytes = -(-self.num_dets // 8)
        self._discard_mask = np.packbits(
            np.array(
                [k in self.postselected_detectors for k in range(self.num_dets)],
                dtype=np.bool_,
            ),
            bitorder="little",
        )
        self.gap_circuit_sampler = self.gap_circuit.compile_detector_sampler()
        self.gap_decoder = pymatching.Matching.from_detector_error_model(self.gap_dem)

        edge = next(iter(self.gap_decoder.to_networkx().edges.values()))
        edge_w = edge["weight"]
        edge_p = edge["error_probability"]
        self.decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w

        # _obs_det_byte masks the obs-detector bit in the decoder's layout:
        # bit num_dets - 1 in full mode, bit len(kept) - 1 in partial mode.
        if kept is None:
            effective_n = self.num_dets
            self._effective_num_det_bytes = self.num_det_bytes
        else:
            effective_n = len(kept)
            self._effective_num_det_bytes = -(-len(kept) // 8)
            # The obs detector must be kept and stay last: decoding toggles it
            # in the last byte.
            obs_old_idx = self.gap_dem_complete.num_detectors - 1
            if old_to_new is None or obs_old_idx not in old_to_new:
                raise RuntimeError(
                    "obs-detector was unexpectedly clipped from the reduced DEM."
                )
            if old_to_new[obs_old_idx] != len(kept) - 1:
                raise RuntimeError(
                    f"obs-detector new index ({old_to_new[obs_old_idx]}) is not "
                    f"the last index (n_kept-1 = {len(kept) - 1}); the bit-flip "
                    f"trick assumes it is."
                )
        self._obs_det_byte = 1 << ((effective_n - 1) % 8)

    @staticmethod
    def from_task(
        task: sinter.Task,
        *,
        partial_mask_bool: Optional[np.ndarray] = None,
        weight_cutoff: float = 15.0,
        connectivity_fix_type: str = "per_detector",
    ) -> "CompiledPartialDesaturationSampler":
        gap_dem_complete, gap_circuit, postselected_detectors = (
            get_desaturated_dem_with_obs_detector(task.circuit, dem=task.detector_error_model)
        )

        # Full mode: decode on the complete gap DEM.
        if partial_mask_bool is None:
            return CompiledPartialDesaturationSampler(
                task=task,
                gap_dem=gap_dem_complete,
                gap_dem_complete=gap_dem_complete,
                postselected_detectors=postselected_detectors,
                gap_circuit=gap_circuit,
                kept=None,
                old_to_new=None,
                partial_mask_bool=None,
                weight_cutoff=weight_cutoff,
                connectivity_fix_type=connectivity_fix_type,
            )

        n_circuit_dets = task.circuit.num_detectors
        if partial_mask_bool.shape != (n_circuit_dets,):
            raise ValueError(
                f"partial_mask_bool must have shape ({n_circuit_dets},); "
                f"got {partial_mask_bool.shape}."
            )

        # Clip the circuit detectors outside the mask. Virtual pair nodes
        # (n_circuit_dets .. n_gap_dets-2) and the obs detector (n_gap_dets-1)
        # are always kept.
        clip_set = {d for d in range(n_circuit_dets) if not bool(partial_mask_bool[d])}

        gap_dem_reduced, kept, old_to_new = _reduce_dem_chain_aware_with_cutoff(
            gap_dem_complete, clip_set,
            weight_cutoff=weight_cutoff,
            connectivity_fix_type=connectivity_fix_type,
        )

        return CompiledPartialDesaturationSampler(
            task=task,
            gap_dem=gap_dem_reduced,
            gap_dem_complete=gap_dem_complete,
            postselected_detectors=postselected_detectors,
            gap_circuit=gap_circuit,
            kept=kept,
            old_to_new=old_to_new,
            partial_mask_bool=partial_mask_bool,
            weight_cutoff=weight_cutoff,
            connectivity_fix_type=connectivity_fix_type,
        )

    # ----------------- sinter.CompiledSampler interface -----------------

    def sample(self, shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs = self.gap_circuit_sampler.sample(
            shots, separate_observables=True, bit_packed=True,
        )

        keep_mask = ~np.any(dets & self._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        assert actual_obs.shape[1] == 1
        actual_obs = actual_obs[:, 0]
        predictions, gaps = self._decode_batch_overwrite_last_byte(bit_packed_dets=dets)
        errors = predictions ^ actual_obs
        counter: collections.Counter = collections.Counter()
        for gap, err in zip(gaps, errors):
            counter[f"E{round(gap)}" if err else f"C{round(gap)}"] += 1
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=int(shots),
            errors=int(np.count_nonzero(errors)),
            discards=int(shots - np.count_nonzero(keep_mask)),
            seconds=t1 - t0,
            custom_counts=counter,
        )

    # ----------------- decoding entry points -----------------

    def _repack_to_decoder_layout(self, bit_packed_dets: np.ndarray) -> np.ndarray:
        """Syndromes in the decoder's layout. Full mode returns the input
        array itself, which decoding then mutates in place. Partial mode
        returns a new array of the `self.kept` bits and leaves the input alone."""
        if self.kept is None:
            return bit_packed_dets
        return _repack_syndromes_to_kept(bit_packed_dets, self.kept, self.num_dets)

    def _decode_batch_overwrite_last_byte(
        self, bit_packed_dets: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self._repack_to_decoder_layout(bit_packed_dets)
        target[:, -1] |= self._obs_det_byte
        _, on_weights = self.gap_decoder.decode_batch(
            target,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        target[:, -1] ^= self._obs_det_byte
        _, off_weights = self.gap_decoder.decode_batch(
            target,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        gaps: np.ndarray = np.abs((on_weights - off_weights) * self.decibels_per_w)
        predictions: np.ndarray = on_weights < off_weights
        return predictions, gaps

    def _decode_batch_overwrite_last_byte_with_time(
        self,
        bit_packed_dets: np.ndarray,
        measure_mode: str = "serial",
    ) -> tuple[np.ndarray, np.ndarray, float]:
        target = self._repack_to_decoder_layout(bit_packed_dets)

        target[:, -1] |= self._obs_det_byte
        t0_on = time.perf_counter_ns()
        _, on_weights = self.gap_decoder.decode_batch(
            target,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        t1_on = time.perf_counter_ns()
        measured_ns_on = float(max(t1_on - t0_on, 0))

        target[:, -1] ^= self._obs_det_byte
        t0_off = time.perf_counter_ns()
        _, off_weights = self.gap_decoder.decode_batch(
            target,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        t1_off = time.perf_counter_ns()
        measured_ns_off = float(max(t1_off - t0_off, 0))

        gaps: np.ndarray = np.abs((on_weights - off_weights) * self.decibels_per_w)
        predictions: np.ndarray = on_weights < off_weights

        if measure_mode == "serial":
            measured_ns = measured_ns_on + measured_ns_off
        elif measure_mode == "parallel":
            measured_ns = max(measured_ns_on, measured_ns_off)
        else:
            raise NotImplementedError("This measure mode is not supported!")
        return predictions, gaps, measured_ns

    def decode_det_set(self, det_set: set[int]) -> tuple[bool, float]:
        dets = np.zeros(shape=(1, self.num_dets), dtype=np.bool_)
        for d in det_set:
            if d in self.postselected_detectors:
                return False, 0
            dets[0][d] = 1
        predictions, gaps = self._decode_batch_overwrite_last_byte(
            np.packbits(dets, bitorder="little", axis=1),
        )
        return predictions[0], math.ceil(gaps[0])

    def decode_det_set_with_time(
        self, det_set: set[int], measure_mode: str = "serial",
    ) -> tuple[bool, float, float]:
        dets = np.zeros(shape=(1, self.num_dets), dtype=np.bool_)
        for d in det_set:
            if d in self.postselected_detectors:
                return False, 0, 0.0
            dets[0][d] = 1
        predictions, gaps, measured_ns = self._decode_batch_overwrite_last_byte_with_time(
            np.packbits(dets, bitorder="little", axis=1),
            measure_mode=measure_mode,
        )
        return predictions[0], math.ceil(gaps[0]), measured_ns
