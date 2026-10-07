# dem_structure.py — DEM error components and set-crossing measures.
#
# A component is one separator-delimited part of an error instruction:
# `error(p) D0 D1 L0 ^ D2 D3` gives two, each with probability p. Duplicate
# detector targets are XOR-merged; parallel components are not merged.
# Support size is 1, 2 or >= 3, so components are not matching-graph edges.
# A mask M is a bool array over the DEM's detectors.
# [used: PartialMaskBuilder.closure_mask, run_eval_partial_mask]

import warnings

import numpy as np
import stim


class DemStructure:
    def __init__(self, dem: stim.DetectorErrorModel):
        self.num_detectors = dem.num_detectors
        components, probs = [], []
        for inst in dem.flattened():
            if inst.type != "error":
                continue
            p = inst.args_copy()[0]
            parts = [[]]
            for t in inst.targets_copy():
                if t.is_separator():
                    parts.append([])
                elif t.is_relative_detector_id():
                    parts[-1].append(t.val)
            for ds in parts:
                parity = {}
                for d in ds:
                    parity[d] = parity.get(d, 0) ^ 1
                ds = sorted(d for d, par in parity.items() if par)
                if ds:
                    components.append(ds)
                    probs.append(p)
        self.components: list[list[int]] = components       # detector support per component
        self.p: np.ndarray = np.asarray(probs, dtype=np.float64)
        # incident components per detector
        self.by_det: list[list[int]] = [[] for _ in range(self.num_detectors)]
        for i, ds in enumerate(components):
            for d in ds:
                self.by_det[d].append(i)
        # neighbours = detectors sharing a component, not matching-graph adjacency
        self.nbrs: list[set[int]] = [set() for _ in range(self.num_detectors)]
        for ds in components:
            for a in ds:
                self.nbrs[a].update(b for b in ds if b != a)

    # ---------------------------------------------------------------- measures
    @staticmethod
    def _crosses(mask: np.ndarray, ds: list[int]) -> bool:
        """True iff the component's full support meets both M and its complement."""
        return any(mask[d] for d in ds) and not all(mask[d] for d in ds)

    def S(self, mask: np.ndarray) -> float:
        """Crossing mass S(M): sum of p_e over components crossing M. A
        first-order proxy for the straddling error mass, not its exact value."""
        return float(sum(self.p[i] for i, ds in enumerate(self.components) if self._crosses(mask, ds)))

    def delta(self, mask: np.ndarray, d: int) -> float:
        """S(M with d toggled) - S(M). Only components containing d can change."""
        m2 = mask.copy()
        m2[d] = not m2[d]
        return float(sum(
            self.p[i] * (float(self._crosses(m2, self.components[i])) - float(self._crosses(mask, self.components[i])))
            for i in self.by_det[d]))

    def boundary(self, mask: np.ndarray) -> list[int]:
        """Selected detectors with at least one hidden component-neighbour."""
        return [d for d in range(self.num_detectors) if mask[d] and any(not mask[n] for n in self.nbrs[d])]

    def frontier(self, mask: np.ndarray) -> list[int]:
        """Hidden detectors with at least one selected component-neighbour."""
        return [d for d in range(self.num_detectors) if not mask[d] and any(mask[n] for n in self.nbrs[d])]

    # ---------------------------------------------------------------- closure
    def closure(self, mask: np.ndarray, *, tau: float = 0.0, max_iter=None, return_info: bool = False):
        """Add-only closure of M under the crossing mass.

        Each iteration evaluates delta(M, d) for every frontier detector
        against the same mask and adds all with delta < -tau at once (batch,
        not sequential greedy). It stops at a fixed point or after max_iter
        iterations (default num_detectors). S(M) strictly decreases on every
        iteration that adds a detector. The comparison has no tolerance, so
        with tau = 0 a delta that is zero up to rounding can admit a fill.

        Returns the closed mask, or (mask, info) when return_info is True.
        info keys: iterations, detectors_added, S_before, S_after, converged,
        S_per_iteration (length iterations + 1, initial to final). Warns if
        max_iter is reached without a fixed point.
        """
        m = np.asarray(mask, dtype=np.bool_).copy()
        if max_iter is None:
            max_iter = self.num_detectors
        S_hist = [self.S(m)]
        iterations = 0
        converged = False
        while iterations < max_iter:
            fills = [d for d in self.frontier(m) if self.delta(m, d) < -tau]
            if not fills:
                converged = True
                break
            m[fills] = True
            iterations += 1
            S_hist.append(self.S(m))
        else:
            converged = not [d for d in self.frontier(m) if self.delta(m, d) < -tau]
        if not converged:
            warnings.warn(f"DemStructure.closure: max_iter={max_iter} reached without a fixed point "
                          f"(S {S_hist[0]:.4f} -> {S_hist[-1]:.4f})")
        if not return_info:
            return m
        info = {
            "iterations": iterations,
            "detectors_added": int(m.sum() - np.asarray(mask, dtype=np.bool_).sum()),
            "S_before": S_hist[0],
            "S_after": S_hist[-1],
            "S_per_iteration": S_hist,
            "converged": converged,
        }
        return m, info
