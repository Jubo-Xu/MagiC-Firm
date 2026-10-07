"""placement.py — circuit-dependent placement over the board tree.

Given a CompilerConfig (topology) and a scanned+refined circuit, decide:
  * each board's measured qubit scope,
  * which board constructs each detector (deepest board covering its qubits),
  * which raw measurements each board must forward upward — two variants:
      forward_normal : only for detectors placed above the board (checked vs raw_out)
      forward_obs    : also the logical-observable measurements (reported, not checked),
  * the stage board for each stage (LCA of that stage's detector placements).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

from config import CompilerConfig


class Placement:
    def __init__(self, config: CompilerConfig, scanner):
        self.cfg = config
        self.s = scanner

        self.used_qubits: Set[int] = set()
        self.leaf_scope: Dict[int, Set[int]] = {}      # leaf_id -> measured qubits
        self.board_scope: Dict[int, Set[int]] = {}     # board_id -> measured qubits in subtree
        self.channel_placement: Dict[tuple, int] = {}  # spatial coord -> board_id
        self.placement: List[int] = []                 # det_idx -> board_id
        self.forward_normal: Dict[int, Set[int]] = {}  # board_id -> qubits forwarded up
        self.forward_obs: Dict[int, Set[int]] = {}     # board_id -> qubits (incl. observable)
        self.stage_board: List[int] = []               # stage_idx -> board_id
        self.observable_qubits: Set[int] = set()
        self.unwired_observable_qubits: Set[int] = set()

        self._compute()

    # ------------------------------------------------------------------ #
    def _det_qubits(self, d: int) -> Set[int]:
        nt = self.s.detector_coord[d][2]
        adj = self.s.detector_to_measurement[nt][d]
        return {q for qs in adj.values() for q in qs}

    def _compute(self) -> None:
        cfg, s = self.cfg, self.s

        # 1) qubits that feed at least one detector
        for d in range(len(s.detector_coord)):
            self.used_qubits |= self._det_qubits(d)

        # 2) leaf scopes + coverage/partition check
        uncovered = self.used_qubits - set(cfg.qubit_to_leaf)
        if uncovered:
            raise ValueError(f"{len(uncovered)} measured qubits are not wired to any "
                             f"leaf board, e.g. {sorted(uncovered)[:8]}")
        for lid in cfg.leaf_ids:
            self.leaf_scope[lid] = set(cfg.boards[lid].qubits) & self.used_qubits

        # 3) board measured scope = subtree qubits ∩ used
        for bid in cfg.boards:
            self.board_scope[bid] = cfg.subtree_qubits[bid] & self.used_qubits

        # 4) placement (Option B): a whole channel (all detectors at one spatial
        #    coord) goes to the LCA of its FULL support, so it is never split —
        #    one kernel = one spatial coordinate, one physical output line.
        for coord, support in s.channel_support.items():
            leaves = {cfg.qubit_to_leaf[q] for q in support}
            self.channel_placement[coord] = cfg.lca(leaves)
        self.placement = [self.channel_placement[s.detector_coord[d][0]]
                          for d in range(len(s.detector_coord))]

        # 5) forwarding (normal): a board forwards q if q feeds a detector placed above it
        self.forward_normal = {bid: set() for bid in cfg.boards}
        for d in range(len(s.detector_coord)):
            P = self.placement[d]
            for q in self._det_qubits(d):
                for b in cfg.path_to_root(cfg.qubit_to_leaf[q]):
                    if b == P:
                        break
                    self.forward_normal[b].add(q)

        # 6) observable qubits + forwarding (with observable, routed to root)
        if s._logical_observable_terms:
            for term in s._logical_observable_terms:
                self.observable_qubits |= {q for q, _p in term}
        else:
            for recs in s._logical_observable_recs:
                self.observable_qubits |= {q for q, _t in recs}
        self.unwired_observable_qubits = self.observable_qubits - set(cfg.qubit_to_leaf)

        self.forward_obs = {bid: set(v) for bid, v in self.forward_normal.items()}
        for q in self.observable_qubits & set(cfg.qubit_to_leaf):
            for b in cfg.path_to_root(cfg.qubit_to_leaf[q]):
                if b == cfg.root_id:
                    break
                self.forward_obs[b].add(q)

        # 7) stage board = LCA of the stage's detector placements
        self.stage_board = [cfg.lca({self.placement[d] for d in st["detectors"]})
                            for st in s.stages]

    # ------------------------------------------------------------------ #
    def report(self) -> str:
        cfg = self.cfg
        lines = [f"placement: {len(self.s.detector_coord)} detectors, "
                 f"{len(self.used_qubits)} measured qubits"]
        # detectors per board
        from collections import Counter
        per_board = Counter(self.placement)
        for b in sorted(cfg.boards.values(), key=lambda x: (x.level, x.board_id)):
            bid = b.board_id
            m = len(self.board_scope[bid])
            dets = per_board.get(bid, 0)
            fn, fo = len(self.forward_normal[bid]), len(self.forward_obs[bid])
            ro = b.raw_out
            flag = ""
            if ro is not None and fn > ro:
                flag = f"  !! forward_normal {fn} > raw_out {ro}"
            ro_s = "-" if ro is None else str(ro)
            lines.append(f"  L{b.level} {b.type:6s} id={bid}: m={m:4d} dets={dets:4d} "
                         f"forward_normal={fn:3d}/{ro_s} forward_obs={fo:3d}{flag}")
        lines.append("stage boards: " +
                     ", ".join(f"stage{i}->{bd}" for i, bd in enumerate(self.stage_board)))
        if self.unwired_observable_qubits:
            lines.append(f"NOTE: {len(self.unwired_observable_qubits)} observable qubits "
                         f"not wired to any leaf (observable path incomplete)")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# test helper: partition a circuit's qubits into a small tree by x-coordinate
# --------------------------------------------------------------------------- #
def make_partition_config(scanner, n_leaves: int, *, name="test_partition",
                          postselect="none") -> CompilerConfig:
    """Build a 2-layer (leaves -> root) config that partitions the circuit's
    physical qubits into `n_leaves` vertical strips by x-coordinate. Kernel
    resources are placeholders (placement doesn't use them)."""
    coords = scanner._physical_qubits_coord_raw
    qubits = [q for q, xy in enumerate(coords) if xy is not None]
    xs = sorted({coords[q][0] for q in qubits})
    # split the x-range into n_leaves contiguous strips
    bounds = [xs[int(round(i * len(xs) / n_leaves))] for i in range(n_leaves)] + [xs[-1] + 1]
    def strip(q):
        x = coords[q][0]
        for i in range(n_leaves):
            if bounds[i] <= x < bounds[i + 1]:
                return i
        return n_leaves - 1
    buckets: Dict[int, List[int]] = {i: [] for i in range(n_leaves)}
    for q in qubits:
        buckets[strip(q)].append(q)
    big_kernels = [{"n": 32, "h": 16} for _ in range(512)]   # generously over-provisioned
    leaves = [{"board_id": i, "qubits": sorted(buckets[i]),
               "kernels": big_kernels, "raw_out": 999}
              for i in range(n_leaves) if buckets[i]]
    root = {"board_id": 1000, "children": [b["board_id"] for b in leaves],
            "kernels": big_kernels}
    data = {"name": name, "version": 1,
            "hardware": {"layers": [
                {"level": 0, "type": "leaf", "boards": leaves},
                {"level": 1, "type": "root", "boards": [root]}]},
            "postselect": postselect}
    return CompilerConfig(data)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from parser import DetectorConstructionScanner

    circ = sys.argv[1] if len(sys.argv) > 1 else (
        "../../../magic_state_cultivation/circuits_dump/end2end_d1=3_d2=9_r1=3_r2=3_b=Y.stim")
    n_leaves = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    s = DetectorConstructionScanner.from_file(circ)
    s.scan(); s.refine()
    cfg = make_partition_config(s, n_leaves)
    print(cfg.summary())
    print()
    pl = Placement(cfg, s)
    print(pl.report())
