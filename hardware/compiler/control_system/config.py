"""config.py — load and validate the compiler configuration, build the board tree.

The config JSON has two sections (see data/config.template.json):
  * hardware.layers : the tree of control boards (leaf -> router -> root)
  * postselect      : WHICH detectors to postselect, as an explicit list of
                      stim detector indices (the decoder workflow decides the
                      set; the compiler just consumes it)

This module is circuit-agnostic: it only builds the topology + the postselect
detector set.  Circuit-dependent work (qubit scopes, placement, forwarding)
lives in placement.py, which consumes both this config and the parsed circuit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# --------------------------------------------------------------------------- #
# postselect detector set
# --------------------------------------------------------------------------- #
def resolve_postselect_detectors(spec, base_dir=None) -> frozenset:
    """Resolve the config's ``postselect`` field into a set of stim detector
    indices. Accepted forms:

      * "none" or []                     -> no postselection
      * [3, 17, 69, ...]                 -> explicit stim detector indices
      * {"from_file": "path.json"}       -> read the list from a JSON file:
            a bare list, or a dict whose "postselected_detectors" key (or the
            key named by an optional "key" entry) holds the list. A relative
            path resolves against `base_dir` (the config file's directory).

    The postselect set is a circuit+decoder property computed by the decoder
    workflow (e.g. algorithms/get_desaturated_dem_with_obs_detector.py writes
    postselected_detectors.json); the compiler consumes it verbatim. Legacy
    preset strings ('cultivation', ...) and predicate objects are rejected.
    """
    if spec is None or spec == "none":
        return frozenset()
    if isinstance(spec, list):
        out = frozenset(int(d) for d in spec)
        if any(d < 0 for d in out):
            raise ValueError(f"postselect detector indices must be >= 0; got {sorted(out)[:8]}")
        return out
    if isinstance(spec, dict) and "from_file" in spec:
        path = spec["from_file"]
        if not os.path.isabs(path) and base_dir is not None:
            path = os.path.join(base_dir, path)
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data[spec.get("key", "postselected_detectors")]
        if not isinstance(data, list):
            raise ValueError(f"postselect from_file {path!r} did not yield a list")
        return resolve_postselect_detectors(data)
    raise ValueError(
        f"postselect must be 'none', a list of stim detector indices, or "
        f"{{'from_file': <json path>}}; got {spec!r}. Legacy presets/predicates "
        f"were removed — generate the explicit list with your decoder workflow "
        f"(for cultivation: algorithms/get_desaturated_dem_with_obs_detector.py)."
    )


# --------------------------------------------------------------------------- #
# board tree
# --------------------------------------------------------------------------- #
@dataclass
class Kernel:
    n: int          # selector width
    h: int          # number of cores


@dataclass
class Board:
    board_id: int
    type: str                       # 'leaf' | 'router' | 'root'
    level: int
    kernels: List[Kernel]
    qubits: Optional[List[int]] = None      # leaf only
    children: List[int] = field(default_factory=list)   # router/root
    raw_out: Optional[int] = None           # leaf/router only


class CompilerConfig:
    def __init__(self, data: dict, base_dir: Optional[str] = None):
        self.data = data
        self.name: str = data.get("name", "<unnamed>")
        self.version: int = data.get("version", 0)
        self.postselect_detectors: frozenset = resolve_postselect_detectors(
            data.get("postselect", "none"), base_dir=base_dir)

        self.boards: Dict[int, Board] = {}
        self.parent: Dict[int, int] = {}
        self.leaf_ids: List[int] = []
        self.root_id: Optional[int] = None
        self.subtree_leaves: Dict[int, Set[int]] = {}
        self.subtree_qubits: Dict[int, Set[int]] = {}
        self.qubit_to_leaf: Dict[int, int] = {}

        self._build()

    @classmethod
    def from_file(cls, path: str) -> "CompilerConfig":
        with open(path) as f:
            return cls(json.load(f), base_dir=os.path.dirname(os.path.abspath(path)))

    # ------------------------------------------------------------------ #
    def _build(self) -> None:
        if "hardware" not in self.data or "layers" not in self.data["hardware"]:
            raise ValueError("config missing 'hardware.layers'")
        layers = sorted(self.data["hardware"]["layers"], key=lambda l: l["level"])

        # levels must be contiguous 0..N-1, with leaf first and root last
        levels = [l["level"] for l in layers]
        if levels != list(range(len(layers))):
            raise ValueError(f"layer levels must be contiguous 0..N-1, got {levels}")
        if layers[-1]["type"] != "root":
            raise ValueError("last layer must be type 'root'")
        # monolithic: a single 'root' layer whose board carries qubits (leaf+root fused)
        monolithic = (len(layers) == 1 and layers[0]["type"] == "root")
        if not monolithic:
            if layers[0]["type"] != "leaf":
                raise ValueError("level-0 layer must be type 'leaf' (or a single 'root' "
                                 "layer with qubits for a monolithic board)")
            for l in layers[1:-1]:
                if l["type"] != "router":
                    raise ValueError(f"intermediate layer {l['level']} must be 'router'")

        # --- parse boards, per-type field checks ---
        for layer in layers:
            ltype, level = layer["type"], layer["level"]
            for b in layer["boards"]:
                bid = b["board_id"]
                if bid in self.boards:
                    raise ValueError(f"duplicate board_id {bid}")
                kspec = b["kernels"]
                if isinstance(kspec, dict):          # compact: {count, n, h} uniform kernels
                    kernels = [Kernel(int(kspec["n"]), int(kspec["h"]))
                               for _ in range(int(kspec["count"]))]
                else:                                # explicit per-kernel list
                    kernels = [Kernel(int(k["n"]), int(k["h"])) for k in kspec]
                board = Board(bid, ltype, level, kernels,
                              qubits=b.get("qubits"),
                              children=list(b.get("children", [])),
                              raw_out=b.get("raw_out"))
                if ltype == "leaf":
                    if board.qubits is None:
                        raise ValueError(f"leaf board {bid} needs 'qubits'")
                    if board.children:
                        raise ValueError(f"leaf board {bid} must not have 'children'")
                    if board.raw_out is None:
                        raise ValueError(f"leaf board {bid} needs 'raw_out'")
                elif ltype == "root" and board.qubits is not None:
                    # monolithic root (leaf+root fused): qubits, no children, endpoint
                    if board.children:
                        raise ValueError(f"monolithic root {bid} must not have 'children'")
                    if board.raw_out is not None:
                        raise ValueError(f"monolithic root {bid} is the endpoint; "
                                         f"must not have 'raw_out'")
                else:
                    if not board.children:
                        raise ValueError(f"{ltype} board {bid} needs 'children'")
                    if board.qubits is not None:
                        raise ValueError(f"{ltype} board {bid} must not have 'qubits'")
                    if ltype == "router" and board.raw_out is None:
                        raise ValueError(f"router board {bid} needs 'raw_out'")
                self.boards[bid] = board

        # --- root ---
        roots = [b for b in self.boards.values() if b.type == "root"]
        if len(roots) != 1:
            raise ValueError(f"expected exactly one root board, got {len(roots)}")
        self.root_id = roots[0].board_id
        # "leaves" = boards wired directly to qubits (real leaves + a monolithic root)
        self.leaf_ids = [b.board_id for b in self.boards.values() if b.qubits is not None]

        # --- parent/child links; each child referenced by exactly one parent,
        #     and only from the immediately-higher layer ---
        for b in self.boards.values():
            for cid in b.children:
                if cid not in self.boards:
                    raise ValueError(f"board {b.board_id} references unknown child {cid}")
                child = self.boards[cid]
                if child.level != b.level - 1:
                    raise ValueError(f"board {b.board_id} (level {b.level}) child {cid} "
                                     f"is level {child.level}, expected {b.level - 1}")
                if cid in self.parent:
                    raise ValueError(f"board {cid} has two parents "
                                     f"({self.parent[cid]} and {b.board_id})")
                self.parent[cid] = b.board_id
        for b in self.boards.values():
            if b.board_id != self.root_id and b.board_id not in self.parent:
                raise ValueError(f"board {b.board_id} has no parent")

        # --- qubit ownership (all declared qubits, measured or not) ---
        for lid in self.leaf_ids:
            for q in self.boards[lid].qubits:
                if q in self.qubit_to_leaf:
                    raise ValueError(f"qubit {q} wired to two leaves "
                                     f"({self.qubit_to_leaf[q]} and {lid})")
                self.qubit_to_leaf[q] = lid

        # --- subtree leaves / qubits (bottom-up by level) ---
        for b in sorted(self.boards.values(), key=lambda x: x.level):
            if b.qubits is not None:
                self.subtree_leaves[b.board_id] = {b.board_id}
                self.subtree_qubits[b.board_id] = set(b.qubits)
            else:
                leaves: Set[int] = set()
                qubits: Set[int] = set()
                for cid in b.children:
                    leaves |= self.subtree_leaves[cid]
                    qubits |= self.subtree_qubits[cid]
                self.subtree_leaves[b.board_id] = leaves
                self.subtree_qubits[b.board_id] = qubits

    # ------------------------------------------------------------------ #
    def path_to_root(self, board_id: int) -> List[int]:
        """Boards from `board_id` up to and including the root."""
        path = [board_id]
        while path[-1] != self.root_id:
            path.append(self.parent[path[-1]])
        return path

    def lca(self, board_ids: Sequence[int]) -> int:
        """Lowest common ancestor of a set of boards (the deepest board whose
        subtree contains all of them)."""
        board_ids = list(board_ids)
        if not board_ids:
            raise ValueError("lca of empty set")
        common = set(self.path_to_root(board_ids[0]))
        for bid in board_ids[1:]:
            common &= set(self.path_to_root(bid))
        # the deepest board in the common ancestor set (leaf level = 0, so deepest = min level)
        return min(common, key=lambda b: self.boards[b].level)

    def summary(self) -> str:
        lines = [f"config '{self.name}' v{self.version}: "
                 f"{len(self.boards)} boards, root={self.root_id}, "
                 f"{len(self.leaf_ids)} leaves"]
        for b in sorted(self.boards.values(), key=lambda x: (x.level, x.board_id)):
            kern = ",".join(f"({k.n},{k.h})" for k in b.kernels)
            extra = (f"qubits={len(b.qubits)}" if b.qubits is not None
                     else f"children={b.children}")
            ro = "" if b.raw_out is None else f" raw_out={b.raw_out}"
            lines.append(f"  L{b.level} {b.type:6s} id={b.board_id}: {extra} "
                         f"kernels=[{kern}]{ro}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/config.template.json"
    cfg = CompilerConfig.from_file(path)
    print(cfg.summary())
    print("\nsubtree_leaves:", {b: sorted(v) for b, v in cfg.subtree_leaves.items()})
    print("path board 0 -> root:", cfg.path_to_root(0))
    print("lca(leaves 0,3):", cfg.lca([0, 3]))
    print("postselect detectors:", sorted(cfg.postselect_detectors)[:16],
          f"({len(cfg.postselect_detectors)} total)")
