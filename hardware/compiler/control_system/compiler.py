"""compiler.py — turn (config + parsed circuit) into per-board hardware programs.

This first stage builds the parts that don't yet need the postselect/observable
specifics:
  * per-board measurement input map  (m, qubit -> local channel index)
  * measurement synchronization masks (per measurement-time, which channels fire)
  * channel -> kernel assignment      (bin-pack, with n/h/k feasibility, the
                                       minimal-h computation, and the strict-
                                       increasing-completion check the circular
                                       counter relies on)
Selector/core mask tables, postselect blocks, root output and the observable
block come next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from config import CompilerConfig
from placement import Placement


@dataclass
class ChannelInstance:
    board_id: int
    coord: tuple
    detectors: List[int]              # placed here, sorted by new_t
    support: List[int]                # distinct qubits (selector set), sorted
    required_h: int                   # minimal cores under the emission-index mod-h scheme
    init_counter: int                 # circular-counter start (0 for emission-index cores)
    kernel: Optional[int] = None      # assigned kernel index on the board


@dataclass
class KernelProgram:
    board_id: int
    kernel: int                       # kernel index on the board
    coord: tuple
    selector: List[int]               # board-local measurement line per select-index (len = n_used)
    n_used: int                       # = len(support)
    h: int                            # cores on this kernel
    init_counter: int
    support: List[int]                # qubit per select-index (support[i] = qubit of select bit i)
    detectors: List[int]              # this channel's detectors in new_t order (emit order)
    # core -> {meas_time: (select_indices, emit)} ; select_indices are 0..n_used-1
    core_steps: Dict[int, Dict[int, Tuple[List[int], bool]]]


@dataclass
class BoardProgram:
    board_id: int
    input_qubits: List[int]                       # local channel index = position
    qubit_local: Dict[int, int]
    sync_masks: List[List[int]]                   # per meas-time: local indices that fire
    channels: List[ChannelInstance]
    kernels: List[KernelProgram] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)

    @property
    def m(self) -> int:
        return len(self.input_qubits)


@dataclass
class DetectorOutput:
    """Detector-time synchronization + output for a stage board (and the root).
    Output lines are the channels constructed in this board's subtree (they flow
    up through it); one line = one spatial coordinate."""
    board_id: int
    out_lines: List[tuple]                              # line index -> channel coord
    line_of_coord: Dict[tuple, int]
    sync_det_mask: Dict[int, List[int]]                 # new_t -> active line indices
    global_det_index: Optional[Dict[int, Dict[int, int]]]  # root: new_t -> {line: stim det_idx}
    postselect_mask: Dict[int, Dict[int, List[int]]]    # stage_idx -> {new_t -> postselect lines}


class Compiler:
    def __init__(self, config: CompilerConfig, scanner):
        self.cfg = config
        self.s = scanner
        self.pl = Placement(config, scanner)
        self.programs: Dict[int, BoardProgram] = {}
        self.stage_boards: Dict[int, List[int]] = {}   # board_id -> stage indices it hosts
        self.outputs: Dict[int, DetectorOutput] = {}
        self._out_lines_cache: Dict[int, list] = {}
        self._build()
        self._build_kernel_programs()
        self._build_outputs()

    # ------------------------------------------------------------------ #
    def _board_input_qubits(self, bid: int) -> List[int]:
        """Measured qubits entering a board. Leaf: its own scope (sorted). Router/
        root: each child's raw outputs concatenated in child order, so a child's
        measurements occupy a contiguous block of the parent's input ports."""
        b = self.cfg.boards[bid]
        if b.qubits is not None:               # leaf or monolithic root: its own scope
            return sorted(self.pl.leaf_scope[bid])
        order = []
        for cid in b.children:
            order += sorted(self.pl.forward_normal[cid])       # child raw-output order
        return order

    def out_lines_of(self, bid: int) -> List[tuple]:
        """Detector output ports of a board: children's outputs concatenated (the
        pass block), then local channels in kernel order (the construct block).
        Never coord-sorted, so a child's detector outputs occupy a contiguous block
        of the parent's detector-input ports."""
        if bid not in self._out_lines_cache:
            b = self.cfg.boards[bid]
            lines = []
            for cid in b.children:                             # pass part (empty for leaf/monolithic)
                lines += self.out_lines_of(cid)
            lines += [kp.coord for kp in sorted(self.programs[bid].kernels,
                                                key=lambda k: k.kernel)]   # construct part
            self._out_lines_cache[bid] = lines
        return self._out_lines_cache[bid]

    def _required_h(self, dets: List[int]) -> int:
        """Minimal h for the mod-h core scheme, where a detector's core is its
        LOCAL time index — its emission rank (position in new_t order) mod h.
        This is what the per-emit circular counter produces; absolute new_t need
        not be contiguous (channels skip rounds), because the detector sync mask
        realigns to absolute new_t downstream. Also verifies the strict
        completion order the counter relies on."""
        win = [self.s.detector_window[d] for d in dets]      # dets already sorted by new_t
        comps = [w[1] for w in win]
        for i in range(1, len(comps)):
            if comps[i] <= comps[i - 1]:
                coord = self.s.detector_coord[dets[0]][0]
                raise ValueError(
                    f"spatial coord {coord}: detectors {dets[i-1]} and {dets[i]} both "
                    f"complete at measurement time {comps[i]}. A channel builds one "
                    f"detector per step, so detectors finishing in the same step must "
                    f"have distinct spatial coordinates (separate channels). The "
                    f"circuit's detector annotation does not satisfy this.")
        for h in range(1, len(dets) + 1):
            buckets: Dict[int, List[Tuple[int, int]]] = {}
            for k, w in enumerate(win):
                buckets.setdefault(k % h, []).append(w)       # k = local time index (emit rank)
            ok = True
            for ivs in buckets.values():
                ivs.sort()
                for a, b in zip(ivs, ivs[1:]):
                    if b[0] <= a[1]:            # overlap on a shared core
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return h
        return len(dets)

    def _assign_kernels(self, bid: int, channels: List[ChannelInstance]) -> List[str]:
        """Best-fit bin-pack channels onto the board's kernels; collect problems."""
        kernels = self.cfg.boards[bid].kernels
        problems: List[str] = []
        if len(channels) > len(kernels):
            problems.append(f"needs {len(channels)} kernels, has {len(kernels)}")
        used = [False] * len(kernels)
        for c in sorted(channels, key=lambda c: (len(c.support), c.required_h), reverse=True):
            best = None
            for ki, k in enumerate(kernels):
                if used[ki] or k.n < len(c.support) or k.h < c.required_h:
                    continue
                if best is None or (k.n, k.h) < (kernels[best].n, kernels[best].h):
                    best = ki
            if best is None:
                problems.append(f"channel {c.coord} (n={len(c.support)}, h={c.required_h}) "
                                f"fits no free kernel")
            else:
                used[best] = True
                c.kernel = best
                c.init_counter = 0          # emission-index cores => counter starts at 0
        return problems

    def _build(self) -> None:
        s, cfg, pl = self.s, self.cfg, self.pl

        # detectors placed at each board, grouped by spatial coord
        from collections import defaultdict
        by_board: Dict[int, Dict[tuple, List[int]]] = defaultdict(lambda: defaultdict(list))
        for d in range(len(s.detector_coord)):
            by_board[pl.placement[d]][s.detector_coord[d][0]].append(d)

        for bid, b in cfg.boards.items():
            input_qubits = self._board_input_qubits(bid)
            qubit_local = {q: i for i, q in enumerate(input_qubits)}

            # measurement sync masks: per meas-time, local channels that fire
            sync_masks = []
            for layer in s.measurement_rec:
                fired = sorted(qubit_local[q] for q, _t in layer if q in qubit_local)
                sync_masks.append(fired)

            # channel instances placed at this board
            channels: List[ChannelInstance] = []
            problems: List[str] = []
            for coord, dets in by_board.get(bid, {}).items():
                dets = sorted(dets, key=lambda d: s.detector_coord[d][2])
                support = sorted({q for d in dets for q in pl._det_qubits(d)})
                # sanity: every qubit of a placed detector must reach this board
                missing = [q for q in support if q not in qubit_local]
                assert not missing, f"board {bid} channel {coord} missing qubits {missing}"
                try:
                    rh = self._required_h(dets)
                except ValueError as e:
                    problems.append(str(e))            # ill-formed annotation; skip channel
                    continue
                channels.append(ChannelInstance(bid, coord, dets, support, rh, init_counter=0))

            problems += self._assign_kernels(bid, channels)
            self.programs[bid] = BoardProgram(bid, input_qubits, qubit_local,
                                              sync_masks, channels, problems=problems)

    # ------------------------------------------------------------------ #
    def _kernel_program(self, c: ChannelInstance, prog: BoardProgram) -> KernelProgram:
        """Turn one assigned channel into its selector + per-core select/emit masks.
        Core of the k-th detector (new_t order) = k mod h (emission-index)."""
        h = self.cfg.boards[c.board_id].kernels[c.kernel].h
        sel_index = {q: i for i, q in enumerate(c.support)}        # qubit -> select bit (0..n-1)
        selector = [prog.qubit_local[q] for q in c.support]        # board line per select bit
        steps: Dict[int, Dict[int, Tuple[set, bool]]] = {core: {} for core in range(h)}
        for k, d in enumerate(c.detectors):                        # emission order == new_t order
            core = k % h
            nt = self.s.detector_coord[d][2]
            adj = self.s.detector_to_measurement[nt][d]
            hi = self.s.detector_window[d][1]
            for tau, qs in adj.items():
                sel, _emit = steps[core].setdefault(tau, (set(), False))
                for q in qs:
                    sel.add(sel_index[q])
            sel, _emit = steps[core].setdefault(hi, (set(), False))
            steps[core][hi] = (sel, True)                          # emit = send + clear at completion
        core_steps = {core: {tau: (sorted(sel), emit) for tau, (sel, emit) in sorted(m.items())}
                      for core, m in steps.items()}
        return KernelProgram(c.board_id, c.kernel, c.coord, selector, len(c.support),
                             h, c.init_counter, list(c.support), list(c.detectors), core_steps)

    def _build_kernel_programs(self) -> None:
        for bid, prog in self.programs.items():
            for c in prog.channels:
                if c.kernel is not None:
                    prog.kernels.append(self._kernel_program(c, prog))

    def dump_kernel(self, board_id: int, coord: tuple) -> str:
        prog = self.programs[board_id]
        kp = next(k for k in prog.kernels if k.coord == coord)
        lines = [f"board {board_id} kernel {kp.kernel} channel {coord}: "
                 f"n={kp.n_used} h={kp.h} init_counter={kp.init_counter}",
                 f"  selector (select-bit -> board line): {kp.selector}"]
        for core in range(kp.h):
            steps = kp.core_steps[core]
            if not steps:
                continue
            lines.append(f"  core {core}:")
            for tau, (sel, emit) in steps.items():
                lines.append(f"    meas_t={tau}: select={sel}{'  EMIT' if emit else ''}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def _build_outputs(self) -> None:
        """Detector-time sync + postselect masks for stage boards, plus the root's
        decoder-facing output (valid mask + global detector index + values)."""
        s, cfg, pl = self.s, self.cfg, self.pl

        bad = sorted(d for d in cfg.postselect_detectors
                     if not (0 <= d < len(s.detector_coord)))
        if bad:
            raise ValueError(
                f"config postselect detectors out of range "
                f"[0, {len(s.detector_coord)}): {bad[:8]}{'...' if len(bad) > 8 else ''} "
                f"— the postselect list likely belongs to a different circuit.")
        is_ps = [d in cfg.postselect_detectors
                 for d in range(len(s.detector_coord))]
        det_at: Dict[tuple, Dict[int, int]] = {}          # coord -> {new_t: det_idx}
        for d in range(len(s.detector_coord)):
            det_at.setdefault(s.detector_coord[d][0], {})[s.detector_coord[d][2]] = d

        # which board hosts each stage's postselect block (= LCA of its detectors)
        for si, bid in enumerate(pl.stage_board):
            self.stage_boards.setdefault(bid, []).append(si)

        for bid in set(self.stage_boards) | {cfg.root_id}:
            # output lines in hardware order: pass (children) ++ construct (kernels)
            coords = self.out_lines_of(bid)
            line_of = {c: i for i, c in enumerate(coords)}

            sync: Dict[int, List[int]] = {}
            gidx: Optional[Dict[int, Dict[int, int]]] = {} if bid == cfg.root_id else None
            for c in coords:
                for nt, d in det_at[c].items():
                    sync.setdefault(nt, []).append(line_of[c])
                    if gidx is not None:
                        gidx.setdefault(nt, {})[line_of[c]] = d
            for nt in sync:
                sync[nt].sort()

            ps_mask: Dict[int, Dict[int, List[int]]] = {}
            for si in self.stage_boards.get(bid, []):
                m: Dict[int, List[int]] = {}
                for d in s.stages[si]["detectors"]:
                    if is_ps[d]:
                        m.setdefault(s.detector_coord[d][2], []).append(
                            line_of[s.detector_coord[d][0]])
                for nt in m:
                    m[nt].sort()
                ps_mask[si] = m

            self.outputs[bid] = DetectorOutput(bid, coords, line_of, sync, gidx, ps_mask)

    def dump_output(self, board_id: int, max_rows: int = 4) -> str:
        o = self.outputs[board_id]
        lines = [f"board {board_id} detector output: {len(o.out_lines)} lines, "
                 f"{len(o.sync_det_mask)} detector-time layers"]
        for nt in sorted(o.sync_det_mask)[:max_rows]:
            active = o.sync_det_mask[nt]
            extra = ""
            if o.global_det_index is not None:
                gi = o.global_det_index[nt]
                extra = "  global_idx=" + str({ln: gi[ln] for ln in active[:6]})
            lines.append(f"  new_t={nt}: {len(active)} valid lines{extra}")
        for si, m in o.postselect_mask.items():
            total = sum(len(v) for v in m.values())
            lines.append(f"  postselect stage {si}: {total} postselect detector-slots "
                         f"over {len(m)} detector-times")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    def report(self) -> str:
        cfg = self.cfg
        lines = [f"compiler '{cfg.name}': {len(self.s.detector_coord)} detectors"]
        feasible = True
        for b in sorted(cfg.boards.values(), key=lambda x: (x.level, x.board_id)):
            p = self.programs[b.board_id]
            nk = len(b.kernels)
            maxn = max((len(c.support) for c in p.channels), default=0)
            maxh = max((c.required_h for c in p.channels), default=0)
            act = sum(1 for layer in p.sync_masks if layer)
            used_k = sum(1 for c in p.channels if c.kernel is not None)
            lines.append(f"  L{b.level} {b.type:6s} id={b.board_id}: m={p.m:4d} "
                         f"sync_layers={act:3d} channels={len(p.channels):3d} "
                         f"kernels_used={used_k}/{nk} max_n={maxn} max_h={maxh}")
            if b.board_id in self.outputs:
                o = self.outputs[b.board_id]
                role = "root" if b.board_id == self.cfg.root_id else ""
                stg = f" postselect_stages={list(o.postselect_mask)}" if o.postselect_mask else ""
                lines.append(f"        output: {len(o.out_lines)} lines, "
                             f"{len(o.sync_det_mask)} det-time layers {role}{stg}")
            for prob in p.problems[:4]:
                feasible = False
                lines.append(f"        !! {prob}")
            if len(p.problems) > 4:
                feasible = False
                lines.append(f"        !! ... and {len(p.problems) - 4} more")
        lines.append(f"feasible: {feasible}")
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from parser import DetectorConstructionScanner
    from placement import make_partition_config

    circ = sys.argv[1] if len(sys.argv) > 1 else (
        "../../../magic_state_cultivation/circuits_dump/end2end_d1=3_d2=9_r1=3_r2=3_b=Y.stim")
    n_leaves = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    s = DetectorConstructionScanner.from_file(circ)
    s.scan(); s.refine()
    cfg = make_partition_config(s, n_leaves)
    comp = Compiler(cfg, s)
    print(comp.report())
