"""serializer.py — write a compiled program to per-board, per-regfile files + a report.

PHYSICAL-PORT layout: regfiles are sized to the board's fixed hardware capacity
(all `k` kernels, `raw_out` forward wires, and the full children output width),
NOT the utilized subset. Idle ports carry no valid data:
  - idle kernels get an empty selector + all-zero core (no detectors),
  - idle raw-output wires select a sentinel input line,
  - idle detector-output ports have coord = null and are never valid.
Utilization ("perfect fit") is still summarized in report.txt / manifest.json.

Layout:
  results/<config-name>__<circuit-key>/
    <circuit>.stim, <config>.json, measurement_map.json, manifest.json, report.txt
    board<id>/ json/<regfile>.json   mem/<regfile>.mem   (hex|bin words, LSB=bit 0)

Per-board regfiles:
  qubit_scope   (leaf)                    used input-port -> qubit + unused (dropped) qubits
  sync          (boards with kernels)     T x m_phys
  k<i>_selector (all k kernels)           n x ceil(log2 m_phys)   (one index/word; idle -> sentinel)
  k<i>_core     (all k kernels)           T x h*(n+1)             (idle -> all zero)
  raw_selector[_obs]  (boards forwarding) raw_out x ceil(log2 m_phys)
  output_sync   (stage boards + root)     ndt x #phys_lines
  global_index  (root)                    ndt x #phys_lines*ceil(log2(ndet+1))  (one word/det-time)
  postselect    (stage boards)            flat: ndt x #phys_lines | per_stage: (#stages*ndt) x #phys_lines
  connections   (non-leaf; JSON only)     input port -> (child, child output port)
"""

from __future__ import annotations

import json
import os
import re
import shutil
from functools import lru_cache

from compiler import Compiler


def _bits(n: int) -> int:
    return max(1, (max(1, n) - 1).bit_length())


def _mask(indices) -> int:
    v = 0
    for i in indices:
        v |= 1 << i
    return v


def _word(v: int, width: int, fmt: str) -> str:
    return format(v, f"0{width}b") if fmt == "bin" else format(v, f"0{-(-width // 4)}x")


def _example_name(circuit_path: str, config_name: str) -> str:
    base = re.sub(r"\.stim$", "", os.path.basename(circuit_path))
    kv = dict(re.findall(r"([A-Za-z0-9_]+)=([^,]+)", base))
    tag = ",".join(f"{k}={kv[k]}" for k in ("d1", "d2", "b", "p") if k in kv)
    return f"{config_name}__{tag or base}"


class Serializer:
    def __init__(self, comp: Compiler):
        self.comp = comp
        self.s = comp.s
        self.cfg = comp.cfg
        self.pl = comp.pl
        self.T = len(self.s.measurement_rec)
        self.ndet = len(self.s.detector_coord)
        self.ndt = max((self.s.detector_coord[d][2] for d in range(self.ndet)), default=-1) + 1
        # coord -> {new_t: det_idx}
        self.det_at = {}
        for d in range(self.ndet):
            self.det_at.setdefault(self.s.detector_coord[d][0], {})[self.s.detector_coord[d][2]] = d

    # ---- physical port layouts (None = idle port) ---- #
    @lru_cache(maxsize=None)
    def _raw_ports(self, bid):
        """Qubits on the board's raw_out forward wires (len = raw_out; None padding)."""
        used = sorted(self.pl.forward_normal[bid])
        ro = self.cfg.boards[bid].raw_out or len(used)
        return tuple(used + [None] * (ro - len(used)))

    @lru_cache(maxsize=None)
    def _in_qubits(self, bid):
        """Physical measurement input ports. Leaf: its used qubits (scope-filtered).
        Router/root: children's raw_out wires concatenated (idle wires -> None)."""
        b = self.cfg.boards[bid]
        if b.qubits is not None:                 # leaf or monolithic root: its used qubits
            return tuple(sorted(self.pl.leaf_scope[bid]))
        order = []
        for c in b.children:
            order += self._raw_ports(c)
        return tuple(order)

    @lru_cache(maxsize=None)
    def _in_local(self, bid):
        return {q: i for i, q in enumerate(self._in_qubits(bid)) if q is not None}

    @lru_cache(maxsize=None)
    def _out_lines(self, bid):
        """Physical detector output ports: children's outputs (pass) ++ all k kernels
        (construct) at their physical index; idle kernel ports are None."""
        b = self.cfg.boards[bid]
        lines = []
        if b.type != "leaf":
            for c in b.children:
                lines += self._out_lines(c)
        kc = {kp.kernel: kp.coord for kp in self.comp.programs[bid].kernels}
        lines += [kc.get(i) for i in range(len(b.kernels))]
        return tuple(lines)

    def _kp(self, bid, ki):
        for kp in self.comp.programs[bid].kernels:
            if kp.kernel == ki:
                return kp
        return None

    # ---- regfile builders: return (json, mem|None) ---- #
    def _qubit_scope(self, bid, fmt):
        used = list(self._in_qubits(bid))
        unused = sorted(set(self.cfg.boards[bid].qubits) - set(used))
        return {"regfile": "qubit_scope",
                "note": "leaf scope filtering: 'used' = input port -> physical qubit; "
                        "'unused' = declared but not measured into any detector (unconnected)",
                "used": used, "unused": unused}, None

    def _sync(self, bid, fmt):
        inq = self._in_qubits(bid); loc = self._in_local(bid); m = len(inq)
        entries = []
        for t in range(self.T):
            fired = sorted(loc[q] for q, _ in self.s.measurement_rec[t] if q in loc)
            entries.append(fired)
        if self.wait_row == "copy-last" and entries:
            entries.append(list(entries[-1]))     # saturating wait meas-time = copy of the last
        j = {"regfile": "sync", "word_width": m, "depth": len(entries),
             "note": "bit i = physical input port i measured this meas_time (LSB=port 0); "
                     "last row is the copy-last saturating wait round when enabled",
             "channel_qubit": list(inq), "entries": entries}
        return j, [_word(_mask(e), m, fmt) for e in entries]

    def _kernel_selector(self, bid, ki, fmt):
        loc = self._in_local(bid); m = len(self._in_qubits(bid)); w = _bits(m)
        n = self.cfg.boards[bid].kernels[ki].n
        kp = self._kp(bid, ki)
        if kp is None:                                   # idle kernel
            lines = [m] * n                              # sentinel = out-of-range input line
            j = {"regfile": "selector", "kernel": ki, "idle": True, "n_used": 0,
                 "word_width": w, "lines": lines}
        else:
            lines = [loc[q] for q in kp.support] + [m] * (n - kp.n_used)
            j = {"regfile": "selector", "kernel": ki, "idle": False, "n_used": kp.n_used,
                 "coord": list(kp.coord), "word_width": w,
                 "note": "select-bit i -> physical input port (one index/word)", "lines": lines}
        return j, [_word(ln, w, fmt) for ln in lines]

    def _kernel_core(self, bid, ki, fmt):
        kcfg = self.cfg.boards[bid].kernels[ki]
        n, h = kcfg.n, kcfg.h
        cw = n + 1; w = h * cw
        kp = self._kp(bid, ki)
        # copy-last appends one saturating wait meas-time (a copy of meas-time T-1). The
        # JSON `steps` stay length T on purpose, so validate_serialized (range(T)) ignores
        # the wait row — its detector value is a placeholder (see module notes).
        n_rows = self.T + (1 if self.wait_row == "copy-last" else 0)
        if kp is None:                                   # idle kernel: all-zero
            j = {"regfile": "core", "kernel": ki, "idle": True, "h": h, "core_word_bits": cw,
                 "word_width": w, "detectors": [], "steps": {}}
            return j, [_word(0, w, fmt) for _ in range(n_rows)]
        steps = {c: {t: {"select": sel, "emit": e} for t, (sel, e) in kp.core_steps[c].items()}
                 for c in range(kp.h)}
        j = {"regfile": "core", "kernel": ki, "idle": False, "coord": list(kp.coord),
             "n": kp.n_used, "h": h, "core_word_bits": cw, "word_width": w,
             "note": "per meas_time: h cores packed (core c at bit c*(n+1)); within a core, "
                     "bits 0..n-1 select, bit n = emit", "detectors": kp.detectors, "steps": steps}
        mem = []
        for t in range(self.T):
            word = 0
            for c in range(kp.h):
                sel, e = kp.core_steps[c].get(t, ([], False))
                word |= (_mask(sel) | (int(e) << n)) << (c * cw)
            mem.append(_word(word, w, fmt))
        if self.wait_row == "copy-last" and mem:
            mem.append(mem[-1])                          # saturating wait row = copy of meas-time T-1
        return j, mem

    def _raw_selector(self, bid, fmt):
        loc = self._in_local(bid); m = len(self._in_qubits(bid)); w = _bits(m)
        ports = self._raw_ports(bid)
        lines = [loc[q] if q is not None else m for q in ports]     # idle wire -> sentinel m
        j = {"regfile": "raw_selector", "word_width": w,
             "note": "raw-output wire k -> physical input port (sentinel = idle wire)",
             "qubits": list(ports), "lines": lines}
        return j, [_word(ln, w, fmt) for ln in lines]

    def _raw_selector_obs(self, bid, fmt):
        order = sorted(self.pl.forward_obs[bid])
        return {"regfile": "raw_selector_obs", "raw_out_needed": len(order),
                "note": "additional raw-out requirement to also forward observable measurements "
                        "(qubit list; not a functional regfile in the normal deployment)",
                "qubits": order}, None

    def _output_sync(self, bid, fmt):
        lines = self._out_lines(bid); w = len(lines)
        entries = []
        for nt in range(self.ndt):
            entries.append([i for i, c in enumerate(lines)
                            if c is not None and nt in self.det_at.get(c, {})])
        if self.wait_row == "copy-last" and entries:
            entries.append(list(entries[-1]))     # saturating wait det-time = copy of the last
        j = {"regfile": "output_sync", "word_width": w, "depth": len(entries),
             "note": "bit i = output port i carries a detector at this det-time (idle ports never set); "
                     "last row is the copy-last saturating wait round when enabled",
             "line_coord": [list(c) if c is not None else None for c in lines], "entries": entries}
        return j, [_word(_mask(e), w, fmt) for e in entries]

    def _gidx_sizing(self, bid):
        """Compact global-index sizing. Regfile holds base indexes 0..ndet-1 (plus
        the copy-last row's ndet..ndet+stride-1) and an ALL-ONES sentinel that must
        stay strictly above every valid base index. stride = detectors per wait
        round = used lines at the last det-time (what the saturating repeat emits).
        Returns (idx_w, stride, sentinel)."""
        lines = self._out_lines(bid)
        last_dt = self.ndt - 1
        stride = sum(1 for c in lines if c is not None and last_dt in self.det_at.get(c, {}))
        copy = (self.wait_row == "copy-last")
        max_base = self.ndet - 1 + (stride if copy else 0)   # copy row adds ndet..ndet+stride-1
        w = (max_base + 1).bit_length()
        if (1 << w) - 1 <= max_base:      # +1 bit if the max base is itself all-ones
            w += 1
        return w, stride, (1 << w) - 1    # idx_w, stride, all-ones sentinel

    def _global_index(self, bid, fmt):
        lines = self._out_lines(bid); nl = len(lines)
        w, stride, sentinel = self._gidx_sizing(bid)
        ww = nl * w                       # one word per det-time: all lines' indices packed
        rows, mem = [], []
        for nt in range(self.ndt):
            r = [self.det_at.get(c, {}).get(nt, sentinel) if c is not None else sentinel
                 for c in lines]
            rows.append(r)
        if self.wait_row == "copy-last":
            # saturating wait row: fresh base indexes ndet.. on the last det-time's used
            # lines (same lines as the copied output_sync row), sentinel elsewhere.
            last_used = [i for i, c in enumerate(lines)
                         if c is not None and (self.ndt - 1) in self.det_at.get(c, {})]
            wait = [sentinel] * nl
            for k, ln in enumerate(last_used):
                wait[ln] = self.ndet + k
            rows.append(wait)
        for r in rows:
            packed = 0
            for i, v in enumerate(r):
                packed |= v << (i * w)    # line i's index at bits [i*w +: w] (LSB = line 0)
            mem.append(_word(packed, ww, fmt))
        j = {"regfile": "global_index", "index_width": w, "word_width": ww,
             "lines": nl, "sentinel": sentinel, "stride": stride,
             "note": "one word per det-time; line i's stim detector index at bits "
                     "[i*index_width +: index_width] (sentinel = all-ones = idle/unused slot). "
                     "Copy-last wait row (appended separately) uses base ndet..ndet+stride-1.",
             "rows": rows}
        return j, mem

    def _round_marker(self, bid, fmt):
        """Root-only per-det-time marker: {is_first_normal, is_last_normal, is_wait}.
        The last `wait_rounds` real det-times are wait rounds; a copy-last row (if any)
        is also a wait round. Depth matches output_sync/global_index (ndt [+1])."""
        W = self.wait_rounds
        extra = 1 if (self.wait_row == "copy-last") else 0
        depth = self.ndt + extra
        last_normal = self.ndt - 1 - W          # last non-wait det-time
        rows, mem = [], []
        for nt in range(depth):
            is_first = 1 if nt == 0 else 0
            is_last  = 1 if nt == last_normal else 0
            is_wait  = 1 if nt > last_normal else 0
            rows.append({"is_first_normal": is_first, "is_last_normal": is_last, "is_wait": is_wait})
            mem.append(_word(is_first | (is_last << 1) | (is_wait << 2), 3, fmt))
        j = {"regfile": "round_marker", "word_width": 3, "depth": depth,
             "wait_rounds": W, "wait_row": self.wait_row,
             "note": "bit0=is_first_normal, bit1=is_last_normal, bit2=is_wait "
                     "(last wait_rounds real det-times + a copy-last row are wait)",
             "rows": rows}
        return j, mem

    def _postselect(self, bid, fmt):
        lines = self._out_lines(bid); w = len(lines)
        line_of = {c: i for i, c in enumerate(lines) if c is not None}
        ps = self.cfg.postselect_detectors
        # per-stage, per-det-time postselect output lines
        stage_e = {}
        for si in self.comp.stage_boards.get(bid, []):
            e = [[] for _ in range(self.ndt)]
            for d in self.s.stages[si]["detectors"]:
                if d in ps:
                    e[self.s.detector_coord[d][2]].append(line_of[self.s.detector_coord[d][0]])
            for x in e:
                x.sort()
            stage_e[si] = e

        # copy-last appends a ZERO postselect row: no postselect during the hold/wait phase.
        wait = 1 if self.wait_row == "copy-last" else 0

        if self.ps_layout == "per_stage":
            # separate ndt-block per stage: (#stages * ndt) words, stage blocks concatenated
            mem = []
            for si in self.comp.stage_boards.get(bid, []):
                mem += [_word(_mask(x), w, fmt) for x in stage_e[si]]
                mem += [_word(0, w, fmt)] * wait     # zero wait row per stage block
            j = {"regfile": "postselect", "layout": "per_stage", "word_width": w,
                 "depth_per_stage": self.ndt + wait, "stages": stage_e,
                 "note": "per stage: ndt mask words (stage blocks concatenated); "
                         "bit = output line carrying that stage's postselect detector at det-time; "
                         "copy-last appends a zero wait row per block"}
            return j, mem

        # flat (default): one word per det-time = OR of every stage's lines at that det-time.
        # Lossless: a board's stages have disjoint det-time ranges, so no det-time merges two.
        flat = [sorted({ln for si in stage_e for ln in stage_e[si][nt]})
                for nt in range(self.ndt)]
        flat += [[]] * wait                          # zero wait row (no postselect during hold)
        mem = [_word(_mask(x), w, fmt) for x in flat]
        j = {"regfile": "postselect", "layout": "flat", "word_width": w, "depth": len(flat),
             "entries": flat, "stages": stage_e,
             "note": "one word per det-time: OR over the board's stages of their postselect "
                     "output lines at that det-time; copy-last appends a zero wait row"}
        return j, mem

    def _connections(self, bid, fmt):
        children = self.cfg.boards[bid].children
        meas = []
        for c in children:
            for jc, q in enumerate(self._raw_ports(c)):
                meas.append({"port": len(meas), "child": c, "child_port": jc, "qubit": q})
        assert [d["qubit"] for d in meas] == list(self._in_qubits(bid)), \
            f"board {bid} measurement-input order != children raw-output concatenation"
        det = []
        for c in children:
            for jc, x in enumerate(self._out_lines(c)):
                det.append({"port": len(det), "coord": list(x) if x is not None else None,
                            "child": c, "child_port": jc})
        return {"board": bid, "children": list(children),
                "measurement_ports": meas, "detector_ports": det}, None

    # ---- per-board params (manifest + report) ---- #
    def _board_params(self, bid):
        p = self.comp.programs[bid]; b = self.cfg.boards[bid]; kn = b.kernels
        d = {"board_id": bid, "type": b.type, "level": b.level,
             "m_phys": len(self._in_qubits(bid)),
             "sync_active": sum(1 for e in self._sync(bid, "hex")[1] if int(e, 16))
             if p.kernels else 0,
             "kernels_used": len(p.kernels), "kernels_avail": len(kn),
             "n_used_max": max((len(c.support) for c in p.channels), default=0),
             "n_cap": max((k.n for k in kn), default=0),
             "h_used_max": max((c.required_h for c in p.channels), default=0),
             "h_cap": max((k.h for k in kn), default=0),
             "raw_out_cap": b.raw_out, "forward_normal": len(self.pl.forward_normal[bid]),
             "forward_obs": len(self.pl.forward_obs[bid])}
        if bid in self.comp.outputs:
            d["output_ports"] = len(self._out_lines(bid))
            if self.comp.outputs[bid].global_det_index is not None:
                w, stride, sentinel = self._gidx_sizing(bid)
                d["global_index_width"] = w                                  # bits per index (regfile)
                d["global_index_word_width"] = len(self._out_lines(bid)) * w  # bits per det-time
                d["global_index_stride"] = stride                            # detectors per wait round
                d["global_index_sentinel"] = sentinel                        # all-ones = unused slot
            if bid in self.comp.stage_boards:
                d["postselect_stages"] = self.comp.stage_boards[bid]
                psd = [det for si in self.comp.stage_boards[bid]
                       for det in self.s.stages[si]["detectors"]
                       if det in self.cfg.postselect_detectors]
                d["postselect_detectors"] = sorted(psd)   # stim indices this board postselects
        if b.qubits is None:                     # non-leaf boards report their input-port widths
            d["measurement_input_ports"] = len(self._in_qubits(bid))
            d["detector_input_ports"] = sum(len(self._out_lines(c)) for c in b.children)
        return d

    def _report(self, params, feasible):
        L = ["=" * 64, f"Compilation report — feasible: {feasible}   (PHYSICAL-port layout)",
             f"  detectors (ndet) : {self.ndet}", f"  meas-times (T)   : {self.T}",
             f"  det-times (ndt)  : {self.ndt}",
             f"  global_index: {self._gidx_sizing(self.cfg.root_id)[0]} bits/index, one word/det-time "
             f"(line i at bit i*width; all-ones sentinel; "
             f"stride={self._gidx_sizing(self.cfg.root_id)[1]})",
             "-" * 64, "Per-board utilization + derived parameters:"]
        for d in params:
            L.append(f"  board {d['board_id']:<5} {d['type']:6} L{d['level']}")
            L.append(f"    measurement m_phys   : {d['m_phys']:4}")
            if d["kernels_avail"]:
                L.append(f"    kernels used/avail   : {d['kernels_used']}/{d['kernels_avail']}"
                         f"   (idle kernels emitted empty)")
                L.append(f"    selector n used/cap  : {d['n_used_max']}/{d['n_cap']}"
                         f"   (index width {_bits(d['m_phys'])} bits)")
                L.append(f"    core     h used/cap  : {d['h_used_max']}/{d['h_cap']}")
            if d["raw_out_cap"] is not None:
                L.append(f"    raw fwd normal/+obs/cap: {d['forward_normal']}/{d['forward_obs']}/"
                         f"{d['raw_out_cap']}")
            if "output_ports" in d:
                L.append(f"    detector out ports   : {d['output_ports']}")
            if "measurement_input_ports" in d:
                L.append(f"    input ports meas/det : {d['measurement_input_ports']}/"
                         f"{d['detector_input_ports']}")
        L += ["-" * 64, "Stage boards (postselect):"]
        for si, bid in enumerate(self.pl.stage_board):
            n = sum(len(v) for v in self.comp.outputs[bid].postselect_mask.get(si, {}).values())
            L.append(f"  stage {si} -> board {bid} : {n} postselect detector-slots")
        L.append("=" * 64)
        return "\n".join(L)

    # ---- driver ---- #
    def _board_regfiles(self, bid):
        p = self.comp.programs[bid]; b = self.cfg.boards[bid]; rf = []
        if b.qubits is not None:                 # leaf or monolithic root
            rf.append(("qubit_scope", lambda fmt: self._qubit_scope(bid, fmt)))
        # MAXIMUM-way emission: emit every regfile the board's fixed HARDWARE provisions, not
        # only the placed/used ones. An idle module gets a valid zero/sentinel regfile (the
        # builders already handle this: idle kernel -> sentinel selector + all-zero core, idle
        # raw wire -> sentinel). So every regfile a real machine would power-on-initialize is
        # present on disk, and every consumer (emulator loader, RTL tb, hardware) loads the
        # complete set without special-casing used vs. unused.
        if b.kernels:                            # all provisioned kernels (+ their sync), idle -> zero
            rf.append(("sync", lambda fmt: self._sync(bid, fmt)))
            for ki in range(len(b.kernels)):
                rf.append((f"k{ki}_selector", lambda fmt, ki=ki: self._kernel_selector(bid, ki, fmt)))
                rf.append((f"k{ki}_core", lambda fmt, ki=ki: self._kernel_core(bid, ki, fmt)))
        if b.raw_out:                            # any provisioned raw-forward path, idle -> sentinel
            rf.append(("raw_selector", lambda fmt: self._raw_selector(bid, fmt)))
        if self.pl.forward_obs[bid]:
            rf.append(("raw_selector_obs", lambda fmt: self._raw_selector_obs(bid, fmt)))
        # output_sync: 'all' (default) emits it on EVERY board so OutputSync is the
        # output stage everywhere (single finish wire per link); 'output-only' keeps
        # the legacy behavior (root/stage boards only). global_index + postselect
        # stay root/stage-only regardless.
        if self.output_sync_all or bid in self.comp.outputs:
            rf.append(("output_sync", lambda fmt: self._output_sync(bid, fmt)))
        if bid in self.comp.outputs:
            o = self.comp.outputs[bid]
            if o.global_det_index is not None:
                rf.append(("global_index", lambda fmt: self._global_index(bid, fmt)))
                rf.append(("round_marker", lambda fmt: self._round_marker(bid, fmt)))
            if bid in self.comp.stage_boards:
                rf.append(("postselect", lambda fmt: self._postselect(bid, fmt)))
        if b.children:                           # only boards with children have wiring
            rf.append(("connections", lambda fmt: self._connections(bid, fmt)))
        return rf

    def write(self, out_root, circuit_path, config_path, mem_fmt="hex", postselect_layout="flat",
              output_sync_all=True, wait_rounds=0, wait_row="normal", gidx_hw_width=24):
        self.ps_layout = postselect_layout
        self.output_sync_all = output_sync_all
        self.wait_rounds = wait_rounds      # last W real det-times are wait rounds (= r2)
        self.wait_row = wait_row            # 'normal' | 'copy-last' (append a saturating copy)
        self.gidx_hw_width = gidx_hw_width  # hardware global-index datapath width (>= regfile idx_w)
        # folder name records the compiler flags so distinct settings don't collide
        flags = (f"osync={'all' if output_sync_all else 'output-only'},"
                 f"wr={wait_rounds},wrow={wait_row},ps={postselect_layout}")
        root = os.path.join(out_root, _example_name(circuit_path, self.cfg.name) + "__" + flags)
        os.makedirs(root, exist_ok=True)
        shutil.copy(circuit_path, os.path.join(root, os.path.basename(circuit_path)))
        shutil.copy(config_path, os.path.join(root, os.path.basename(config_path)))
        feasible = all(not self.comp.programs[b].problems for b in self.cfg.boards)
        with open(os.path.join(root, "measurement_map.json"), "w") as f:
            json.dump([[q, t] for q, t in self.s._measurement_rec_raw_flat], f)

        params = []
        for b in sorted(self.cfg.boards.values(), key=lambda x: (x.level, x.board_id)):
            bid = b.board_id
            jdir = os.path.join(root, f"board{bid}", "json"); mdir = os.path.join(root, f"board{bid}", "mem")
            os.makedirs(jdir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
            names = []
            for name, builder in self._board_regfiles(bid):
                jobj, mem = builder(mem_fmt)
                with open(os.path.join(jdir, f"{name}.json"), "w") as f:
                    json.dump(jobj, f, indent=1)
                if mem is not None:
                    with open(os.path.join(mdir, f"{name}.mem"), "w") as f:
                        f.write("\n".join(mem) + ("\n" if mem else ""))
                names.append(name)
            d = self._board_params(bid); d["regfiles"] = names; params.append(d)

        ps_dets = [d for d in range(self.ndet)
                   if d in self.cfg.postselect_detectors]
        manifest = {"circuit": os.path.basename(circuit_path), "config": os.path.basename(config_path),
                    "mem_format": mem_fmt, "layout": "physical", "postselect_layout": postselect_layout,
                    "output_sync": "all" if output_sync_all else "output-only",
                    "wait_rounds": wait_rounds, "wait_row": wait_row,
                    "global_index_hw_width": gidx_hw_width,   # hardware output width (host sentinel = 2^hw-1)
                    "detectors": self.ndet,
                    "meas_times": self.T, "det_times": self.ndt, "feasible": feasible,
                    "stage_boards": {si: bid for si, bid in enumerate(self.pl.stage_board)},
                    "postselect_detectors": ps_dets,   # stim detector indices that are postselect
                    "boards": params}
        with open(os.path.join(root, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        with open(os.path.join(root, "report.txt"), "w") as f:
            f.write(self._report(params, feasible) + "\n")
        return root


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, ".")
    from parser import DetectorConstructionScanner
    from config import CompilerConfig

    ap = argparse.ArgumentParser()
    ap.add_argument("circuit"); ap.add_argument("config")
    ap.add_argument("--mem", choices=["hex", "bin"], default="bin"); ap.add_argument("--out", default="results")
    ap.add_argument("--postselect-layout", choices=["flat", "per_stage"], default="flat",
                    help="postselect regfile when a board hosts multiple stages: "
                         "'flat' (default) = one ndt-deep mask (OR of the stages, per det-time); "
                         "'per_stage' = a separate ndt block per stage")
    ap.add_argument("--output-sync", choices=["all", "output-only"], default="all",
                    help="which boards get an output_sync regfile: "
                         "'all' (default) = every board (OutputSync is the output stage "
                         "everywhere); 'output-only' = legacy (root + stage boards only)")
    ap.add_argument("--wait-rounds", type=int, default=0,
                    help="number of trailing wait rounds baked into the circuit (= r2, "
                         "user-provided): the last W real det-times are marked is_wait "
                         "(real stim detectors). 0 = no wait rounds.")
    ap.add_argument("--wait-row", choices=["normal", "copy-last"], default="normal",
                    help="'normal' = no appended row (wait rounds, if any, come from the "
                         "circuit's last W det-times); 'copy-last' = also append a saturating "
                         "wait row copied from the last normal round (fresh global_index) "
                         "for finish arriving beyond the baked-in W")
    args = ap.parse_args()

    s = DetectorConstructionScanner.from_file(args.circuit); s.scan(); s.refine()
    comp = Compiler(CompilerConfig.from_file(args.config), s)
    root = Serializer(comp).write(args.out, args.circuit, args.config, mem_fmt=args.mem,
                                  postselect_layout=args.postselect_layout,
                                  output_sync_all=(args.output_sync == "all"),
                                  wait_rounds=args.wait_rounds, wait_row=args.wait_row)
    print("wrote", root)
    print(open(os.path.join(root, "report.txt")).read())
