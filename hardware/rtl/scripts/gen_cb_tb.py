#!/usr/bin/env python3
"""gen_cb_tb.py — generate a self-checking SystemVerilog testbench for a TREE of
ControlBoards + per-leaf StimReadout loops, from a compiled results/<example> dir.

RTL analog of emulator/tests/control_system/test_control_board_stim.cpp. Where
gen_dcb_tb.py wires only the DATA plane (bare DetectorConstructBlocks), this wires the
whole CONTROL plane: full ControlBoards (control FSM + PhysicalMMIO + DCB), events
ripple DOWN (parent.out_ev -> child.ev), data / attempt / finish / post-select ripple
UP, and each LEAF closes its own mmio_out -> StimReadout -> in_meas loop. START kicks
the root; the tree then runs itself. Handles MONOLITHIC (1 board = root+leaf) and
DISTRIBUTED (root / routers / leaves, wired from each board's connections.json).

The generated tb runs the SAME multi-trial state machine as the emulator driver:
  * INTERNAL post-select reject  -> board aborts + auto-advances every leaf's readout;
    check the oracle (OR of all stage boards' expected_postselect) agreed.
  * CLEAN completion (last_normal + wait_rounds real rounds) -> compare recon[global
    index] vs expected_dets[s], then gap_post_select to force-advance to the next shot.
  * The LAST shot ends the trial conditionally (like real operation): clean + copy-last
    -> host finish (drain via out_finish, which resets the readouts); else -> rst.

Output: <out_dir>/<example_name>/
    tb_cb.sv               the generated, self-checking testbench
    mem/*.mem              flat, board-prefixed copy of every board's regfiles + the
                           leaf MMIO programs + sim_readout memories
    files.f               filelist (module sources + tb) for verilator / xsim

Usage: gen_cb_tb.py <results/example_dir> [--out-dir DIR] [--shots N] [--trials T]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil


def _load(path):
    with open(path) as f:
        return json.load(f)


def _rows(path):
    with open(path) as f:
        return sum(1 for ln in f if ln.strip())


def _clog2(n):
    return 1 if n <= 1 else int(math.ceil(math.log2(n)))


def _used_lines(path, width):
    """Bit positions (LSB-indexed) whose column is ever 1 across the .mem rows. A FIFO
    on any other line is never popped, so its occupancy is not a depth requirement."""
    used = set()
    if not os.path.exists(path):
        return used
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            assert len(ln) == width, f"{path}: row width {len(ln)} != {width}"
            for i in range(width):
                if ln[width - 1 - i] == "1":      # MSB-first text, bit i = LSB-indexed
                    used.add(i)
    return used


def _board_params(R, bd, mem_src, man):
    """Per-board ControlBoard + StimReadout parameters from the manifest entry + mems."""
    rf = set(bd["regfiles"])
    copy = 1 if man.get("wait_row") == "copy-last" else 0
    sync_path = os.path.join(mem_src, "sync.mem")
    T = _rows(sync_path) if os.path.exists(sync_path) else man["meas_times"] + copy
    bid = bd["board_id"]
    is_leaf = "qubit_scope" in rf
    is_root = bd["type"] == "root"
    p = {
        "id":      bid,
        "M":       bd["m_phys"],
        "D_IN":    bd.get("detector_input_ports") or 0,
        "K":       bd["kernels_avail"],
        "N":       bd["n_cap"],
        "H":       bd["h_cap"],
        "RAW_OUT": bd.get("raw_out_cap") or 0,
        "IDX_W":   bd.get("global_index_width") or 1,
        "STRIDE":  bd.get("global_index_stride") or 0,
        "SENT":    bd.get("global_index_sentinel") or 0,
        "T":       T,
        "NDT":     _rows(os.path.join(mem_src, "output_sync.mem")),
        "IS_ROOT": 1 if is_root else 0,
        "IS_LEAF": 1 if is_leaf else 0,
        "HAS_PS":  1 if "postselect" in rf else 0,
        "HAS_OSY": 1 if "output_sync" in rf else 0,
        "COPY":    copy,
        "is_leaf": is_leaf,
        # BoardControl behaviour axes (mirror control_board_loader.hpp)
        "EVENT_MODE": 0 if is_root else 1,     # 0=ORIGINATE (root/mono), 1=FORWARD
        "DATA_SRC":   1 if is_leaf else 0,     # 1=INTERNAL (leaf/mono), 0=EXTERNAL
    }
    p["D_OUT"] = p["D_IN"] + p["K"]
    p["DIN_W"] = max(p["D_IN"], 1)
    p["RAW_W"] = max(p["RAW_OUT"], 1)

    # child topology (routers/root) from connections.json: children in the SAME order the
    # emulator loader derives — detector-port children first-seen, then measurement-port
    # children first-seen; child_dw/child_raw are that child's det/raw line counts.
    p["NCHILD"], p["CHILD_DW"], p["CHILD_RAW"], p["children"] = 0, [], [], []
    p["det_ports"], p["meas_ports"] = [], []
    cj = os.path.join(R, f"board{bid}", "json", "connections.json")
    if os.path.exists(cj):
        c = _load(cj)
        order, seen, dw, raw = [], set(), {}, {}
        def note(ch):
            if ch not in seen:
                seen.add(ch); order.append(ch)
        for dp in c.get("detector_ports", []):
            ch = dp["child"]; note(ch); dw[ch] = dw.get(ch, 0) + 1
            p["det_ports"].append((dp["port"], ch, dp["child_port"]))
        for mp in c.get("measurement_ports", []):
            ch = mp["child"]; note(ch); raw[ch] = raw.get(ch, 0) + 1
            p["meas_ports"].append((mp["port"], ch, mp["child_port"]))
        p["children"]  = order
        p["NCHILD"]    = len(order)
        p["CHILD_DW"]  = [dw.get(ch, 0) for ch in order]
        p["CHILD_RAW"] = [raw.get(ch, 0) for ch in order]

    # leaf: one PhysicalMMIO (P=1) + a StimReadout replay memory
    p["P"] = 1 if is_leaf else 0
    p["INSTR_DEPTH"], p["CW_DEPTH"], p["DATA_W"], p["WT_W"] = 4, 16, 8, 8
    p["SR_SHOTS"], p["SR_DEPTH"], p["SR_M"] = 0, 0, p["M"]
    if is_leaf:
        ij  = _load(os.path.join(R, f"board{bid}", "json", "MMIO_instr_0.json"))
        cwj = _load(os.path.join(R, f"board{bid}", "json", "MMIO_cw_0.json"))
        p["INSTR_DEPTH"] = ij["depth"]; p["CW_DEPTH"] = cwj["depth"]
        p["DATA_W"] = cwj["word_width"]; p["WT_W"] = ij["wt_w"]
        srj = _load(os.path.join(R, f"board{bid}", "json", "sim_readout_meas.json"))
        p["SR_SHOTS"] = srj["shots"]; p["SR_DEPTH"] = srj["depth"]; p["SR_M"] = srj["word_width"]
        p["SR_IDX_W"] = _clog2(p["SR_DEPTH"])
        assert p["DATA_W"] >= p["SR_IDX_W"], \
            f"board{bid}: mmio DATA_W {p['DATA_W']} < readout index width {p['SR_IDX_W']}"
    p["P_W"] = max(p["P"], 1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example_dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--io-only", action="store_true",
                    help="also emit clean host<->root interface aliases (io_*) and dump ONLY those to "
                         "the VCD, so the waveform shows just the host<->root signals instead of every "
                         "internal port (run the sim with --trace to produce tb_cb.vcd).")
    ap.add_argument("--fifo-stats", action="store_true",
                    help="track the max occupancy of every MeasurementSync / OutputSync per-line FIFO "
                         "(used lines vs all lines) and print one 'FIFO board=..' line per board at the "
                         "end — the measurement that sizes SYNC_FIFO / OUT_FIFO for synthesis.")
    ap.add_argument("--latency-trace", action="store_true",
                    help="print cycle-stamped 'LT <cycle> <event> ...' lines for every board event "
                         "(leaf MMIO round command / measurement arrival, det/raw finish, post-select, "
                         "events down, root output/discard) — the per-board latency measurement that "
                         "feeds the control-system latency model.")
    args = ap.parse_args()

    R = os.path.abspath(args.example_dir)
    man = _load(os.path.join(R, "manifest.json"))
    ex_name = os.path.basename(R.rstrip("/"))
    HW_W  = man["global_index_hw_width"]
    TSTIM = man["meas_times"]
    NDET  = man["detectors"]
    WAIT_ROUNDS = man.get("wait_rounds", 0)

    boards  = {b["board_id"]: b for b in man["boards"]}
    root_bd = next(b for b in man["boards"] if b["type"] == "root")
    root_id = root_bd["board_id"]

    out_root = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "testbench", "control_system", "generated")
    out = os.path.join(out_root, ex_name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)
    mem_out = os.path.join(out, "mem")
    os.makedirs(mem_out)

    # per-board params + flat, board-prefixed copy of every mem file
    P = {}
    for bid, bd in boards.items():
        mem_src = os.path.join(R, f"board{bid}", "mem")
        P[bid] = _board_params(R, bd, mem_src, man)
        P[bid]["ms_used"] = _used_lines(os.path.join(mem_src, "sync.mem"), P[bid]["M"])
        P[bid]["os_used"] = _used_lines(os.path.join(mem_src, "output_sync.mem"), P[bid]["D_OUT"])
        for fn in os.listdir(mem_src):
            shutil.copy(os.path.join(mem_src, fn), os.path.join(mem_out, f"board{bid}_{fn}"))

    # parent-of map (for event bus down)
    parent_of = {}
    for bid in P:
        for ch in P[bid]["children"]:
            parent_of[ch] = bid

    # stage boards: ps fast-path order (excl root) and the oracle set (ALL stage boards).
    stage_map = {int(k): v for k, v in man.get("stage_boards", {}).items()}  # stage_idx -> board_id
    stage_order = [stage_map[k] for k in sorted(stage_map)]            # in stage order
    ps_fast = [b for b in stage_order if b != root_id]                 # feed root.ps_in
    oracle_boards = list(dict.fromkeys(stage_order))                   # OR-ed for reject (dedup: mono maps both stages to one board)

    ed = _load(os.path.join(R, f"board{root_id}", "json", "expected_dets.json"))
    SHOTS_AVAIL = ed["shots"]
    SHOTS = SHOTS_AVAIL if args.shots <= 0 else min(args.shots, SHOTS_AVAIL)

    leaves = [bid for bid in P if P[bid]["is_leaf"]]

    sv = _emit_tb(ex_name, P, root_id, leaves, parent_of, ps_fast, oracle_boards,
                  HW_W=HW_W, SHOTS=SHOTS, SHOTS_AVAIL=SHOTS_AVAIL, TSTIM=TSTIM,
                  NDET=NDET, WAIT_ROUNDS=WAIT_ROUNDS, TRIALS=args.trials, io_only=args.io_only,
                  latency_trace=args.latency_trace, fifo_stats=args.fifo_stats)
    with open(os.path.join(out, "tb_cb.sv"), "w") as f:
        f.write(sv)

    _write_expected(out, root_id, P[root_id], oracle_boards, SHOTS, TSTIM, NDET,
                    WAIT_ROUNDS, args.trials, ex_name, HW_W)

    # filelist: every module the tree needs
    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    cs, dc, cc, lib = (os.path.join(src, "control_system"),
                       os.path.join(src, "control_system", "detector_construct"),
                       os.path.join(src, "control_system", "cultiv_control"),
                       os.path.join(src, "lib"))
    files = ([os.path.join(lib, m + ".sv") for m in ("RegFileROM", "SyncROM")]
             + [os.path.join(dc, m + ".sv") for m in ("RawSelector", "MeasurementSync", "Kernel",
                "DetectorPass", "OutputSync", "RootOutputSync", "Postselect", "DetectorConstructBlock")]
             + [os.path.join(cc, m + ".sv") for m in ("InstrDecode", "InstrUnpack", "InstrSequencer",
                "DrainAggregator", "PhysicalMMIO", "BoardControl")]
             + [os.path.join(cs, m + ".sv") for m in ("StimReadout", "ControlBoard")]
             + [os.path.join(out, "tb_cb.sv")])
    with open(os.path.join(out, "files.f"), "w") as f:
        f.write("\n".join(files) + "\n")

    # io-only trace caps (Verilator defaults would silently drop signals otherwise):
    #   --trace-max-array: io_out_global_indexes is an unpacked array of ROOT_DOUT elements (default 32)
    #   --trace-max-width: applied to the array's FLATTENED width (ROOT_DOUT*HW_W bits), not per
    #                      element — so it must clear ROOT_DOUT*HW_W, else the whole array is dropped.
    RD = P[root_id]["D_OUT"]
    trace = (f" --trace --trace-max-array {max(32, RD)} --trace-max-width {max(256, RD * HW_W)}"
             if args.io_only else "")
    print(f"  wrote {out}/tb_cb.sv  ({len(P)} boards: {len(leaves)} leaves, root=board{root_id}, "
          f"ps_fast={ps_fast}, oracle={oracle_boards}){'  [io-only: host<->root aliases + VCD]' if args.io_only else ''}")
    print(f"  stim: shots={SHOTS}/{SHOTS_AVAIL} T={TSTIM} ndet={NDET} wait_rounds={WAIT_ROUNDS} trials={args.trials}")
    print(f"  run:  cd {out}/mem && verilator --binary --timing{trace} -Wno-fatal -f {out}/files.f --top-module tb_cb -o sim && ./obj_dir/sim"
          + (f"   # -> {out}/mem/tb_cb.vcd (io_* only)" if args.io_only else ""))


# --------------------------------------------------------------------- golden reference

def _mem_ints(path):
    """each .mem row (MSB-first binary) -> integer (bit i = LSB-indexed bit i)."""
    with open(path) as f:
        return [int(ln.strip(), 2) for ln in f if ln.strip()]


def _write_expected(out, root_id, rp, oracle_boards, SHOTS, TSTIM, NDET,
                    WAIT_ROUNDS, TRIALS, ex, HW_W):
    """Golden reference for the WHOLE-SYSTEM tb: the expected ROOT output per shot, per
    out_valid pulse (emission order), PLUS the per-shot classification that the DCB tb
    lacks. Each shot is CLEAN or REJECTED per the post-select oracle (OR of the stage
    boards' expected_postselect): CLEAN shots get the full detector prediction (same as
    the DCB golden ref); REJECTED shots get a one-line annotation (the board aborts +
    advances every leaf readout, no clean emission). The sequence is identical across the
    TRIALS trials, so it is listed once; the LAST shot's action ends the trial."""
    mem = os.path.join(out, "mem")
    pre = f"board{root_id}_"
    D_OUT, IDX_W, SENT = rp["D_OUT"], rp["IDX_W"], rp["SENT"]
    STRIDE, COPY = rp["STRIDE"], rp["COPY"]
    imask = (1 << IDX_W) - 1

    osync = _mem_ints(os.path.join(mem, pre + "output_sync.mem"))    # NDT rows, D_OUT bits
    gidx  = _mem_ints(os.path.join(mem, pre + "global_index.mem"))   # NDT rows, D_OUT*IDX_W bits
    exp   = _mem_ints(os.path.join(mem, pre + "expected_dets.mem"))  # SHOTS rows, NDET bits
    rm    = _mem_ints(os.path.join(mem, pre + "round_marker.mem"))   # NDT rows, 3 bits
    ps_path = os.path.join(mem, pre + "postselect.mem")
    ps = _mem_ints(ps_path) if os.path.exists(ps_path) else None     # NDT rows, D_OUT bits (stage only)
    ndt = len(osync)
    real_ndt = ndt - COPY          # real det-times (copy-last appends 1 saturating row)
    sat_row  = ndt - 1

    # post-select oracle: OR bit0 of every stage board's expected_postselect -> reject[s]
    reject = [0] * SHOTS
    for b in oracle_boards:
        v = _mem_ints(os.path.join(mem, f"board{b}_expected_postselect.mem"))
        for s in range(SHOTS):
            if s < len(v) and (v[s] & 1):
                reject[s] = 1
    n_rej = sum(reject[:SHOTS])

    allones = (1 << HW_W) - 1

    def round_line(s, nt, offset=0, tag=""):
        m, g = osync[nt], gidx[nt]
        items = []
        for l in range(D_OUT):
            if not ((m >> l) & 1):
                continue
            idx = (g >> (l * IDX_W)) & imask
            if idx == SENT:
                continue
            hw = idx + offset
            val = str((exp[s] >> hw) & 1) if hw < NDET else "W"
            items.append(f"line {l} -> gidx {hw} = {val}")
        body = " | ".join(items) if items else "(fast-forward, no detectors)"
        return f"  round {tag or nt}: {body}"

    def action(s):
        is_last = (s == SHOTS - 1)
        if reject[s]:
            act = "discard -> EV_ABORT -> every leaf StimReadout advances to the next shot"
            return act + ("  [LAST -> rst; trial ends]" if is_last else "")
        if not is_last:
            return "gap_post_select -> EV_ABORT -> every leaf StimReadout advances to the next shot"
        return ("host finish -> out_finish on the last wait round -> readouts reset to shot 0; trial ends"
                if COPY else "rst to restart; trial ends")

    lines = [
        f"# Expected WHOLE-SYSTEM (tb_cb) ROOT output for '{ex}'  (generated by gen_cb_tb.py)",
        f"# D_OUT={D_OUT} IDX_W={IDX_W} det-times={ndt} (real={real_ndt}{', +1 copy-last wait row' if COPY else ''})"
        f"  SHOTS={SHOTS}  wait_rounds={WAIT_ROUNDS}  copy_last={bool(COPY)}",
        f"#",
        f"# The host driver runs TRIALS={TRIALS}, each replaying all {SHOTS} shots IDENTICALLY, so the",
        f"# sequence below is listed ONCE (per trial). Each shot is classified by the post-select",
        f"# ORACLE = OR of the stage boards' expected_postselect ({oracle_boards}):",
        f"#   CLEAN   [oracle=0]: the root emits all real + wait detector rounds (listed); the driver",
        f"#             compares recon[gidx] vs expected_dets[s], then advances (see each shot's action).",
        f"#   REJECTED[oracle=1]: a stage board's post_select fires -> the root asserts `discard`, an",
        f"#             EV_ABORT ripples down, every leaf StimReadout advances; NO clean emission, NO finish.",
        f"# This example: {SHOTS - n_rej} CLEAN + {n_rej} REJECTED per trial.",
        f"#",
        f"# Rounds are in EMISSION ORDER = successive root out_valid pulses (absolute cycle is data-",
        f"# dependent through the tree — count out_valid pulses in the waveform). Each used line:",
        f"# 'line L -> gidx G = value' from expected_dets[s]; copy-last wait rounds re-emit the",
        f"# saturating row at offset w*STRIDE (value 'W' = wait row, no ground truth).",
        "",
        "============================ SECTION 1 — per-shot detector emission (per trial) ============================",
    ]
    for s in range(SHOTS):
        tag = "REJECTED" if reject[s] else "CLEAN"
        last = "  (LAST)" if s == SHOTS - 1 else ""
        lines.append(f"======================== SHOT {s}  [{tag}, oracle_ps={reject[s]}]{last} ========================")
        if reject[s]:
            lines.append("  (discarded internally — no clean root detector emission)")
        else:
            for nt in range(real_ndt):
                lines.append(round_line(s, nt))
            if COPY:
                for w in range(WAIT_ROUNDS):
                    lines.append(round_line(s, sat_row, offset=w * STRIDE, tag=f"wait{w}"))
        lines.append(f"  -> action: {action(s)}")
        lines.append("")

    # ---- SECTION 2: root DUT output signal values per out_valid pulse (clean shots only) ----
    def sig_round(s, nt, offset, is_sat, w, prev_wait, is_last_shot):
        m, g = osync[nt], gidx[nt]
        r = rm[nt]
        is_first, is_last_r, is_wait = r & 1, (r >> 1) & 1, (r >> 2) & 1
        first_wait = 1 if (is_wait and not prev_wait) else 0
        # out_finish fires ONLY on the last shot of a trial, and only when it is clean AND
        # copy-last (host finish rides the last wait round). Non-last clean shots advance via
        # gap_post_select (abort), so they never assert out_finish.
        finish = 1 if (is_last_shot and COPY and is_sat and w == WAIT_ROUNDS - 1) else 0
        last_wait = 1 if (is_wait and finish) else 0
        det_int, gi, ps_fire = 0, [], 0
        pm = ps[nt] if ps is not None else 0
        for l in range(D_OUT):
            used = (m >> l) & 1
            idx = (g >> (l * IDX_W)) & imask
            if used and idx != SENT:
                gi.append(idx + offset)
                bit = 0 if is_sat else ((exp[s] >> (idx + offset)) & 1) if (idx + offset) < NDET else 0
                if bit:
                    det_int |= (1 << l)
                    if (pm >> l) & 1:
                        ps_fire = 1
            else:
                gi.append(allones)
        det = "(placeholder, no ground truth)" if is_sat else format(det_int, f"0{D_OUT}b")
        hdr = (f"round {('wait%d' % w) if is_sat else nt}: out_valid=1 out_finish={finish} "
               f"first_normal={is_first} last_normal={is_last_r} first_wait={first_wait} last_wait={last_wait}"
               + (f" post_select={ps_fire}" if ps is not None else ""))
        return [hdr,
                f"  out_used            = {format(m, f'0{D_OUT}b')}",
                f"  out_det             = {det}",
                f"  out_global_indexes  = {' '.join(str(x) for x in gi)}"], is_wait

    emission = [(nt, 0, False, None) for nt in range(real_ndt)]
    if COPY:
        emission += [(sat_row, w * STRIDE, True, w) for w in range(WAIT_ROUNDS)]

    lines += [
        "",
        "============================================================================",
        "# SECTION 2 — root DUT output signals per out_valid pulse (CLEAN shots only; compare to the waveform).",
        f"# out_used / out_det: BINARY, MSB = line {D_OUT-1} (as a [{D_OUT-1}:0] vector).",
        f"# out_global_indexes: DECIMAL per line, line 0 first; unused/sentinel slot reads {allones} ({HW_W} bits).",
        "# out_finish is 1 ONLY on the LAST shot's last wait round (clean + copy-last); non-last clean shots",
        "#   advance via gap_post_select (abort), so out_finish=0 there. REJECTED shots emit no clean rounds.",
        "============================================================================",
    ]
    for s in range(SHOTS):
        if reject[s]:
            lines.append(f"-------------------------------- SHOT {s}  [REJECTED — no clean emission] --------------------------------")
            lines.append("")
            continue
        lines.append(f"-------------------------------- SHOT {s}  [CLEAN] --------------------------------")
        prev_wait = 0
        for (nt, off, is_sat, w) in emission:
            body, prev_wait = sig_round(s, nt, off, is_sat, w, prev_wait, s == SHOTS - 1)
            lines += body
        lines.append("")

    with open(os.path.join(out, "expected_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- emit

MAX_CHILD = 64   # must match ControlBoard.sv's MAX_CHILD (CHILD_* literals are padded to it)


def _arr(vals):
    """fixed-size (MAX_CHILD) unpacked-array literal, zero-padded (Verilator needs an exact count)."""
    padded = list(vals) + [0] * (MAX_CHILD - len(vals))
    return "'{" + ", ".join(str(v) for v in padded) + "}"


def _inst(p, HW_W):
    """per-board nets + ControlBoard instance."""
    b, M, DINW, DOUT, RAWW = p["id"], p["M"], p["DIN_W"], p["D_OUT"], p["RAW_W"]
    NCW = max(p["NCHILD"], 1)
    NPS = p["NCHILD"] and 0  # placeholder; NPS set below from root
    role = ("leaf " if p["is_leaf"] else "") + ("root " if p["IS_ROOT"] else "") + \
           ("stage " if p["HAS_PS"] else "") + ("router" if not p["is_leaf"] and not p["IS_ROOT"] else "")
    dataw, pw = p["DATA_W"], p["P_W"]
    s = [f"    // ---- board{b} ({role.strip()}) ----"]
    s.append(f"    logic [1:0] b{b}_ev_type, b{b}_out_ev_type;")
    s.append(f"    logic [{p['PAYLOAD_W']-1 if 'PAYLOAD_W' in p else 1}:0] b{b}_ev_payload, b{b}_out_ev_payload;")
    s.append(f"    logic b{b}_ev_valid, b{b}_ev_attempt, b{b}_out_ev_valid, b{b}_out_ev_attempt;")
    s.append(f"    logic [{DINW-1}:0] b{b}_in_det, b{b}_in_det_valid;")
    s.append(f"    logic [{M-1}:0] b{b}_in_meas, b{b}_in_meas_valid, b{b}_in_meas_finish;")
    s.append(f"    logic [{NCW-1}:0] b{b}_in_det_fin_ch, b{b}_in_raw_fin_ch, b{b}_in_child_att;")
    s.append(f"    logic [{max(p.get('NPS',0),1)-1}:0] b{b}_ps_in, b{b}_ps_in_att;")
    s.append(f"    logic [{DOUT-1}:0] b{b}_fwd_det, b{b}_fwd_det_valid;")
    s.append(f"    logic b{b}_fwd_det_finish, b{b}_fwd_raw_finish, b{b}_out_attempt, b{b}_ps_out, b{b}_ps_out_att;")
    s.append(f"    logic [{RAWW-1}:0] b{b}_fwd_raw, b{b}_fwd_raw_valid;")
    s.append(f"    logic [{DOUT-1}:0] b{b}_out_det, b{b}_out_used;")
    s.append(f"    logic b{b}_out_valid, b{b}_out_finish, b{b}_first_normal, b{b}_last_normal, b{b}_first_wait, b{b}_last_wait, b{b}_discard;")
    s.append(f"    logic [{DOUT*HW_W-1}:0] b{b}_out_gidx;")
    s.append(f"    logic [{pw-1}:0] b{b}_cw_gen_finish, b{b}_mmio_out_valid;")
    s.append(f"    logic [{pw*dataw-1}:0] b{b}_mmio_out_data;")

    # parameter list (conditionally include topology params)
    par = [f".IS_ROOT(1'b{p['IS_ROOT']})", f".IS_LEAF(1'b{p['IS_LEAF']})",
           f".EVENT_MODE({p['EVENT_MODE']})", f".DATA_SRC({p['DATA_SRC']})", f".HAS_PS(1'b{p['HAS_PS']})"]
    if p["NCHILD"] > 0:
        par += [f".NCHILD({p['NCHILD']})", f".CHILD_DW({_arr(p['CHILD_DW'])})", f".CHILD_RAW({_arr(p['CHILD_RAW'])})"]
    if p.get("NPS", 0) > 0:
        par += [f".NPS({p['NPS']})"]
    par += [f".M({M})", f".D_IN({p['D_IN']})", f".K({p['K']})", f".N({p['N']})", f".H({p['H']})",
            f".RAW_OUT({p['RAW_OUT']})", f".IDX_W({p['IDX_W']})", f".HW_WIDTH({HW_W})",
            f".STRIDE({p['STRIDE']})", f".SENTINEL({p['SENT']})", f".T({p['T']})", f".NDT({p['NDT']})",
            f".PC_W(16)", f".SYNC_FIFO(64)", f".OUT_FIFO(64)",
            f".HAS_OSYNC(1'b{p['HAS_OSY']})", f".COPY_LAST(1'b{p['COPY']})", f'.MEM_DIR("board{b}_")',
            f".PAYLOAD_W({p['PAYLOAD_W']})"]
    if p["is_leaf"]:
        par += [f".P({p['P']})", f".INSTR_DEPTH({p['INSTR_DEPTH']})", f".CW_DEPTH({p['CW_DEPTH']})",
                f".DATA_W({p['DATA_W']})", f".WT_W({p['WT_W']})"]

    # host controls: root <- h_*, others tied 0
    start  = "h_start"  if p["IS_ROOT"] else "1'b0"
    finish = "h_finish" if p["IS_ROOT"] else "1'b0"
    gap    = "h_gap_ps" if p["IS_ROOT"] else "1'b0"

    s.append(f"    ControlBoard #(\n        " + ",\n        ".join(par) + "\n    ) b{b} (".replace("{b}", str(b)))
    s.append(f"        .clk(clk), .rst(rst),")
    s.append(f"        .ev_valid(b{b}_ev_valid), .ev_type(b{b}_ev_type), .ev_payload(b{b}_ev_payload), .ev_attempt(b{b}_ev_attempt),")
    s.append(f"        .start({start}), .finish({finish}), .gap_post_select({gap}),")
    s.append(f"        .in_det(b{b}_in_det), .in_det_valid(b{b}_in_det_valid),")
    s.append(f"        .in_meas(b{b}_in_meas), .in_meas_valid(b{b}_in_meas_valid), .in_meas_finish(b{b}_in_meas_finish),")
    s.append(f"        .in_det_finish_child(b{b}_in_det_fin_ch), .in_raw_finish_child(b{b}_in_raw_fin_ch), .in_child_attempt(b{b}_in_child_att),")
    s.append(f"        .ps_in(b{b}_ps_in), .ps_in_attempt(b{b}_ps_in_att),")
    s.append(f"        .fwd_det(b{b}_fwd_det), .fwd_det_valid(b{b}_fwd_det_valid), .fwd_det_finish(b{b}_fwd_det_finish),")
    s.append(f"        .fwd_raw(b{b}_fwd_raw), .fwd_raw_valid(b{b}_fwd_raw_valid), .fwd_raw_finish(b{b}_fwd_raw_finish),")
    s.append(f"        .out_attempt(b{b}_out_attempt), .ps_out(b{b}_ps_out), .ps_out_attempt(b{b}_ps_out_att),")
    s.append(f"        .out_ev_valid(b{b}_out_ev_valid), .out_ev_type(b{b}_out_ev_type), .out_ev_payload(b{b}_out_ev_payload), .out_ev_attempt(b{b}_out_ev_attempt),")
    s.append(f"        .out_det(b{b}_out_det), .out_used(b{b}_out_used), .out_valid(b{b}_out_valid), .out_finish(b{b}_out_finish),")
    s.append(f"        .out_global_indexes(b{b}_out_gidx),")
    s.append(f"        .first_normal(b{b}_first_normal), .last_normal(b{b}_last_normal), .first_wait(b{b}_first_wait), .last_wait(b{b}_last_wait), .discard(b{b}_discard),")
    s.append(f"        .cw_gen_finish(b{b}_cw_gen_finish), .mmio_out_data(b{b}_mmio_out_data), .mmio_out_valid(b{b}_mmio_out_valid)")
    s.append(f"    );\n")
    return "\n".join(s)


def _wiring(P, root_id, parent_of, ps_fast):
    """assign statements: tree data plane + per-child finish/attempt + events down + ps fast path."""
    L = []
    for b in sorted(P):
        p = P[b]
        L.append(f"    // board{b} wiring")
        # event bus down: non-root ev_* <- parent.out_ev_*; root ev_* tied 0
        if p["IS_ROOT"]:
            L.append(f"    assign b{b}_ev_valid = 1'b0; assign b{b}_ev_type = 2'd0; assign b{b}_ev_payload = '0; assign b{b}_ev_attempt = 1'b0;")
        else:
            par = parent_of[b]
            L.append(f"    assign b{b}_ev_valid = b{par}_out_ev_valid; assign b{b}_ev_type = b{par}_out_ev_type;")
            L.append(f"    assign b{b}_ev_payload = b{par}_out_ev_payload; assign b{b}_ev_attempt = b{par}_out_ev_attempt;")

        if p["is_leaf"]:
            # leaf: no children; measurement comes from its StimReadout (assigned later), det tied 0
            L.append(f"    assign b{b}_in_det = '0; assign b{b}_in_det_valid = '0;")
            L.append(f"    assign b{b}_in_det_fin_ch = '0; assign b{b}_in_raw_fin_ch = '0; assign b{b}_in_child_att = '0;")
        else:
            # router/root data plane: in_det[port] <- child.fwd_det[child_port]; in_meas[port] <- child.fwd_raw[child_port]
            L.append(f"    assign b{b}_in_det = '0;" if not p["det_ports"] else "")
            for (port, ch, cp) in p["det_ports"]:
                L.append(f"    assign b{b}_in_det[{port}] = b{ch}_fwd_det[{cp}]; assign b{b}_in_det_valid[{port}] = b{ch}_fwd_det_valid[{cp}];")
            for (port, ch, cp) in p["meas_ports"]:
                L.append(f"    assign b{b}_in_meas[{port}] = b{ch}_fwd_raw[{cp}]; assign b{b}_in_meas_valid[{port}] = b{ch}_fwd_raw_valid[{cp}]; assign b{b}_in_meas_finish[{port}] = b{ch}_fwd_raw_finish;")
            # per-child single-bit finish / attempt buses (child order)
            for c, ch in enumerate(p["children"]):
                L.append(f"    assign b{b}_in_det_fin_ch[{c}] = b{ch}_fwd_det_finish; assign b{b}_in_raw_fin_ch[{c}] = b{ch}_fwd_raw_finish; assign b{b}_in_child_att[{c}] = b{ch}_out_attempt;")

        # ps_in: only the root consumes the fast path; everyone else tie 0
        if p["IS_ROOT"] and ps_fast:
            for i, st in enumerate(ps_fast):
                L.append(f"    assign b{b}_ps_in[{i}] = b{st}_ps_out; assign b{b}_ps_in_att[{i}] = b{st}_ps_out_att;")
        else:
            L.append(f"    assign b{b}_ps_in = '0; assign b{b}_ps_in_att = '0;")
    return "\n".join(x for x in L if x != "") + "\n"


def _leaf_readouts(P, leaves):
    """per-leaf StimReadout + close the mmio_out -> readout -> in_meas loop."""
    L = []
    for b in leaves:
        p = P[b]
        siw, m = p["SR_IDX_W"], p["SR_M"]
        # abort source mirrors the MMIO's: ORIGINATE (root+leaf mono) -> out_ev, FORWARD (leaf) -> ev
        ev_v = f"b{b}_out_ev_valid" if p["EVENT_MODE"] == 0 else f"b{b}_ev_valid"
        ev_t = f"b{b}_out_ev_type"  if p["EVENT_MODE"] == 0 else f"b{b}_ev_type"
        L.append(f"    // board{b} StimReadout loop")
        L.append(f"    logic [{m-1}:0] b{b}_sr_meas, b{b}_sr_valid, b{b}_sr_finish;")
        L.append(f"    logic [{_clog2(p['SR_SHOTS']+1)-1}:0] b{b}_sr_shot;")
        L.append(f"    StimReadout #(")
        L.append(f"        .M({m}), .SHOTS({p['SR_SHOTS']}), .DEPTH({p['SR_DEPTH']}),")
        L.append(f'        .MEAS_FILE("board{b}_sim_readout_meas.mem"), .VALID_FILE("board{b}_sim_readout_valid.mem"), .INIT_HEX(1\'b0)')
        L.append(f"    ) sr{b} (")
        L.append(f"        .clk(clk), .rst(rst),")
        L.append(f"        .mmio_out_valid(b{b}_mmio_out_valid[0]), .mmio_out_data(b{b}_mmio_out_data[{siw-1}:0]), .cw_gen_finish(b{b}_cw_gen_finish[0]),")
        L.append(f"        .ev_valid({ev_v}), .ev_type({ev_t}),")
        L.append(f"        .out_meas(b{b}_sr_meas), .out_meas_valid(b{b}_sr_valid), .out_meas_finish(b{b}_sr_finish), .shot_reg(b{b}_sr_shot)")
        L.append(f"    );")
        L.append(f"    assign b{b}_in_meas = b{b}_sr_meas; assign b{b}_in_meas_valid = b{b}_sr_valid; assign b{b}_in_meas_finish = b{b}_sr_finish;")
    return "\n".join(L) + "\n"


def _fifo_stats(P):
    """--fifo-stats: per board, track the max occupancy of the MeasurementSync and OutputSync
    per-line FIFOs, split into USED lines (mask column ever 1 -> the depth the hardware needs)
    and ALL lines (idle lines only buffer until the per-shot board reset). Prints one
    'FIFO board=..' line per board at the end of the run."""
    def mask_lit(width, used):
        return f"{width}'b" + "".join("1" if i in used else "0" for i in range(width - 1, -1, -1))
    decl, upd, rep = [], [], []
    for b in sorted(P):
        p = P[b]
        M, D = p["M"], p["D_OUT"]
        ms = f"b{b}.u_dcb.g_construct.ms"
        if p["IS_ROOT"]:
            osy = f"b{b}.u_dcb.g_out.g_root.rosync.os"
        elif p["HAS_PS"] or p["HAS_OSY"]:
            osy = f"b{b}.u_dcb.g_out.g_nonroot.osync"
        else:
            osy = None
        decl.append(f"    int fs_ms_used{b} = 0, fs_ms_all{b} = 0, fs_os_used{b} = 0, fs_os_all{b} = 0;")
        decl.append(f"    localparam logic [{M-1}:0] FS_MS_USED{b} = {mask_lit(M, p['ms_used'])};")
        decl.append(f"    localparam logic [{D-1}:0] FS_OS_USED{b} = {mask_lit(D, p['os_used'])};")
        upd.append(f"        for (int i = 0; i < {M}; i++) begin")
        upd.append(f"            automatic int c = int'({ms}.count[i]);")
        upd.append(f"            if (c > fs_ms_all{b}) fs_ms_all{b} = c;")
        upd.append(f"            if (FS_MS_USED{b}[i] && c > fs_ms_used{b}) fs_ms_used{b} = c;")
        upd.append(f"        end")
        if osy:
            upd.append(f"        for (int i = 0; i < {D}; i++) begin")
            upd.append(f"            automatic int c = int'({osy}.count[i]);")
            upd.append(f"            if (c > fs_os_all{b}) fs_os_all{b} = c;")
            upd.append(f"            if (FS_OS_USED{b}[i] && c > fs_os_used{b}) fs_os_used{b} = c;")
            upd.append(f"        end")
        role = "root" if p["IS_ROOT"] else ("leaf" if p["is_leaf"] else "router")
        rep.append(f'        $display("FIFO board={b} type={role} M={M} ms_used=%0d ms_all=%0d '
                   f'D={D} os_used=%0d os_all=%0d", fs_ms_used{b}, fs_ms_all{b}, fs_os_used{b}, fs_os_all{b});')
    return ("    // ============================ fifo stats ============================\n"
            + "\n".join(decl) + "\n    always @(posedge clk) begin\n" + "\n".join(upd) + "\n    end\n\n",
            "\n".join(rep) + "\n")


def _emit_tb(ex, P, root_id, leaves, parent_of, ps_fast, oracle_boards, **g):
    HW_W = g["HW_W"]
    for b in P:
        P[b]["PAYLOAD_W"] = 2
    P[root_id]["NPS"] = len(ps_fast)
    root = P[root_id]
    RD = root["D_OUT"]

    insts   = "\n".join(_inst(P[b], HW_W) for b in sorted(P))
    wiring  = _wiring(P, root_id, parent_of, ps_fast)
    readout = _leaf_readouts(P, leaves)

    # --io-only: clean, grouped aliases for EXACTLY the host<->root interface, and a VCD dump of
    # only those (+clk/rst) so the waveform isn't buried under every internal ControlBoard port.
    io_block = ""
    trace_off = ""
    if g.get("io_only"):
        r = root_id
        # Restrict the trace to the host<->root interface. Two mechanisms, so BOTH sim tools
        # narrow the same way: (1) the explicit $dumpvars list below (honored by Vivado XSim);
        # (2) tracing_off/tracing_on metacomments (honored by Verilator, which ignores the
        # $dumpvars signal list and would otherwise dump the whole design). clk/rst are declared
        # before the tracing_off, so they stay traced for context.
        trace_off = "\n    /*verilator tracing_off*/"
        io_block = f"""    /*verilator tracing_on*/
    // ==================== host <-> root INTERFACE (io-only view) ====================
    // Aliases for exactly the signals exchanged between the host driver and the ROOT board:
    //   inputs  (host -> root): start, finish, gap_post_select
    //   outputs (root -> host): discard, out_valid, out_finish, out_det, out_used,
    //                           out_global_indexes, first_normal, last_normal, first_wait, last_wait
    // Only these (+ clk/rst) are dumped to tb_cb.vcd, so the waveform shows just the interface.
    wire                          io_start              = h_start;
    wire                          io_finish             = h_finish;
    wire                          io_gap_post_select    = h_gap_ps;
    wire                          io_discard            = b{r}_discard;
    wire                          io_out_valid          = b{r}_out_valid;
    wire                          io_out_finish         = b{r}_out_finish;
    wire [ROOT_DOUT-1:0]          io_out_det            = b{r}_out_det;
    wire [ROOT_DOUT-1:0]          io_out_used           = b{r}_out_used;
    // out_global_indexes split PER LINE (like the DCB tb's gi_out[d]): io_out_global_indexes[l]
    // is line l's HW_WIDTH-bit global detector index, so each wire's index is readable in the
    // waveform instead of one flattened ROOT_DOUT*HW_WIDTH-bit number. Unused slot = all-ones.
    wire [HW_WIDTH-1:0]           io_out_global_indexes [ROOT_DOUT];
    for (genvar gl = 0; gl < ROOT_DOUT; gl++) begin : g_io_gi
        assign io_out_global_indexes[gl] = b{r}_out_gidx[gl*HW_WIDTH +: HW_WIDTH];
    end
    wire                          io_first_normal       = b{r}_first_normal;
    wire                          io_last_normal        = b{r}_last_normal;
    wire                          io_first_wait         = b{r}_first_wait;
    wire                          io_last_wait          = b{r}_last_wait;
    initial begin
        $dumpfile("tb_cb.vcd");
        $dumpvars(0, clk, rst,
                  io_start, io_finish, io_gap_post_select,
                  io_discard, io_out_valid, io_out_finish, io_out_det, io_out_used,
                  io_out_global_indexes, io_first_normal, io_last_normal, io_first_wait, io_last_wait);
    end
    /*verilator tracing_off*/

"""

    # --latency-trace: cycle-stamped event lines (parsed offline into per-board latencies)
    lat_block = ""
    if g.get("latency_trace"):
        r = root_id
        ev = []
        for b in sorted(P):
            p = P[b]
            if p["is_leaf"]:
                siw = p["SR_IDX_W"]
                ev.append(f'        if (b{b}_mmio_out_valid[0]) $display("LT %0d mmio board={b} round=%0d", cyc, b{b}_mmio_out_data[{siw-1}:0]);')
                ev.append(f'        if (|b{b}_in_meas_valid) $display("LT %0d meas_in board={b} finish=%0d", cyc, |b{b}_in_meas_finish);')
            ev.append(f'        if (b{b}_fwd_det_finish) $display("LT %0d det_finish board={b}", cyc);')
            ev.append(f'        if (b{b}_fwd_raw_finish) $display("LT %0d raw_finish board={b}", cyc);')
            if not p["IS_ROOT"]:   # per-round forward: rising edge of any fwd_det_valid line
                ev.append(f'        if ((|b{b}_fwd_det_valid) && !lt_fv{b}) $display("LT %0d fwd_valid board={b}", cyc);')
                ev.append(f'        lt_fv{b} <= |b{b}_fwd_det_valid;')
            if p["HAS_PS"]:
                ev.append(f'        if (b{b}_ps_out) $display("LT %0d ps_out board={b}", cyc);')
            if not p["IS_ROOT"]:
                ev.append(f'        if (b{b}_ev_valid) $display("LT %0d ev_in board={b} type=%0d", cyc, b{b}_ev_type);')
            ev.append(f'        if (b{b}_out_ev_valid) $display("LT %0d ev_out board={b} type=%0d", cyc, b{b}_out_ev_type);')
        ev.append(f'        if (b{r}_out_valid) $display("LT %0d root_out_valid last_normal=%0d", cyc, b{r}_last_normal);')
        ev.append(f'        if (b{r}_discard) $display("LT %0d root_discard", cyc);')
        fv_decl = " ".join(f"logic lt_fv{b} = 1'b0;" for b in sorted(P) if not P[b]["IS_ROOT"])
        lat_block = ("    // ============================ latency trace ============================\n"
                     "    int cyc = 0;\n"
                     f"    {fv_decl}\n"
                     "    always @(posedge clk) cyc <= cyc + 1;\n"
                     "    always @(posedge clk) begin\n" + "\n".join(ev) + "\n    end\n\n")

    fifo_block, fifo_report = _fifo_stats(P) if g.get("fifo_stats") else ("", "")

    # oracle: OR of every stage board's expected_postselect
    orc_decl = "\n".join(f"    logic [0:0] st{b}_ps [SHOTS_AVAIL];" for b in oracle_boards)
    orc_load = "\n".join(f'        $readmemb("board{b}_expected_postselect.mem", st{b}_ps);' for b in oracle_boards)
    orc_expr = " | ".join(f"st{b}_ps[s][0]" for b in oracle_boards) if oracle_boards else "1'b0"

    # leaf shot_reg list for the diagnostic sync check
    sr_names = ", ".join(f"b{b}_sr_shot" for b in leaves)
    sync_chk = ""
    if len(leaves) > 1:
        first = leaves[0]
        checks = " || ".join(f"(b{b}_sr_shot !== b{first}_sr_shot)" for b in leaves[1:])
        sync_chk = f"""
        // diagnostic: are all leaves' readouts on the same shot? (desync = a real bug, printed)
        if ({checks}) begin
            total_sync_mismatch++;
            $display("  trial %0d shot %0d: leaf readout DESYNC  shot_regs = {' '.join('%0d' for _ in leaves)}",
                     trial, s, {sr_names});
        end"""

    return f"""// tb_cb.sv — AUTO-GENERATED by gen_cb_tb.py from '{ex}'. DO NOT EDIT.
//
// Whole-system, self-checking testbench for a TREE of ControlBoards + per-leaf
// StimReadout loops (RTL analog of test_control_board_stim.cpp). Events ripple DOWN
// (parent.out_ev -> child.ev), data/attempt/finish/post-select ripple UP, each leaf
// closes its own mmio_out -> StimReadout -> in_meas loop. START kicks the root. A
// multi-trial host FSM watches the ROOT: internal reject (discard) -> check the
// post-select oracle; clean completion (last_normal + wait_rounds rounds) -> compare
// recon[global_index] vs expected_dets, then gap_post_select to advance. The last shot
// ends the trial via host finish (clean copy-last) or reset. All mem files are flat +
// board-prefixed in mem/ and referenced by FILENAME ($readmemb), so run with the sim
// working directory set to mem/ (verilator) or add mem/*.mem to the Vivado sim set.
`timescale 1ns / 1ps

module tb_cb;
    localparam int HW_WIDTH = {HW_W};
    localparam int SHOTS = {g['SHOTS']}, SHOTS_AVAIL = {g['SHOTS_AVAIL']};
    localparam int TSTIM = {g['TSTIM']}, NDET = {g['NDET']}, WAIT_ROUNDS = {g['WAIT_ROUNDS']};
    localparam int TRIALS = {g['TRIALS']};
    localparam int ROOT_DOUT = {RD};
    localparam int SENSE_TIMEOUT = 12000;   // per-shot sense-loop guard

    logic clk = 1'b0;
    initial forever #5 clk = ~clk;
    logic rst;
{trace_off}
    // ROOT host controls (the only board the test drives)
    logic h_start, h_finish, h_gap_ps;

    // ============================ boards ============================
{insts}
    // ============================ tree wiring ============================
{wiring}
    // ============================ leaf readouts ============================
{readout}
{io_block}{lat_block}{fifo_block}    // ============================ expected ============================
    logic [NDET-1:0] exp_dets [SHOTS_AVAIL];
    logic            oracle_ps [SHOTS_AVAIL];
{orc_decl}
    initial begin
        $readmemb("board{root_id}_expected_dets.mem", exp_dets);
{orc_load}
        for (int s = 0; s < SHOTS_AVAIL; s++) oracle_ps[s] = {orc_expr};
    end

    // ============================ monitor state ============================
    int   recon [NDET];
    logic ln_seen, disc_seen, complete, fin_seen;
    int   post_ln_rounds;

    // one monitor step: sample the ROOT this cycle -> recon[global_index] = detector,
    // track last_normal / discard / out_finish / completion (mirrors monitor()).
    task automatic mon_step;
        automatic int real_used = 0;
        automatic logic ln_this = b{root_id}_last_normal;
        if (b{root_id}_discard)    disc_seen = 1'b1;
        if (ln_this)               ln_seen   = 1'b1;
        if (b{root_id}_out_finish) fin_seen  = 1'b1;
        if (b{root_id}_out_valid) begin
            for (int l = 0; l < ROOT_DOUT; l++) if (b{root_id}_out_used[l]) begin
                automatic int idx = int'(b{root_id}_out_gidx[l*HW_WIDTH +: HW_WIDTH]);
                if (idx >= 0 && idx < NDET) begin real_used++; recon[idx] = b{root_id}_out_det[l]; end
            end
            if (ln_seen && !ln_this && real_used > 0) post_ln_rounds++;
            if (ln_seen && post_ln_rounds >= WAIT_ROUNDS) complete = 1'b1;
        end
    endtask

    int checked = 0, passed = 0, det_mismatch = 0, det_missing = 0, ps_mismatch = 0;
    int internal = 0, finishes = 0, finish_attempts = 0, total_sync_mismatch = 0;

    initial begin
        automatic int w, fw, mism, miss;
        automatic logic exp_ps, was_discard, is_last;
        h_start = 0; h_finish = 0; h_gap_ps = 0;
        $display("tb_cb[{ex}]: boards={len(P)} leaves={len(leaves)} root=board{root_id} ps_fast={ps_fast} SHOTS=%0d T=%0d NDET=%0d wait_rounds=%0d trials=%0d",
                 SHOTS, TSTIM, NDET, WAIT_ROUNDS, TRIALS);

        rst = 1'b1; repeat (3) @(negedge clk); rst = 1'b0;   // power-on reset (once)

        for (int trial = 0; trial < TRIALS; trial++) begin
            @(negedge clk); h_start = 1'b1; @(negedge clk); h_start = 1'b0;   // pulse START

            for (int s = 0; s < SHOTS; s++) begin
                is_last = (s == SHOTS - 1);
                ln_seen = 0; disc_seen = 0; complete = 0; post_ln_rounds = 0;
                for (int d = 0; d < NDET; d++) recon[d] = -1;

                // sense: run until an internal reject or a clean completion
                w = 0;
                while (!disc_seen && !complete && w < SENSE_TIMEOUT) begin
                    @(posedge clk); mon_step(); w++;
                end
{sync_chk}
                exp_ps      = oracle_ps[s];
                was_discard = disc_seen && !complete;

                if (was_discard) begin                       // internal post-select reject
                    internal++;
                    if (!exp_ps) begin ps_mismatch++;
                        $display("  trial %0d shot %0d: discarded but oracle_ps=0", trial, s); end
                    repeat (80) @(negedge clk);              // let reset/advance settle
                end else begin                               // clean completion -> compare
                    mism = 0; miss = 0;
                    for (int d = 0; d < NDET; d++) begin
                        if (recon[d] < 0)                          miss++;
                        else if (recon[d] != int'(exp_dets[s][d])) mism++;
                    end
                    checked++; det_mismatch += mism; det_missing += miss;
                    if (mism == 0 && miss == 0) passed++;
                    else $display("  trial %0d shot %0d: det_mismatch=%0d det_missing=%0d", trial, s, mism, miss);
                    if (exp_ps) begin ps_mismatch++;
                        $display("  trial %0d shot %0d: ran clean but oracle_ps=1", trial, s); end
                    if (!is_last) begin                      // gap post-select -> discard + advance
                        @(negedge clk); h_gap_ps = 1'b1; @(negedge clk); h_gap_ps = 1'b0;
                        repeat (80) @(negedge clk);
                    end
                end

                if (is_last) begin
                    // End the trial like real operation: finish only an ACCEPTED (clean) attempt,
                    // and only when a saturating detector-emit round exists to ride (copy-last wait
                    // rows). Otherwise restart via reset.
                    if (!was_discard && WAIT_ROUNDS > 0) begin
                        finish_attempts++; fin_seen = 0;
                        @(negedge clk); h_finish = 1'b1; @(negedge clk); h_finish = 1'b0;
                        fw = 0;
                        while (!fin_seen && fw < SENSE_TIMEOUT) begin @(posedge clk); mon_step(); fw++; end
                        if (fin_seen) finishes++;
                        else $display("  trial %0d: out_finish never fired after finish", trial);
                        repeat (10) @(negedge clk);
                    end else begin
                        rst = 1'b1; repeat (2) @(negedge clk); rst = 1'b0;
                    end
                end
            end
        end

{fifo_report}        $display("tb_cb: trials=%0d shots/trial=%0d checked=%0d passed=%0d internal_ps=%0d finishes=%0d/%0d",
                 TRIALS, SHOTS, checked, passed, internal, finishes, finish_attempts);
        $display("tb_cb: det_mismatch=%0d det_missing=%0d ps_mismatch=%0d sync_mismatch=%0d",
                 det_mismatch, det_missing, ps_mismatch, total_sync_mismatch);
        if (passed == checked && checked > 0 && ps_mismatch == 0 && finishes == finish_attempts)
             $display("tb_cb: PASS");
        else $display("tb_cb: FAIL");
        $finish;
    end

endmodule
"""


if __name__ == "__main__":
    main()
