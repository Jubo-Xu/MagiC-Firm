
#!/usr/bin/env python3
"""gen_dcb_tb.py — generate a self-checking SystemVerilog testbench for a tree of
DetectorConstructBlocks from a compiled results/<example> directory.

RTL analog of emulator/tests/.../test_dcb_stim.cpp. SV parameters are compile-time,
so this bakes every board's dimensions (from manifest.json) into the generated tb and
copies each board's mem/ dir alongside it, producing a SELF-CONTAINED folder that can
be dropped into a Vivado project (add the .sv sources; set the simulation working
directory to this folder so the relative $readmemb paths resolve).

Handles both the MONOLITHIC case (single root+leaf board, no wiring) and the
DISTRIBUTED case (N boards, tree wiring from each parent's connections.json). Leaves
(qubit_scope) are driven by their own stim vectors; parents' inputs are wired from
their children's fwd_raw (measurements) / fwd_det (detectors), with the single-bit
finish broadcast. The root's output is reconstructed by global index and checked vs
expected_dets; stage boards' postselect is OR-checked vs expected_postselect.

Output: <out_dir>/<example_name>/
    tb_dcb.sv                 the generated, self-checking testbench
    board<id>/mem/*.mem       copy of each board's regfiles + stim vectors
    files.f                   filelist (module sources + tb) for verilator/xsim

Gaps:  --input-gap  idle cycles between measurement rounds within a shot
       --shot-gap   idle cycles between shots (after result captured + board reset)
Both bake tb defaults and are overridable at sim time via +input_gap / +shot_gap.

Usage: gen_dcb_tb.py <results/example_dir> [--out-dir DIR] [--input-gap G] [--shot-gap G] [--shots N]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil


def _load(path):
    with open(path) as f:
        return json.load(f)


def _rows(path):
    with open(path) as f:
        return sum(1 for ln in f if ln.strip())


def _board_params(bd, mem_src, man):
    rf = set(bd["regfiles"])
    copy = 1 if man.get("wait_row") == "copy-last" else 0
    # sync.mem may be absent when a board has kernels_avail (D_OUT width) but 0 real
    # detectors placed on them (spare capacity -> idle kernels). Fall back on meas_times.
    sync_path = os.path.join(mem_src, "sync.mem")
    T = _rows(sync_path) if os.path.exists(sync_path) else man["meas_times"] + copy
    p = {
        "id":      bd["board_id"],
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
        "IS_ROOT": 1 if bd["type"] == "root" else 0,
        "HAS_PS":  1 if "postselect" in rf else 0,
        "HAS_OSY": 1 if "output_sync" in rf else 0,
        "COPY":    copy,
        "is_leaf": "qubit_scope" in rf,
    }
    p["D_OUT"] = p["D_IN"] + p["K"]
    p["DIN_W"] = max(p["D_IN"], 1)
    p["RAW_W"] = max(p["RAW_OUT"], 1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("example_dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--input-gap", type=int, default=0)
    ap.add_argument("--shot-gap", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0)
    args = ap.parse_args()

    R = os.path.abspath(args.example_dir)
    man = _load(os.path.join(R, "manifest.json"))
    ex_name = os.path.basename(R.rstrip("/"))
    HW_W = man["global_index_hw_width"]
    TSTIM = man["meas_times"]
    NDET = man["detectors"]
    WAIT_ROUNDS = man.get("wait_rounds", 0)

    boards = {b["board_id"]: b for b in man["boards"]}
    root_bd = next(b for b in man["boards"] if b["type"] == "root")
    root_id = root_bd["board_id"]

    # ---- per-board params + role, and copy each board's mem/ dir ----
    out_root = args.out_dir or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "testbench", "control_system", "detector_construct", "generated")
    out = os.path.join(out_root, ex_name)
    if os.path.isdir(out):
        shutil.rmtree(out)
    os.makedirs(out)

    # all boards' mem files go FLAT into one mem/ dir, board-prefixed (board<id>_<name>.mem),
    # so Vivado can add them to the sim set (which loses folder structure) and $readmemb
    # resolves them by basename. The tb / block reference files by prefixed FILENAME only.
    mem_out = os.path.join(out, "mem")
    os.makedirs(mem_out)

    P = {}
    conns = {}
    for bid, bd in boards.items():
        mem_src = os.path.join(R, f"board{bid}", "mem")
        P[bid] = _board_params(bd, mem_src, man)
        for fn in os.listdir(mem_src):
            shutil.copy(os.path.join(mem_src, fn), os.path.join(mem_out, f"board{bid}_{fn}"))
        cj = os.path.join(R, f"board{bid}", "json", "connections.json")
        if os.path.exists(cj):
            conns[bid] = _load(cj)

    ed = _load(os.path.join(R, f"board{root_id}", "json", "expected_dets.json"))
    SHOTS_AVAIL = ed["shots"]
    SHOTS = SHOTS_AVAIL if args.shots <= 0 else min(args.shots, SHOTS_AVAIL)

    leaves  = [bid for bid in P if P[bid]["is_leaf"]]
    stages  = [bid for bid in P if P[bid]["HAS_PS"]]

    sv = _emit_tb(ex_name, P, conns, root_id, leaves, stages,
                  HW_W=HW_W, SHOTS=SHOTS, SHOTS_AVAIL=SHOTS_AVAIL, TSTIM=TSTIM,
                  NDET=NDET, WAIT_ROUNDS=WAIT_ROUNDS,
                  input_gap=args.input_gap, shot_gap=args.shot_gap)
    with open(os.path.join(out, "tb_dcb.sv"), "w") as f:
        f.write(sv)

    _write_expected(out, root_id, P[root_id], SHOTS, TSTIM, NDET, WAIT_ROUNDS,
                    args.input_gap, args.shot_gap, ex_name, HW_W)

    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "src", "control_system", "detector_construct"))
    lib = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "lib"))
    modules = ["DetectorConstructBlock", "RawSelector", "MeasurementSync", "Kernel",
               "DetectorPass", "OutputSync", "RootOutputSync", "Postselect"]
    with open(os.path.join(out, "files.f"), "w") as f:   # absolute paths: sim runs from mem/ (filename $readmemb)
        for m in modules:
            f.write(os.path.join(src, m + ".sv") + "\n")
        f.write(os.path.join(lib, "RegFileROM.sv") + "\n")
        f.write(os.path.join(out, "tb_dcb.sv") + "\n")

    print(f"  wrote {out}/tb_dcb.sv  ({len(P)} boards: {len(leaves)} leaves, root=board{root_id}, stages={stages})")
    print(f"  stim: shots={SHOTS}/{SHOTS_AVAIL} T={TSTIM} ndet={NDET} wait_rounds={WAIT_ROUNDS}")
    print(f"  mem:  {out}/mem/  (flat, board-prefixed; add to Vivado sim set -> $readmemb by filename)")
    print(f"  run:  cd {out}/mem && verilator --binary --timing -f {out}/files.f --top-module tb_dcb -o sim && ./obj_dir/sim")


# --------------------------------------------------------------------- golden reference

def _mem_ints(path):
    """each .mem row (MSB-first binary) -> integer (bit i = LSB-indexed bit i)."""
    with open(path) as f:
        return [int(ln.strip(), 2) for ln in f if ln.strip()]


def _write_expected(out, root_id, rp, SHOTS, TSTIM, NDET, WAIT_ROUNDS,
                    input_gap, shot_gap, ex, HW_W):
    """Golden reference: expected ROOT output per shot, per output round (= out_valid
    pulse, in emission order). Absolute cycle is data-dependent, so match by counting
    out_valid pulses. Values come from expected_dets; copy-last wait rounds have no
    ground truth (marked 'W')."""
    mem = os.path.join(out, "mem")
    pre = f"board{root_id}_"
    D_OUT, IDX_W, SENT = rp["D_OUT"], rp["IDX_W"], rp["SENT"]
    STRIDE, COPY = rp["STRIDE"], rp["COPY"]
    imask = (1 << IDX_W) - 1

    osync = _mem_ints(os.path.join(mem, pre + "output_sync.mem"))    # NDT rows, D_OUT bits
    gidx  = _mem_ints(os.path.join(mem, pre + "global_index.mem"))   # NDT rows, D_OUT*IDX_W bits
    exp   = _mem_ints(os.path.join(mem, pre + "expected_dets.mem"))  # SHOTS rows, NDET bits
    ndt   = len(osync)
    real_ndt = ndt - COPY          # real det-times (copy-last appends 1 saturating row)
    sat_row  = ndt - 1             # appended saturating row index (copy-last)

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

    # per-shot drive-cycle landmark (negedge index; APPROXIMATE — the drain is now event-driven:
    # after the drive we wait for out_finish, then shot_gap, then reset, then 1 cycle).
    per_round = 2 + input_gap
    FIN_EST = 24   # rough post-drive latency until out_finish (data-dependent)
    r_neg = TSTIM * per_round + (WAIT_ROUNDS * per_round if COPY else 0)
    shot_period = r_neg + FIN_EST + shot_gap + 3   # + ~finish + shot_gap + reset(2) + 1

    lines = [
        f"# Expected ROOT output for '{ex}'  (generated by gen_dcb_tb.py)",
        f"# D_OUT={D_OUT} IDX_W={IDX_W} det-times={ndt} (real={real_ndt}{', +1 copy-last wait row' if COPY else ''})",
        f"#",
        f"# Rounds are listed in EMISSION ORDER = successive out_valid pulses. The absolute",
        f"# cycle of each pulse is data-dependent (FIFO/sync latency), so count out_valid",
        f"# pulses in the waveform to match. Each used line: 'line L -> gidx G = value'.",
        f"# gaps: input_gap={input_gap} shot_gap={shot_gap}. Clk period 10ns; a shot's drive",
        f"# begins ~negedge {3} + shot*{shot_period} (APPROX: reset 3 at start; per shot = drive",
        f"# + wait-for-finish + shot_gap({shot_gap}) + reset(2) + 1). Output pulses trail the drive.",
        "",
    ]
    for s in range(SHOTS):
        start = 3 + s * shot_period
        lines.append(f"======================== SHOT {s}  (drive starts ~negedge {start}) ========================")
        for nt in range(real_ndt):
            lines.append(round_line(s, nt))
        if COPY:
            for w in range(WAIT_ROUNDS):
                lines.append(round_line(s, sat_row, offset=w * STRIDE, tag=f"wait{w}"))
        lines.append("")

    # ---- SECTION 2: ALL root DUT output signal values per out_valid pulse (compare to waveform) ----
    allones = (1 << HW_W) - 1
    rm = _mem_ints(os.path.join(mem, pre + "round_marker.mem"))        # NDT rows, 3 bits
    ps_path = os.path.join(mem, pre + "postselect.mem")
    ps = _mem_ints(ps_path) if os.path.exists(ps_path) else None  # NDT rows, D_OUT bits (stage only)

    # emission order: real det-times, then (copy-last) the saturating row repeated WAIT_ROUNDS times
    emission = [(nt, 0, False, None) for nt in range(real_ndt)]
    if COPY:
        emission += [(sat_row, w * STRIDE, True, w) for w in range(WAIT_ROUNDS)]

    def sig_round(s, nt, offset, is_sat, w, prev_wait):
        m, g = osync[nt], gidx[nt]
        r = rm[nt]
        is_first, is_last, is_wait = r & 1, (r >> 1) & 1, (r >> 2) & 1
        first_wait = 1 if (is_wait and not prev_wait) else 0
        finish = (1 if w == WAIT_ROUNDS - 1 else 0) if is_sat else (1 if (not COPY and nt == real_ndt - 1) else 0)
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
               f"first_normal={is_first} last_normal={is_last} first_wait={first_wait} last_wait={last_wait}"
               + (f" post_select={'?' if is_sat else ps_fire}" if ps is not None else ""))
        return [hdr,
                f"  out_used            = {format(m, f'0{D_OUT}b')}",
                f"  out_det             = {det}",
                f"  out_global_indexes  = {' '.join(str(x) for x in gi)}"], is_wait

    lines.append("")
    lines.append("============================================================================")
    lines.append("# SECTION 2 — expected ROOT DUT output signals per out_valid pulse (compare to the waveform).")
    lines.append(f"# out_used / out_det: BINARY, MSB = line {D_OUT-1} (as Vivado shows a [{D_OUT-1}:0] vector).")
    lines.append(f"# out_global_indexes: DECIMAL per detector, line 0 first (matches the gi_out[l] signals);")
    lines.append(f"#   an unused / sentinel slot reads {allones} (all-ones, {HW_W} bits).")
    lines.append("# out_valid / out_finish / first_normal / last_normal / first_wait / last_wait / post_select:")
    lines.append("#   1-bit, on the header line. out_finish/last_wait assume finish rides the LAST driven round")
    lines.append("#   (last real round for wr=0; last wait round for copy-last) -> its detectors' emit round.")
    lines.append("============================================================================")
    for s in range(SHOTS):
        lines.append(f"-------------------------------- SHOT {s} --------------------------------")
        prev_wait = 0
        for (nt, off, is_sat, w) in emission:
            body, prev_wait = sig_round(s, nt, off, is_sat, w, prev_wait)
            lines += body
        lines.append("")

    with open(os.path.join(out, "expected_output.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------------- emit

def _inst(p):
    """per-board signals + DetectorConstructBlock instance."""
    b = p["id"]
    s = f"""
    // ---- board{b} ({'leaf ' if p['is_leaf'] else ''}{'root ' if p['IS_ROOT'] else ''}{'stage ' if p['HAS_PS'] else ''}{'router' if not p['is_leaf'] and not p['IS_ROOT'] else ''}) ----
    logic [{p['M']-1}:0]           b{b}_in_meas, b{b}_in_valid, b{b}_in_meas_finish;
    logic [{p['DIN_W']-1}:0]       b{b}_in_det, b{b}_in_det_valid, b{b}_in_det_finish;
    logic [{p['D_OUT']-1}:0]       b{b}_fwd_det, b{b}_fwd_det_valid;
    logic                          b{b}_fwd_det_finish;
    logic [{p['RAW_W']-1}:0]       b{b}_fwd_raw, b{b}_fwd_raw_valid;
    logic                          b{b}_fwd_raw_finish;
    logic [{p['D_OUT']-1}:0]       b{b}_out_det, b{b}_out_used;
    logic                          b{b}_out_valid, b{b}_out_finish;
    logic [{p['D_OUT']*HW_GLOBAL-1}:0] b{b}_out_gi;
    logic                          b{b}_first_normal, b{b}_last_normal, b{b}_first_wait, b{b}_last_wait, b{b}_post_select;

    DetectorConstructBlock #(
        .M({p['M']}), .D_IN({p['D_IN']}), .K({p['K']}), .N({p['N']}), .H({p['H']}), .RAW_OUT({p['RAW_OUT']}),
        .IDX_W({p['IDX_W']}), .HW_WIDTH({HW_GLOBAL}), .STRIDE({p['STRIDE']}), .SENTINEL({p['SENT']}),
        .T({p['T']}), .NDT({p['NDT']}), .PC_W(16), .SYNC_FIFO(64), .OUT_FIFO(64),
        .IS_ROOT(1'b{p['IS_ROOT']}), .HAS_PS(1'b{p['HAS_PS']}), .HAS_OSYNC(1'b{p['HAS_OSY']}), .COPY_LAST(1'b{p['COPY']}),
        .MEM_DIR("board{b}_")
    ) b{b} (
        .clk(clk), .rst(rst),
        .in_meas(b{b}_in_meas), .in_valid(b{b}_in_valid), .in_meas_finish(b{b}_in_meas_finish),
        .in_det(b{b}_in_det), .in_det_valid(b{b}_in_det_valid), .in_det_finish(b{b}_in_det_finish),
        .fwd_det(b{b}_fwd_det), .fwd_det_valid(b{b}_fwd_det_valid), .fwd_det_finish(b{b}_fwd_det_finish),
        .fwd_raw(b{b}_fwd_raw), .fwd_raw_valid(b{b}_fwd_raw_valid), .fwd_raw_finish(b{b}_fwd_raw_finish),
        .out_det(b{b}_out_det), .out_used(b{b}_out_used), .out_valid(b{b}_out_valid), .out_finish(b{b}_out_finish),
        .out_global_indexes(b{b}_out_gi),
        .first_normal(b{b}_first_normal), .last_normal(b{b}_last_normal),
        .first_wait(b{b}_first_wait), .last_wait(b{b}_last_wait), .post_select(b{b}_post_select)
    );
"""
    return s


def _wiring(p, conn):
    """parent inputs <- children fwd buses (measurements from fwd_raw, detectors from fwd_det)."""
    b = p["id"]
    lines = [f"    // board{b} tree wiring (inputs from children)"]
    for e in conn.get("measurement_ports", []):
        if e.get("child") is None:
            continue
        c, cp, port = e["child"], e["child_port"], e["port"]
        lines.append(f"    assign b{b}_in_meas[{port}]        = b{c}_fwd_raw[{cp}];")
        lines.append(f"    assign b{b}_in_valid[{port}]       = b{c}_fwd_raw_valid[{cp}];")
        lines.append(f"    assign b{b}_in_meas_finish[{port}] = b{c}_fwd_raw_finish;")   # single-bit broadcast
    for e in conn.get("detector_ports", []):
        if e.get("child") is None:
            continue
        c, cp, port = e["child"], e["child_port"], e["port"]
        lines.append(f"    assign b{b}_in_det[{port}]        = b{c}_fwd_det[{cp}];")
        lines.append(f"    assign b{b}_in_det_valid[{port}]  = b{c}_fwd_det_valid[{cp}];")
        lines.append(f"    assign b{b}_in_det_finish[{port}] = b{c}_fwd_det_finish;")     # single-bit broadcast
    return "\n".join(lines) + "\n"


def _tie_leaf_det(p):
    """leaf boards have no children: tie the (unused) detector inputs to 0."""
    b = p["id"]
    return (f"    assign b{b}_in_det = '0; assign b{b}_in_det_valid = '0; assign b{b}_in_det_finish = '0;\n")


def _leaf_drive(leaves, P, round_expr, finish_expr):
    """SV statements to drive every leaf's meas/valid/finish for one round."""
    out = []
    for b in leaves:
        out.append(f"            b{b}_in_meas  = b{b}_in_meas_vec [{round_expr}];")
        out.append(f"            b{b}_in_valid = b{b}_in_valid_vec[{round_expr}];")
        out.append(f"            b{b}_in_meas_finish = {finish_expr.format(b=b)};")
    return "\n".join(out)


def _leaf_clear(leaves):
    out = []
    for b in leaves:
        out.append(f"            b{b}_in_meas = '0; b{b}_in_valid = '0; b{b}_in_meas_finish = '0;")
    return "\n".join(out)


def _emit_tb(ex, P, conns, root_id, leaves, stages, **g):
    global HW_GLOBAL
    HW_GLOBAL = g["HW_W"]
    root = P[root_id]
    RD = root["D_OUT"]

    insts   = "".join(_inst(P[bid]) for bid in sorted(P))
    wiring  = ""
    for bid in sorted(P):
        if P[bid]["is_leaf"]:
            wiring += _tie_leaf_det(P[bid])
        elif bid in conns:
            wiring += _wiring(P[bid], conns[bid])

    # leaf stim vector declarations
    leaf_decl = []
    for b in leaves:
        m = P[b]["M"]
        leaf_decl.append(f"    logic [{m-1}:0] b{b}_in_meas_vec  [SHOTS_AVAIL*TSTIM];")
        leaf_decl.append(f"    logic [{m-1}:0] b{b}_in_valid_vec [SHOTS_AVAIL*TSTIM];")
    leaf_load = []
    for b in leaves:
        leaf_load.append(f'        $readmemb("board{b}_in_meas.mem",  b{b}_in_meas_vec);')
        leaf_load.append(f'        $readmemb("board{b}_in_valid.mem", b{b}_in_valid_vec);')

    # stage postselect: expected + per-shot accumulator
    stage_decl, stage_load, stage_acc, stage_reset, stage_check = [], [], [], [], []
    for b in stages:
        stage_decl.append(f"    logic [0:0] b{b}_exp_ps [SHOTS_AVAIL];")
        stage_decl.append(f"    logic       b{b}_ps_fired;")
        stage_load.append(f'        $readmemb("board{b}_expected_postselect.mem", b{b}_exp_ps);')
        stage_acc.append(f"    always @(posedge clk) if (!rst && b{b}_post_select) b{b}_ps_fired <= 1'b1;")
        stage_reset.append(f"        b{b}_ps_fired = 1'b0;")
        stage_check.append(
            f"            if (b{b}_ps_fired !== b{b}_exp_ps[s][0]) begin total_ps_mismatch++;\n"
            f'                $display("  shot %0d board{b}: postselect emu=%0b exp=%0b", s, b{b}_ps_fired, b{b}_exp_ps[s][0]); end')

    real_finish = "(!COPY_LAST && t == TSTIM-1) ? b{b}_in_valid_vec[s*TSTIM + t] : '0"
    wait_finish = "(w == WAIT_ROUNDS-1) ? b{b}_in_valid_vec[s*TSTIM + (TSTIM-1)] : '0"

    return f"""// tb_dcb.sv — AUTO-GENERATED by gen_dcb_tb.py from '{ex}'. DO NOT EDIT.
//
// Self-checking testbench for a tree of DetectorConstructBlocks: drives the leaf
// stim vectors, wires each parent from its children, reconstructs recon[global_index]
// = out_det at the root over each shot, and checks vs expected_dets (+ stage
// postselect vs expected_postselect) -> PASS/FAIL. Gaps overridable via +input_gap= /
// +shot_gap=. All mem files are flat + board-prefixed in mem/ and referenced by FILENAME
// only ($readmemb), so add mem/*.mem to the Vivado sim set (folder structure not needed);
// for verilator, run the sim with the working directory set to mem/.
`timescale 1ns / 1ps

module tb_dcb;
    localparam int HW_WIDTH = {HW_GLOBAL};
    localparam int SHOTS = {g['SHOTS']}, SHOTS_AVAIL = {g['SHOTS_AVAIL']};
    localparam int TSTIM = {g['TSTIM']}, NDET = {g['NDET']}, WAIT_ROUNDS = {g['WAIT_ROUNDS']};
    localparam int MAX_WAIT = 4000;   // safety timeout waiting for out_finish (normal path exits in ~10-20)
    localparam int ROOT_DOUT = {RD};
    localparam bit COPY_LAST = 1'b{root['COPY']};

    int input_gap = {g['input_gap']};
    int shot_gap  = {g['shot_gap']};

    logic clk = 1'b0;
    initial forever #5 clk = ~clk;
    logic rst;

    // ============================ boards ============================
{insts}
    // ============================ tree wiring ============================
{wiring}
    // ============================ leaf stim ============================
{chr(10).join(leaf_decl)}
    initial begin
{chr(10).join(leaf_load)}
    end

    // ============================ root reconstruction ============================
    logic [HW_WIDTH-1:0] gi_out [ROOT_DOUT];
    for (genvar d = 0; d < ROOT_DOUT; d++) begin : g_gi
        assign gi_out[d] = b{root_id}_out_gi[d*HW_WIDTH +: HW_WIDTH];
    end

    int  recon [NDET];
    logic shot_finish_seen;
    always @(posedge clk) if (!rst && b{root_id}_out_valid) begin
        for (int l = 0; l < ROOT_DOUT; l++) if (b{root_id}_out_used[l]) begin
            automatic int idx = int'(gi_out[l]);
            if (idx >= 0 && idx < NDET) recon[idx] = b{root_id}_out_det[l];
        end
    end
    // single driver: set on the shot's out_finish, cleared by the inter-shot reset (rst)
    always @(posedge clk)
        if (rst)                                                    shot_finish_seen <= 1'b0;
        else if (b{root_id}_out_valid && b{root_id}_out_finish)     shot_finish_seen <= 1'b1;

    // ============================ expected ============================
    logic [NDET-1:0] exp_dets [SHOTS_AVAIL];
    initial $readmemb("board{root_id}_expected_dets.mem", exp_dets);
{chr(10).join(stage_decl)}
    initial begin
{chr(10).join(stage_load) if stage_load else "        // no stage boards"}
    end
{chr(10).join(stage_acc)}

    int total_mismatch = 0, total_missing = 0, total_ps_mismatch = 0, shots_run = 0, shots_finish = 0;

    task automatic run_shot(input int s);
        for (int d = 0; d < NDET; d++) recon[d] = -1;   // (shot_finish_seen cleared by rst between shots)
{chr(10).join(stage_reset)}
        // real rounds
        for (int t = 0; t < TSTIM; t++) begin
            @(negedge clk);
{_leaf_drive(leaves, P, "s*TSTIM + t", real_finish)}
            @(negedge clk);
{_leaf_clear(leaves)}
            repeat (input_gap) @(negedge clk);
        end
        // copy-last: extra saturating wait rounds (re-feed last real round; finish on the last)
        for (int w = 0; COPY_LAST && w < WAIT_ROUNDS; w++) begin
            @(negedge clk);
{_leaf_drive(leaves, P, "s*TSTIM + (TSTIM-1)", wait_finish)}
            @(negedge clk);
{_leaf_clear(leaves)}
            repeat (input_gap) @(negedge clk);
        end
        // drain is event-driven: wait for the shot's out_finish (last detector round emitted),
        // bounded by MAX_WAIT. The finish round's detectors are scattered at that posedge.
        begin
            automatic int guard = 0;
            while (!shot_finish_seen && guard < MAX_WAIT) begin @(negedge clk); guard++; end
        end
        @(negedge clk);   // settle: let the finish round's scatter land in recon
        begin
            automatic int mism = 0, miss = 0;
            for (int d = 0; d < NDET; d++) begin
                if (recon[d] < 0)                          miss++;
                else if (recon[d] != int'(exp_dets[s][d])) mism++;
            end
            total_mismatch += mism; total_missing += miss; shots_run++;
            if (shot_finish_seen) shots_finish++;
            if (mism || miss) $display("  shot %0d: mismatch=%0d missing=%0d", s, mism, miss);
{chr(10).join(stage_check)}
        end
    endtask

    initial begin
        void'($value$plusargs("input_gap=%d", input_gap));
        void'($value$plusargs("shot_gap=%d",  shot_gap));
        $display("tb_dcb[{ex}]: boards={len(P)} leaves={len(leaves)} root=board{root_id} SHOTS=%0d T=%0d NDET=%0d input_gap=%0d shot_gap=%0d",
                 SHOTS, TSTIM, NDET, input_gap, shot_gap);
{_leaf_clear(leaves).replace("            ", "        ")}
        rst = 1'b1; repeat (3) @(negedge clk); rst = 1'b0;

        for (int s = 0; s < SHOTS; s++) begin
            run_shot(s);                             // drives + waits for out_finish + checks
            repeat (shot_gap) @(negedge clk);        // gap AFTER finish
            rst = 1'b1; repeat (2) @(negedge clk); rst = 1'b0;   // reset for next shot
            @(negedge clk);                          // one cycle later -> next shot's data
        end

        $display("tb_dcb: shots=%0d det_mismatch=%0d det_missing=%0d postselect_mismatch=%0d finish_ok=%0d/%0d",
                 shots_run, total_mismatch, total_missing, total_ps_mismatch, shots_finish, shots_run);
        if (total_mismatch == 0 && total_missing == 0 && total_ps_mismatch == 0 && shots_finish == shots_run)
             $display("tb_dcb: PASS");
        else $display("tb_dcb: FAIL");
        $finish;
    end

endmodule
"""


if __name__ == "__main__":
    main()
