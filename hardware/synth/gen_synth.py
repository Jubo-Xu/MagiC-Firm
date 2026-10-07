#!/usr/bin/env python3
"""gen_synth.py — build a self-contained Vivado synthesis bundle for the ControlBoards of one
compiled tree (results/<example> or out/control_system/<circuit>/compiled*/<tree>).

Per selected board it emits a wrapper module that instantiates ControlBoard with that board's
literal parameters (the same derivation as gen_cb_tb.py, so the synthesized board is the one
the whole-system tb validated), a flat copy of the board's regfiles, and a Vivado tcl that runs
synth (out-of-context) + opt + place + route and writes utilization/timing reports plus a JSON
summary. A driver script runs the tcl files on the Vivado host.

Board selection (--select): the paper reports, per board TYPE (leaf / router / root), the
maximum resource usage and the minimum Fmax. Routers and the root are few, so all of them are
synthesized. Leaves are many; only the ones that are maximal in at least one size parameter
(the Pareto front over M, K, N, H, RAW_OUT) can be the maximum, so only those are synthesized.
--select all synthesizes every board.

Usage: gen_synth.py <tree_dir> --out-dir DIR [--part xcvu19p-fsva3824-2-e] [--fifo 4]
                    [--period-ns 2.0] [--select pareto|all]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rtl", "scripts"))
from gen_cb_tb import _board_params, _load, _arr, MAX_CHILD   # noqa: E402

SIZE_KEYS = ("M", "D_IN", "K", "N", "H", "RAW_OUT", "T", "NDT")


def _role(p):
    return "root" if p["IS_ROOT"] else ("leaf" if p["is_leaf"] else "router")


def select_boards(P, mode="pareto"):
    """Board ids to synthesize, grouped by role. 'pareto': every router and the root, plus the
    leaves not dominated (<= in every SIZE_KEY, < in at least one) by another leaf. 'all':
    everything. Returns {role: [ids]} in ascending id order."""
    by_role = {"leaf": [], "router": [], "root": []}
    for b in sorted(P):
        by_role[_role(P[b])].append(b)
    if mode == "all":
        return by_role

    def dominated(b, others):
        pb = P[b]
        for o in others:
            if o == b:
                continue
            po = P[o]
            if all(po[k] >= pb[k] for k in SIZE_KEYS) and any(po[k] > pb[k] for k in SIZE_KEYS):
                return True
        return False

    leaves = by_role["leaf"]
    by_role["leaf"] = [b for b in leaves if not dominated(b, leaves)]
    return by_role


def _param_list(p, HW_W, fifo, mem_prefix):
    """ControlBoard parameter overrides for board p — the same list gen_cb_tb._inst emits,
    with the FIFO depths from --fifo and MEM_DIR = the flat regfile prefix (the tcl cd's
    into the bundle's mem/ dir before synth, so $readmemb resolves the bare filenames)."""
    par = [f".IS_ROOT(1'b{p['IS_ROOT']})", f".IS_LEAF(1'b{p['IS_LEAF']})",
           f".EVENT_MODE({p['EVENT_MODE']})", f".DATA_SRC({p['DATA_SRC']})", f".HAS_PS(1'b{p['HAS_PS']})"]
    if p["NCHILD"] > 0:
        par += [f".NCHILD({p['NCHILD']})", f".CHILD_DW({_arr(p['CHILD_DW'])})",
                f".CHILD_RAW({_arr(p['CHILD_RAW'])})"]
    if p.get("NPS", 0) > 0:
        par += [f".NPS({p['NPS']})"]
    par += [f".M({p['M']})", f".D_IN({p['D_IN']})", f".K({p['K']})", f".N({p['N']})", f".H({p['H']})",
            f".RAW_OUT({p['RAW_OUT']})", f".IDX_W({p['IDX_W']})", f".HW_WIDTH({HW_W})",
            f".STRIDE({p['STRIDE']})", f".SENTINEL({p['SENT']})", f".T({p['T']})", f".NDT({p['NDT']})",
            f".PC_W(16)", f".SYNC_FIFO({fifo})", f".OUT_FIFO({fifo})",
            f".HAS_OSYNC(1'b{p['HAS_OSY']})", f".COPY_LAST(1'b{p['COPY']})",
            f'.MEM_DIR("{mem_prefix}")', f".PAYLOAD_W(2)"]
    if p["is_leaf"]:
        par += [f".P({p['P']})", f".INSTR_DEPTH({p['INSTR_DEPTH']})", f".CW_DEPTH({p['CW_DEPTH']})",
                f".DATA_W({p['DATA_W']})", f".WT_W({p['WT_W']})"]
    return par


def _wrapper(p, HW_W, fifo, mem_prefix):
    """cb_board<N>.sv: synthesis top with ControlBoard's exact port list at this board's
    widths, instantiating ControlBoard with the literal parameters (Vivado -generic cannot
    pass the CHILD_DW/CHILD_RAW unpacked arrays, hence a wrapper)."""
    b = p["id"]
    DINW, M, DOUT, RAWW = p["DIN_W"], p["M"], p["D_OUT"], p["RAW_W"]
    NCW, NPSW = max(p["NCHILD"], 1), max(p.get("NPS", 0), 1)
    PW, DW = p["P_W"], p["DATA_W"]
    ports = [
        ("input", 1, "clk"), ("input", 1, "rst"),
        ("input", 1, "ev_valid"), ("input", 2, "ev_type"), ("input", 2, "ev_payload"),
        ("input", 1, "ev_attempt"),
        ("input", 1, "start"), ("input", 1, "finish"), ("input", 1, "gap_post_select"),
        ("input", DINW, "in_det"), ("input", DINW, "in_det_valid"),
        ("input", M, "in_meas"), ("input", M, "in_meas_valid"), ("input", M, "in_meas_finish"),
        ("input", NCW, "in_det_finish_child"), ("input", NCW, "in_raw_finish_child"),
        ("input", NCW, "in_child_attempt"),
        ("input", NPSW, "ps_in"), ("input", NPSW, "ps_in_attempt"),
        ("output", DOUT, "fwd_det"), ("output", DOUT, "fwd_det_valid"), ("output", 1, "fwd_det_finish"),
        ("output", RAWW, "fwd_raw"), ("output", RAWW, "fwd_raw_valid"), ("output", 1, "fwd_raw_finish"),
        ("output", 1, "out_attempt"), ("output", 1, "ps_out"), ("output", 1, "ps_out_attempt"),
        ("output", 1, "out_ev_valid"), ("output", 2, "out_ev_type"), ("output", 2, "out_ev_payload"),
        ("output", 1, "out_ev_attempt"),
        ("output", DOUT, "out_det"), ("output", DOUT, "out_used"), ("output", 1, "out_valid"),
        ("output", 1, "out_finish"), ("output", DOUT * HW_W, "out_global_indexes"),
        ("output", 1, "first_normal"), ("output", 1, "last_normal"), ("output", 1, "first_wait"),
        ("output", 1, "last_wait"), ("output", 1, "discard"),
        ("output", PW, "cw_gen_finish"), ("output", PW * DW, "mmio_out_data"),
        ("output", PW, "mmio_out_valid"),
    ]
    decls = []
    for d, w, n in ports:
        rng = "" if w == 1 else f"[{w - 1}:0] "
        decls.append(f"    {d:6s} logic {rng}{n}")
    port_decls = ",\n".join(decls)
    conns = ",\n        ".join(f".{n}({n})" for _, _, n in ports)
    pars = ",\n        ".join(_param_list(p, HW_W, fifo, mem_prefix))
    return f"""// cb_board{b}.sv — AUTO-GENERATED by gen_synth.py. DO NOT EDIT.
// Synthesis top for board{b} ({_role(p)}): ControlBoard with this board's literal parameters.
`timescale 1ns / 1ps
module cb_board{b} (
{port_decls}
);
    ControlBoard #(
        {pars}
    ) u_cb (
        {conns}
    );
endmodule
"""


TCL_TEMPLATE = r"""# board@BOARD@.tcl — AUTO-GENERATED by gen_synth.py. DO NOT EDIT.
# Vivado batch flow for board@BOARD@ (@ROLE@): out-of-context synth + opt + place + phys_opt +
# route; reports + summary.json under <bundle>/runs/board@BOARD@/. Fmax method: the clock target
# is aggressive on purpose, so the post-route WNS is (slightly) negative and
# Fmax = 1 / (period - WNS); a confirmation run at that period closes timing.
set here   [file dirname [file normalize [info script]]]
set bundle [file dirname $here]
set top    cb_board@BOARD@
set run    [file join $bundle runs board@BOARD@]
file mkdir $run
set_param general.maxThreads @THREADS@
# $readmemb resolves the bare board@BOARD@_*.mem filenames relative to the cwd
cd [file join $bundle mem]
foreach f [lsort [glob [file join $bundle src *.sv]]] { read_verilog -sv $f }
read_verilog -sv [file join $bundle boards $top.sv]
synth_design -top $top -part @PART@ -mode out_of_context
create_clock -period @PERIOD@ -name clk [get_ports clk]
write_checkpoint -force [file join $run post_synth.dcp]
report_utilization -file [file join $run util_synth.rpt]
opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $run post_route.dcp]
report_utilization -file [file join $run util_route.rpt]
report_utilization -hierarchical -hierarchical_depth 3 -file [file join $run util_hier.rpt]
report_timing_summary -max_paths 5 -file [file join $run timing.rpt]
report_timing -max_paths 3 -nworst 1 -file [file join $run critical_path.rpt]
set path [get_timing_paths -max_paths 1 -nworst 1 -setup]
set wns  [get_property SLACK $path]
set fmax [format %.2f [expr {1000.0 / (@PERIOD@ - $wns)}]]
set f [open [file join $run summary.json] w]
puts $f "{\"board\": @BOARD@, \"role\": \"@ROLE@\", \"top\": \"$top\", \"part\": \"@PART@\", \"period_ns\": @PERIOD@, \"wns_ns\": $wns, \"fmax_mhz\": $fmax, \"crit_start\": \"[get_property STARTPOINT_PIN $path]\", \"crit_end\": \"[get_property ENDPOINT_PIN $path]\"}"
close $f
puts "SYNTH_DONE board@BOARD@ wns=$wns fmax=$fmax"
"""


def _tcl(p, part, period_ns, threads):
    """Vivado batch script for one board (see TCL_TEMPLATE)."""
    return (TCL_TEMPLATE.replace("@BOARD@", str(p["id"])).replace("@ROLE@", _role(p))
            .replace("@PART@", part).replace("@PERIOD@", f"{period_ns:g}")
            .replace("@THREADS@", str(threads)))


RTL_MODULES = {
    "lib": ("RegFileROM", "SyncROM"),
    "control_system/detector_construct": ("RawSelector", "MeasurementSync", "Kernel", "DetectorPass",
                                          "OutputSync", "RootOutputSync", "Postselect",
                                          "DetectorConstructBlock"),
    "control_system/cultiv_control": ("InstrDecode", "InstrUnpack", "InstrSequencer", "DrainAggregator",
                                      "PhysicalMMIO", "BoardControl"),
    "control_system": ("ControlBoard",),
}
SIM_ONLY_MEMS = ("sim_readout_", "expected_", "in_meas", "in_valid")   # tb stimulus, not hardware


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree_dir", help="compiled tree dir (contains manifest.json + board<N>/)")
    ap.add_argument("--out-dir", required=True, help="bundles go to <out-dir>/<name>/")
    ap.add_argument("--name", default=None, help="bundle folder name (default: the tree dir's basename)")
    ap.add_argument("--part", default="xcvu19p-fsva3824-2-e")
    ap.add_argument("--fifo", type=int, default=4, help="SYNC_FIFO = OUT_FIFO depth (measured need: 1)")
    ap.add_argument("--period-ns", type=float, default=2.0, help="clock target (aggressive on purpose)")
    ap.add_argument("--threads", type=int, default=8, help="Vivado maxThreads per run")
    ap.add_argument("--select", choices=["pareto", "all"], default="pareto")
    args = ap.parse_args()

    R = os.path.abspath(args.tree_dir)
    man = _load(os.path.join(R, "manifest.json"))
    HW_W = man["global_index_hw_width"]
    root_id = next(b["board_id"] for b in man["boards"] if b["type"] == "root")
    P = {b["board_id"]: _board_params(R, b, os.path.join(R, f"board{b['board_id']}", "mem"), man)
         for b in man["boards"]}
    # root post-select fast path width = stage boards other than the root (as gen_cb_tb)
    stage_map = {int(k): v for k, v in man.get("stage_boards", {}).items()}
    P[root_id]["NPS"] = sum(1 for k in sorted(stage_map) if stage_map[k] != root_id)
    sel = select_boards(P, args.select)
    chosen = [b for role in ("leaf", "router", "root") for b in sel[role]]

    out = os.path.join(os.path.abspath(args.out_dir), args.name or os.path.basename(R.rstrip("/")))
    if os.path.isdir(out):
        shutil.rmtree(out)
    for d in ("src", "boards", "mem", "tcl"):
        os.makedirs(os.path.join(out, d))

    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    for sub, mods in RTL_MODULES.items():
        for m in mods:
            shutil.copy(os.path.join(src, sub, m + ".sv"), os.path.join(out, "src", m + ".sv"))

    for b in chosen:
        mem_src = os.path.join(R, f"board{b}", "mem")
        for fn in os.listdir(mem_src):
            if fn.endswith(".mem") and not fn.startswith(SIM_ONLY_MEMS):
                shutil.copy(os.path.join(mem_src, fn), os.path.join(out, "mem", f"board{b}_{fn}"))
        with open(os.path.join(out, "boards", f"cb_board{b}.sv"), "w") as f:
            f.write(_wrapper(P[b], HW_W, args.fifo, f"board{b}_"))
        with open(os.path.join(out, "tcl", f"board{b}.tcl"), "w") as f:
            f.write(_tcl(P[b], args.part, args.period_ns, args.threads))

    info = {"tree": os.path.basename(R.rstrip("/")), "circuit": man["circuit"], "config": man["config"],
            "part": args.part, "period_ns": args.period_ns, "fifo": args.fifo, "select": args.select,
            "selected": sel,
            "boards": {b: {"role": _role(P[b]), **{k: P[b][k] for k in SIZE_KEYS},
                           "D_OUT": P[b]["D_OUT"], "NCHILD": P[b]["NCHILD"],
                           "HAS_PS": P[b]["HAS_PS"], "NPS": P[b].get("NPS", 0)} for b in chosen}}
    with open(os.path.join(out, "boards.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"  wrote {out}: {len(chosen)} boards  leaf={sel['leaf']} router={sel['router']} "
          f"root={sel['root']}  part={args.part} period={args.period_ns}ns fifo={args.fifo}")


if __name__ == "__main__":
    main()
