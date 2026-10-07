#!/usr/bin/env python3
"""cli.py — general entry point for the control-system compiler.

The compiler is a four-stage pipeline (see README.md):

    parser.py    DetectorConstructionScanner   stim circuit -> parsed structs
    config.py    CompilerConfig                hardware board-tree config
    compiler.py  Compiler                      (config + circuit) -> in-memory
                                               per-board programs   [pure library]
    serializer.py Serializer                   programs -> per-board regfiles on disk

`compiler.py` is deliberately a pure, file-I/O-free library, and `serializer.py`
is the *serialization* stage — neither is a natural front door. This CLI owns the
orchestration and is the single place new generation stages plug in:

    1. detector-construct regfiles         (serializer.write, always)
    2. MMIO instruction regfile + CW mem   (--instr-reg / --cw-mem, Phase B)
    3. stim simulation measurement mems    (--sim, Phase C)

Stages 2 and 3 exist for simulation/testing today (the "sim" type); real
deployment types come later. This is the detector-construct compiler for now;
the folder will be renamed to mirror emulator/rtl `control_system/`.

Usage:
    python cli.py <circuit.stim> <config.json> [detector opts] [sim opts]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gen_instr_cw
import gen_sim
from compiler import Compiler
from config import CompilerConfig
from parser import DetectorConstructionScanner
from serializer import Serializer


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Compile a stim circuit + hardware config into per-board programs.")

    # ---- inputs ----
    ap.add_argument("circuit", help="stim circuit (.stim)")
    ap.add_argument("config", help="hardware board-tree config (.json)")

    # ---- detector-construct serialization (mirrors serializer.py) ----
    ap.add_argument("--mem", choices=["hex", "bin"], default="bin",
                    help="regfile .mem radix (default bin; everything downstream reads binary)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
                    help="output root dir (default <compiler>/results, CWD-independent)")
    ap.add_argument("--postselect-layout", choices=["flat", "per_stage"], default="flat",
                    help="multi-stage postselect regfile layout (default flat)")
    ap.add_argument("--output-sync", choices=["all", "output-only"], default="all",
                    help="which boards get an output_sync regfile (default all)")
    ap.add_argument("--wait-rounds", type=int, default=0,
                    help="trailing wait rounds baked into the circuit (= r2). 0 = none.")
    ap.add_argument("--wait-row", choices=["normal", "copy-last"], default="normal",
                    help="'normal' = no appended row; 'copy-last' = append a saturating "
                         "wait row (also adds one instr/CW/measurement entry, Phase B/C).")

    # ---- MMIO command generation (Phase B; only the 'sim' type exists today) ----
    ap.add_argument("--instr-reg", choices=["sim"], default=None,
                    help="generate the per-board MMIO instruction regfile of this type "
                         "(only 'sim' today: start=end=index, wt from circuit timing).")
    ap.add_argument("--cw-mem", choices=["sim"], default=None,
                    help="generate the per-board MMIO command-word memory of this type "
                         "(only 'sim' today: each entry = its round index).")
    ap.add_argument("--clock-freq", type=float, default=100.0,
                    help="control clock frequency in MHz (default 100); sets the instr "
                         "wt field = inter-measurement circuit time / clock period.")

    # ---- stim simulation test data (Phase C) ----
    ap.add_argument("--sim", nargs="+", choices=["dcb", "readout"], default=None, metavar="LAYOUT",
                    help="generate stim simulation test data for the given measurement "
                         "layout(s): 'dcb' (in_meas/in_valid depth T, DCB test) and/or "
                         "'readout' (sim_readout_meas/valid depth N, whole-system StimReadout). "
                         "expected_dets/expected_postselect are emitted for either. "
                         "e.g. --sim dcb readout")
    ap.add_argument("--shots", type=int, default=100, help="shots to sample for --sim (default 100).")
    ap.add_argument("--seed", type=int, default=1, help="stim sampler seed for --sim (default 1).")
    return ap


def main(argv=None) -> str:
    args = build_arg_parser().parse_args(argv)

    # ---- stage 1: parse -> compile -> serialize detector-construct regfiles ----
    scanner = DetectorConstructionScanner.from_file(args.circuit)
    scanner.scan()
    scanner.refine()
    comp = Compiler(CompilerConfig.from_file(args.config), scanner)
    root = Serializer(comp).write(
        args.out, args.circuit, args.config, mem_fmt=args.mem,
        postselect_layout=args.postselect_layout,
        output_sync_all=(args.output_sync == "all"),
        wait_rounds=args.wait_rounds, wait_row=args.wait_row)
    print(f"[cli] detector-construct programs -> {root}")

    # ---- stage 2: MMIO instruction regfile + command-word memory (Phase B) ----
    if args.instr_reg or args.cw_mem:
        mmio = gen_instr_cw.generate(
            comp, scanner, root, instr_type=args.instr_reg, cw_type=args.cw_mem,
            clock_freq_mhz=args.clock_freq, wait_row=args.wait_row, mem_fmt=args.mem)
        print(f"[cli] MMIO instr+CW ({args.instr_reg or args.cw_mem}, "
              f"clock={args.clock_freq}MHz) -> {len(mmio)} board(s)")

    # ---- stage 3: stim simulation test data (Phase C) ----
    if args.sim:
        sim = gen_sim.generate(root, args.sim, shots=args.shots, seed=args.seed,
                               wait_row=args.wait_row)
        print(f"[cli] sim test data ({'+'.join(args.sim)}, shots={args.shots}, "
              f"seed={args.seed}) -> {len(sim)} leaf board(s)")

    return root


if __name__ == "__main__":
    main()
