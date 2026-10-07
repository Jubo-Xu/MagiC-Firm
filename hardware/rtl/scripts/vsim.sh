#!/usr/bin/env bash
# vsim.sh — compile+run one SystemVerilog testbench with Verilator, print output.
#
# Usage: vsim.sh <tb.sv> <src.sv> [<src.sv> ...]
# The top module is the tb's basename (e.g. tb_DetectorPass.sv -> tb_DetectorPass).
# Build artifacts go to a scratch dir (kept off the home inode quota), auto-removed.
#
# Requires verilator on PATH (conda env basic_quantum). Exits non-zero if the run
# prints "FAIL" or Verilator errors out.
set -euo pipefail

TB="${1:?usage: vsim.sh <tb.sv> <src.sv>...}"; shift
TOP="$(basename "$TB" .sv)"
BUILD="$(mktemp -d "${TMPDIR:-/tmp}/vsim.XXXXXX")"
trap 'rm -rf "$BUILD"' EXIT

verilator --binary --timing -Wall -Wno-DECLFILENAME -Wno-UNUSEDSIGNAL -Wno-WIDTHTRUNC -Wno-WIDTHEXPAND -Wno-UNUSEDPARAM -Wno-PINCONNECTEMPTY \
    --top-module "$TOP" -Mdir "$BUILD" -o sim \
    "$TB" "$@" 2>&1 | sed 's/^/[verilator] /'

# `|| true` matters: a testbench assertion calls $stop, which exits non-zero,
# and under `set -e` a bare assignment would abort the script here — printing
# nothing at all. An assertion failure would then look identical to silence.
OUT="$("$BUILD/sim" 2>&1)" || true
echo "$OUT"
# %Error covers $error/$fatal from assertions, which never reach the PASS/FAIL
# line the testbench prints at the end.
echo "$OUT" | grep -qE "FAIL|%Error" && { echo "vsim: FAIL"; exit 1; } || true
echo "$OUT" | grep -q "PASS" || { echo "vsim: no PASS marker"; exit 1; }
