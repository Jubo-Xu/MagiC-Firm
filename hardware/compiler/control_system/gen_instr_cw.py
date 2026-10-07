"""gen_instr_cw.py — per-board MMIO instruction regfile + command-word memory.

Phase B of the control-system compiler (see cli.py). For every board that owns
physical qubits (leaf / monolithic — routers/root have no control cores), emit ONE
instruction regfile and ONE command-word memory that drive the board's PhysicalMMIO.
P = 1 core per board today, so the files carry the `_0` core index; the flat
board-prefix (`board<id>_...`) is applied later by the test generator.

EXTENSIBLE BY TYPE. The instruction/CW *content* varies by program type; only the
"sim" type exists now, but real deployment types will follow. So the content logic
lives in per-type PLAN functions registered in `_PLAN` (keyed by type), each
returning per-board `(start, end, wt)` instructions + `cw` indices. Everything
downstream — field-width sizing, word packing, file writing — is TYPE-AGNOSTIC and
shared in `_emit_board`. Adding a type = write `_plan_<type>` and register it.

Sizing (any type): a board only measures its own qubits, at a SPARSE set of the
circuit's measurement time-indexes (e.g. {2,5,9}); N = that count, depth = N (+1 for
copy-last).

The 'sim' plan:
  instr[i] = (start = end = i, wt = wait cycles before round i's readout)
             wt[0]  = circuit start -> the board's first round,
             wt[i]  = board round i-1 -> round i,   in clock cycles.
  cw[i]    = i (round index; the sim readout model decodes it).
  copy-last appends one entry: (start=end=N, wt copied from N-1), cw index N.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

from serializer import _bits, _word
from timing import round_completion_ns

# A per-board plan: the instruction triples and CW indices for one board.
#   instr : list of (start_addr, end_addr, wt)
#   cw    : list of command words (round indices)
BoardPlan = Dict[str, list]


# --------------------------------------------------------------------------- #
# type-specific plan functions (content only; packing/writing is shared)
# --------------------------------------------------------------------------- #
def _plan_sim(comp, scanner, bid: int, times: List[float],
              clock_freq_mhz: float, wait_row: str) -> Optional[BoardPlan]:
    """'sim' program: one instruction per measurement round, wt from circuit timing;
    CW word = round index. Returns None for a board with no measured rounds."""
    period_ns = 1000.0 / clock_freq_mhz                    # 100 MHz -> 10 ns
    mrr_flat = scanner._measurement_rec_raw_flat            # [(qubit, meas_time)]
    qubits = set(comp._board_input_qubits(bid))             # this board's used scope
    board_rounds = sorted({mt for q, mt in mrr_flat if q in qubits})
    if not board_rounds:
        return None

    wt, prev_ns = [], 0.0
    for r in board_rounds:                                  # wt[0] from start, else delta
        t_r = times[r]
        wt.append(int(round((t_r - prev_ns) / period_ns)))
        prev_ns = t_r

    N = len(board_rounds)
    instr: List[Tuple[int, int, int]] = [(i, i, wt[i]) for i in range(N)]
    cw: List[int] = list(range(N))
    if wait_row == "copy-last":                             # saturating wait entry
        instr.append((N, N, wt[-1]))
        cw.append(N)
    return {"instr": instr, "cw": cw, "board_rounds": board_rounds}


# type -> plan function. Register new program types here.
_PLAN = {
    "sim": _plan_sim,
}


# --------------------------------------------------------------------------- #
# shared: sizing, packing, writing (identical for every type)
# --------------------------------------------------------------------------- #
def _emit_board(root: str, bid: int, plan: BoardPlan, mem_fmt: str,
                clock_freq_mhz: float, wait_row: str, reg_type: str) -> dict:
    instr, cw = plan["instr"], plan["cw"]
    depth = len(cw)

    # field widths -> MMIO params (ADDR_W is derived from CW_DEPTH in the module)
    addr_w = _bits(depth)                                   # index 0..depth-1
    wt_w = max(1, max((w for _, _, w in instr), default=0).bit_length())
    data_w = _bits(depth)                                   # command word = round index
    instr_w = 2 * addr_w + wt_w

    # pack: instr word = end | start<<addr_w | wt<<2*addr_w  (InstrUnpack layout)
    instr_mem, cw_mem = [], []
    for (s, e, w) in instr:
        word = e | (s << addr_w) | (w << (2 * addr_w))
        assert (word & ((1 << addr_w) - 1)) == e                      # round-trip end
        assert ((word >> addr_w) & ((1 << addr_w) - 1)) == s          # round-trip start
        assert ((word >> (2 * addr_w)) & ((1 << wt_w) - 1)) == w      # round-trip wt
        instr_mem.append(_word(word, instr_w, mem_fmt))
    for v in cw:
        cw_mem.append(_word(v, data_w, mem_fmt))

    jdir = os.path.join(root, f"board{bid}", "json")
    mdir = os.path.join(root, f"board{bid}", "mem")
    note_i = (f"MMIO instruction regfile ({reg_type}), 1 core. word = "
              f"[wt:{wt_w} | start_addr:{addr_w} | end_addr:{addr_w}]; "
              f"start=end=round index; wt = wait cycles before that round "
              f"@ {clock_freq_mhz}MHz. copy-last row: {wait_row == 'copy-last'}.")
    note_c = (f"MMIO command-word memory ({reg_type}), 1 core. word = round index; "
              "the sim readout model uses index==0 to jump to the next shot.")
    _write_regfile(jdir, mdir, "MMIO_instr_0",
                   {"regfile": "MMIO_instr", "type": reg_type, "core": 0,
                    "word_width": instr_w, "depth": depth, "wt_w": wt_w, "addr_w": addr_w,
                    "start": [s for s, _, _ in instr], "end": [e for _, e, _ in instr],
                    "wt": [w for _, _, w in instr], "note": note_i},
                   instr_mem, mem_fmt)
    _write_regfile(jdir, mdir, "MMIO_cw_0",
                   {"regfile": "MMIO_cw", "type": reg_type, "core": 0,
                    "word_width": data_w, "depth": depth, "index": cw, "note": note_c},
                   cw_mem, mem_fmt)

    return {"N": len(plan["board_rounds"]), "depth": depth,
            "instr_depth": depth, "cw_depth": depth,
            "data_w": data_w, "wt_w": wt_w, "addr_w": addr_w,
            "board_rounds": plan["board_rounds"], "wt": [w for _, _, w in instr]}


def _write_regfile(jdir: str, mdir: str, name: str, jobj: dict, mem, mem_fmt: str) -> None:
    os.makedirs(jdir, exist_ok=True)
    os.makedirs(mdir, exist_ok=True)
    with open(os.path.join(jdir, f"{name}.json"), "w") as f:
        json.dump(jobj, f, indent=1)
    with open(os.path.join(mdir, f"{name}.mem"), "w") as f:
        f.write("\n".join(mem) + ("\n" if mem else ""))


# --------------------------------------------------------------------------- #
# entry point: dispatch by type, then shared emit
# --------------------------------------------------------------------------- #
def generate(comp, scanner, root: str, instr_type: Optional[str] = "sim",
             cw_type: Optional[str] = "sim", clock_freq_mhz: float = 100.0,
             wait_row: str = "normal", mem_fmt: str = "hex") -> Dict[int, dict]:
    """Emit MMIO instr + CW for every qubit-owning board under `root`. `instr_type`
    and `cw_type` select the program type (only 'sim' today; they must match).
    Returns a per-board param summary (also written to results/mmio_params.json)."""
    reg_type = instr_type or cw_type
    if instr_type and cw_type and instr_type != cw_type:
        raise ValueError(f"instr-reg ({instr_type}) and cw-mem ({cw_type}) must match; "
                         "mixed program types are not supported yet")
    if reg_type not in _PLAN:
        raise ValueError(f"unknown MMIO program type {reg_type!r}; have {sorted(_PLAN)}")
    plan_fn = _PLAN[reg_type]

    times = round_completion_ns(scanner.raw_circuit)        # ns per global round
    expected_T = max((mt for _, mt in scanner._measurement_rec_raw_flat), default=-1) + 1
    if len(times) < expected_T:                             # indices must line up with meas_time
        raise AssertionError(
            f"timing produced {len(times)} rounds but the parser has {expected_T}; "
            "meas_time convention mismatch in timing.py")

    summary: Dict[int, dict] = {}
    for bid, b in sorted(comp.cfg.boards.items()):
        if b.qubits is None:                                # only boards with control cores
            continue
        plan = plan_fn(comp, scanner, bid, times, clock_freq_mhz, wait_row)
        if plan is None:
            continue
        summary[bid] = _emit_board(root, bid, plan, mem_fmt, clock_freq_mhz, wait_row, reg_type)

    with open(os.path.join(root, "mmio_params.json"), "w") as f:
        json.dump({"type": reg_type, "clock_freq_mhz": clock_freq_mhz, "wait_row": wait_row,
                   "boards": {str(k): v for k, v in summary.items()}}, f, indent=1)
    return summary
