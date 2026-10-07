"""gen_sim.py — stim simulation test data (Phase C of the control-system compiler).

Samples ground-truth shots for a compiled example and writes the measurement
stimulus + expected detector/postselect outcomes, beside the regfiles under each
board folder. This is the compiler-owned home for what used to be the emulator's
gen_stim_vectors.py (which wrote into the compiler's folder from outside).

Two measurement-memory LAYOUTS, both derived from ONE shared sample so they stay
consistent; the ground truth is layout-independent and emitted once:

  dcb      in_meas / in_valid           depth shots*T   (every meas-time; in_valid=0
                                         padding for non-participating times) — the
                                         DCB-only test (test_dcb_stim, gen_dcb_tb).
  readout  sim_readout_meas / _valid    depth shots*N   (only the board's N meas-times;
                                         non-participation handled by cycle sync) — the
                                         whole-system test (ControlBoard + MMIO +
                                         StimReadout). copy-last appends one all-1s wait
                                         row per shot (depth shots*(N+1)).

Shared ground truth: expected_dets (root), expected_postselect (stage boards).

Extensible: register a new measurement layout in _LAYOUTS. Words are binary,
MSB-first, LSB=bit 0 (same as the regfiles) so $readmemb / load_mem read them directly.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import stim


def _load(path):
    with open(path) as f:
        return json.load(f)


def _word(bits) -> str:
    """bits: iterable of 0/1, index = bit position (LSB=bit0) -> MSB-first bin string."""
    v = 0
    for i, b in enumerate(bits):
        if b:
            v |= 1 << i
    return format(v, f"0{len(bits)}b")


def _write(bdir: str, name: str, rows: List[str], width: int, shots: int, seed: int,
           note: str, extra: dict) -> None:
    with open(os.path.join(bdir, "mem", f"{name}.mem"), "w") as f:
        f.write("\n".join(rows) + ("\n" if rows else ""))
    j = {"regfile": name, "shots": shots, "seed": seed, "word_width": width,
         "rows": len(rows), "note": note, **extra}
    with open(os.path.join(bdir, "json", f"{name}.json"), "w") as f:
        json.dump(j, f, indent=1)


def _board_rounds(chq: dict, T: int, qt2rec: dict) -> List[int]:
    """The sparse set of meas-times this board actually participates in (any port)."""
    return sorted({t for t in range(T) for p in range(len(chq)) if (chq[p], t) in qt2rec})


# --------------------------------------------------------------------------- #
# measurement-memory layouts (leaf boards). Signature is uniform so _LAYOUTS
# can dispatch; each writes its own mem/json and returns (name, depth).
# --------------------------------------------------------------------------- #
def _emit_dcb(bdir, chq, m, M, qt2rec, T, shots, seed, wait_row, sync_mem, mem_fmt):
    base = 16 if mem_fmt == "hex" else 2            # sync.mem radix follows --mem
    meas_rows, valid_rows = [], []
    for s in range(shots):
        for t in range(T):
            mv = [1 if (chq[p], t) in qt2rec and M[s, qt2rec[(chq[p], t)]] else 0 for p in range(m)]
            vv = [1 if (chq[p], t) in qt2rec else 0 for p in range(m)]
            if s == 0 and sync_mem:                 # sanity: valid schedule == sync mask (value, not string)
                assert int(_word(vv), 2) == int(sync_mem[t], base), f"in_valid[t={t}] != sync.mem[t]"
            meas_rows.append(_word(mv))
            valid_rows.append(_word(vv))
    note = "row = shot*T + t; bit i = input port i (channel_qubit[i]); LSB=bit0"
    _write(bdir, "in_meas",  meas_rows,  m, shots, seed, note, {"T": T, "m": m})
    _write(bdir, "in_valid", valid_rows, m, shots, seed, note, {"T": T, "m": m})
    return ("in_meas/in_valid", len(meas_rows))


def _emit_readout(bdir, chq, m, M, qt2rec, T, shots, seed, wait_row, sync_mem, mem_fmt):
    rounds = _board_rounds(chq, T, qt2rec)          # the board's N sparse meas-times
    N = len(rounds)
    copy = (wait_row == "copy-last")
    meas_rows, valid_rows = [], []
    for s in range(shots):
        for t in rounds:                            # row = shot*(N[+1]) + i  (shot-major)
            mv = [1 if (chq[p], t) in qt2rec and M[s, qt2rec[(chq[p], t)]] else 0 for p in range(m)]
            vv = [1 if (chq[p], t) in qt2rec else 0 for p in range(m)]
            meas_rows.append(_word(mv))
            valid_rows.append(_word(vv))
        if copy:                                    # copy-last virtual wait round = all 1s
            meas_rows.append(_word([1] * m))
            valid_rows.append(_word([1] * m))
    depth = N + (1 if copy else 0)
    note = ("row = shot*depth + i (shot-major); i = board's i-th meas-time (rounds "
            f"{rounds}); bit p = input port p; LSB=bit0. copy-last all-1s wait row: {copy}.")
    _write(bdir, "sim_readout_meas",  meas_rows,  m, shots, seed, note,
           {"N": N, "depth": depth, "rounds": rounds})
    _write(bdir, "sim_readout_valid", valid_rows, m, shots, seed, note,
           {"N": N, "depth": depth, "rounds": rounds})
    return ("sim_readout_meas/valid", len(meas_rows))


_LAYOUTS = {
    "dcb": _emit_dcb,
    "readout": _emit_readout,
}


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def generate(root: str, layouts: List[str], shots: int = 100, seed: int = 1,
             wait_row: str = "normal") -> Dict[int, list]:
    """Sample `shots` shots for the compiled example at `root` and emit the shared
    ground truth + each requested measurement layout. Self-contained on `root`."""
    bad = [l for l in layouts if l not in _LAYOUTS]
    if bad:
        raise ValueError(f"unknown sim layout(s) {bad}; have {sorted(_LAYOUTS)}")

    man = _load(os.path.join(root, "manifest.json"))
    mrr = _load(os.path.join(root, "measurement_map.json"))    # record -> [qubit|null, meas_time]
    T, ndet = man["meas_times"], man["detectors"]
    mem_fmt = man.get("mem_format", "hex")                     # regfile radix (sync.mem etc.)
    circuit = stim.Circuit.from_file(os.path.join(root, man["circuit"]))

    M = circuit.compile_sampler(seed=seed).sample(shots)                        # shots x records
    D = circuit.compile_m2d_converter().convert(measurements=M, separate_observables=False)
    assert D.shape[1] == ndet, (D.shape, ndet)

    qt2rec = {}
    for i, (q, t) in enumerate(mrr):
        if q is None:
            continue
        assert (q, t) not in qt2rec, f"qubit {q} measured twice at meas_time {t}"
        qt2rec[(q, t)] = i

    done: Dict[int, list] = {}
    for bd in man["boards"]:
        bid = bd["board_id"]
        bdir = os.path.join(root, f"board{bid}")
        regs = bd["regfiles"]

        # shared ground truth
        if "global_index" in regs:                 # root
            det_rows = [_word([int(D[s, d]) for d in range(ndet)]) for s in range(shots)]
            _write(bdir, "expected_dets", det_rows, ndet, shots, seed,
                   "row = shot; bit i = stim detector index i; LSB=bit0", {"ndet": ndet})
        if "postselect" in regs:                    # stage
            psd = bd.get("postselect_detectors", [])
            ps_rows = [_word([1 if any(int(D[s, d]) for d in psd) else 0]) for s in range(shots)]
            _write(bdir, "expected_postselect", ps_rows, 1, shots, seed,
                   "row = shot; 1 = OR of THIS board's postselect detectors fired (reject)",
                   {"num_postselect_dets": len(psd)})

        # per-layout measurement memories (leaf boards only)
        if "qubit_scope" in regs:
            chq = _load(os.path.join(bdir, "json", "qubit_scope.json"))["used"]   # port -> qubit
            m = len(chq)
            sync_path = os.path.join(bdir, "mem", "sync.mem")
            sync_mem = open(sync_path).read().split() if os.path.exists(sync_path) else None
            emitted = []
            for lay in layouts:
                name, depth = _LAYOUTS[lay](bdir, chq, m, M, qt2rec, T, shots, seed, wait_row, sync_mem, mem_fmt)
                emitted.append((lay, name, depth))
            done[bid] = emitted

    return done
