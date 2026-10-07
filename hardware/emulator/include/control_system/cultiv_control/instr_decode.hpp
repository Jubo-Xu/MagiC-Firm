// instr_decode.hpp — instruction decoder for PhysicalMMIO.
//
// Turns ONE instruction {w_t, start, end} into a command-word read stream for
// the CW memory (a SyncROM). The state machine feeds instructions via a 1-cycle
// `en` pulse; this block generates the CW read address sequence and tells the
// state machine when an instruction is done (one_instr_end).
//
//   en pulse + {w_t, start, end}
//        |  wait w_t idle cycles
//        v
//   r_addr:  start, start+1, ..., end     (one per cycle, r_en high)
//        |
//        v  on the cycle r_addr == end
//   one_instr_end = 1   -> state machine hands over the next instruction
//
// r_en/r_addr drive SyncROM.en/.addr; the CW data + valid come back from SyncROM
// one cycle later (this block does NOT touch the data path — SyncROM already
// pairs data with data_valid).
//
// Wait-count convention: w_t = number of idle cycles before the first read.
//   w_t = 0 -> read `start` on the en cycle itself (fast path, no stall)
//   w_t = N -> N idle cycles, read `start` on en+N
//   start == end -> exactly one read, one_instr_end on that same cycle.
//
// OUTPUTS ARE COMBINATIONAL (Mealy). This is required for the no-stall handoff:
// at the cycle r_addr == end the state machine registers en=1 for the NEXT
// cycle, and the next instruction's first read (w_t==0) must appear on that same
// en cycle so CW reads stay contiguous (end_i at T, start_{i+1} at T+1). The
// internal FSM state (phase/addr/end/wait) is registered; a separate clocked
// process registers the next-state the comb process computes.
//
// Parameters:
//   wt_w    width of in_wt   (max wait count)
//   addr_w  width of in_start / in_end (CW address; start and end share it)
#pragma once

#include <systemc>

#include <cstdint>

#include "signals.hpp"

namespace emu {

SC_MODULE(InstrDecode) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high (state machine's local reset)

    // --- instruction input (from the state machine) ---
    sc_core::sc_in<bool>     en;        // 1-cycle pulse: start decoding this instruction
    sc_core::sc_in<uint32_t> in_wt;     // idle cycles before first read
    sc_core::sc_in<uint32_t> in_start;  // first CW address
    sc_core::sc_in<uint32_t> in_end;    // last CW address (>= in_start)

    // --- CW-memory read control (to SyncROM) ---
    sc_core::sc_out<bool>     r_en;
    sc_core::sc_out<uint32_t> r_addr;

    // --- handshake back to the state machine ---
    sc_core::sc_out<bool> one_instr_end;  // 1 for one cycle when r_addr == end

    InstrDecode(sc_core::sc_module_name nm, int wt_w, int addr_w);

    enum Phase { IDLE = 0, WAIT = 1, STREAM = 2 };

  private:
    void comb();  // combinational: outputs + next-state (switch on phase)
    void seq();   // clocked: register next-state (async reset)

    const int wt_w_;
    const int addr_w_;

    // registered FSM state
    sc_core::sc_signal<int>      phase_;
    sc_core::sc_signal<uint32_t> addr_;   // holds start during WAIT, walks during STREAM
    sc_core::sc_signal<uint32_t> end_;    // latched end address
    sc_core::sc_signal<uint32_t> wait_;   // remaining idle cycles

    // next-state (comb -> seq)
    sc_core::sc_signal<int>      nphase_;
    sc_core::sc_signal<uint32_t> naddr_;
    sc_core::sc_signal<uint32_t> nend_;
    sc_core::sc_signal<uint32_t> nwait_;
};

}  // namespace emu
