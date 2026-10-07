// instr_sequencer.hpp — instruction sequencer (state machine) for PhysicalMMIO.
//
// Owns the instruction pointer into the instruction regfile and the event FSM
// that reacts to START / ABORT / FINISH from the parent board. It hands one
// instruction at a time to InstrDecode (which streams the CW addresses) and
// consumes `one_instr_end` to advance.
//
//   ev_* ---> [ InstrSequencer ] --instr_en/instr_addr--> instr regfile (async)
//                   ^     |                                     |
//     one_instr_end |     +--rst_out--> InstrDecode <-----------+
//                   +-----------------------'
//
// NO-STALL HANDOFF. `instr_en`/`instr_addr` are registered: on the cycle
// InstrDecode raises one_instr_end, this block registers the next pointer with
// en=1, so the next instruction is fetched the very next cycle and its first CW
// read (w_t==0) lands with no bubble.
//
// WAIT BEHAVIOUR. `instr_ptr` SATURATES at DEPTH-1 instead of running off the
// end, so the last instruction repeats forever. That last entry is the compiled
// "wait" instruction (one syndrome-extraction round), which is what keeps the
// protocol idling until FINISH arrives.
//
// ADDRESS CONVENTION. `instr_addr` is meaningful ONLY when `instr_en` is high:
// instr_addr = instr_en ? instr_ptr : 0. `instr_ptr` is a separate maintained
// register so the pointer survives the cycles where en is low.
//
// RESET OUTPUT (combinational, deliberately):
//     rst_out = rst || is_abort || rst_for_finish
//   * rst      — the system reset also resets InstrDecode, so rst_out is the
//                SINGLE reset path into it (no OR needed outside).
//   * abort    — high on the SAME cycle as the event, so InstrDecode is cleared
//                one cycle BEFORE instr_addr returns to 0 and restarts.
//   * finish   — rst_for_finish is registered, so it rises one cycle AFTER the
//                final CW address was issued. SyncROM (which has no reset) still
//                delivers that last command word on that cycle; cw_gen_finish
//                pulses in exactly the same cycle as the final out_valid.
//
// Parameters:
//   depth      instruction regfile depth (pointer saturates at depth-1)
//   payload_w  width of the reserved ev_payload bus
#pragma once

#include <systemc>

#include <cstdint>

#include "signals.hpp"

namespace emu {

SC_MODULE(InstrSequencer) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;   // asynchronous, active high (system reset)

    // --- control event bus (from the parent board) ---
    sc_core::sc_in<bool>     ev_valid;
    sc_core::sc_in<uint32_t> ev_type;     // see EvType
    sc_core::sc_in<Bits>     ev_payload;  // RESERVED (JUMP target); unused today

    // --- from InstrDecode ---
    sc_core::sc_in<bool> one_instr_end;

    // --- to the instruction regfile ---
    sc_core::sc_out<bool>     instr_en;
    sc_core::sc_out<uint32_t> instr_addr;   // valid only while instr_en

    // --- to InstrDecode / the outside world ---
    sc_core::sc_out<bool> rst_out;         // local reset (combinational)
    sc_core::sc_out<bool> cw_gen_finish;   // 1-cycle pulse: CW generation complete

    InstrSequencer(sc_core::sc_module_name nm, int depth, int payload_w);

    enum State  { IDLE = 0, EXEC = 1, DRAIN = 2 };
    enum EvType { EV_START = 0, EV_ABORT = 1, EV_FINISH = 2, EV_JUMP = 3 };

  private:
    void seq();   // clocked: state / instr_ptr / instr_en / rst_for_finish
    void comb();  // combinational: instr_addr, rst_out, cw_gen_finish

    const int depth_;
    const int payload_w_;

    sc_core::sc_signal<int>      state_;
    sc_core::sc_signal<uint32_t> ptr_;       // maintained instruction pointer
    sc_core::sc_signal<bool>     en_;        // registered instr_en
    sc_core::sc_signal<bool>     rst_fin_;   // 1-cycle flag set when CW gen completes
};

}  // namespace emu
