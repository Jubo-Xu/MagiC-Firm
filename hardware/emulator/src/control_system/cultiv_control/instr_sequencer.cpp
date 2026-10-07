// instr_sequencer.cpp — implementation of InstrSequencer.
//
// seq()  : one clocked switch on the state, async reset (all state registered).
// comb() : the three combinational outputs (instr_addr mux, rst_out, cw_gen_finish).
#include "control_system/cultiv_control/instr_sequencer.hpp"

namespace emu {

InstrSequencer::InstrSequencer(sc_core::sc_module_name nm, int depth, int payload_w)
    : sc_module(nm), depth_(depth), payload_w_(payload_w),
      state_("state"), ptr_("ptr"), en_("en"), rst_fin_("rst_fin") {
    SC_HAS_PROCESS(InstrSequencer);

    SC_METHOD(seq);
    sensitive << clk.pos() << rst;
    dont_initialize();

    SC_METHOD(comb);
    sensitive << en_ << ptr_ << rst_fin_ << rst << ev_valid << ev_type;

    (void)payload_w_;  // ev_payload is a reserved stub (JUMP); nothing reads it yet
}

void InstrSequencer::seq() {
    if (rst.read()) {
        state_.write(IDLE);
        ptr_.write(0);
        en_.write(false);
        rst_fin_.write(false);
        return;
    }
    if (!clk.posedge()) return;

    const bool     valid = ev_valid.read();
    const uint32_t type  = ev_type.read();
    const bool is_start  = valid && (type == EV_START);
    const bool is_abort  = valid && (type == EV_ABORT);
    const bool is_finish = valid && (type == EV_FINISH);
    const bool oie       = one_instr_end.read();
    const uint32_t ptr   = ptr_.read();

    switch (state_.read()) {
        case IDLE:
            // START is the only thing that wakes us; everything else stays idle.
            rst_fin_.write(false);
            if (is_start) {
                state_.write(EXEC);
                ptr_.write(0);
                en_.write(true);        // fetch instruction 0 next cycle
            } else {
                ptr_.write(0);
                en_.write(false);
            }
            break;

        case EXEC:
            if (is_abort) {
                // restart from instruction 0; rst_out is already high THIS cycle
                // (combinational), so InstrDecode is clear before the refetch.
                ptr_.write(0);
                en_.write(true);
                state_.write(EXEC);
                rst_fin_.write(false);
            } else if (is_finish) {
                if (oie) {
                    // the last instruction finished in the very same cycle:
                    // nothing left to drain, go straight to IDLE.
                    state_.write(IDLE);
                    ptr_.write(0);
                    en_.write(false);
                    rst_fin_.write(true);
                } else {
                    // an instruction is mid-flight: let it finish in DRAIN
                    state_.write(DRAIN);
                    ptr_.write(0);
                    en_.write(false);
                    rst_fin_.write(false);
                }
            } else {
                // normal: advance on one_instr_end, saturating at the wait entry
                if (oie) {
                    ptr_.write((ptr == static_cast<uint32_t>(depth_ - 1)) ? ptr : ptr + 1);
                    en_.write(true);
                } else {
                    en_.write(false);   // ptr_ holds
                }
                state_.write(EXEC);
                rst_fin_.write(false);
            }
            break;

        case DRAIN:
            // no new instruction is fetched; wait for the in-flight one to end.
            if (oie) {
                state_.write(IDLE);
                ptr_.write(0);
                en_.write(false);
                rst_fin_.write(true);
            } else {
                ptr_.write(0);
                en_.write(false);
                rst_fin_.write(false);
            }
            break;

        default:
            state_.write(IDLE);
            ptr_.write(0);
            en_.write(false);
            rst_fin_.write(false);
            break;
    }
}

void InstrSequencer::comb() {
    const bool en = en_.read();
    const bool is_abort = ev_valid.read() && (ev_type.read() == EV_ABORT);

    instr_en.write(en);
    instr_addr.write(en ? ptr_.read() : 0u);           // address valid only with en
    rst_out.write(rst.read() || is_abort || rst_fin_.read());
    cw_gen_finish.write(rst_fin_.read());
}

}  // namespace emu
