// instr_decode.cpp — implementation of InstrDecode.
//
// Two-process Mealy FSM: comb() computes the (combinational) outputs and the
// next-state via a switch on the current phase; seq() registers the next-state
// on the clock edge with an async reset to IDLE.
#include "control_system/cultiv_control/instr_decode.hpp"

namespace emu {

InstrDecode::InstrDecode(sc_core::sc_module_name nm, int wt_w, int addr_w)
    : sc_module(nm), wt_w_(wt_w), addr_w_(addr_w),
      phase_("phase"), addr_("addr"), end_("end"), wait_("wait"),
      nphase_("nphase"), naddr_("naddr"), nend_("nend"), nwait_("nwait") {
    SC_HAS_PROCESS(InstrDecode);

    SC_METHOD(comb);
    sensitive << phase_ << addr_ << end_ << wait_
              << en << in_wt << in_start << in_end;

    SC_METHOD(seq);
    sensitive << clk.pos() << rst;
    dont_initialize();

    (void)wt_w_; (void)addr_w_;  // widths documented for RTL parity; SystemC uses uint32_t
}

// Combinational: default the outputs to idle and next-state to "hold current",
// then override per phase. Only IDLE consumes `en` — by construction (no-stall
// handoff + abort resets first) `en` never arrives mid-instruction.
void InstrDecode::comb() {
    const int      phase = phase_.read();
    const uint32_t addr  = addr_.read();
    const uint32_t end   = end_.read();
    const uint32_t wait  = wait_.read();

    // defaults
    bool     r_en_o = false;
    uint32_t r_addr_o = 0;
    bool     oie_o = false;
    int      nphase = phase;
    uint32_t naddr = addr, nend = end, nwait = wait;

    switch (phase) {
        case IDLE:
            if (en.read()) {
                const uint32_t wt = in_wt.read();
                const uint32_t st = in_start.read();
                const uint32_t ed = in_end.read();
                if (wt == 0) {
                    // fast path: read `start` this very cycle
                    r_en_o = true;
                    r_addr_o = st;
                    if (st == ed) {
                        oie_o = true;         // single-word instruction, done now
                        nphase = IDLE;
                    } else {
                        nphase = STREAM;
                        naddr = st + 1;       // next address to stream
                        nend = ed;
                    }
                } else {
                    // wait first; latch start/end
                    nphase = WAIT;
                    nwait = wt - 1;           // this cycle is the first idle cycle
                    naddr = st;
                    nend = ed;
                }
            }
            break;

        case WAIT:
            if (wait == 0) {
                // idle done: read `start` (held in addr_) now
                r_en_o = true;
                r_addr_o = addr;
                if (addr == end) {
                    oie_o = true;
                    nphase = IDLE;
                } else {
                    nphase = STREAM;
                    naddr = addr + 1;
                }
            } else {
                nwait = wait - 1;
                nphase = WAIT;
            }
            break;

        case STREAM:
            r_en_o = true;
            r_addr_o = addr;
            if (addr == end) {
                oie_o = true;
                nphase = IDLE;
            } else {
                naddr = addr + 1;
                nphase = STREAM;
            }
            break;

        default:
            nphase = IDLE;
            break;
    }

    r_en.write(r_en_o);
    r_addr.write(r_addr_o);
    one_instr_end.write(oie_o);

    nphase_.write(nphase);
    naddr_.write(naddr);
    nend_.write(nend);
    nwait_.write(nwait);
}

// Clocked: async reset to IDLE, else register the next-state.
void InstrDecode::seq() {
    if (rst.read()) {
        phase_.write(IDLE);
        addr_.write(0);
        end_.write(0);
        wait_.write(0);
        return;
    }
    if (!clk.posedge()) return;

    phase_.write(nphase_.read());
    addr_.write(naddr_.read());
    end_.write(nend_.read());
    wait_.write(nwait_.read());
}

}  // namespace emu
