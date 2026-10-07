// board_control.cpp — implementation of BoardControl.
//
// comb() : combinational outputs (reset, set_in_data_to_zero, out_attempt,
//          cur_attempt, discard).
// seq()  : one clocked switch on the FSM state with async reset; drives the
//          registered attempt, rst_for_finish, and out_ev bus.
//
// Behaviour keys off two independent axes: event_mode (ORIGINATE vs FORWARD)
// and data_src (EXTERNAL vs INTERNAL). See the header for the board mapping.
#include "control_system/cultiv_control/board_control.hpp"

namespace emu {

BoardControl::BoardControl(sc_core::sc_module_name nm, int event_mode, int data_src,
                           int m, int payload_w)
    : sc_module(nm), event_mode_(event_mode), data_src_(data_src), m_(m), payload_w_(payload_w),
      state_("state"), attempt_("attempt"), rst_for_finish_("rst_for_finish"),
      abort_det_q_("abort_det_q") {
    SC_HAS_PROCESS(BoardControl);

    // comb reads the REGISTERED board-abort (abort_det_q_), not post_select directly, so
    // the abort->reset path no longer clears its own trigger within a delta.
    SC_METHOD(comb);
    sensitive << attempt_ << rst_for_finish_ << rst << in_valid << data_attempt
              << ev_valid << ev_type << data_valid_output << abort_det_q_;

    SC_METHOD(seq);
    sensitive << clk.pos() << rst;
    dont_initialize();
}

// ORIGINATE: is there any post_select line whose stamped attempt equals ours?
// XNOR == equality (a plain AND would miss the attempt==0 case entirely).
bool BoardControl::detect_abort() {
    if (event_mode_ != ORIGINATE) return false;
    const Bits ps  = post_select.read();
    const Bits pa  = post_select_attempt.read();
    const bool att = attempt_.read();
    for (int i = 0; i < m_ && i < (int)ps.size(); ++i)
        if (ps[i] && (static_cast<bool>(pa[i]) == att)) return true;
    return false;
}

void BoardControl::comb() {
    const bool att = attempt_.read();
    const bool is_abort_ev = ev_valid.read() && (ev_type.read() == EV_ABORT);
    // Board-generated abort is the REGISTERED value (breaks the self-clearing loop);
    // the external event abort stays combinational (it never resets its own source).
    const bool abort_now = abort_det_q_.read() || is_abort_ev;

    reset.write(rst.read() || abort_now || rst_for_finish_.read());

    const bool data_att_eff = (data_src_ == INTERNAL) ? att : data_attempt.read();
    const bool mism = (data_att_eff != att);
    set_in_data_to_zero.write(in_valid.read() && (mism || abort_now));

    out_attempt.write(data_valid_output.read() ? att : false);
    cur_attempt.write(att);          // raw attempt — post_select fast-path / gap stamp
    discard.write(abort_now);        // notify the outside (root) to reset on abort
}

void BoardControl::seq() {
    if (rst.read()) {
        state_.write(IDLE);
        attempt_.write(false);
        rst_for_finish_.write(false);
        abort_det_q_.write(false);
        out_ev_valid.write(false);
        out_ev_type.write(0);
        out_ev_payload.write(Bits(payload_w_));
        out_ev_attempt.write(false);
        return;
    }
    if (!clk.posedge()) return;

    const bool     valid = ev_valid.read();
    const uint32_t type  = ev_type.read();
    const bool is_start  = valid && (type == EV_START);
    const bool is_finish = valid && (type == EV_FINISH);
    // The board-generated abort is the registered one-shot; the event abort is direct.
    const bool ab        = abort_det_q_.read() || (valid && (type == EV_ABORT));

    // Re-arm the one-shot: pulse for exactly one cycle on a rising board-abort. The
    // reset/attempt-flip it triggers only become visible next cycle, so guarding with
    // !abort_det_q_ prevents a spurious second pulse (which would double-flip attempt).
    abort_det_q_.write(detect_abort() && !abort_det_q_.read());
    const bool dd        = drain_done.read();
    const bool att       = attempt_.read();
    const Bits pay       = ev_payload.read();
    const int  st        = state_.read();
    const bool origin    = (event_mode_ == ORIGINATE);

    // out_ev defaults:
    //   ORIGINATE : no event (generated only on transitions below)
    //   FORWARD   : relay the received event, delayed one cycle
    bool     ov = origin ? false : valid;
    uint32_t ot = origin ? 0u    : type;
    bool     oa = origin ? false : ev_attempt.read();
    Bits     op = pay;

    int  nstate = st;
    bool natt   = att;
    bool nrff   = false;

    switch (st) {
        case IDLE:
            if (is_start) {
                natt   = origin ? false : ev_attempt.read();
                nstate = EXEC;
                if (origin) { ov = true; ot = EV_START; oa = natt; }
            }
            break;

        case EXEC:
            if (ab) {
                // abort: flip (ORIGINATE) or adopt (FORWARD); stay in EXEC
                natt   = origin ? (!att) : ev_attempt.read();
                nstate = EXEC;
                if (origin) { ov = true; ot = EV_ABORT; oa = natt; }
            } else if (is_finish) {
                if (dd) {
                    // children are already IDLE (drain_done came from them): nothing
                    // left below to notify -> emit NO event, just finish ourselves.
                    ov = false; ot = 0u; oa = false;
                    nstate = IDLE; natt = false; nrff = true;
                } else {
                    // first FINISH: ride it DOWN once, then drain
                    if (origin) { ov = true; ot = EV_FINISH; oa = att; }
                    nstate = DRAIN;
                }
            }
            // else normal: hold, no event
            break;

        case DRAIN:
            // no downward event during drain; FORWARD relays whatever is on in_ev
            // (parent is silent, so that is 0). Wait for completion to ripple up.
            if (dd) {
                ov = false; ot = 0u; oa = false;
                nstate = IDLE; natt = false; nrff = true;
            }
            break;

        default:
            nstate = IDLE;
            break;
    }

    state_.write(nstate);
    attempt_.write(natt);
    rst_for_finish_.write(nrff);
    out_ev_valid.write(ov);
    out_ev_type.write(ot);
    out_ev_attempt.write(oa);
    out_ev_payload.write(op);
}

}  // namespace emu
