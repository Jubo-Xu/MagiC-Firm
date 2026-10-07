// board_control.hpp — per-board control / attempt state machine.
//
// One parameterized module for every board kind. Behaviour varies along TWO
// INDEPENDENT axes (so the monolithic root+leaf board is just one combination,
// not a special case):
//
//   event_mode  ORIGINATE : make events from post_select + outside start/finish,
//                           and drive the out_ev bus from the FSM (root, monolithic)
//               FORWARD   : relay the received in_ev to children, delayed one
//                           cycle, and adopt its attempt (router, leaf)
//
//   data_src    EXTERNAL  : incoming data carries an attempt tag from children
//                           (root, router)
//               INTERNAL  : inputs are physical measurements with no tag, so use
//                           our own attempt (leaf, monolithic)
//
//   board              event_mode   data_src
//   distributed root   ORIGINATE    EXTERNAL
//   distributed mid    FORWARD      EXTERNAL
//   distributed leaf   FORWARD      INTERNAL
//   monolithic         ORIGINATE    INTERNAL
//
// Two concerns:
//   DOWNWARD (parent -> children): the control events (START / ABORT / FINISH).
//   UPWARD   (children -> parent): the attempt tag that lets a parent drop stale
//     data from a killed trial. (Drain completion is NO LONGER signalled on the
//     control path — see DRAIN below.)
//
// ATTEMPT (1 bit). START -> 0. Abort flips it (ORIGINATE) or adopts the event's
// attempt (FORWARD). A parent always changes attempt one cycle before its
// children can, so a mismatch on incoming data means "stale": gating is
// drop-only and no purge window is needed.
//
// KEY EQUATIONS (see the .cpp):
//   abort_detected      = |( post_select & (post_select_attempt ~^ {m{attempt}}) ) // XNOR = equality
//   reset               = rst | abort_now | rst_for_finish        // LOCAL only, never propagated
//   set_in_data_to_zero = in_valid & ( (data_att_eff != attempt) | abort_now )
//   out_attempt         = data_valid_output ? attempt : 0
//   cur_attempt         = attempt                                 // RAW; stamps the post_select fast path
//   discard             = abort_now                               // root routes it out to the host
//
// FSM (all boards): IDLE -> EXEC -> DRAIN -> IDLE. FINISH rides DOWN once
// (EXEC->DRAIN). DRAIN then waits for `drain_done` and resets the board locally.
// `drain_done` is DATA-PATH driven: it is wired at assembly to this board's own
// DetectorConstructBlock DET out_finish, which only fires after every child's
// finish-tagged detector has arrived (finish propagates UP the DCB tree). So the
// DCB det-finish IS "my whole subtree has drained" — no control-path drain ripple
// (out_drain_done) is needed, and the FSM's only product of drain_done is `reset`.
//
// LEAF/MONO NOTE. On a distributed leaf, PhysicalMMIOs are fed the SAME in_ev
// this block gets (DCB and PhysicalMMIO reset align on the same cycle), so the
// leaf's out_ev is unused. On a monolithic board (ORIGINATE) the PhysicalMMIOs
// are instead fed the generated out_ev (registered), a 1-cycle skew vs the DCB
// reset — handled at assembly.
//
// Parameters:
//   event_mode  ORIGINATE / FORWARD
//   data_src    EXTERNAL / INTERNAL
//   m           number of post_select lines (ORIGINATE; 0 otherwise)
//   payload_w   width of the (reserved) ev_payload bus
#pragma once

#include <systemc>

#include <cstdint>

#include "signals.hpp"

namespace emu {

SC_MODULE(BoardControl) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- upward-data status (from this board's own input side) ---
    sc_core::sc_in<bool> in_valid;      // OR of all incoming data valids
    sc_core::sc_in<bool> data_attempt;  // attempt tag on incoming child data (unused if INTERNAL)

    // --- event bus in (from parent; for ORIGINATE this is the outside start/finish) ---
    sc_core::sc_in<bool>     ev_valid;
    sc_core::sc_in<uint32_t> ev_type;     // EvType
    sc_core::sc_in<Bits>     ev_payload;  // RESERVED (JUMP); forwarded, unused today
    sc_core::sc_in<bool>     ev_attempt;  // attempt carried by the event (unused for ORIGINATE)

    // --- post-select (ORIGINATE only; tie to 0 otherwise) ---
    sc_core::sc_in<Bits> post_select;          // [m] one bit per stage-board / source
    sc_core::sc_in<Bits> post_select_attempt;  // [m] attempt stamped at each source

    // --- drain-complete pulse (this board's whole subtree has drained; wired at
    //     assembly to the internal DetectorConstructBlock's DET out_finish) ---
    sc_core::sc_in<bool> drain_done;

    // --- this board's own output-valid status ---
    sc_core::sc_in<bool> data_valid_output;  // OR of this board's output valids

    // --- upward: attempt tag to parent ---
    sc_core::sc_out<bool> out_attempt;

    // --- raw current attempt: stamps the post_select fast path (this board's ps_out_attempt)
    //     and the own/gap post_select_attempt slots on a root ---
    sc_core::sc_out<bool> cur_attempt;

    // --- discard pulse (= abort_now): root routes it out so the host can reset the
    //     outside decoder / gap estimator on an abort ---
    sc_core::sc_out<bool> discard;

    // --- downward: event bus to children (unused on a distributed leaf) ---
    sc_core::sc_out<bool>     out_ev_valid;
    sc_core::sc_out<uint32_t> out_ev_type;
    sc_core::sc_out<Bits>     out_ev_payload;
    sc_core::sc_out<bool>     out_ev_attempt;

    // --- resets / gating for the rest of THIS board (never propagated) ---
    sc_core::sc_out<bool> reset;                // reset the detector-construct block etc.
    sc_core::sc_out<bool> set_in_data_to_zero;  // mask stale input valids to 0

    BoardControl(sc_core::sc_module_name nm, int event_mode, int data_src, int m, int payload_w);

    enum EventMode { ORIGINATE = 0, FORWARD = 1 };
    enum DataSrc   { EXTERNAL = 0, INTERNAL = 1 };
    enum EvType    { EV_START = 0, EV_ABORT = 1, EV_FINISH = 2 };
    enum State     { IDLE = 0, EXEC = 1, DRAIN = 2 };

  private:
    void comb();          // combinational: reset, set_in_data_to_zero, out_attempt, cur_attempt, discard
    void seq();           // clocked: FSM state, attempt, rst_for_finish, out_ev
    bool detect_abort();  // ORIGINATE: any post_select whose attempt matches internal attempt

    const int event_mode_;
    const int data_src_;
    const int m_;
    const int payload_w_;

    sc_core::sc_signal<int>  state_;
    sc_core::sc_signal<bool> attempt_;
    sc_core::sc_signal<bool> rst_for_finish_;
    // Registered abort_detected: breaks the self-clearing combinational loop
    // (post_select -> abort -> reset -> clears post_select). Delaying the board-
    // generated abort by one cycle makes it a stable single-cycle pulse the FSM can
    // latch, so EV_ABORT actually propagates. is_abort_ev (external) stays combinational.
    sc_core::sc_signal<bool> abort_det_q_;
};

}  // namespace emu
