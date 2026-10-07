// BoardControl.sv — per-board control / attempt state machine.
// 1:1 port of emulator/include/control_system/cultiv_control/board_control/board_control.hpp.
//
// One parameterized module for every board kind. Behaviour varies along TWO
// INDEPENDENT axes (so the monolithic root+leaf board is one combination, not a
// special case):
//
//   EVENT_MODE  ORIGINATE : make events from post_select + outside start/finish,
//                           drive out_ev from the FSM (root, monolithic)
//               FORWARD   : relay the received in_ev to children, delayed one
//                           cycle, and adopt its attempt (router, leaf)
//
//   DATA_SRC    EXTERNAL  : incoming data carries an attempt tag from children
//               INTERNAL  : physical-measurement inputs with no tag -> use our
//                           own attempt (leaf, monolithic)
//
//   board              EVENT_MODE   DATA_SRC
//   distributed root   ORIGINATE    EXTERNAL
//   distributed mid    FORWARD      EXTERNAL
//   distributed leaf   FORWARD      INTERNAL
//   monolithic         ORIGINATE    INTERNAL
//
// ATTEMPT (1 bit). START -> 0. Abort flips it (ORIGINATE) or adopts the event's
// attempt (FORWARD). A parent always changes attempt one cycle before its
// children can, so a mismatch on incoming data means "stale": gating is
// drop-only and no purge window is needed.
//
//   abort_detected      = |( post_select & (post_select_attempt ~^ {m{attempt}}) ) // XNOR = equality
//   abort_det_q         <= abort_detected & ~abort_det_q          // registered one-shot (breaks the
//                                                                 //   reset->clears-post_select loop)
//   abort_now           = abort_det_q | is_abort_ev              // board-abort registered; event-abort direct
//   reset               = rst | abort_now | rst_for_finish        // LOCAL only, never propagated
//   set_in_data_to_zero = in_valid & ( (data_att_eff != attempt) | abort_now )
//   out_attempt         = data_valid_output ? attempt : 0
//   cur_attempt         = attempt                                 // RAW; stamps the post_select fast path
//   discard             = abort_now                               // root routes it out to the host
//
// FSM (all boards): IDLE -> EXEC -> DRAIN -> IDLE. FINISH rides DOWN once
// (EXEC->DRAIN). DRAIN then waits for `drain_done` and resets the board locally.
// `drain_done` is DATA-PATH driven: wired at assembly to this board's own
// DetectorConstructBlock DET out_finish, which only fires after every child's
// finish-tagged detector has arrived (finish propagates UP the DCB tree). So the
// DCB det-finish IS "my whole subtree has drained" — no control-path drain ripple
// (out_drain_done) is needed, and drain_done's only product is `reset`.
//
// OUT_EV STYLE. Each cycle the out_ev bus DEFAULTS to the FORWARD (relay)
// behaviour; the ORIGINATE-only overrides sit inside `if (ORIGIN)` blocks with
// no else, and the "normal"/"drain-hold" cases need no else at all because the
// defaults are already correct for both modes.
`timescale 1ns / 1ps

module BoardControl #(
    parameter int EVENT_MODE = 0,   // 0 = ORIGINATE, 1 = FORWARD
    parameter int DATA_SRC   = 0,   // 0 = EXTERNAL,  1 = INTERNAL
    parameter int M          = 1,   // number of post_select lines (ORIGINATE; >=1)
    parameter int PAYLOAD_W  = 2    // reserved ev_payload width
) (
    input  logic                 clk,
    input  logic                 rst,

    // upward-data status (from this board's own input side)
    input  logic                 in_valid,       // OR of all incoming data valids
    input  logic                 data_attempt,   // attempt tag on incoming child data (unused if INTERNAL)

    // event bus in (from parent; ORIGINATE: outside start/finish)
    input  logic                 ev_valid,
    input  logic [1:0]           ev_type,        // 00=START 01=ABORT 10=FINISH
    input  logic [PAYLOAD_W-1:0] ev_payload,     // RESERVED (JUMP); forwarded, unused today
    input  logic                 ev_attempt,     // attempt carried by the event (unused for ORIGINATE)

    // post-select (ORIGINATE only; tie to 0 otherwise)
    input  logic [M-1:0]         post_select,
    input  logic [M-1:0]         post_select_attempt,

    // drain-complete pulse (this board's whole subtree has drained; wired at
    // assembly to the internal DetectorConstructBlock's DET out_finish)
    input  logic                 drain_done,

    // this board's own output-valid status
    input  logic                 data_valid_output,

    // upward: attempt tag to parent
    output logic                 out_attempt,

    // raw current attempt: stamps the post_select fast path (this board's
    // ps_out_attempt) and the own/gap post_select_attempt slots on a root
    output logic                 cur_attempt,

    // discard pulse (= abort_now): root routes it out so the host can reset the
    // outside decoder / gap estimator on an abort
    output logic                 discard,

    // downward: event bus to children (unused on a distributed leaf)
    output logic                 out_ev_valid,
    output logic [1:0]           out_ev_type,
    output logic [PAYLOAD_W-1:0] out_ev_payload,
    output logic                 out_ev_attempt,

    // resets / gating for the rest of THIS board (never propagated)
    output logic                 reset,
    output logic                 set_in_data_to_zero
);

    // ---- encodings ----
    // EVENT_MODE: 0 = ORIGINATE, 1 = FORWARD.  DATA_SRC: 0 = EXTERNAL, 1 = INTERNAL.
    localparam int ORIGINATE = 0;   // the value that selects event origination
    localparam int INTERNAL  = 1;   // the value that selects internal attempt
    localparam logic [1:0] EV_START = 2'd0, EV_ABORT = 2'd1, EV_FINISH = 2'd2;
    typedef enum logic [1:0] { IDLE = 2'd0, EXEC = 2'd1, DRAINS = 2'd2 } state_e;

    localparam bit ORIGIN = (EVENT_MODE == ORIGINATE);
    localparam bit INTERN = (DATA_SRC   == INTERNAL);

    // ---- state ----
    state_e state;
    logic   attempt;
    logic   rst_for_finish;
    // Registered one-shot of the board-generated abort. abort_detected feeds `reset`,
    // and `reset` async-clears post_select (its own source) — a self-clearing combinational
    // loop the FSM edge can never latch. Registering it decouples the reset by one cycle so
    // the FSM reliably captures the abort and emits EV_ABORT. is_abort_ev (external) is NOT
    // part of this loop and stays combinational.
    logic   abort_det_q;

    // ---- decoded events ----
    logic is_start, is_abort_ev, is_finish;
    assign is_start    = ev_valid && (ev_type == EV_START);
    assign is_abort_ev = ev_valid && (ev_type == EV_ABORT);
    assign is_finish   = ev_valid && (ev_type == EV_FINISH);

    // ---- abort detection (ORIGINATE): XNOR == attempt equality, then OR-reduce ----
    // abort_now uses the REGISTERED board-abort (abort_det_q, see below), not the raw
    // combinational abort_detected — so the abort->reset path no longer clears its own
    // trigger within the cycle. The external event-abort (is_abort_ev) stays combinational.
    logic abort_detected, abort_now;
    assign abort_detected = ORIGIN ? |( post_select & (post_select_attempt ~^ {M{attempt}}) )
                                   : 1'b0;
    assign abort_now = abort_det_q | is_abort_ev;

    // ---- combinational outputs ----
    logic data_att_eff;
    assign data_att_eff        = INTERN ? attempt : data_attempt;
    assign reset               = rst | abort_now | rst_for_finish;
    assign set_in_data_to_zero = in_valid & ( (data_att_eff != attempt) | abort_now );
    assign out_attempt         = data_valid_output ? attempt : 1'b0;
    assign cur_attempt         = attempt;      // raw attempt — post_select fast-path / gap stamp
    assign discard             = abort_now;    // notify the outside (root) to reset on abort

    // ---- clocked FSM: state, attempt, rst_for_finish, out_ev ----
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state          <= IDLE;
            attempt        <= 1'b0;
            rst_for_finish <= 1'b0;
            abort_det_q    <= 1'b0;
            out_ev_valid   <= 1'b0;
            out_ev_type    <= 2'd0;
            out_ev_payload <= '0;
            out_ev_attempt <= 1'b0;
        end else begin
            // Re-arm the abort one-shot: pulse for exactly one cycle on a rising board-abort.
            // The reset/attempt-flip it triggers only become visible next cycle, so the
            // ~abort_det_q guard blocks a spurious second pulse (double attempt-flip).
            abort_det_q <= abort_detected & ~abort_det_q;

            // DEFAULT out_ev = the FORWARD (relay) behaviour; ORIGINATE overrides
            // below. ORIGINATE's default is "no event".
            out_ev_valid   <= ORIGIN ? 1'b0    : ev_valid;
            out_ev_type    <= ORIGIN ? 2'd0    : ev_type;
            out_ev_attempt <= ORIGIN ? 1'b0    : ev_attempt;
            out_ev_payload <= ev_payload;

            rst_for_finish <= 1'b0;   // default: not finishing this cycle

            unique case (state)
                IDLE: begin
                    if (is_start) begin
                        attempt <= ORIGIN ? 1'b0 : ev_attempt;
                        state   <= EXEC;
                        if (ORIGIN) begin
                            out_ev_valid   <= 1'b1;
                            out_ev_type    <= EV_START;
                            out_ev_attempt <= 1'b0;      // ORIGINATE START attempt is always 0
                        end
                    end
                end

                EXEC: begin
                    if (abort_now) begin
                        attempt <= ORIGIN ? ~attempt : ev_attempt;
                        state   <= EXEC;
                        if (ORIGIN) begin
                            out_ev_valid   <= 1'b1;
                            out_ev_type    <= EV_ABORT;
                            out_ev_attempt <= ~attempt;  // NEW attempt
                        end
                    end else if (is_finish) begin
                        if (drain_done) begin
                            // children already IDLE: nothing left below to notify
                            out_ev_valid   <= 1'b0;
                            out_ev_type    <= 2'd0;
                            out_ev_attempt <= 1'b0;
                            state          <= IDLE;
                            attempt        <= 1'b0;
                            rst_for_finish <= 1'b1;
                        end else begin
                            // first FINISH: ride it DOWN once, then drain.
                            // FORWARD already relayed FINISH via the default above.
                            if (ORIGIN) begin
                                out_ev_valid   <= 1'b1;
                                out_ev_type    <= EV_FINISH;
                                out_ev_attempt <= attempt;
                            end
                            state <= DRAINS;
                        end
                    end
                    // else normal running: defaults already correct for both modes
                end

                DRAINS: begin
                    if (drain_done) begin
                        out_ev_valid   <= 1'b0;
                        out_ev_type    <= 2'd0;
                        out_ev_attempt <= 1'b0;
                        state          <= IDLE;
                        attempt        <= 1'b0;
                        rst_for_finish <= 1'b1;
                    end
                    // else draining: hold; defaults already correct (parent silent)
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule
