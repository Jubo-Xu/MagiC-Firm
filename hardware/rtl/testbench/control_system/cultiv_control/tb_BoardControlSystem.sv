// tb_BoardControlSystem.sv — REAL-TREE system testbench for BoardControl.
//
// Wires four BoardControl instances the way a real machine would and runs one
// realistic trial: START -> run -> post-select ABORT -> run -> FINISH -> drain.
// Events ripple DOWN the tree; attempt tags ripple UP.
//
//   host --> ROOT --out_ev--> MID --out_ev--> LEAF        (events ripple DOWN)
//            <--out_attempt-- <--out_attempt--            (attempt tags ripple UP)
//
//   host --> MONO                                          (1-node tree: root+leaf)
//
//   role   EVENT_MODE  DATA_SRC
//   ROOT   ORIGINATE   EXTERNAL
//   MID    FORWARD     EXTERNAL
//   LEAF   FORWARD     INTERNAL
//   MONO   ORIGINATE   INTERNAL
//
// Signal names use the FULL BoardControl port names, prefixed by board role.
//
// DRAIN (new model): completion NO LONGER ripples up the CONTROL path (out_drain_done
// is retired). Each board's drain_done is its own DetectorConstructBlock DET
// out_finish, which fires only after every child's finish-tagged detector has
// arrived — so it naturally lands leaf -> mid -> root as finish propagates up the
// DATA path, and its only product is the board's local reset. This pure-BoardControl
// test has no DCB, so it DRIVES each board's drain_done directly in that order and
// checks each board's reset. The real DCB-driven drain is validated at the
// board-assembly level (P2d).
//
// WHAT TO WATCH IN THE WAVEFORM (all deterministic, 1 cycle per hop):
//   * START: root_out_ev=START, then one cycle later mid_out_ev=START (ripples down).
//   * ABORT: root_attempt flips to 1 first; while MID is still on attempt 0 and
//     sending data up, ROOT gates it (root_set_in_data_to_zero=1). One cycle later
//     MID gates LEAF the same way. Stale-data drop with REAL timing.
//   * FINISH: rides down (ROOT->MID->LEAF); then each board's drain_done (its DCB
//     det-finish) pulses one cycle later as reset, in leaf->mid->root order.
//
//  cyc | stimulus (host / tb)         | observed on the wires
// -----+------------------------------+------------------------------------------------
//   1  | root ev=START                | (root samples it)
//   2  | -                            | root_out_ev=START(att0)      -> ripples to MID
//   3  | -                            | mid_out_ev=START(att0)       -> ripples to LEAF
//   4  | -                            | leaf_out_ev=START; all out_attempt = 0
//  5-6 | in_valid=1 (data flows up)   | steady: no board asserts set_in_data_to_zero
//   7  | root_post_select=1 (att0)    | root_out_attempt->1, root_out_ev=ABORT(att1);
//      |   (asserted late in cyc6)    | root_set_in_data_to_zero=1 (gates MID stale att0)
//   8  | -                            | mid_out_attempt->1, mid_out_ev=ABORT(att1);
//      |                              | mid_set_in_data_to_zero=1 (gates LEAF)
//   9  | -                            | leaf_out_attempt->1; all attempt 1
// 10-11| run on attempt 1             | steady: no gating
//  12  | root ev=FINISH               | root_out_ev=FINISH(att1); ROOT -> DRAIN
//  13  | -                            | mid_out_ev=FINISH;         MID  -> DRAIN
//  14  | -                            | leaf gets FINISH;          LEAF -> DRAIN
//  15+ | leaf/mid/root drain_done     | each board's reset pulses (drain_done -> reset)
`timescale 1ns / 1ps

module tb_BoardControlSystem;

    localparam int PW = 2;
    localparam logic [1:0] EV_START = 2'd0, EV_ABORT = 2'd1, EV_FINISH = 2'd2;

    logic clk; initial begin clk = 1'b0; forever #5 clk = ~clk; end
    logic system_reset;
    int   errors = 0;

    // ===================== ROOT nets (host-driven) =====================
    logic       root_ev_valid;   logic [1:0] root_ev_type;   logic [PW-1:0] root_ev_payload;
    logic       root_post_select, root_post_select_attempt;
    logic       root_in_valid, root_drain_done;
    logic       root_out_attempt, root_cur_attempt, root_discard;
    logic       root_out_ev_valid, root_out_ev_attempt;
    logic [1:0] root_out_ev_type;   logic [PW-1:0] root_out_ev_payload;
    logic       root_reset, root_set_in_data_to_zero;

    // ===================== MID nets =====================
    logic       mid_in_valid, mid_drain_done;
    logic       mid_out_attempt;
    logic       mid_out_ev_valid, mid_out_ev_attempt;
    logic [1:0] mid_out_ev_type;   logic [PW-1:0] mid_out_ev_payload;
    logic       mid_reset, mid_set_in_data_to_zero;

    // ===================== LEAF nets =====================
    logic       leaf_in_valid, leaf_drain_done;
    logic       leaf_out_attempt;
    logic       leaf_out_ev_valid, leaf_out_ev_attempt;
    logic [1:0] leaf_out_ev_type;   logic [PW-1:0] leaf_out_ev_payload;
    logic       leaf_reset, leaf_set_in_data_to_zero;

    // ROOT: data_attempt <- MID's attempt up; event bus <- host
    BoardControl #(.EVENT_MODE(0), .DATA_SRC(0), .M(1), .PAYLOAD_W(PW)) u_root (
        .clk, .rst(system_reset),
        .in_valid            (root_in_valid),
        .data_attempt        (mid_out_attempt),          // <- MID's attempt, up the tree
        .ev_valid            (root_ev_valid),
        .ev_type             (root_ev_type),
        .ev_payload          (root_ev_payload),
        .ev_attempt          (1'b0),                     // ORIGINATE ignores this
        .post_select         (root_post_select),
        .post_select_attempt (root_post_select_attempt),
        .drain_done          (root_drain_done),
        .data_valid_output   (1'b1),
        .out_attempt         (root_out_attempt),
        .cur_attempt         (root_cur_attempt),
        .discard             (root_discard),
        .out_ev_valid        (root_out_ev_valid),
        .out_ev_type         (root_out_ev_type),
        .out_ev_payload      (root_out_ev_payload),
        .out_ev_attempt      (root_out_ev_attempt),
        .reset               (root_reset),
        .set_in_data_to_zero (root_set_in_data_to_zero));

    // MID: event bus <- ROOT.out_ev; data_attempt <- LEAF's attempt up
    BoardControl #(.EVENT_MODE(1), .DATA_SRC(0), .M(1), .PAYLOAD_W(PW)) u_mid (
        .clk, .rst(system_reset),
        .in_valid            (mid_in_valid),
        .data_attempt        (leaf_out_attempt),         // <- LEAF's attempt, up the tree
        .ev_valid            (root_out_ev_valid),        // <- ROOT's event, down the tree
        .ev_type             (root_out_ev_type),
        .ev_payload          (root_out_ev_payload),
        .ev_attempt          (root_out_ev_attempt),
        .post_select         (1'b0),
        .post_select_attempt (1'b0),
        .drain_done          (mid_drain_done),
        .data_valid_output   (1'b1),
        .out_attempt         (mid_out_attempt),
        .cur_attempt         (),
        .discard             (),
        .out_ev_valid        (mid_out_ev_valid),
        .out_ev_type         (mid_out_ev_type),
        .out_ev_payload      (mid_out_ev_payload),
        .out_ev_attempt      (mid_out_ev_attempt),
        .reset               (mid_reset),
        .set_in_data_to_zero (mid_set_in_data_to_zero));

    // LEAF: event bus <- MID.out_ev; INTERNAL (no incoming attempt tag)
    BoardControl #(.EVENT_MODE(1), .DATA_SRC(1), .M(1), .PAYLOAD_W(PW)) u_leaf (
        .clk, .rst(system_reset),
        .in_valid            (leaf_in_valid),
        .data_attempt        (1'b0),                     // INTERNAL: unused
        .ev_valid            (mid_out_ev_valid),         // <- MID's event, down the tree
        .ev_type             (mid_out_ev_type),
        .ev_payload          (mid_out_ev_payload),
        .ev_attempt          (mid_out_ev_attempt),
        .post_select         (1'b0),
        .post_select_attempt (1'b0),
        .drain_done          (leaf_drain_done),
        .data_valid_output   (1'b1),
        .out_attempt         (leaf_out_attempt),
        .cur_attempt         (),
        .discard             (),
        .out_ev_valid        (leaf_out_ev_valid),
        .out_ev_type         (leaf_out_ev_type),
        .out_ev_payload      (leaf_out_ev_payload),
        .out_ev_attempt      (leaf_out_ev_attempt),
        .reset               (leaf_reset),
        .set_in_data_to_zero (leaf_set_in_data_to_zero));

    // ===================== MONO (1-node tree: root+leaf fused) =====================
    logic       mono_ev_valid;   logic [1:0] mono_ev_type;   logic [PW-1:0] mono_ev_payload;
    logic       mono_post_select, mono_post_select_attempt;
    logic       mono_in_valid, mono_drain_done;
    logic       mono_out_attempt;
    logic       mono_out_ev_valid, mono_out_ev_attempt;
    logic [1:0] mono_out_ev_type;   logic [PW-1:0] mono_out_ev_payload;
    logic       mono_reset, mono_set_in_data_to_zero;

    BoardControl #(.EVENT_MODE(0), .DATA_SRC(1), .M(1), .PAYLOAD_W(PW)) u_mono (
        .clk, .rst(system_reset),
        .in_valid            (mono_in_valid),
        .data_attempt        (1'b0),                     // INTERNAL: unused
        .ev_valid            (mono_ev_valid),
        .ev_type             (mono_ev_type),
        .ev_payload          (mono_ev_payload),
        .ev_attempt          (1'b0),
        .post_select         (mono_post_select),
        .post_select_attempt (mono_post_select_attempt),
        .drain_done          (mono_drain_done),
        .data_valid_output   (1'b1),
        .out_attempt         (mono_out_attempt),
        .cur_attempt         (),
        .discard             (),
        .out_ev_valid        (mono_out_ev_valid),
        .out_ev_type         (mono_out_ev_type),
        .out_ev_payload      (mono_out_ev_payload),
        .out_ev_attempt      (mono_out_ev_attempt),
        .reset               (mono_reset),
        .set_in_data_to_zero (mono_set_in_data_to_zero));

    // free-running cycle counter (waveform landmark)
    int cycle;
    always_ff @(posedge clk or posedge system_reset)
        if (system_reset) cycle <= 0; else cycle <= cycle + 1;

    // ===================== helpers =====================
    task automatic tick; @(posedge clk); #1; endtask
    task automatic chk(input logic cond, input string msg);
        if (!cond) begin errors++; $display("  [%0t] FAIL: %s", $time, msg); end
        else                        $display("           ok: %s", msg);
    endtask

    initial begin
        // --- power-on reset ---
        system_reset = 1'b1;
        root_ev_valid=0; root_ev_type=0; root_ev_payload=0;
        root_post_select=0; root_post_select_attempt=0; root_in_valid=0; root_drain_done=0;
        mid_in_valid=0; mid_drain_done=0; leaf_in_valid=0; leaf_drain_done=0;
        mono_ev_valid=0; mono_ev_type=0; mono_ev_payload=0;
        mono_post_select=0; mono_post_select_attempt=0; mono_in_valid=0; mono_drain_done=0;
        tick; tick; system_reset = 1'b0; tick;

        $display("=== CHAIN: root -> mid -> leaf ===");

        // ---------- 1. START ripples DOWN (cyc 1-4) ----------
        root_ev_valid=1; root_ev_type=EV_START; tick;   // root samples START
        root_ev_valid=0;                                // deassert
        #1; chk(root_out_ev_valid && root_out_ev_type==EV_START && root_out_ev_attempt==0,
                "root emits START(att=0)");
        tick;
        #1; chk(mid_out_ev_valid && mid_out_ev_type==EV_START, "START reached MID (forwards to LEAF)");
        tick;                                           // leaf now EXEC
        #1; chk(root_out_attempt==0 && mid_out_attempt==0 && leaf_out_attempt==0,
                "all boards attempt=0 after START");

        // ---------- 2. run: data flowing up, attempts match, no gating (cyc 5-6) ----------
        root_in_valid=1; mid_in_valid=1; leaf_in_valid=1; tick; tick;
        #1; chk(!root_set_in_data_to_zero && !mid_set_in_data_to_zero && !leaf_set_in_data_to_zero,
                "steady run: no gating (attempts match)");

        // ---------- 3. post-select ABORT at ROOT (flip + gating ripple) ----------
        // The board-generated abort is REGISTERED (one-shot), so reset/discard assert the
        // cycle AFTER post_select, and the ABORT out_ev one cycle after that.
        root_post_select=1; root_post_select_attempt=0;   // stage board reports, attempt 0 == root's 0
        #1; chk(!root_reset,   "root abort registered (reset asserts next cycle)");
        chk(!root_discard, "root discard not yet (registered)");
        tick;                                             // abort_det_q now 1
        #1; chk(root_reset,   "root detects abort (reset pulse, XNOR match att=0)");
        chk(root_discard, "root discard pulses on abort (= abort_now)");
        root_post_select=0;                               // drop ps; the register carries it
        tick;                                             // abort acted on: flip + ABORT emitted
        #1; chk(!root_discard, "root discard clears after abort");
        chk(root_out_attempt==1 && root_out_ev_valid && root_out_ev_type==EV_ABORT && root_out_ev_attempt==1,
                "root flips to att=1, emits ABORT down");
        // ROOT now att=1 but MID still att=0 and still streaming data up -> ROOT gates it
        chk(root_set_in_data_to_zero, "ROOT gates MID's stale att=0 data (real-timing stale drop)");
        tick;
        #1; chk(mid_out_attempt==1, "ABORT reached MID: adopts att=1");
        // MID now att=1 but LEAF still att=0 -> MID gates LEAF
        chk(mid_set_in_data_to_zero, "MID gates LEAF's stale att=0 data");
        chk(!root_set_in_data_to_zero, "ROOT gating cleared (MID caught up to att=1)");
        tick;
        #1; chk(leaf_out_attempt==1, "ABORT reached LEAF: adopts att=1");
        chk(!mid_set_in_data_to_zero, "MID gating cleared (LEAF caught up)");

        // ---------- 4. run again on attempt 1 (cyc 10-11) ----------
        tick; tick;
        #1; chk(!root_set_in_data_to_zero && !mid_set_in_data_to_zero && !leaf_set_in_data_to_zero,
                "steady run on att=1: no gating");
        root_in_valid=0; mid_in_valid=0; leaf_in_valid=0;

        // ---------- 5. FINISH down, then per-board drain (DCB det-finish) -> reset ----------
        root_ev_valid=1; root_ev_type=EV_FINISH; tick;  // root -> DRAIN, emits FINISH
        root_ev_valid=0;
        #1; chk(root_out_ev_valid && root_out_ev_type==EV_FINISH, "root emits FINISH down, enters DRAIN");
        tick; tick;                                     // FINISH reaches MID then LEAF (all DRAIN)

        // each board's drain_done = its own DCB det-finish; arrives leaf -> mid -> root
        // (finish propagates up the DATA path). reset pulses the cycle AFTER drain_done.
        leaf_drain_done=1; tick; leaf_drain_done=0;
        #1; chk(leaf_reset, "LEAF drain_done (DCB det-finish) -> leaf reset");
        mid_drain_done=1;  tick; mid_drain_done=0;
        #1; chk(mid_reset,  "MID drain_done -> mid reset");
        root_drain_done=1; tick; root_drain_done=0;
        #1; chk(root_reset, "ROOT drain_done -> root reset");
        tick;
        #1; chk(!root_reset, "root reset 1-cycle only");

        // ================= MONO (root+leaf fused) =================
        $display("=== MONO: single-board tree ===");
        system_reset = 1'b1; tick; system_reset = 1'b0; tick;   // fresh trial

        mono_ev_valid=1; mono_ev_type=EV_START; tick; mono_ev_valid=0;
        #1; chk(mono_out_ev_valid && mono_out_ev_type==EV_START && mono_out_ev_attempt==0,
                "mono START(att=0)");
        tick;
        #1; chk(mono_out_attempt==0, "mono attempt=0");

        mono_post_select=1; mono_post_select_attempt=0;   // abort at attempt 0 (the XNOR case again)
        #1; chk(!mono_reset, "mono abort registered (reset next cycle)");
        tick;                                             // abort_det_q now 1
        #1; chk(mono_reset, "mono detects abort@0");
        mono_post_select=0;                               // drop ps; the register carries it
        tick;                                             // abort acted on: flip + ABORT emitted
        #1; chk(mono_out_attempt==1 && mono_out_ev_valid && mono_out_ev_type==EV_ABORT && mono_out_ev_attempt==1,
                "mono flips to att=1, emits ABORT");
        tick;

        mono_ev_valid=1; mono_ev_type=EV_FINISH; tick; mono_ev_valid=0;
        #1; chk(mono_out_ev_valid && mono_out_ev_type==EV_FINISH, "mono FINISH, enters DRAIN");
        tick;
        mono_drain_done=1; tick; mono_drain_done=0;
        #1; chk(mono_reset, "mono drain_done (self DCB det-finish) -> reset");
        tick;
        #1; chk(!mono_reset, "mono reset 1-cycle only");

        if (errors == 0) $display("tb_BoardControlSystem: PASS");
        else             $display("tb_BoardControlSystem: FAIL (%0d errors)", errors);
        $finish;
    end

endmodule
