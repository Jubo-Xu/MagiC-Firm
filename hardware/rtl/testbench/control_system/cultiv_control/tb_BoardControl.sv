// tb_BoardControl.sv — testbench for BoardControl, all four board combinations.
//
// Written in BOTH styles:
//   (1) a per-cycle expected-output TABLE before each scenario, so the Vivado
//       waveform can be read off directly and compared against the comments;
//   (2) self-checking: each cycle compares DUT outputs against those expected
//       values, counts mismatches, prints PASS/FAIL (for Verilator / xsim).
//
// Mirrors the SystemC unit test
//   emulator/tests/control_system/cultiv_control/board_control/test_board_control.cpp
//
// Four DUTs, one per (EVENT_MODE, DATA_SRC) combination:
//   root : ORIGINATE + EXTERNAL     mono : ORIGINATE + INTERNAL
//   mid  : FORWARD   + EXTERNAL     leaf : FORWARD   + INTERNAL
//
// TIMING MODEL. reset / set_in_data_to_zero / out_attempt are
// COMBINATIONAL (reflect THIS cycle's inputs); attempt and out_ev are REGISTERED
// (reflect the PREVIOUS cycle's inputs). Each cycle:
//     adv();               // @(posedge); #1  -> registered outputs = prev cycle
//     <check registered outputs produced by the previous cycle>
//     <drive this cycle's inputs>;  #1;      // settle combinational
//     <check combinational outputs for this cycle>
// So a registered response to inputs driven in cycle N is checked at cycle N+1;
// a combinational response is checked in cycle N itself.
`timescale 1ns / 1ps

module tb_BoardControl;

    localparam int PW = 2;
    localparam logic [1:0] EV_START = 2'd0, EV_ABORT = 2'd1, EV_FINISH = 2'd2;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    int errors = 0;
    int cyc    = 0;

    logic rst;
    // root (ORIGINATE+EXTERNAL)
    logic r_inv, r_datt, r_evv, r_eva, r_dd, r_dvo;  logic [1:0] r_evt;  logic [PW-1:0] r_evp;
    logic r_ps, r_psa;
    logic r_oatt, r_oevv, r_oeva, r_rst, r_setz, r_cur, r_disc;  logic [1:0] r_oevt;  logic [PW-1:0] r_oevp;
    // mono (ORIGINATE+INTERNAL)
    logic m_inv, m_datt, m_evv, m_eva, m_dd, m_dvo;  logic [1:0] m_evt;  logic [PW-1:0] m_evp;
    logic m_ps, m_psa;
    logic m_oatt, m_oevv, m_oeva, m_rst, m_setz, m_cur, m_disc;  logic [1:0] m_oevt;  logic [PW-1:0] m_oevp;
    // mid (FORWARD+EXTERNAL)
    logic d_inv, d_datt, d_evv, d_eva, d_dd, d_dvo;  logic [1:0] d_evt;  logic [PW-1:0] d_evp;
    logic d_oatt, d_oevv, d_oeva, d_rst, d_setz, d_cur, d_disc;  logic [1:0] d_oevt;  logic [PW-1:0] d_oevp;
    // leaf (FORWARD+INTERNAL)
    logic l_inv, l_datt, l_evv, l_eva, l_dd, l_dvo;  logic [1:0] l_evt;  logic [PW-1:0] l_evp;
    logic l_oatt, l_oevv, l_oeva, l_rst, l_setz, l_cur, l_disc;  logic [1:0] l_oevt;  logic [PW-1:0] l_oevp;

    BoardControl #(.EVENT_MODE(0), .DATA_SRC(0), .M(1), .PAYLOAD_W(PW)) u_root (
        .clk, .rst, .in_valid(r_inv), .data_attempt(r_datt), .ev_valid(r_evv), .ev_type(r_evt),
        .ev_payload(r_evp), .ev_attempt(r_eva), .post_select(r_ps), .post_select_attempt(r_psa),
        .drain_done(r_dd), .data_valid_output(r_dvo), .out_attempt(r_oatt),
        .cur_attempt(r_cur), .discard(r_disc),
        .out_ev_valid(r_oevv), .out_ev_type(r_oevt), .out_ev_payload(r_oevp), .out_ev_attempt(r_oeva),
        .reset(r_rst), .set_in_data_to_zero(r_setz));

    BoardControl #(.EVENT_MODE(0), .DATA_SRC(1), .M(1), .PAYLOAD_W(PW)) u_mono (
        .clk, .rst, .in_valid(m_inv), .data_attempt(m_datt), .ev_valid(m_evv), .ev_type(m_evt),
        .ev_payload(m_evp), .ev_attempt(m_eva), .post_select(m_ps), .post_select_attempt(m_psa),
        .drain_done(m_dd), .data_valid_output(m_dvo), .out_attempt(m_oatt),
        .cur_attempt(m_cur), .discard(m_disc),
        .out_ev_valid(m_oevv), .out_ev_type(m_oevt), .out_ev_payload(m_oevp), .out_ev_attempt(m_oeva),
        .reset(m_rst), .set_in_data_to_zero(m_setz));

    BoardControl #(.EVENT_MODE(1), .DATA_SRC(0), .M(1), .PAYLOAD_W(PW)) u_mid (
        .clk, .rst, .in_valid(d_inv), .data_attempt(d_datt), .ev_valid(d_evv), .ev_type(d_evt),
        .ev_payload(d_evp), .ev_attempt(d_eva), .post_select(1'b0), .post_select_attempt(1'b0),
        .drain_done(d_dd), .data_valid_output(d_dvo), .out_attempt(d_oatt),
        .cur_attempt(d_cur), .discard(d_disc),
        .out_ev_valid(d_oevv), .out_ev_type(d_oevt), .out_ev_payload(d_oevp), .out_ev_attempt(d_oeva),
        .reset(d_rst), .set_in_data_to_zero(d_setz));

    BoardControl #(.EVENT_MODE(1), .DATA_SRC(1), .M(1), .PAYLOAD_W(PW)) u_leaf (
        .clk, .rst, .in_valid(l_inv), .data_attempt(l_datt), .ev_valid(l_evv), .ev_type(l_evt),
        .ev_payload(l_evp), .ev_attempt(l_eva), .post_select(1'b0), .post_select_attempt(1'b0),
        .drain_done(l_dd), .data_valid_output(l_dvo), .out_attempt(l_oatt),
        .cur_attempt(l_cur), .discard(l_disc),
        .out_ev_valid(l_oevv), .out_ev_type(l_oevt), .out_ev_payload(l_oevp), .out_ev_attempt(l_oeva),
        .reset(l_rst), .set_in_data_to_zero(l_setz));

    task automatic adv; @(posedge clk); #1; cyc++; endtask
    task automatic settle; #1; endtask

    task automatic cbit(input logic got, input logic exp, input string tag);
        if (got !== exp) begin errors++;
            $display("  cyc %0d MISMATCH %s: got %0b exp %0b", cyc, tag, got, exp); end
    endtask
    task automatic cev(input logic v, input logic [1:0] t, input logic a,
                       input logic ev, input logic [1:0] et, input logic ea, input string tag);
        if (v !== ev || (ev && (t !== et || a !== ea))) begin errors++;
            $display("  cyc %0d MISMATCH %s: got v=%0b t=%0d a=%0b exp v=%0b t=%0d a=%0b",
                     cyc, tag, v, t, a, ev, et, ea); end
    endtask

    task automatic clear_inputs;
        r_inv=0; r_datt=0; r_evv=0; r_evt=0; r_evp=0; r_eva=0; r_ps=0; r_psa=0; r_dd=0; r_dvo=1;
        m_inv=0; m_datt=0; m_evv=0; m_evt=0; m_evp=0; m_eva=0; m_ps=0; m_psa=0; m_dd=0; m_dvo=1;
        d_inv=0; d_datt=0; d_evv=0; d_evt=0; d_evp=0; d_eva=0; d_dd=0; d_dvo=1;
        l_inv=0; l_datt=0; l_evv=0; l_evt=0; l_evp=0; l_eva=0; l_dd=0; l_dvo=1;
    endtask

    initial begin
        rst = 1'b1; clear_inputs();
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_BoardControl: four boards root/mono/mid/leaf");

        // ============================ ORIGINATE: root & mono ============================
        // (both start IDLE, attempt=0. reg = out_ev; comb = reset)
        //
        // NOTE: the BOARD-GENERATED abort is REGISTERED (one-shot abort_det_q) to break the
        // self-clearing reset->clears-post_select loop. So a post_select match drives
        // reset/discard ONE CYCLE LATER than it is asserted, and the ABORT out_ev one cycle
        // after that. (The event-abort path, mid/leaf below, is combinational and unchanged.)
        //
        //  cyc | drive (ps/psa, ev)      | COMB this cyc       | REG (from prev cyc)
        // -----+-------------------------+--------------------+--------------------------------
        //  C1  | ev=START                | reset=0            | out_ev=0
        //  C2  | -                       | reset=0            | out_ev=START att=0   <-- from C1
        //  C3  | -                       | reset=0            | out_ev=0
        //  C4  | ps=1 psa=0 (att==0)     | reset=0 (regd)     | out_ev=0
        //  C5  | ps=0 (abort_det_q=1)    | reset=1  <<XNOR    | out_ev=0
        //  C6  | -                       | reset=0            | out_ev=ABORT att=1   <-- flip
        //  C7  | -                       | reset=0            | out_ev=0   (attempt now 1)
        //  C8  | ps=1 psa=0 (mismatch)   | reset=0  IGNORED   | out_ev=0
        //  C9  | ps=0                    | reset=0            | out_ev=0
        //  C10 | ps=1 psa=1 (att==1)     | reset=0 (regd)     | out_ev=0
        //  C11 | ps=0 (abort_det_q=1)    | reset=1            | out_ev=0
        //  C12 | -                       | reset=0            | out_ev=ABORT att=0   <-- flip back
        //  C13 | -                       | reset=0            | out_ev=0   (attempt now 0)
        //  C14 | ev=FINISH               | reset=0            | out_ev=0
        //  C15 | -                       | reset=0            | out_ev=FINISH att=0  <-- ride down
        //  C16 | - (draining)            | reset=0            | out_ev=0
        //  C17 | drain_done=1            | reset=0            | out_ev=0
        //  C18 | -                       | reset=1            | out_ev=0   <-- rst_for_finish pulse
        //  C19 | -                       | reset=0            | out_ev=0   (1-cycle pulse)

        adv;                                                     // C1
        r_evv=1; r_evt=EV_START; m_evv=1; m_evt=EV_START; settle;
        cbit(r_rst, 0, "C1 root reset");
        adv;                                                     // C2
        cev(r_oevv,r_oevt,r_oeva, 1,EV_START,0, "C2 root START");
        cev(m_oevv,m_oevt,m_oeva, 1,EV_START,0, "C2 mono START");
        r_evv=0; m_evv=0; settle;
        adv;                                                     // C3
        adv;                                                     // C4  ps match -> abort registers (reset NEXT cyc)
        r_ps=1; r_psa=0; m_ps=1; m_psa=0; settle;
        cbit(r_rst, 0, "C4 root abort@0 registered (reset next cyc)");
        cbit(m_rst, 0, "C4 mono abort@0 registered");
        cbit(r_disc, 0, "C4 root no discard yet");
        adv;                                                     // C5  abort_det_q=1 -> reset+discard
        r_ps=0; r_psa=0; m_ps=0; m_psa=0; settle;                // drop ps; the register carries it
        cbit(r_rst, 1, "C5 root abort@0 (XNOR) reset");
        cbit(m_rst, 1, "C5 mono abort@0 (XNOR) reset");
        cbit(r_disc, 1, "C5 root discard on abort");
        cbit(m_disc, 1, "C5 mono discard on abort");
        adv;                                                     // C6  ABORT out_ev, attempt flips ->1
        cev(r_oevv,r_oevt,r_oeva, 1,EV_ABORT,1, "C6 root ABORT flip->1");
        cev(m_oevv,m_oevt,m_oeva, 1,EV_ABORT,1, "C6 mono ABORT flip->1");
        settle;
        adv;                                                     // C7  (attempt now 1)
        adv;                                                     // C8  stale post_select (att 1, stamped 0)
        r_ps=1; r_psa=0; m_ps=1; m_psa=0; settle;
        cbit(r_rst, 0, "C8 root stale post_select ignored");
        cbit(m_rst, 0, "C8 mono stale post_select ignored");
        cbit(r_disc, 0, "C8 root no discard (stale)");
        adv;                                                     // C9
        r_ps=0; m_ps=0; settle;
        adv;                                                     // C10 ps match @att1 -> registers
        r_ps=1; r_psa=1; m_ps=1; m_psa=1; settle;
        cbit(r_rst, 0, "C10 root abort@1 registered (reset next cyc)");
        adv;                                                     // C11 abort_det_q=1 -> reset+discard
        r_ps=0; m_ps=0; settle;
        cbit(r_rst, 1, "C11 root abort@1 reset");
        cbit(r_disc, 1, "C11 root discard on abort@1");
        adv;                                                     // C12 ABORT out_ev, attempt flips ->0
        cev(r_oevv,r_oevt,r_oeva, 1,EV_ABORT,0, "C12 root ABORT flip->0");
        settle;
        adv;                                                     // C13 (attempt now 0)
        adv;                                                     // C14
        r_evv=1; r_evt=EV_FINISH; m_evv=1; m_evt=EV_FINISH; settle;
        adv;                                                     // C15
        cev(r_oevv,r_oevt,r_oeva, 1,EV_FINISH,0, "C15 root FINISH down");
        r_evv=0; m_evv=0; settle;
        adv;                                                     // C16 draining
        adv;                                                     // C17
        r_dd=1; m_dd=1; settle;
        adv;                                                     // C18
        cbit(r_rst, 1, "C18 root finish reset");
        cbit(m_rst, 1, "C18 mono finish reset");
        cbit(r_disc, 0, "C18 root discard 0 on finish reset");
        cbit(m_disc, 0, "C18 mono discard 0 on finish reset");
        r_dd=0; m_dd=0; settle;
        adv;                                                     // C19
        cbit(r_rst, 0, "C19 root finish reset 1-cycle only");

        // ============================ FORWARD: mid & leaf ============================
        // (both IDLE, attempt=0. reg = out_ev forwarded; comb = reset / set_in_data_to_zero)
        //
        //  cyc | drive                       | COMB this cyc                    | REG (prev cyc)
        // -----+-----------------------------+---------------------------------+------------------
        //  D1  | ev=START att=0              | reset=0                         | out_ev=0
        //  D2  | -                           | reset=0                         | out_ev=START att=0
        //  D3  | -                           | reset=0                         | out_ev=0
        //  D4  | ev=ABORT att=1              | reset=1  (is_abort_ev)          | out_ev=0
        //  D5  | -                           | reset=0                         | out_ev=ABORT att=1
        //  D6  | -                           | (attempt now 1 on both)         | out_ev=0
        //  D7  | in_valid=1 data_attempt=0   | mid setz=1 (mismatch) ;         | out_ev=0
        //      |                             | leaf setz=0 (INTERNAL ignores)  |
        //  D8  | -                           | -                               | -
        //  D9  | leaf in_valid=1, ev=ABORT   | leaf setz=1 (abort_now)         | -

        adv;                                                     // D1
        d_evv=1; d_evt=EV_START; d_eva=0; l_evv=1; l_evt=EV_START; l_eva=0; settle;
        adv;                                                     // D2
        cev(d_oevv,d_oevt,d_oeva, 1,EV_START,0, "D2 mid START fwd");
        cev(l_oevv,l_oevt,l_oeva, 1,EV_START,0, "D2 leaf START fwd");
        d_evv=0; l_evv=0; settle;
        adv;                                                     // D3
        adv;                                                     // D4
        d_evv=1; d_evt=EV_ABORT; d_eva=1; l_evv=1; l_evt=EV_ABORT; l_eva=1; settle;
        cbit(d_rst, 1, "D4 mid abort reset");
        cbit(l_rst, 1, "D4 leaf abort reset");
        cbit(d_disc, 1, "D4 mid discard on abort ev");
        cbit(l_disc, 1, "D4 leaf discard on abort ev");
        adv;                                                     // D5
        cev(d_oevv,d_oevt,d_oeva, 1,EV_ABORT,1, "D5 mid ABORT fwd att=1");
        cev(l_oevv,l_oevt,l_oeva, 1,EV_ABORT,1, "D5 leaf ABORT fwd att=1");
        d_evv=0; l_evv=0; settle;
        adv;                                                     // D6
        adv;                                                     // D7
        d_inv=1; d_datt=0; l_inv=1; l_datt=0; settle;            // valid data tagged 0, internal=1
        cbit(d_setz, 1, "D7 mid gates mismatch (EXTERNAL)");
        cbit(l_setz, 0, "D7 leaf ignores data_attempt (INTERNAL)");
        adv;                                                     // D8
        d_inv=0; l_inv=0; settle;
        adv;                                                     // D9
        l_inv=1; l_evv=1; l_evt=EV_ABORT; l_eva=1; settle;
        cbit(l_setz, 1, "D9 leaf gates on abort");
        adv;
        l_inv=0; l_evv=0; settle;

        if (errors == 0) $display("tb_BoardControl: PASS");
        else             $display("tb_BoardControl: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
