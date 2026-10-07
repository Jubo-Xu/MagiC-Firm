// tb_Postselect.sv — testbench for Postselect (D=4).
//
// Written in BOTH styles:
//   (1) a per-cycle table below, so the Vivado waveform can be read off directly;
//   (2) self-checking: mismatch count + PASS/FAIL for Verilator / xsim.
//
// Mirrors emulator/tests/.../test_postselect.cpp. Postselect has NO internal pc:
// it registers the mask one cycle (mask_reg) to align with OutputSync's REGISTERED
// detector word, then post_select = in_valid & |(mask_reg & in_det) combinationally.
//
// REALISTIC TIMING: in the real block Postselect's in_det/in_valid are OutputSync's
// REGISTERED outputs (update at the posedge). This tb registers the driven
// in_det/in_valid so they co-update with mask_reg at the posedge — so post_select
// transitions only at the posedge (clock-to-Q), holds the full cycle, and does not
// glitch. (Driving them combinationally 1 ns after the posedge, as an early draft
// did, produces cosmetic mid-cycle glitches that the real registered inputs cannot.)
//
// Each cyc() presents ONE det-time's (mask, det, valid) together; the mask is
// registered into mask_reg and the det/valid into in_det/in_valid, so their result
// appears on post_select the NEXT cycle. So post_select checked at cyc k reflects
// cyc (k-1)'s inputs.
//
// ==========================================================================
//  cyc | mask det  valid | post_select (=prev cyc result) | note
// -----+-----------------+--------------------------------+-------------------
//   0  | 0011 1010 1     |     0  (reset)                 | l1 masked & set
//   1  | 0000 1111 1     |     1  <- c0: |(0011 & 1010)   | mask empty
//   2  | 1000 0000 1     |     0  <- c1: mask empty       | l3 masked, det 0
//   3  | 0100 0100 1     |     0  <- c2: det l3=0         | l2 masked & set
//   4  | 0000 0000 0     |     1  <- c3: |(0100 & 0100)   | valid=0
//   5  | 0000 0000 0     |     0  <- c4: valid=0          | idle
// ==========================================================================
// Bit order: bit i == line i. "0011" = lines 0,1.
`timescale 1ns / 1ps

module tb_Postselect;

    localparam int D = 4;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    logic         rst, in_valid, post_select;
    logic [D-1:0] in_det, postselect_mask;

    // driven inputs registered so they update at the posedge like OutputSync's outputs
    logic         nd_valid;
    logic [D-1:0] nd_det;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin in_det <= '0; in_valid <= 1'b0; end
        else     begin in_det <= nd_det; in_valid <= nd_valid; end
    end

    int errors = 0, cyc_i = 0;

    Postselect #(.D(D)) dut (
        .clk, .rst, .in_valid, .in_det, .postselect_mask, .post_select);

    // Present THIS det-time's (mask, det, valid). They are registered (mask into
    // mask_reg, det/valid into in_det/in_valid) so the result lands on post_select
    // next cycle. Check the CURRENT post_select = the PREVIOUS cyc's result.
    task automatic cyc(input logic [D-1:0] mask,
                       input logic [D-1:0] det,
                       input logic         valid,
                       input logic         exp_ps,
                       input string        note);
        @(posedge clk);
        #1;
        if (post_select !== exp_ps) begin
            errors++;
            $display("  cyc %0d MISMATCH: post_select=%b (exp %b)  now driving mask=%b det=%b valid=%b  %s",
                     cyc_i, post_select, exp_ps, mask, det, valid, note);
        end else begin
            $display("  cyc %0d ok: post_select=%b  now driving mask=%b det=%b valid=%b  %s",
                     cyc_i, post_select, mask, det, valid, note);
        end
        postselect_mask = mask; nd_det = det; nd_valid = valid;
        cyc_i++;
    endtask

    initial begin
        rst = 1'b1; postselect_mask = '0; nd_det = '0; nd_valid = 1'b0;
        @(posedge clk); @(posedge clk); #1; rst = 1'b0;
        $display("tb_Postselect: D=%0d", D);

        //   mask     det      valid  exp_ps  note
        cyc(4'b0011, 4'b1010, 1'b1, 1'b0, "l1 masked & set (result next cyc)");
        cyc(4'b0000, 4'b1111, 1'b1, 1'b1, "prev l1 -> FIRE; mask empty now");
        cyc(4'b1000, 4'b0000, 1'b1, 1'b0, "prev empty mask -> 0; l3 masked/det0 now");
        cyc(4'b0100, 4'b0100, 1'b1, 1'b0, "prev det l3=0 -> 0; l2 masked & set now");
        cyc(4'b0000, 4'b0000, 1'b0, 1'b1, "prev l2 -> FIRE; valid=0 now");
        cyc(4'b0000, 4'b0000, 1'b0, 1'b0, "prev valid=0 -> 0");

        repeat (4) @(posedge clk);   // trailing cycles: full waveform tail

        if (errors == 0) $display("tb_Postselect: PASS");
        else             $display("tb_Postselect: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
