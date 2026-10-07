// tb_StimReadout.sv — unit test for the sim-only StimReadout model.
//
// Hand-built BRAM: SHOTS=2, DEPTH=3, M=3, meas[addr]=addr, valid[addr]=all-ones
// (stimreadout_ut_meas.mem / stimreadout_ut_valid.mem). Drive the MMIO command stream
// (round index + cw_gen_finish) and the leaf event bus (ev_valid/ev_type) and check the
// measurement outputs + shot_reg. Covers: read address = base + index, ABORT advancing to
// the next shot's base (decoupled from the read), the last-shot advance guard, cw_gen_finish
// (finish bit + reset), and mmio_out_valid=0 (idle).
//
// Registered outputs are sampled #1 after the posedge, so each step checks the read it just
// drove (this differs from the SystemC tb's deferred check, a simulator quirk).
`timescale 1ns / 1ps

module tb_StimReadout;
    localparam int M = 3, SHOTS = 2, DEPTH = 3, IDX_W = 2;
    localparam logic [1:0] EV_ABORT = 2'd1;

    logic clk;
    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    int   errors = 0;
    logic rst, mmio_v, cw_fin, ev_v;
    logic [1:0]       ev_t;
    logic [IDX_W-1:0] mmio_data;
    logic [M-1:0]     o_meas, o_valid, o_finish;
    logic [$clog2(SHOTS+1)-1:0] shot;

    StimReadout #(
        .M(M), .SHOTS(SHOTS), .DEPTH(DEPTH),
        .MEAS_FILE ("testbench/control_system/stimreadout_ut_meas.mem"),
        .VALID_FILE("testbench/control_system/stimreadout_ut_valid.mem"),
        .INIT_HEX(1'b0)
    ) dut (
        .clk(clk), .rst(rst),
        .mmio_out_valid(mmio_v), .mmio_out_data(mmio_data), .cw_gen_finish(cw_fin),
        .ev_valid(ev_v), .ev_type(ev_t),
        .out_meas(o_meas), .out_meas_valid(o_valid), .out_meas_finish(o_finish), .shot_reg(shot)
    );

    task automatic cint(input int got, input int exp, input string tag);
        if (got !== exp) begin errors++; $display("  MISMATCH %s: got %0d exp %0d", tag, got, exp); end
    endtask

    // drive one cycle (read idx + finish, and/or an abort); the registered outputs settle #1
    // after the posedge and reflect it: meas[base+idx], valid, finish bit, shot_reg after update.
    task automatic step(input logic v, input int idx, input logic abort, input logic fin,
                        input int em, input int ev, input int ef, input int es, input string tag);
        mmio_v = v; mmio_data = idx[IDX_W-1:0]; cw_fin = fin;
        ev_v = abort; ev_t = abort ? EV_ABORT : 2'd0;
        @(posedge clk); #1;
        cint(o_meas,   em, {tag, " meas"});
        cint(o_valid,  ev, {tag, " valid"});
        cint(o_finish, ef, {tag, " finish"});
        cint(shot,     es, {tag, " shot_reg"});
    endtask

    initial begin
        rst = 1; mmio_v = 0; cw_fin = 0; ev_v = 0; mmio_data = '0; ev_t = '0;
        @(posedge clk); @(posedge clk); #1; rst = 0;

        // shot 0 (base 0): read rounds 0,1,2 -> addr base+idx = 0,1,2. No abort -> shot holds 0.
        step(1, 0, 0, 0,  0, 7, 0, 0, "s0 r0");
        step(1, 1, 0, 0,  1, 7, 0, 0, "s0 r1");
        step(1, 2, 0, 0,  2, 7, 0, 0, "s0 r2");
        // ABORT (no read): jump to shot 1's base (3), shot 0->1.
        step(0, 0, 1, 0,  0, 0, 0, 1, "abort->s1");
        // shot 1 (base 3): read rounds 0,1,2 -> addr 3,4,5.
        step(1, 0, 0, 0,  3, 7, 0, 1, "s1 r0");
        step(1, 1, 0, 0,  4, 7, 0, 1, "s1 r1");
        step(1, 2, 0, 0,  5, 7, 0, 1, "s1 r2");
        // ABORT at the last shot (shot+1 == SHOTS) -> GUARD: no advance, base/shot hold.
        step(0, 0, 1, 0,  0, 0, 0, 1, "abort-guard");
        // live round + cw_gen_finish: reads base(3)+2=5, finish=valid, shot_reg resets to 0.
        step(1, 2, 0, 1,  5, 7, 7, 0, "finish");
        // idle: mmio_out_valid=0 -> outputs 0.
        step(0, 0, 0, 0,  0, 0, 0, 0, "idle");

        repeat (4) @(posedge clk);
        if (errors == 0) $display("tb_StimReadout: PASS");
        else             $display("tb_StimReadout: FAIL (%0d mismatches)", errors);
        $finish;
    end
endmodule
