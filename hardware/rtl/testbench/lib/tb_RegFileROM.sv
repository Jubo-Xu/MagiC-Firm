// tb_RegFileROM.sv — directed testbench for RegFileROM.
//
// Written in BOTH styles:
//   (1) an explicit table below with the expected values also passed inline to
//       every check() call, so the Vivado waveform can be read off directly;
//   (2) self-checking with mismatch counting and PASS/FAIL for Verilator / xsim.
//
// Mirrors the SystemC unit test emulator/tests/lib/test_regfile_rom.cpp.
//
// The module is purely combinational (asynchronous read), so there is no clock:
// each step applies an address and checks the data after a settling delay. In
// the waveform `data` should track `addr` with no clock edge in between — that
// is the whole difference from SyncROM.
//
// ROM contents from regfile_rom_init.mem via $readmemb (DEPTH=8, WIDTH=6):
//
// ============================================================================
//  step | addr | mem word (bin) | data (dec) | note
// ------+------+----------------+------------+------------------------------
//   0   |   0  |    000001      |      1     | bit 0 only
//   1   |   1  |    000010      |      2     | bit 1 only
//   2   |   2  |    000100      |      4     |
//   3   |   3  |    001000      |      8     |
//   4   |   4  |    010000      |     16     |
//   5   |   5  |    100000      |     32     | bit 5 (MSB) only
//   6   |   6  |    101010      |     42     | alternating pattern
//   7   |   7  |    111111      |     63     | all ones
//   8   |   0  |    000001      |      1     | re-read: no state, no latency
// ============================================================================
//
// Steps 5-7 are the ones that matter: a walking single bit through the MSB plus
// an alternating pattern would expose a reversed bit order or a width error,
// which the small values in steps 0-3 would not.
`timescale 1ns / 1ps

module tb_RegFileROM;

    localparam int DEPTH  = 8;
    localparam int WIDTH  = 6;
    localparam int ADDR_W = 3;

    logic [ADDR_W-1:0] addr;
    logic [WIDTH-1:0]  data;

    int errors = 0;
    int step   = 0;

    RegFileROM #(
        .DEPTH     (DEPTH),
        .WIDTH     (WIDTH),
        .INIT_FILE ("regfile_rom_init.mem"),
        .INIT_HEX  (1'b0)
    ) dut (
        .addr (addr),
        .data (data)
    );

    task automatic check(input logic [ADDR_W-1:0] a,
                         input logic [WIDTH-1:0]  exp_data,
                         input string             note);
        addr = a;
        #5;
        if (data !== exp_data) begin
            errors++;
            $display("  step %0d MISMATCH  addr=%0d  got data=%0d (%b) | exp %0d (%b)   %s",
                     step, a, data, data, exp_data, exp_data, note);
        end else begin
            $display("  step %0d ok        addr=%0d  data=%0d (%b)   %s",
                     step, a, data, data, note);
        end
        step++;
    endtask

    initial begin
        $display("tb_RegFileROM: DEPTH=%0d WIDTH=%0d", DEPTH, WIDTH);

        check(3'd0, 6'd1,  "bit 0 only");
        check(3'd1, 6'd2,  "bit 1 only");
        check(3'd2, 6'd4,  "");
        check(3'd3, 6'd8,  "");
        check(3'd4, 6'd16, "");
        check(3'd5, 6'd32, "bit 5 (MSB) only");
        check(3'd6, 6'd42, "alternating pattern");
        check(3'd7, 6'd63, "all ones");
        check(3'd0, 6'd1,  "re-read: no state, no latency");

        if (errors == 0) $display("tb_RegFileROM: PASS");
        else             $display("tb_RegFileROM: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
