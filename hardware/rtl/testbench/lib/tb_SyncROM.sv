// tb_SyncROM.sv — directed testbench for SyncROM.
//
// Mirrors the SystemC unit test (emulator/tests/lib/test_sync_rom.cpp).
// ROM contents come from sync_rom_init.mem via $readmemb: mem[i] = 3*i + 1
//   mem[0..7] = 1, 4, 7, 10, 13, 16, 19, 22
//
// Each step drives (en, addr) on a negedge and checks the registered outputs
// after the following posedge — i.e. the 1-cycle read latency.
//
//   step  en addr   EXPECTED data  EXPECTED data_valid   note
//   ----  -- ----   -------------  -------------------   -------------------------
//    1     0   0          0                0             idle
//    2     1   2          7                1             read mem[2]
//    3     1   3         10                1             back-to-back, NO bubble
//    4     1   4         13                1             back-to-back, NO bubble
//    5     0   0          0                0             en low -> bus forced to 0
//    6     1   7         22                1             read mem[7]
//    7     0   0          0                0             en low -> bus forced to 0
//
// Step 5 is the key check that `data` is ZEROED (not held) when data_valid is
// low, matching the SystemC model so consumers never mask the bus themselves.
`timescale 1ns / 1ps

module tb_SyncROM;

    localparam int DEPTH  = 8;
    localparam int WIDTH  = 8;
    localparam int ADDR_W = 3;

    logic clk;
    initial begin                  // 100 MHz — single driving process
        clk = 1'b0;
        forever #5 clk = ~clk;
    end

    logic              en;
    logic [ADDR_W-1:0] addr;
    logic [WIDTH-1:0]  data;
    logic              data_valid;

    int errors = 0;

    SyncROM #(
        .DEPTH     (DEPTH),
        .WIDTH     (WIDTH),
        .INIT_FILE ("sync_rom_init.mem"),
        .INIT_HEX  (1'b0)
    ) dut (
        .clk        (clk),
        .en         (en),
        .addr       (addr),
        .data       (data),
        .data_valid (data_valid)
    );

    // drive (en, addr) for one cycle, then check the registered outputs
    task automatic step(input logic              e,
                        input logic [ADDR_W-1:0] a,
                        input logic              exp_valid,
                        input logic [WIDTH-1:0]  exp_data,
                        input string             tag);
        @(negedge clk);
        en   = e;
        addr = a;
        @(posedge clk);
        #1;
        if (data_valid !== exp_valid || data !== exp_data) begin
            errors++;
            $display("  MISMATCH %-22s got valid=%0b data=%3d   exp valid=%0b data=%3d",
                     tag, data_valid, data, exp_valid, exp_data);
        end else begin
            $display("  ok       %-22s     valid=%0b data=%3d", tag, data_valid, data);
        end
    endtask

    initial begin
        en   = 1'b0;
        addr = '0;
        $display("tb_SyncROM: DEPTH=%0d WIDTH=%0d", DEPTH, WIDTH);

        step(1'b0, 3'd0, 1'b0, 8'd0,  "idle");
        step(1'b1, 3'd2, 1'b1, 8'd7,  "read mem[2]");
        step(1'b1, 3'd3, 1'b1, 8'd10, "back-to-back mem[3]");
        step(1'b1, 3'd4, 1'b1, 8'd13, "back-to-back mem[4]");
        step(1'b0, 3'd0, 1'b0, 8'd0,  "en low -> data 0");
        step(1'b1, 3'd7, 1'b1, 8'd22, "read mem[7]");
        step(1'b0, 3'd0, 1'b0, 8'd0,  "en low -> data 0");

        if (errors == 0) $display("tb_SyncROM: PASS");
        else             $display("tb_SyncROM: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
