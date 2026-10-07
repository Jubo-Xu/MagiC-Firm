// tb_BankedRAM.sv — directed testbench for BankedRAM.
//
// Written in BOTH styles:
//   (1) an explicit step table below with the expected values also passed inline
//       to every check, so the Vivado waveform can be read off directly;
//   (2) self-checking with mismatch counting and PASS/FAIL for Verilator / xsim.
//
// Mirrors the SystemC unit test emulator/tests/lib/test_banked_ram.cpp case for
// case, since the RTL is a 1:1 port of that module.
//
// Small geometry (BANKS=4, DEPTH=4, WIDTH=8) so the waveform is readable; the
// real M_T is 16 x 40 x 640. The bit-slice property under test is independent
// of size: index = offset*BANKS + bank.
//
// ============================================================================
//  step | what                                  | expect
// ------+---------------------------------------+-----------------------------
//   0   | rst pulse                             | array wiped, outputs zero
//   1   | read any bank after reset             | 0
//   2   | write idx 9 (bank 1, off 2) = 8'hA5   | -
//   3   | read idx 9, SAME cycle as the write   | 0  <- READ_FIRST, old word
//   4   | read idx 9 a cycle later              | 8'hA5
//   5   | write one distinct word per bank      | -
//   6   | read ALL banks at offset 3, one cycle | each bank its own word
//   7   | drop rd_en, one more cycle            | 0  <- drives zero, not held
//   8   | rst again, then read idx 9            | 0  <- contents wiped
// ============================================================================
//
// Steps 3 and 7 are the ones that matter: they are the two behaviours that
// distinguish this from a plain BRAM wrapper, and a naive implementation
// (WRITE_FIRST, or holding the output when disabled) passes everything else.
`timescale 1ns / 1ps

module tb_BankedRAM;

    localparam int BANKS  = 4;
    localparam int DEPTH  = 4;
    localparam int WIDTH  = 8;
    localparam int ADDR_W = 2;

    logic clk;
    logic rst;

    logic [BANKS-1:0]             rd_en,   wr_en;
    logic [BANKS-1:0][ADDR_W-1:0] rd_addr, wr_addr;
    logic [BANKS-1:0][WIDTH-1:0]  rd_data, wr_data;

    int errors = 0;

    initial begin clk = 1'b0; forever #5 clk = ~clk; end

    BankedRAM #(.BANKS(BANKS), .DEPTH(DEPTH), .WIDTH(WIDTH), .ADDR_W(ADDR_W))
    dut (
        .clk(clk), .rst(rst),
        .rd_en(rd_en), .rd_addr(rd_addr), .rd_data(rd_data),
        .wr_en(wr_en), .wr_addr(wr_addr), .wr_data(wr_data)
    );

    // index -> (bank, offset): the compiler's bit-slice
    function automatic int bank_of(input int idx);   return idx % BANKS;  endfunction
    function automatic int off_of (input int idx);   return idx / BANKS;  endfunction

    task automatic idle();
        rd_en = '0; wr_en = '0;
        for (int i = 0; i < BANKS; i++) begin
            rd_addr[i] = '0; wr_addr[i] = '0; wr_data[i] = '0;
        end
    endtask

    task automatic step();  @(posedge clk); #1;  endtask

    task automatic check(input logic [WIDTH-1:0] got,
                         input logic [WIDTH-1:0] exp,
                         input string what);
        if (got !== exp) begin
            errors++;
            $display("  MISMATCH %-42s got=%02h exp=%02h", what, got, exp);
        end
    endtask

    // write one word, one cycle
    task automatic wr(input int idx, input logic [WIDTH-1:0] d);
        idle();
        wr_en[bank_of(idx)]   = 1'b1;
        wr_addr[bank_of(idx)] = ADDR_W'(off_of(idx));
        wr_data[bank_of(idx)] = d;
        step();
        idle();
    endtask

    // present an address, take a cycle, return what the bank drove
    task automatic rd(input int idx, output logic [WIDTH-1:0] d);
        idle();
        rd_en[bank_of(idx)]   = 1'b1;
        rd_addr[bank_of(idx)] = ADDR_W'(off_of(idx));
        step();
        idle();
        d = rd_data[bank_of(idx)];
    endtask

    logic [WIDTH-1:0] d;

    initial begin
        idle();
        rst = 1'b0;
        @(posedge clk); #1;

        // ---- step 0: rst pulse wipes the array and the outputs ----
        rst = 1'b1; step();
        rst = 1'b0;
        for (int i = 0; i < BANKS; i++)
            check(rd_data[i], 8'h00,
                  $sformatf("bank %0d output zero after reset", i));

        // ---- step 1: array is zero after the reset ----
        rd(9, d);  check(d, 8'h00, "read after reset is zero");

        // ---- steps 2-3: read-during-write returns the OLD word ----
        idle();
        wr_en[bank_of(9)]   = 1'b1;
        wr_addr[bank_of(9)] = ADDR_W'(off_of(9));
        wr_data[bank_of(9)] = 8'hA5;
        rd_en[bank_of(9)]   = 1'b1;         // same address, same cycle
        rd_addr[bank_of(9)] = ADDR_W'(off_of(9));
        step();
        idle();
        check(rd_data[bank_of(9)], 8'h00, "read-during-write returns old word");

        // ---- step 4: the written word is there a cycle later ----
        rd(9, d);  check(d, 8'hA5, "written word reads back");

        // ---- step 5: one distinct word per bank, all at offset 3 ----
        for (int i = 0; i < BANKS; i++)
            wr(3 * BANKS + i, 8'h10 + WIDTH'(i));

        // ---- step 6: all banks read in ONE cycle ----
        idle();
        for (int i = 0; i < BANKS; i++) begin
            rd_en[i]   = 1'b1;
            rd_addr[i] = ADDR_W'(3);
        end
        step();
        for (int i = 0; i < BANKS; i++)
            check(rd_data[i], 8'h10 + WIDTH'(i),
                  $sformatf("bank %0d read alongside all others", i));

        // ---- step 7: enables dropped -> drives ZERO, does not hold ----
        idle();
        step();
        for (int i = 0; i < BANKS; i++)
            check(rd_data[i], 8'h00,
                  $sformatf("bank %0d drives zero when disabled", i));

        // ---- step 8: rst wipes the contents ----
        rst = 1'b1; step();
        rst = 1'b0;
        rd(9, d);  check(d, 8'h00, "contents cleared by rst");

        if (errors == 0) $display("tb_BankedRAM: PASS");
        else             $display("tb_BankedRAM: FAIL (%0d mismatches)", errors);
        $finish;
    end

endmodule
