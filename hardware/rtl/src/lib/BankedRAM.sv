// BankedRAM.sv — banked read/write RAM (the PGE's M_T column store).
// 1:1 port of the SystemC emulator module emulator/include/lib/banked_ram.hpp.
//
// BANKS * DEPTH words of WIDTH bits, addressed by one index whose BIT-SLICES
// pick bank and offset:
//
//     bank   = index[$clog2(BANKS)-1:0]
//     offset = index[INDEX_W-1:$clog2(BANKS)]
//
// valid only for a power-of-two BANKS. The PGE compiler enforces that, and its
// relabelling pi(c) = offset*BANKS + bank is what turns the split into fixed
// wiring instead of a runtime address lookup. The decode is NOT done here — the
// caller presents a per-bank offset, because the whole point of the relabelling
// is that the split is free wiring at the consumer.
//
// One read port and one write port per bank (simple dual port). That is exactly
// what the compiler's banking rule buys: for every fault, at most P=1 of its
// checks lands in any one bank, so all of a fault's columns are readable in a
// single cycle.
//
// SYNCHRONOUS read, one cycle of latency — the synchronous sibling of
// RegFileROM, same relationship as SyncROM. Not a preference: at ndet=640 the
// store is 640*640 bits = 50 KiB, which is BRAM on any FPGA, and BRAM registers
// its output.
//
// A bank whose rd_en was low drives ZERO, not its previous word, so a consumer's
// XOR tree can take all banks unconditionally — rd_en is the mask.
//
// STRUCTURE / BRAM INFERENCE. As in SyncROM the memory read is kept pure
// (`if (rd_en) mem_q <= mem[addr];`, holding when the enable is low) and the
// zeroing is a separate fabric mux on the output. Folding it in as
// `else mem_q <= '0;` would ask the BRAM to clear its output register while EN
// is low, which the primitive cannot do (RSTRAM is qualified by EN), and
// synthesis would fall back to distributed RAM.
//
// READ_FIRST. Read and write are both nonblocking in one always_ff, so every
// right-hand side samples pre-edge state and a same-address read/write pair
// returns the OLD word — matching the SystemC model. For hardware to agree the
// inferred SDP BRAM must be configured READ_FIRST; an inferred simple-dual-port
// defaults to don't-care on read-during-write in some Vivado versions, so if
// the design ever relies on this, check the synthesis report rather than
// assuming.
//
// RESET. Asynchronous, active high: zeroes every bank and the output registers,
// matching the SystemC model exactly. Note for synthesis: a reset that clears
// the whole array prevents BRAM inference and maps the store to flip-flops —
// revisit here if the target needs true BRAM.
//
// Parameters:
//   BANKS   number of banks (power of two)
//   DEPTH   words per bank
//   WIDTH   bits per word
`timescale 1ns / 1ps

module BankedRAM #(
    parameter int BANKS     = 16,
    parameter int DEPTH     = 40,
    parameter int WIDTH     = 640,
    parameter int ADDR_W    = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter     RAM_STYLE = "block"   // Vivado ram_style: block / distributed / auto
) (
    input  logic clk,
    input  logic rst,                                // asynchronous, active high

    input  logic [BANKS-1:0]             rd_en,
    input  logic [BANKS-1:0][ADDR_W-1:0] rd_addr,
    output logic [BANKS-1:0][WIDTH-1:0]  rd_data,    // valid one cycle after rd_en

    input  logic [BANKS-1:0]             wr_en,
    input  logic [BANKS-1:0][ADDR_W-1:0] wr_addr,
    input  logic [BANKS-1:0][WIDTH-1:0]  wr_data
);

    // ------------------------------------------------------------------
    // banks
    // ------------------------------------------------------------------
    genvar b;
    generate
        for (b = 0; b < BANKS; b++) begin : g_bank

            (* ram_style = RAM_STYLE *)
            logic [WIDTH-1:0] mem [0:DEPTH-1];

            logic [WIDTH-1:0] mem_q;
            logic             rd_valid_q;

            always_ff @(posedge clk or posedge rst) begin
                if (rst) begin
                    for (int i = 0; i < DEPTH; i++) mem[i] <= '0;
                    mem_q      <= '0;
                    rd_valid_q <= 1'b0;
                end else begin
                    // Both nonblocking, so the read samples pre-edge state:
                    // READ_FIRST, same as the SystemC model.
                    if (rd_en[b]) mem_q <= mem[rd_addr[b]];
                    if (wr_en[b]) mem[wr_addr[b]] <= wr_data[b];
                    rd_valid_q <= rd_en[b];
                end
            end

            // Fabric mux, deliberately outside the always_ff — see the header.
            assign rd_data[b] = rd_valid_q ? mem_q : {WIDTH{1'b0}};

        end
    endgenerate

endmodule
