// SyncROM.sv — synchronous-read ROM (a true BRAM). 1:1 port of the SystemC
// emulator module emulator/include/lib/sync_rom.hpp.
//
// Synchronous sibling of the async RegFileROM:
//   RegFileROM  data = mem[addr] combinationally  -> LUTRAM / distributed RAM
//   SyncROM     data = mem[addr] one cycle later  -> block RAM (BRAM)
//
// Protocol: assert `en` with `addr`; one cycle later `data` holds mem[addr] and
// `data_valid` is high. Back-to-back reads (en held with a walking addr) give a
// contiguous data stream with no bubble.
//
// NO RESET, deliberately — this models a BRAM output register, which typically
// has none. `data_valid` simply follows `en` by one cycle, so a consumer that
// stops driving `en` sees data_valid fall one cycle later on its own. Callers
// rely on this: resetting the output register would drop the word that was
// already fetched before the reset (in PhysicalMMIO that word is the final
// command of a round).
//
// STRUCTURE / BRAM INFERENCE. The memory read is kept pure (`if (en) mem_q <=
// mem[addr];` — holds when en is low) so Vivado infers a clean BRAM with a read
// enable. The "zero the bus when invalid" behaviour is then a separate fabric
// mux on the BRAM output. Folding it into the always_ff as `else mem_q <= '0;`
// would ask the BRAM to clear its output register while EN is low, which the
// primitive cannot do (RSTRAM is qualified by EN), and synthesis would fall back
// to distributed RAM or bolt on extra logic anyway.
//
// INITIALISATION. INIT_FILE is read by $readmemb/$readmemh at elaboration; Vivado
// synthesis honours this and bakes the contents into the BRAM INIT strings. The
// file format is the compiler's .mem: one word per line, MSB-first, LSB = bit 0
// — identical to what the SystemC SyncROM::load_mem_file() parses, so both
// models consume the same files.
//
// NOTE (SystemC parity): the SystemC model returns zeros for an out-of-range
// address as a debugging aid. Here `addr` is ADDR_W = $clog2(DEPTH) bits, so
// out-of-range is impossible when DEPTH is a power of two. For a non-power-of-two
// DEPTH an out-of-range read yields X in simulation; add a `(addr < DEPTH)` guard
// on the read if you need exact parity (costs a comparator in the BRAM address
// path).
`timescale 1ns / 1ps

module SyncROM #(
    parameter int DEPTH     = 8,
    parameter int WIDTH     = 8,
    parameter int ADDR_W    = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter     INIT_FILE = "",       // "" = leave zero-initialised
    parameter bit INIT_HEX  = 1'b0,     // 0 = $readmemb, 1 = $readmemh
    parameter     RAM_STYLE = "block"   // Vivado ram_style: block / distributed / registers / auto
) (
    input  logic              clk,
    input  logic              en,          // read enable
    input  logic [ADDR_W-1:0] addr,        // read address (sampled when en)

    output logic [WIDTH-1:0]  data,        // mem[addr] one cycle after en (0 when !data_valid)
    output logic              data_valid   // en, delayed one cycle
);

    // ------------------------------------------------------------------
    // storage
    // ------------------------------------------------------------------
    (* ram_style = RAM_STYLE *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
        if (INIT_FILE != "") begin
            if (INIT_HEX) $readmemh(INIT_FILE, mem);
            else          $readmemb(INIT_FILE, mem);
        end
    end

    // ------------------------------------------------------------------
    // BRAM read stage: pure, holds when en is low -> cleanly inferrable
    // ------------------------------------------------------------------
    logic [WIDTH-1:0] mem_q;
    logic             vld_q;

    always_ff @(posedge clk) begin
        if (en) mem_q <= mem[addr];
        vld_q <= en;
    end

    // ------------------------------------------------------------------
    // output stage: keep (data, data_valid) self-consistent so consumers
    // never have to mask the bus themselves
    // ------------------------------------------------------------------
    assign data_valid = vld_q;
    assign data       = vld_q ? mem_q : '0;

endmodule
