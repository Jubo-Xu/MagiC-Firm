// RegFileROM.sv — read-only regfile / ROM with ASYNCHRONOUS read.
// 1:1 port of the SystemC emulator module emulator/include/lib/regfile_rom.hpp.
//
// Asynchronous sibling of SyncROM:
//   RegFileROM  data = mem[addr] combinationally  -> LUTRAM / distributed RAM
//   SyncROM     data = mem[addr] one cycle later  -> block RAM (BRAM)
//
// Used for the small compiler regfiles whose registered pc/address drives `addr`
// and whose word is consumed in the SAME cycle (sync masks, kernel cores,
// output_sync, global_index, postselect, and the PhysicalMMIO instruction
// regfile). There is no clock and no enable — it is a pure lookup table.
//
// SIZING. Distributed RAM costs LUTs in proportion to DEPTH*WIDTH, so this is
// only appropriate for shallow memories. Anything large (e.g. a command-word
// memory) should use SyncROM instead and absorb the one-cycle latency.
//
// INITIALISATION. INIT_FILE is read by $readmemb/$readmemh at elaboration;
// Vivado synthesis honours this and bakes the contents into the LUTRAM INIT
// strings. The file format is the compiler's .mem: one word per line, MSB-first,
// LSB = bit 0 — identical to what the SystemC RegFileROM::load_mem_file()
// parses, so both models consume the same files.
//
// NOTE (SystemC parity): the SystemC model returns zeros for an out-of-range
// address as a debugging aid. Here `addr` is ADDR_W = $clog2(DEPTH) bits, so
// out-of-range is impossible when DEPTH is a power of two. For a non-power-of-two
// DEPTH an out-of-range read yields X in simulation; add an `(addr < DEPTH)`
// guard if you need exact parity (costs a comparator in the address path).
`timescale 1ns / 1ps

module RegFileROM #(
    parameter int DEPTH     = 4,
    parameter int WIDTH     = 8,
    parameter int ADDR_W    = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter     INIT_FILE = "",             // "" = leave zero-initialised
    parameter bit INIT_HEX  = 1'b0,           // 0 = $readmemb, 1 = $readmemh
    parameter     RAM_STYLE = "distributed"   // Vivado ram_style: distributed / registers / block / auto
) (
    input  logic [ADDR_W-1:0] addr,
    output logic [WIDTH-1:0]  data
);

    // NOTE: async read -> only distributed/registers/auto map cleanly; true BRAM
    // ("block") needs a synchronous read (use SyncROM instead).
    (* ram_style = RAM_STYLE *)
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    initial begin
        for (int i = 0; i < DEPTH; i++) mem[i] = '0;
        if (INIT_FILE != "") begin
            if (INIT_HEX) $readmemh(INIT_FILE, mem);
            else          $readmemb(INIT_FILE, mem);
        end
    end

    assign data = mem[addr];   // asynchronous read

endmodule
