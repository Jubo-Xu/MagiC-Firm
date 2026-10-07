// InstrUnpack.sv — instruction word field extraction for PhysicalMMIO.
// 1:1 port of emulator/include/cultiv_control/leaf/physical_mmio/instr_unpack.hpp.
//
// Pure combinational. Splits one packed instruction word (as read from the
// instruction regfile) into its fields. This is the SINGLE place that knows the
// instruction encoding: when instructions gain fields (branch select, round
// boundary flag, ...), only this module and the compiler's emitter change —
// PhysicalMMIO's wiring and InstrDecode stay untouched.
//
// LAYOUT (LSB-first, matching how the compiler packs kernel selectors):
//
//   bits [ADDR_W-1 : 0]                 end_addr    last CW address
//   bits [2*ADDR_W-1 : ADDR_W]          start_addr  first CW address
//   bits [2*ADDR_W+WT_W-1 : 2*ADDR_W]   wt          idle cycles before first read
//
//   INSTR_W = 2*ADDR_W + WT_W
//
// INSTR_W is exposed as a derived parameter so the instantiating module can size
// the instruction regfile from it rather than recomputing the layout.
`timescale 1ns / 1ps

module InstrUnpack #(
    parameter int WT_W    = 8,                    // width of the wt field
    parameter int ADDR_W  = 16,                   // width of start_addr / end_addr
    parameter int INSTR_W = 2*ADDR_W + WT_W       // derived: packed word width
) (
    input  logic [INSTR_W-1:0] instr,

    output logic [WT_W-1:0]    wt,
    output logic [ADDR_W-1:0]  start_addr,
    output logic [ADDR_W-1:0]  end_addr
);

    assign end_addr   = instr[ADDR_W-1 : 0];
    assign start_addr = instr[2*ADDR_W-1 : ADDR_W];
    assign wt         = instr[2*ADDR_W+WT_W-1 : 2*ADDR_W];

endmodule
