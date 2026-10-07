// PhysicalMMIO.sv — command-word generator for one physical control channel.
// 1:1 port of emulator/include/cultiv_control/leaf/physical_mmio/physical_mmio.hpp.
//
// One PhysicalMMIO per control core on a leaf board (P per board; a core may
// drive one qubit or a group — a user configuration choice). It turns the
// board's control events into a stream of command words for the physical
// control core, and reports when its command generation has finished.
//
// Pure assembly of four verified blocks:
//
//   ev_valid/ev_type/ev_payload
//         |
//         v
//   +-----------------+  instr_en ------------------------------+
//   | InstrSequencer  |  instr_addr --> InstrRegFile            |
//   |  IDLE/EXEC/DRAIN|                 (RegFileROM, async)     |
//   +-----------------+                      | instr[INSTR_W]   |
//         ^       | rst_out                  v                  |
//         |       |                    +------------+           |
//         |       |                    | InstrUnpack|           |
//         |       |                    +------------+           |
//         |       |                     wt/start/end            |
//         |       |                          |                  |
//         |       +--------------->  +----------------+ <-------+
//         |                          |  InstrDecode   |  (en)
//         +---- one_instr_end -------|                |
//                                    +----------------+
//                                      r_en | r_addr
//                                           v
//                                     +-----------+
//                                     | CwMemory  |  (SyncROM, 1-cycle read)
//                                     +-----------+
//                                           |
//                                           v
//                               out_data / out_valid    cw_gen_finish
//
// OUTPUT CONTRACT. out_valid alone carries "there is a command this cycle"; when
// it is low out_data is 0 and the control core does nothing. No explicit NOP
// command word is needed, so idle cycles cost no CW memory.
//
// MEMORIES. The instruction memory is small and read combinationally at a
// registered address (RegFileROM -> LUTRAM); the command-word memory is large
// and read with one cycle of latency (SyncROM -> BRAM). Both are initialised
// from .mem files via $readmemb/$readmemh, which Vivado bakes into the LUTRAM /
// BRAM INIT strings at synthesis. The same .mem files drive the SystemC model.
//
// NO COMBINATIONAL LOOP: one_instr_end feeds only InstrSequencer's *clocked*
// process, and instr_en/instr_addr come only from its registers, so the ring is
// broken at the flops.
//
// TIMING NOTE. The path
//   InstrSequencer regs -> instr_addr mux -> InstrRegFile (async) -> InstrUnpack
//   -> InstrDecode (comb) -> CwMemory address
// is combinational end to end. That is the price of the zero-stall instruction
// handoff and is expected to be this module's critical path.
`timescale 1ns / 1ps

module PhysicalMMIO #(
    parameter int INSTR_DEPTH = 4,     // instruction regfile depth (last entry = wait instr)
    parameter int CW_DEPTH    = 16,    // command-word memory depth
    parameter int DATA_W      = 8,     // command word width
    parameter int WT_W        = 8,     // width of the instruction's wt field
    parameter int PAYLOAD_W   = 8,     // reserved ev_payload width
    parameter     INSTR_FILE  = "",    // .mem for the instruction regfile
    parameter     CW_FILE     = "",    // .mem for the command-word memory
    parameter bit INIT_HEX    = 1'b0,  // 0 = $readmemb, 1 = $readmemh (both memories)

    // derived — keeping these in one place stops the compiler's packing and the
    // hardware's decoding from drifting apart
    parameter int ADDR_W  = (CW_DEPTH    <= 1) ? 1 : $clog2(CW_DEPTH),
    parameter int PTR_W   = (INSTR_DEPTH <= 1) ? 1 : $clog2(INSTR_DEPTH),
    parameter int INSTR_W = 2*ADDR_W + WT_W
) (
    input  logic                 clk,
    input  logic                 rst,

    // control event bus (from the board's control block)
    input  logic                 ev_valid,
    input  logic [1:0]           ev_type,       // 00=START 01=ABORT 10=FINISH 11=reserved
    input  logic [PAYLOAD_W-1:0] ev_payload,    // RESERVED (JUMP); unused today

    // to the physical control core
    output logic [DATA_W-1:0]    out_data,      // command word (0 when !out_valid)
    output logic                 out_valid,

    // to the board's control block
    output logic                 cw_gen_finish  // 1-cycle pulse, aligned with the LAST out_valid
);

    // ------------------------------------------------------------------
    // internal nets
    // ------------------------------------------------------------------
    logic               instr_en;
    logic [PTR_W-1:0]   instr_addr;
    logic               rst_out;
    logic               one_instr_end;

    logic [INSTR_W-1:0] instr_word;
    logic [WT_W-1:0]    wt;
    logic [ADDR_W-1:0]  start_addr, end_addr;

    logic               r_en;
    logic [ADDR_W-1:0]  r_addr;

    // ------------------------------------------------------------------
    // sequencer: events in, instruction fetch + local reset out
    // ------------------------------------------------------------------
    InstrSequencer #(
        .DEPTH     (INSTR_DEPTH),
        .PAYLOAD_W (PAYLOAD_W)
    ) u_seq (
        .clk           (clk),
        .rst           (rst),
        .ev_valid      (ev_valid),
        .ev_type       (ev_type),
        .ev_payload    (ev_payload),
        .one_instr_end (one_instr_end),
        .instr_en      (instr_en),
        .instr_addr    (instr_addr),
        .rst_out       (rst_out),
        .cw_gen_finish (cw_gen_finish)
    );

    // ------------------------------------------------------------------
    // instruction regfile (async read at the registered address)
    // ------------------------------------------------------------------
    RegFileROM #(
        .DEPTH     (INSTR_DEPTH),
        .WIDTH     (INSTR_W),
        .INIT_FILE (INSTR_FILE),
        .INIT_HEX  (INIT_HEX)
    ) u_instr_rom (
        .addr (instr_addr),
        .data (instr_word)
    );

    // ------------------------------------------------------------------
    // field extraction (the single place that knows the instruction encoding)
    // ------------------------------------------------------------------
    InstrUnpack #(
        .WT_W   (WT_W),
        .ADDR_W (ADDR_W)
    ) u_unpack (
        .instr      (instr_word),
        .wt         (wt),
        .start_addr (start_addr),
        .end_addr   (end_addr)
    );

    // ------------------------------------------------------------------
    // decoder: instruction -> CW read stream.
    // rst_out is its SINGLE reset source (it already ORs the system reset,
    // abort and finish).
    // ------------------------------------------------------------------
    InstrDecode #(
        .WT_W   (WT_W),
        .ADDR_W (ADDR_W)
    ) u_decode (
        .clk           (clk),
        .rst           (rst_out),
        .en            (instr_en),
        .in_wt         (wt),
        .in_start      (start_addr),
        .in_end        (end_addr),
        .r_en          (r_en),
        .r_addr        (r_addr),
        .one_instr_end (one_instr_end)
    );

    // ------------------------------------------------------------------
    // command-word memory (BRAM, 1-cycle). No reset by design: that is what
    // lets the final command word still be delivered on a finish.
    // ------------------------------------------------------------------
    SyncROM #(
        .DEPTH     (CW_DEPTH),
        .WIDTH     (DATA_W),
        .INIT_FILE (CW_FILE),
        .INIT_HEX  (INIT_HEX)
    ) u_cw_rom (
        .clk        (clk),
        .en         (r_en),
        .addr       (r_addr),
        .data       (out_data),
        .data_valid (out_valid)
    );

endmodule
