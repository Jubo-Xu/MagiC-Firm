// InstrDecode.sv — instruction decoder for PhysicalMMIO. 1:1 port of the SystemC
// emulator module emulator/include/cultiv_control/leaf/physical_mmio/instr_decode.hpp.
//
// Turns ONE instruction {w_t, start, end} into a command-word read stream for the
// CW memory (a SyncROM). The state machine feeds instructions via a 1-cycle `en`
// pulse; this block generates the CW read address sequence and reports when the
// instruction is done (one_instr_end).
//
//   en pulse + {w_t, start, end}
//        |  wait w_t idle cycles
//        v
//   r_addr:  start, start+1, ..., end     (one per cycle, r_en high)
//        |
//        v  on the cycle r_addr == end
//   one_instr_end = 1   -> state machine hands over the next instruction
//
// r_en/r_addr drive SyncROM.en/.addr; CW data + data_valid come back one cycle
// later. This block never touches the data path.
//
// Wait-count convention: w_t = number of idle cycles before the first read.
//   w_t = 0 -> read `start` on the en cycle itself (fast path, no stall)
//   w_t = N -> N idle cycles, read `start` on en+N
//   start == end -> exactly one read, one_instr_end on that same cycle.
//
// OUTPUTS ARE COMBINATIONAL (Mealy), which the no-stall handoff requires: at the
// cycle r_addr == end the state machine registers en=1 for the NEXT cycle, and
// that instruction's first read (w_t==0) must appear on the same en cycle so CW
// reads stay contiguous (end_i at T, start_{i+1} at T+1). Registered outputs
// would insert a bubble. FSM state is registered; `rst` is asynchronous (the
// state machine's local reset, so an abort restarts the decoder cleanly).
`timescale 1ns / 1ps

module InstrDecode #(
    parameter int WT_W   = 8,   // width of in_wt
    parameter int ADDR_W = 16   // width of in_start / in_end (CW address)
) (
    input  logic                clk,
    input  logic                rst,            // asynchronous, active high

    // instruction input (from the state machine)
    input  logic                en,             // 1-cycle pulse: decode this instruction
    input  logic [WT_W-1:0]     in_wt,          // idle cycles before first read
    input  logic [ADDR_W-1:0]   in_start,       // first CW address
    input  logic [ADDR_W-1:0]   in_end,         // last CW address (>= in_start)

    // CW-memory read control (to SyncROM)
    output logic                r_en,
    output logic [ADDR_W-1:0]   r_addr,

    // handshake back to the state machine
    output logic                one_instr_end   // 1 for one cycle when r_addr == end
);

    // ------------------------------------------------------------------
    // state
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE   = 2'd0,
        WAITC  = 2'd1,   // "WAIT" is a reserved-ish name in some tools; WAITC = wait counting
        STREAM = 2'd2
    } phase_e;

    phase_e            phase, nphase;
    logic [ADDR_W-1:0] addr,  naddr;    // holds start during WAITC, walks during STREAM
    logic [ADDR_W-1:0] end_a, nend_a;   // latched end address ("end" is a keyword)
    logic [WT_W-1:0]   wcnt,  nwcnt;    // remaining idle cycles

    // ------------------------------------------------------------------
    // combinational: outputs + next-state (switch on phase)
    // ------------------------------------------------------------------
    always_comb begin
        // defaults: idle outputs, hold state
        r_en          = 1'b0;
        r_addr        = '0;
        one_instr_end = 1'b0;
        nphase        = phase;
        naddr         = addr;
        nend_a        = end_a;
        nwcnt         = wcnt;

        case (phase)
            IDLE: begin
                if (en) begin
                    if (in_wt == '0) begin
                        // fast path: read `start` this very cycle
                        r_en   = 1'b1;
                        r_addr = in_start;
                        if (in_start == in_end) begin
                            one_instr_end = 1'b1;          // single-word instruction
                            nphase        = IDLE;
                        end else begin
                            nphase = STREAM;
                            naddr  = in_start + ADDR_W'(1);
                            nend_a = in_end;
                        end
                    end else begin
                        // wait first; latch start/end
                        nphase = WAITC;
                        nwcnt  = in_wt - WT_W'(1);         // this cycle is the first idle cycle
                        naddr  = in_start;
                        nend_a = in_end;
                    end
                end
            end

            WAITC: begin
                if (wcnt == '0) begin
                    // idle done: read `start` (held in addr) now
                    r_en   = 1'b1;
                    r_addr = addr;
                    if (addr == end_a) begin
                        one_instr_end = 1'b1;
                        nphase        = IDLE;
                    end else begin
                        nphase = STREAM;
                        naddr  = addr + ADDR_W'(1);
                    end
                end else begin
                    nwcnt  = wcnt - WT_W'(1);
                    nphase = WAITC;
                end
            end

            STREAM: begin
                r_en   = 1'b1;
                r_addr = addr;
                if (addr == end_a) begin
                    one_instr_end = 1'b1;
                    nphase        = IDLE;
                end else begin
                    naddr  = addr + ADDR_W'(1);
                    nphase = STREAM;
                end
            end

            default: nphase = IDLE;
        endcase
    end

    // ------------------------------------------------------------------
    // clocked: register next-state, asynchronous reset to IDLE
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            phase <= IDLE;
            addr  <= '0;
            end_a <= '0;
            wcnt  <= '0;
        end else begin
            phase <= nphase;
            addr  <= naddr;
            end_a <= nend_a;
            wcnt  <= nwcnt;
        end
    end

endmodule
