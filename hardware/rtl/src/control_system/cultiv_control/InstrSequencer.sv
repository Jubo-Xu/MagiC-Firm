// InstrSequencer.sv — instruction sequencer (state machine) for PhysicalMMIO.
// 1:1 port of emulator/include/cultiv_control/leaf/physical_mmio/instr_sequencer.hpp.
//
// Owns the instruction pointer into the instruction regfile and the event FSM
// that reacts to START / ABORT / FINISH from the parent board. Hands one
// instruction at a time to InstrDecode and consumes `one_instr_end` to advance.
//
//   ev_* ---> [ InstrSequencer ] --instr_en/instr_addr--> instr regfile (async)
//                   ^     |                                     |
//     one_instr_end |     +--rst_out--> InstrDecode <-----------+
//                   +-----------------------'
//
// NO-STALL HANDOFF. instr_en/instr_addr are registered: on the cycle InstrDecode
// raises one_instr_end this block registers the next pointer with en=1, so the
// next instruction is fetched the very next cycle and its first CW read (w_t==0)
// lands with no bubble.
//
// WAIT BEHAVIOUR. instr_ptr SATURATES at DEPTH-1 rather than running off the end,
// so the last instruction repeats forever. That entry is the compiled "wait"
// instruction (one syndrome-extraction round) that idles the protocol until
// FINISH arrives.
//
// ADDRESS CONVENTION. instr_addr is meaningful ONLY while instr_en is high:
// instr_addr = instr_en ? instr_ptr : 0. instr_ptr is a separate maintained
// register so the pointer survives the cycles where en is low.
//
// RESET OUTPUT (combinational, deliberately):
//     rst_out = rst || is_abort || rst_for_finish
//   * rst      - the system reset also resets InstrDecode, so rst_out is the
//                SINGLE reset path into it (no OR needed outside).
//   * abort    - high on the SAME cycle as the event, so InstrDecode is cleared
//                one cycle BEFORE instr_addr returns to 0 and refetches.
//   * finish   - rst_for_finish is registered, so it rises one cycle AFTER the
//                final CW address was issued. SyncROM (no reset) still delivers
//                that last command word on that cycle; cw_gen_finish pulses in
//                exactly the same cycle as the final out_valid.
`timescale 1ns / 1ps

module InstrSequencer #(
    parameter int DEPTH     = 4,                                  // instruction regfile depth
    parameter int PTR_W     = (DEPTH <= 1) ? 1 : $clog2(DEPTH),
    parameter int PAYLOAD_W = 8                                   // reserved ev_payload width
) (
    input  logic                 clk,
    input  logic                 rst,             // asynchronous, active high (system reset)

    // control event bus (from the parent board)
    input  logic                 ev_valid,
    input  logic [1:0]           ev_type,         // 00=START 01=ABORT 10=FINISH 11=reserved(JUMP)
    /* verilator lint_off UNUSED */
    input  logic [PAYLOAD_W-1:0] ev_payload,      // RESERVED (JUMP target); unused today
    /* verilator lint_on UNUSED */

    // from InstrDecode
    input  logic                 one_instr_end,

    // to the instruction regfile
    output logic                 instr_en,
    output logic [PTR_W-1:0]     instr_addr,      // valid only while instr_en

    // to InstrDecode / the outside world
    output logic                 rst_out,         // local reset (combinational)
    output logic                 cw_gen_finish    // 1-cycle pulse: CW generation complete
);

    // ------------------------------------------------------------------
    // encodings
    // ------------------------------------------------------------------
    typedef enum logic [1:0] {
        IDLE  = 2'd0,
        EXEC  = 2'd1,
        DRAINS = 2'd2                 // "DRAIN"; suffixed to avoid tool keyword clashes
    } state_e;

    localparam logic [1:0] EV_START  = 2'd0;
    localparam logic [1:0] EV_ABORT  = 2'd1;
    localparam logic [1:0] EV_FINISH = 2'd2;

    state_e          state;
    logic [PTR_W-1:0] ptr;            // maintained instruction pointer
    logic             en_q;           // registered instr_en
    logic             rst_fin;        // 1-cycle flag set when CW generation completes

    logic is_start, is_abort, is_finish;
    assign is_start  = ev_valid && (ev_type == EV_START);
    assign is_abort  = ev_valid && (ev_type == EV_ABORT);
    assign is_finish = ev_valid && (ev_type == EV_FINISH);

    // ------------------------------------------------------------------
    // clocked: one switch on the state (all state registered)
    // ------------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state   <= IDLE;
            ptr     <= '0;
            en_q    <= 1'b0;
            rst_fin <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    // START is the only thing that wakes us
                    rst_fin <= 1'b0;
                    if (is_start) begin
                        state <= EXEC;
                        ptr   <= '0;
                        en_q  <= 1'b1;         // fetch instruction 0 next cycle
                    end else begin
                        ptr   <= '0;
                        en_q  <= 1'b0;
                    end
                end

                EXEC: begin
                    if (is_abort) begin
                        // restart from instruction 0; rst_out is already high THIS
                        // cycle (combinational) so InstrDecode is clear before the refetch
                        ptr     <= '0;
                        en_q    <= 1'b1;
                        state   <= EXEC;
                        rst_fin <= 1'b0;
                    end else if (is_finish) begin
                        if (one_instr_end) begin
                            // last instruction ended in the very same cycle:
                            // nothing to drain, go straight to IDLE
                            state   <= IDLE;
                            ptr     <= '0;
                            en_q    <= 1'b0;
                            rst_fin <= 1'b1;
                        end else begin
                            // an instruction is mid-flight: let it finish in DRAIN
                            state   <= DRAINS;
                            ptr     <= '0;
                            en_q    <= 1'b0;
                            rst_fin <= 1'b0;
                        end
                    end else begin
                        // normal: advance on one_instr_end, saturating at the wait entry
                        if (one_instr_end) begin
                            ptr  <= (ptr == PTR_W'(DEPTH-1)) ? ptr : ptr + PTR_W'(1);
                            en_q <= 1'b1;
                        end else begin
                            en_q <= 1'b0;      // ptr holds
                        end
                        state   <= EXEC;
                        rst_fin <= 1'b0;
                    end
                end

                DRAINS: begin
                    // no new instruction is fetched; wait for the in-flight one to end
                    if (one_instr_end) begin
                        state   <= IDLE;
                        ptr     <= '0;
                        en_q    <= 1'b0;
                        rst_fin <= 1'b1;
                    end else begin
                        ptr     <= '0;
                        en_q    <= 1'b0;
                        rst_fin <= 1'b0;
                    end
                end

                default: begin
                    state   <= IDLE;
                    ptr     <= '0;
                    en_q    <= 1'b0;
                    rst_fin <= 1'b0;
                end
            endcase
        end
    end

    // ------------------------------------------------------------------
    // combinational outputs
    // ------------------------------------------------------------------
    assign instr_en      = en_q;
    assign instr_addr    = en_q ? ptr : '0;          // address valid only with en
    assign rst_out       = rst || is_abort || rst_fin;
    assign cw_gen_finish = rst_fin;

endmodule
