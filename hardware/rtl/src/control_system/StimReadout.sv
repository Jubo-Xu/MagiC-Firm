// StimReadout.sv — measurement readout model for whole-system stim testing.
// 1:1 port of emulator/include/control_system/cultiv_control/stim_readout.hpp.
//
// A REAL, synthesizable module (lives in src/, compiled onto the FPGA): for a
// hardware-in-the-loop stim test it STANDS IN for the qubits + readout, replaying
// RECORDED measurements from BRAM back into a leaf's ControlBoard to exercise the
// whole control system. It is test/sim infrastructure, not part of the mission
// datapath, but it IS hardware. Driven by the GLOBAL rst (+ cw_gen_finish), never
// by the board's internal reset.
//
// The leaf's PhysicalMMIO drives one command per measurement round; mmio_out_data is
// the ROUND INDEX. Recorded measurements live in per-board BRAM (SyncROM) laid out
// shot-major (SHOTS*DEPTH rows). 1-cycle BRAM read latency; the module is synchronous
// around it (harmless once the loop is closed — the datapath is pipelined).
//
// SHOT ADVANCE (on ABORT). An abort (ev_valid & ev_type==EV_ABORT — the SAME event that restarts
// the CW program in BoardControl/PhysicalMMIO) DISCARDS the current shot: jump base_reg to the NEXT
// recorded shot's first row and bump shot_reg. This is DECOUPLED from the read — the abort may
// arrive several cycles BEFORE the MMIO's first command (a leaf used in a later cultivation stage
// idles a large wt before its first read), so it only moves the pointer and waits there; the read
// still happens on each mmio_out_valid at base_reg + index. Advancing on the (global, in-sync)
// abort — instead of on the MMIO's wt-delayed index==0 — keeps every leaf's readout jumping to the
// next shot on the SAME cycle, in lockstep.
//
//   reset (rst | cw_gen_finish) : shot_reg<=0, base_reg<=0
//   on abort (guarded, only while shot_reg+1 < SHOTS):
//     shot_reg<=shot_reg+1, base_reg<=(shot_reg+1)*DEPTH
//   on mmio_out_valid : addr = base_reg + index   (the read; data one cycle later)
//   out_meas/out_meas_valid = mem[addr] (SyncROM zeroes when !enabled);
//   out_meas_finish = (cw_gen_finish at the read cycle) ? out_meas_valid : 0.
`timescale 1ns / 1ps

module StimReadout #(
    parameter int M      = 3,          // measurement width (board ports)
    parameter int SHOTS  = 2,          // recorded shots
    parameter int DEPTH  = 3,          // rows per shot (N, or N+1 with copy-last)
    parameter     MEAS_FILE  = "",     // sim_readout_meas .mem
    parameter     VALID_FILE = "",     // sim_readout_valid .mem
    parameter bit INIT_HEX   = 1'b0,   // 0 = $readmemb, 1 = $readmemh
    // ---- derived (do not override) ----
    parameter int IDX_W  = (DEPTH <= 1) ? 1 : $clog2(DEPTH),        // round-index width (= mmio_out_data)
    parameter int TOTAL  = SHOTS * DEPTH,
    parameter int ADDR_W = (TOTAL <= 1) ? 1 : $clog2(TOTAL),
    parameter int SHOT_W = $clog2(SHOTS + 1)                        // holds 0..SHOTS
) (
    input  logic              clk,
    input  logic              rst,               // GLOBAL reset (not board_reset)

    input  logic              mmio_out_valid,
    input  logic [IDX_W-1:0]  mmio_out_data,      // command word = round index
    input  logic              cw_gen_finish,

    // the leaf's event bus (the SAME ev that feeds BoardControl + PhysicalMMIO): an EV_ABORT
    // advances the readout to the next shot, in sync across all leaves
    input  logic              ev_valid,
    input  logic [1:0]        ev_type,            // 00=START 01=ABORT 10=FINISH

    output logic [M-1:0]      out_meas,
    output logic [M-1:0]      out_meas_valid,
    output logic [M-1:0]      out_meas_finish,
    output logic [SHOT_W-1:0] shot_reg            // shots started (== SHOTS while last active)
);

    localparam logic [1:0] EV_ABORT = 2'd1;   // matches BoardControl EV_ABORT

    logic [SHOT_W-1:0] shot_q;
    logic [ADDR_W-1:0] base_q;
    logic              finish_q;

    // ---- combinational: read enable + address = base_reg + index (base moved by abort, below) ----
    logic is_abort, active;
    assign is_abort = ev_valid && (ev_type == EV_ABORT);
    assign active   = mmio_out_valid && !rst;

    logic [ADDR_W-1:0] addr, rom_addr;
    assign addr     = base_q + mmio_out_data;
    assign rom_addr = active ? addr : '0;

    // ---- BRAMs: measurement + valid (1-cycle read; auto-zero when !en) ----
    logic [M-1:0] meas_data, valid_data;
    logic         meas_dv, valid_dv;
    SyncROM #(.DEPTH(TOTAL), .WIDTH(M), .INIT_FILE(MEAS_FILE),  .INIT_HEX(INIT_HEX), .RAM_STYLE("block"))
        u_meas  (.clk(clk), .en(active), .addr(rom_addr), .data(meas_data),  .data_valid(meas_dv));
    SyncROM #(.DEPTH(TOTAL), .WIDTH(M), .INIT_FILE(VALID_FILE), .INIT_HEX(INIT_HEX), .RAM_STYLE("block"))
        u_valid (.clk(clk), .en(active), .addr(rom_addr), .data(valid_data), .data_valid(valid_dv));

    // ---- registered: shot progression + finish pipe (aligned to the BRAM data). The POINTER
    //      (shot_q / base_q) advances on the ABORT event — global and in-sync across leaves — NOT
    //      on the MMIO index; the read address just follows base_q. ----
    logic [SHOT_W-1:0] next_shot;
    assign next_shot = shot_q + 1'b1;
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            shot_q   <= '0;
            base_q   <= '0;
            finish_q <= 1'b0;
        end else begin
            finish_q <= mmio_out_valid && cw_gen_finish; // delay finish to align with the read data
            if (cw_gen_finish) begin                     // trial done -> reset for next trial
                shot_q <= '0;
                base_q <= '0;
            end else if (is_abort) begin                 // discard current shot -> jump to next shot's base
                if (next_shot < SHOT_W'(SHOTS)) begin    // guard: hold at the last shot
                    shot_q <= next_shot;
                    base_q <= next_shot * DEPTH;
                end
            end
        end
    end

    assign out_meas        = meas_data;
    assign out_meas_valid  = valid_data;
    assign out_meas_finish = finish_q ? valid_data : '0;
    assign shot_reg        = shot_q;

endmodule
