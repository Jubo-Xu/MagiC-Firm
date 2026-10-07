// MeasurementSync.sv — input measurement synchronization block (RTL).
//
// 1:1 RTL mapping of the SystemC model (emulator/src/detector_construct/
// measurement_sync.cpp). One FIFO per raw measurement line buffers incoming
// (valid) measurements. A shared regfile pointer `sync_regfile_pc` walks the
// sync-mask program; each step the mask says which lines this measurement-time
// needs. When every needed line is available it emits one synced word and
// advances the pointer.
//
// Per needed line, the value comes from the FIFO head if the FIFO is non-empty
// (oldest buffered = this meas-time's value), otherwise straight from the input
// if a measurement arrives this cycle (BYPASS, saves a FIFO round-trip). A
// bypassed arrival is consumed, not buffered; a FIFO-head consume still buffers
// any co-arriving measurement (it belongs to a later meas-time). An all-zero
// mask has no needed lines, so it is trivially ready: emit all-zero, advance
// (fast-forward). Outputs are registered (one-cycle pulse per accepted step).
//
// Finish (P2b): a per-line in_finish rides with each measurement through a PARALLEL
// finish FIFO (fmem, sharing the same head/tail/count as the value FIFO). The
// emitted word carries a SINGLE out_finish = OR over the used lines of their finish
// tag (fifo head or bypass), i.e. this meas-time is the last round.
//
// Saturation (P2c): with copy-last the sync regfile has an appended saturating wait
// row; if SAT_PC >= 0 the pc holds at SAT_PC instead of advancing past it. -1 = off.
//
// Parameters:
//   M       number of raw measurement input lines
//   D_FIFO  depth of each per-line sync FIFO
//   PC_W    width of the sync regfile address
//   SAT_PC  copy-last saturating row index (-1 = disabled)
module MeasurementSync #(
    parameter int M      = 3,
    parameter int D_FIFO = 4,
    parameter int PC_W   = 16,
    parameter int SAT_PC = -1
) (
    input  logic             clk,
    input  logic             rst,              // asynchronous, active high
    input  logic [M-1:0]     in_meas,          // raw measurement bits
    input  logic [M-1:0]     in_valid,         // per-line valid (1 for one cycle)
    input  logic [M-1:0]     in_finish,        // per-line finish (rides with each measurement)
    input  logic [M-1:0]     sync_mask,        // mask from the sync regfile @ pc
    output logic [M-1:0]     out_meas,         // synced measurements (0 where unused)
    output logic [M-1:0]     out_used,         // which outputs are used (= mask)
    output logic             out_valid,        // 1 for one cycle when out_meas is valid
    output logic             out_finish,       // 1 when the emitted meas-time is the last round
    output logic [PC_W-1:0]  sync_regfile_pc   // read address into the sync regfile
);

    localparam int PTR_W = (D_FIFO < 2) ? 1 : $clog2(D_FIFO);
    // saturation: hold pc at SAT_PC (the copy-last wait row) instead of advancing past it
    localparam bit              SAT_EN  = (SAT_PC >= 0);
    localparam logic [PC_W-1:0] SAT_VAL = SAT_PC[PC_W-1:0];

    // --- per-line FIFO state (value + parallel finish tag, shared pointers) ---
    logic                mem   [M][D_FIFO];   // 1-bit entries (measurement value)
    logic                fmem  [M][D_FIFO];   // parallel finish tag per entry
    logic [PTR_W-1:0]    head  [M];
    logic [PTR_W-1:0]    tail  [M];
    logic [PTR_W:0]      count [M];           // occupancy 0..D_FIFO
    logic [PC_W-1:0]     pc;

    assign sync_regfile_pc = pc;

    // --- combinational: resolve sources + readiness + next output word ---
    logic [M-1:0] empty, from_fifo, bypass_sel, head_val, head_fin;
    logic         ready;
    logic [M-1:0] out_meas_c, out_used_c;
    logic         out_valid_c, out_finish_c;

    always_comb begin
        ready = 1'b1;
        for (int i = 0; i < M; i++) begin
            empty[i]      = (count[i] == 0);
            head_val[i]   = mem[i][head[i]];
            head_fin[i]   = fmem[i][head[i]];
            from_fifo[i]  = sync_mask[i] & ~empty[i];
            bypass_sel[i] = sync_mask[i] &  empty[i] & in_valid[i];
            if (sync_mask[i] & ~(from_fifo[i] | bypass_sel[i]))
                ready = 1'b0;                     // needed but unavailable -> stall
        end

        out_meas_c   = '0;
        out_used_c   = '0;
        out_valid_c  = 1'b0;
        out_finish_c = 1'b0;
        if (ready) begin
            out_valid_c = 1'b1;
            for (int i = 0; i < M; i++) begin
                if (sync_mask[i]) begin
                    out_used_c[i] = 1'b1;
                    out_meas_c[i] = from_fifo[i] ? head_val[i] : in_meas[i];
                    out_finish_c  = out_finish_c | (from_fifo[i] ? head_fin[i] : in_finish[i]);
                end
            end
        end
    end

    // --- sequential: update FIFOs, pc, registered outputs (async reset) ---
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            pc         <= '0;
            out_meas   <= '0;
            out_used   <= '0;
            out_valid  <= 1'b0;
            out_finish <= 1'b0;
            for (int i = 0; i < M; i++) begin
                head[i]  <= '0;
                tail[i]  <= '0;
                count[i] <= '0;
            end
        end else begin
            for (int i = 0; i < M; i++) begin
                automatic logic do_pop    = ready & from_fifo[i];
                automatic logic full_i    = (count[i] == D_FIFO[PTR_W:0]);
                // a push fits if there is room, or a pop frees a slot this cycle
                automatic logic can_push  = ~full_i | do_pop;
                automatic logic want_push = in_valid[i] & ~(ready & bypass_sel[i]);
                automatic logic do_push   = want_push & can_push;

                if (do_pop)
                    head[i] <= (head[i] == PTR_W'(D_FIFO-1)) ? '0 : head[i] + 1'b1;
                if (do_push) begin
                    mem[i][tail[i]]  <= in_meas[i];
                    fmem[i][tail[i]] <= in_finish[i];   // finish rides in lockstep with the value
                    tail[i] <= (tail[i] == PTR_W'(D_FIFO-1)) ? '0 : tail[i] + 1'b1;
                end
                unique case ({do_push, do_pop})
                    2'b10:   count[i] <= count[i] + 1'b1;
                    2'b01:   count[i] <= count[i] - 1'b1;
                    default: count[i] <= count[i];   // 00 idle, 11 push+pop net 0
                endcase

                // synthesis translate_off
                if (want_push & ~can_push)
                    $error("MeasurementSync: FIFO overflow on line %0d", i);
                // synthesis translate_on
            end

            out_meas   <= out_meas_c;
            out_used   <= out_used_c;
            out_valid  <= out_valid_c;
            out_finish <= out_finish_c;
            if (ready) pc <= (SAT_EN && (pc + 1'b1) > SAT_VAL) ? SAT_VAL : pc + 1'b1;
        end
    end

endmodule
