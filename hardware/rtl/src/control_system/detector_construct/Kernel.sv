// Kernel.sv — one detector-construction channel (RTL).
//
// 1:1 RTL mapping of the SystemC model (emulator/src/detector_construct/
// kernel.cpp). A fixed selector picks n of the m measurement lines; h cores each
// XOR-accumulate a masked subset and emit-and-clear when their detector window
// completes. At most one core emits per meas-time (compiler guarantee), so the
// emit bits form a one-hot select for the single detector output.
//
// Datapath (selector -> mask-AND -> XOR-tree -> XOR-with-reg) is combinational;
// core registers, the core-regfile pointer, and the outputs are registered, so
// the block has one-cycle latency. The pointer advances once per in_valid.
//
// Finish (P2b): a single in_finish (MeasurementSync.out_finish) rides in; the
// emitted detector carries the finish of the meas-time it completed on
// (out_finish <= any_emit ? in_finish : 0).
//
// Saturation (P2c): with copy-last the core regfile has an appended saturating
// wait row. The core pc advances once per in_valid in lockstep with
// MeasurementSync, so it takes the SAME saturation index: if SAT_PC >= 0 the pc
// holds at SAT_PC instead of advancing past it. SAT_PC = -1 disables (normal).
//
// Regfile field formats (match the compiler serializer):
//   selector_indexes[n*INDEX_W-1:0] = {index n-1, ..., index 0}
//   core_mask[h*(n+1)-1:0]          = {core h-1, ..., core 0}, core c at bit
//                                     c*(n+1); within a core bits[n-1:0]=select,
//                                     bit[n]=emit.
//
// Parameters:
//   N     selector width (measurements selected per kernel)
//   H     number of cores
//   M     number of board measurement input lines
//   PC_W  width of the core regfile address
//   SAT_PC copy-last saturating core-row index (-1 = disabled)
module Kernel #(
    parameter int N      = 2,
    parameter int H      = 2,
    parameter int M      = 4,
    parameter int PC_W   = 16,
    parameter int SAT_PC = -1,
    // derived — do not override: bits per selector index = ceil(log2 M), min 1
    parameter int INDEX_W = (M <= 1) ? 1 : $clog2(M)
) (
    input  logic                 clk,
    input  logic                 rst,              // asynchronous, active high
    input  logic                 in_valid,         // out_valid of MeasurementSync
    input  logic [M-1:0]         in_used,          // out_used of MeasurementSync
    input  logic [M-1:0]         in_meas,          // out_meas of MeasurementSync
    input  logic                 in_finish,        // out_finish of MeasurementSync (consumed meas-time)
    input  logic [N*INDEX_W-1:0] selector_indexes, // static (preconfigured)
    input  logic [H*(N+1)-1:0]   core_mask,        // read @ core_regfile_pc
    output logic                 out_valid,        // 1 for one cycle when a detector is emitted
    output logic                 out_det,          // the constructed detector bit
    output logic                 out_finish,       // finish of the meas-time this detector completed on
    output logic [PC_W-1:0]      core_regfile_pc   // read address into the core regfile
);

    // saturation: hold pc at SAT_PC (the copy-last wait row) instead of advancing past it
    localparam bit              SAT_EN  = (SAT_PC >= 0);
    localparam logic [PC_W-1:0] SAT_VAL = SAT_PC[PC_W-1:0];

    // --- registered state ---
    logic [H-1:0]    acc;   // per-core accumulators
    logic [PC_W-1:0] pc;
    assign core_regfile_pc = pc;

    // --- combinational datapath ---
    logic [N-1:0] b_m;        // selected (and used) measurement bits
    logic [H-1:0] emit;       // emit bit per core (raw from mask)
    logic [H-1:0] b_core;     // accumulated bit per core (b_c)
    logic         any_emit;
    logic         out_det_c;

    always_comb begin
        // selector: gather the n selected, used measurements
        b_m = '0;
        for (int j = 0; j < N; j++) begin
            automatic logic [INDEX_W-1:0] idx = selector_indexes[j*INDEX_W +: INDEX_W];
            b_m[j] = in_valid & in_used[idx] & in_meas[idx];
        end
        // cores: masked XOR-tree, accumulate; one-hot detector select
        out_det_c = 1'b0;
        for (int c = 0; c < H; c++) begin
            automatic logic [N-1:0] sel = core_mask[c*(N+1) +: N];
            automatic logic         res = ^(sel & b_m);
            emit[c]   = core_mask[c*(N+1) + N];
            b_core[c] = res ^ acc[c];
            out_det_c = out_det_c | (emit[c] & b_core[c]);
        end
        any_emit = in_valid & (|emit);
    end

    // --- sequential: accumulate / emit-and-clear, pc, registered outputs ---
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            pc         <= '0;
            acc        <= '0;
            out_valid  <= 1'b0;
            out_det    <= 1'b0;
            out_finish <= 1'b0;
        end else if (in_valid) begin
            automatic logic [PC_W-1:0] next_pc = pc + 1'b1;
            if (SAT_EN && next_pc > SAT_VAL) next_pc = SAT_VAL;   // saturate at the copy-last wait row
            for (int c = 0; c < H; c++)
                acc[c] <= emit[c] ? 1'b0 : b_core[c];   // emit-and-clear, else store
            out_valid  <= any_emit;
            out_det    <= out_det_c;
            out_finish <= any_emit ? in_finish : 1'b0;  // emitted detector carries this meas-time's finish
            pc         <= next_pc;
        end else begin
            out_valid  <= 1'b0;   // no synced word: hold acc & pc, outputs idle
            out_det    <= 1'b0;
            out_finish <= 1'b0;
        end
    end

endmodule
