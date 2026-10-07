// RawSelector.sv — raw-measurement forwarding selector (RTL).
//
// 1:1 RTL mapping of the SystemC model (emulator/src/detector_construct/
// raw_selector.cpp). Forwards r of the m raw measurement lines (chosen by a
// fixed selector) up the tree, each with its valid. Pure select + one-cycle
// register — independent of the detector path (each line carries its own valid).
//
// Finish (P2b): the raw link carries a SINGLE finish bit — out_finish = OR over
// the forwarded lines of (in_valid[idx] & in_finish[idx]), i.e. the finish of the
// round these raw measurements belong to. The parent broadcasts it back across
// this child's raw lines at board assembly.
//
// Regfile format (matches the compiler serializer):
//   selector_indexes[r*INDEX_W-1:0] = {index r-1, ..., index 0}
//
// Parameters:
//   R  number of raw output measurements
//   M  number of raw input measurement lines
module RawSelector #(
    parameter int R = 2,
    parameter int M = 4,
    // derived — do not override: bits per selector index = ceil(log2 M), min 1
    parameter int INDEX_W = (M <= 1) ? 1 : $clog2(M)
) (
    input  logic                 clk,
    input  logic                 rst,               // asynchronous, active high
    input  logic [R*INDEX_W-1:0] selector_indexes,  // static (preconfigured)
    input  logic [M-1:0]         in_meas,           // raw measurements
    input  logic [M-1:0]         in_valid,          // per-line valid
    input  logic [M-1:0]         in_finish,         // per-line finish
    output logic [R-1:0]         out_meas,          // out_meas[i]  = in_meas[index i]
    output logic [R-1:0]         out_valid,         // out_valid[i] = in_valid[index i]
    output logic                 out_finish         // OR over forwarded (valid) lines of their finish
);

    // --- combinational selection ---
    logic [R-1:0] sel_meas, sel_valid;
    logic         sel_finish;
    always_comb begin
        sel_meas   = '0;
        sel_valid  = '0;
        sel_finish = 1'b0;
        for (int i = 0; i < R; i++) begin
            automatic logic [INDEX_W-1:0] idx = selector_indexes[i*INDEX_W +: INDEX_W];
            sel_meas[i]  = in_meas[idx];
            sel_valid[i] = in_valid[idx];
            sel_finish   = sel_finish | (in_valid[idx] & in_finish[idx]);
        end
    end

    // --- one-cycle register ---
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            out_meas   <= '0;
            out_valid  <= '0;
            out_finish <= 1'b0;
        end else begin
            out_meas   <= sel_meas;
            out_valid  <= sel_valid;
            out_finish <= sel_finish;
        end
    end

endmodule
