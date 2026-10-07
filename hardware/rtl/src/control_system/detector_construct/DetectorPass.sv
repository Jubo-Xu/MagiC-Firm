// DetectorPass.sv — detector forwarding block (RTL).
//
// 1:1 RTL mapping of the SystemC model (emulator/src/detector_construct/
// detector_pass.cpp). Forwards the children's input detectors up the tree,
// registered one cycle. Pure 1-cycle register of the whole d-line bus + its
// per-line valids + per-line finish tags — no selection, no regfile (the identity
// form of RawSelector).
//
// Finish rides with the data like valid: in_finish[i] is the child det-link's
// finish bit broadcast onto line i at board assembly; it is registered one cycle
// exactly like in_valid/in_det so it stays aligned.
//
// Parameters:
//   D  number of detector lines
module DetectorPass #(
    parameter int D = 4
) (
    input  logic         clk,
    input  logic         rst,         // asynchronous, active high
    input  logic [D-1:0] in_valid,    // per-line valid
    input  logic [D-1:0] in_det,      // input detectors
    input  logic [D-1:0] in_finish,   // per-line finish
    output logic [D-1:0] out_valid,   // in_valid  delayed one cycle
    output logic [D-1:0] out_det,     // in_det    delayed one cycle
    output logic [D-1:0] out_finish   // in_finish delayed one cycle
);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            out_det    <= '0;
            out_valid  <= '0;
            out_finish <= '0;
        end else begin
            out_det    <= in_det;
            out_valid  <= in_valid;
            out_finish <= in_finish;
        end
    end

endmodule
