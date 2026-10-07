// RootOutputSync.sv — root-node output sync: global indices, round markers, and
// the copy-last wait offset (RTL).
//
// 1:1 RTL mapping of the SystemC model (emulator/src/detector_construct/
// root_output_sync.cpp). Wraps OutputSync and adds, on top of the registered
// synced detector word:
//   * global indices — global_index[nt] read at sync_regfile_pc, registered one
//     cycle (gi_reg) to line up with OutputSync's registered out_det;
//   * round markers  — round_marker[nt]={is_first_normal,is_last_normal,is_wait}
//     read/registered the same way (rm_reg), decoded into first_normal /
//     last_normal / first_wait / last_wait for the control core;
//   * copy-last wait offset — the output pc saturates at SAT_PC (the appended
//     copy-last row). On the saturating row the global index is the regfile base
//     plus w*STRIDE (w = # saturating rounds so far), so each wait repeat gets a
//     fresh detector-number block. Non-saturating rounds get offset 0.
//
// Output global-index datapath is HW_WIDTH bits/line (>= INDEX_W): the regfile
// value is zero-extended and the offset added; unused slots (regfile == SENTINEL,
// or not in out_used) are one-extended to all-ones.
//
// pc_reg registers sync_regfile_pc one cycle: at the posedge sync_regfile_pc still
// holds the pc that PRODUCED the current out_det (OutputSync updates it to pc+1 on
// the same edge via NBA), so pc_reg aligns with out_det just like gi_reg/rm_reg.
//
// Parameters:
//   D        number of detector lines
//   D_FIFO   depth of each per-line sync FIFO (passed to OutputSync)
//   PC_W     width of the regfile address (shared by all output regfiles)
//   INDEX_W  bits per global index in the REGFILE
//   HW_WIDTH bits per global index on the OUTPUT datapath (>= INDEX_W)
//   STRIDE   detectors per wait round (used lines at the last det-time)
//   SENTINEL all-ones regfile index value marking an unused slot
//   SAT_PC   copy-last saturating row index (-1 = no wait row)
module RootOutputSync #(
    parameter int D        = 3,
    parameter int D_FIFO   = 4,
    parameter int PC_W     = 16,
    parameter int INDEX_W  = 10,
    parameter int HW_WIDTH = 24,
    parameter int STRIDE   = 0,
    parameter int SENTINEL = (1 << INDEX_W) - 1,
    parameter int SAT_PC   = -1
) (
    input  logic                  clk,
    input  logic                  rst,
    input  logic [D-1:0]          in_det,
    input  logic [D-1:0]          in_valid,
    input  logic [D-1:0]          in_finish,           // per-line finish (rides with each detector)
    input  logic [D-1:0]          sync_mask,           // output-sync mask @ pc
    input  logic [D*INDEX_W-1:0]  global_indexes,      // global-index word @ pc
    input  logic [2:0]            round_marker,        // {is_first_normal,is_last_normal,is_wait} @ pc
    output logic [D-1:0]          out_det,
    output logic [D-1:0]          out_used,
    output logic                  out_valid,
    output logic                  out_finish,          // 1 when the emitted det-time is the last round
    output logic [PC_W-1:0]       sync_regfile_pc,     // addresses sync + global-index + round-marker regfiles
    output logic [D*HW_WIDTH-1:0] out_global_indexes,  // aligned with out_det
    // boundary markers for the control core (one-cycle pulses, aligned w/ out_det)
    output logic                  first_normal,        // is_first_normal & valid
    output logic                  last_normal,         // is_last_normal  & valid
    output logic                  first_wait,          // rising edge of is_wait
    output logic                  last_wait            // is_wait & finish
);

    localparam bit                 SAT_EN  = (SAT_PC >= 0);
    localparam logic [PC_W-1:0]    SAT_VAL = SAT_PC[PC_W-1:0];
    localparam logic [INDEX_W-1:0] SENT    = SENTINEL[INDEX_W-1:0];

    logic                 ov_int;      // OutputSync.out_valid
    logic                 of_int;      // OutputSync.out_finish
    logic [D-1:0]         used_int;    // OutputSync.out_used (also forwarded)
    logic [D*INDEX_W-1:0] gi_reg;      // global_indexes delayed one cycle
    logic [2:0]           rm_reg;      // round_marker delayed one cycle
    logic [PC_W-1:0]      pc_reg;      // producing pc, delayed one cycle (= current det-time index)
    logic                 is_wait_q;   // is_wait of the previous VALID output round
    logic [PC_W-1:0]      w;           // saturating-round counter (wait repeats)

    // shared detector-sync front-end (out_det passes straight through)
    OutputSync #(.D(D), .D_FIFO(D_FIFO), .PC_W(PC_W), .SAT_PC(SAT_PC)) os (
        .clk             (clk),
        .rst             (rst),
        .in_det          (in_det),
        .in_valid        (in_valid),
        .in_finish       (in_finish),
        .sync_mask       (sync_mask),
        .out_det         (out_det),
        .out_used        (used_int),
        .out_valid       (ov_int),
        .out_finish      (of_int),
        .sync_regfile_pc (sync_regfile_pc)
    );

    assign out_valid  = ov_int;
    assign out_finish = of_int;
    assign out_used   = used_int;

    // this round on the saturating (repeated copy-last) row?
    logic sat_here;
    assign sat_here = SAT_EN & (pc_reg == SAT_VAL);

    // --- reg_pipe: align gi/rm/pc with out_det; track is_wait edge + wait counter ---
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            gi_reg    <= '0;
            rm_reg    <= '0;
            pc_reg    <= '0;
            is_wait_q <= 1'b0;
            w         <= '0;
        end else begin
            gi_reg <= global_indexes;   // NBA: reads below see the pre-edge (current-round) values
            rm_reg <= round_marker;
            pc_reg <= sync_regfile_pc;
            if (ov_int) begin
                is_wait_q <= rm_reg[2];              // this round's is_wait -> "previous" next round
                if (sat_here) w <= w + 1'b1;         // count saturating (wait-repeat) rounds
            end
        end
    end

    // --- markers (combinational, aligned with out_det) ---
    assign first_normal = ov_int & rm_reg[0];
    assign last_normal  = ov_int & rm_reg[1];
    assign first_wait   = ov_int & rm_reg[2] & ~is_wait_q;   // rising edge of is_wait
    assign last_wait    = ov_int & rm_reg[2] & of_int;       // is_wait & finish

    // --- global-index bus (HW_WIDTH bits/line) with the copy-last offset ---
    logic [HW_WIDTH-1:0] offset;
    assign offset = sat_here ? HW_WIDTH'(w * STRIDE) : '0;

    always_comb begin
        for (int l = 0; l < D; l++) begin
            automatic logic [INDEX_W-1:0] idx = gi_reg[l*INDEX_W +: INDEX_W];
            automatic logic is_used = ov_int & used_int[l];
            out_global_indexes[l*HW_WIDTH +: HW_WIDTH] =
                (is_used && idx != SENT) ? (HW_WIDTH'(idx) + offset) : {HW_WIDTH{1'b1}};
        end
    end

endmodule
