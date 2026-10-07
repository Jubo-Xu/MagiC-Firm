// DetectorConstructBlock.sv — the universal per-board detector-construct block (RTL).
//
// 1:1 structural port of the SystemC model (emulator/src/detector_construct/
// detector_construct_block.cpp). ONE parameterized module for every board kind
// (leaf / router / root / stage / monolithic); which sub-blocks exist is chosen by
// `generate` from the params, and each board instance points its regfile ROMs at
// its own mem/ directory via MEM_DIR + $readmemb.
//
//   raw path    (RAW_OUT>0) : RawSelector                  -> fwd_raw*
//   construct   (K>0)       : MeasurementSync -> K Kernels  -> construct dets
//   pass        (D_IN>0)    : DetectorPass                  -> pass dets
//   out bus                 : D_OUT = pass(D_IN) ++ construct(K)  -> fwd_det (if !root)
//   output sync (root|stage): RootOutputSync (root: +global index/round marker) / OutputSync
//   postselect  (stage)     : Postselect (shared osync_pc)
//
// Static selector indices (kernel + raw) are $readmemb'd into packed constants
// (the RTL of drive_constants). Address-driven regfiles are RegFileROMs read at a
// running pc. Per-regfile RAM_STYLE lets Vivado map each differently (e.g. selectors
// are just registers; large global_index can be distributed/auto — true BRAM needs
// SyncROM, a follow-up).
`timescale 1ns / 1ps

module DetectorConstructBlock #(
    parameter int M         = 4,          // raw measurement input lines on this board
    parameter int D_IN      = 0,          // input detector lines from children (0 for leaf/monolithic)
    parameter int K         = 1,          // number of construction kernels (local detector channels)
    parameter int N         = 2,          // selector width per kernel (measurements selected)
    parameter int H         = 2,          // cores per kernel
    parameter int RAW_OUT   = 0,          // raw measurements forwarded up to the parent (0 for root)
    parameter int IDX_W     = 10,         // global-index width in the REGFILE (root)
    parameter int HW_WIDTH  = 24,         // global-index width on the OUTPUT datapath (root, >= IDX_W)
    parameter int STRIDE    = 0,          // detectors per wait round (used lines at the last det-time, root)
    parameter int SENTINEL  = (1 << IDX_W) - 1,  // all-ones regfile index = unused slot (root)
    parameter int T         = 4,          // sync/core regfile rows = meas-times (incl. copy-last wait row)
    parameter int NDT       = 4,          // output regfile rows = det-times (incl. copy-last wait row)
    parameter int PC_W      = 16,         // regfile address width (shared by all regfiles on the board)
    parameter int SYNC_FIFO = 4,          // depth of each MeasurementSync per-line FIFO
    parameter int OUT_FIFO  = 4,          // depth of each OutputSync per-line FIFO
    parameter bit IS_ROOT   = 1'b1,       // this board is the tree root (final output vs forward-up)
    parameter bit HAS_PS    = 1'b0,       // this board has a postselect (reject) stage
    parameter bit HAS_OSYNC = 1'b0,       // this board carries an output_sync regfile (--output-sync all)
    parameter bit COPY_LAST = 1'b0,       // copy-last wait row present -> pc saturates at each regfile's last row
    parameter     MEM_DIR   = "",         // prefix prepended to each regfile filename in $readmemb,
                                          //   e.g. "board0_" (flat, Vivado-friendly) -> "board0_sync.mem"
    // per-regfile Vivado ram_style (synthesis attribute; ignored in simulation)
    parameter     RAM_STYLE_SYNC  = "distributed",  // sync-mask regfile
    parameter     RAM_STYLE_CORE  = "distributed",  // per-kernel core regfiles
    parameter     RAM_STYLE_OSYNC = "distributed",  // output-sync mask regfile
    parameter     RAM_STYLE_GIDX  = "distributed",  // global-index regfile (large; consider BRAM via SyncROM later)
    parameter     RAM_STYLE_RM    = "distributed",  // round-marker regfile
    parameter     RAM_STYLE_PS    = "distributed",  // postselect-mask regfile
    // ---- derived (do not override) ----
    parameter int D_OUT  = D_IN + K,                    // output detector lines = pass ++ construct
    parameter int DIN_W  = (D_IN   < 1) ? 1 : D_IN,     // safe (non-zero) width for the in_det* ports
    parameter int RAW_W  = (RAW_OUT < 1) ? 1 : RAW_OUT, // safe (non-zero) width for the fwd_raw* ports
    parameter int KW     = (K       < 1) ? 1 : K,       // safe (non-zero) width for per-kernel arrays
    parameter int SEL_IW = (M <= 1) ? 1 : $clog2(M),    // bits per selector index = ceil(log2 M)
    parameter int CORE_W = H * (N + 1),                 // core-mask word width = H*(N+1)
    parameter int SYNC_AW  = (T   <= 1) ? 1 : $clog2(T),   // sync/core regfile address width
    parameter int OSYNC_AW = (NDT <= 1) ? 1 : $clog2(NDT)  // output regfile address width
) (
    input  logic                    clk,
    input  logic                    rst,

    // raw measurements in
    input  logic [M-1:0]            in_meas,
    input  logic [M-1:0]            in_valid,
    input  logic [M-1:0]            in_meas_finish,
    // children detectors in
    input  logic [DIN_W-1:0]        in_det,
    input  logic [DIN_W-1:0]        in_det_valid,
    input  logic [DIN_W-1:0]        in_det_finish,

    // forward-up to parent (meaningful when !IS_ROOT / RAW_OUT>0)
    output logic [D_OUT-1:0]        fwd_det,
    output logic [D_OUT-1:0]        fwd_det_valid,
    output logic                    fwd_det_finish,
    output logic [RAW_W-1:0]        fwd_raw,
    output logic [RAW_W-1:0]        fwd_raw_valid,
    output logic                    fwd_raw_finish,

    // root final output (meaningful when IS_ROOT)
    output logic [D_OUT-1:0]        out_det,
    output logic [D_OUT-1:0]        out_used,
    output logic                    out_valid,
    output logic                    out_finish,
    output logic [D_OUT*HW_WIDTH-1:0] out_global_indexes,
    output logic                    first_normal,
    output logic                    last_normal,
    output logic                    first_wait,
    output logic                    last_wait,

    // stage postselect (meaningful when HAS_PS)
    output logic                    post_select
);

    localparam int MS_SAT  = COPY_LAST ? (T   - 1) : -1;   // MeasurementSync/Kernel saturating pc index
    localparam int OUT_SAT = COPY_LAST ? (NDT - 1) : -1;   // OutputSync saturating pc index
    localparam bit HAS_OUTPUT_SYNC = IS_ROOT | HAS_PS | HAS_OSYNC;

    // ---- construct-path signals ----
    logic [PC_W-1:0]   sync_pc;
    logic [M-1:0]      sync_mask, ms_meas, ms_used;
    logic              ms_valid, ms_finish;
    logic [PC_W-1:0]   core_pc   [KW];
    logic [CORE_W-1:0] core_mask [KW];
    logic [N*SEL_IW-1:0] sel_const [KW];
    logic [KW-1:0]     kern_det, kern_valid, kern_finish;

    // ---- pass-path signals ----
    logic [DIN_W-1:0]  pass_det, pass_valid, pass_finish;

    // ---- the d_out bus ----
    logic [D_OUT-1:0]  dbus_det, dbus_valid, dbus_finish;

    // ---- output-path signals ----
    logic [PC_W-1:0]         osync_pc;
    logic [D_OUT-1:0]        osync_mask, osync_det, osync_used;
    logic [D_OUT*IDX_W-1:0]  gidx_word;
    logic [2:0]              rm_word;
    logic                    osync_valid, osync_finish;
    logic [D_OUT-1:0]        ps_mask;

    // ---- raw selector constant ----
    logic [RAW_W*SEL_IW-1:0] raw_sel_const;

    // ============================ raw path ============================
    generate
        if (RAW_OUT > 0) begin : g_raw
            logic [SEL_IW-1:0] raw_sel_mem [RAW_OUT];
            initial $readmemb({MEM_DIR, "raw_selector.mem"}, raw_sel_mem);
            for (genvar i = 0; i < RAW_OUT; i++) begin : g_pack
                assign raw_sel_const[i*SEL_IW +: SEL_IW] = raw_sel_mem[i];
            end

            RawSelector #(.R(RAW_OUT), .M(M)) raw (
                .clk              (clk),
                .rst              (rst),
                .selector_indexes (raw_sel_const),
                .in_meas          (in_meas),
                .in_valid         (in_valid),
                .in_finish        (in_meas_finish),
                .out_meas         (fwd_raw),
                .out_valid        (fwd_raw_valid),
                .out_finish       (fwd_raw_finish)
            );
        end else begin : g_no_raw
            assign raw_sel_const  = '0;
            assign fwd_raw        = '0;
            assign fwd_raw_valid  = '0;
            assign fwd_raw_finish = 1'b0;
        end
    endgenerate

    // ========================= construct path =========================
    generate
        if (K > 0) begin : g_construct
            RegFileROM #(.DEPTH(T), .WIDTH(M),
                         .INIT_FILE({MEM_DIR, "sync.mem"}), .RAM_STYLE(RAM_STYLE_SYNC)) sync_rom (
                .addr (sync_pc[SYNC_AW-1:0]),
                .data (sync_mask)
            );

            MeasurementSync #(.M(M), .D_FIFO(SYNC_FIFO), .PC_W(PC_W), .SAT_PC(MS_SAT)) ms (
                .clk             (clk),
                .rst             (rst),
                .in_meas         (in_meas),
                .in_valid        (in_valid),
                .in_finish       (in_meas_finish),
                .sync_mask       (sync_mask),
                .out_meas        (ms_meas),
                .out_used        (ms_used),
                .out_valid       (ms_valid),
                .out_finish      (ms_finish),
                .sync_regfile_pc (sync_pc)
            );

            for (genvar i = 0; i < K; i++) begin : g_kern
                RegFileROM #(.DEPTH(T), .WIDTH(CORE_W),
                             .INIT_FILE($sformatf("%sk%0d_core.mem", MEM_DIR, i)),
                             .RAM_STYLE(RAM_STYLE_CORE)) core_rom (
                    .addr (core_pc[i][SYNC_AW-1:0]),
                    .data (core_mask[i])
                );

                // static selector: N rows x SEL_IW -> packed constant {idx N-1..idx 0}
                logic [SEL_IW-1:0] sel_mem [N];
                initial $readmemb($sformatf("%sk%0d_selector.mem", MEM_DIR, i), sel_mem);
                for (genvar j = 0; j < N; j++) begin : g_pack
                    assign sel_const[i][j*SEL_IW +: SEL_IW] = sel_mem[j];
                end

                Kernel #(.N(N), .H(H), .M(M), .PC_W(PC_W), .SAT_PC(MS_SAT)) kern (
                    .clk              (clk),
                    .rst              (rst),
                    .in_valid         (ms_valid),
                    .in_used          (ms_used),
                    .in_meas          (ms_meas),
                    .in_finish        (ms_finish),
                    .selector_indexes (sel_const[i]),
                    .core_mask        (core_mask[i]),
                    .out_valid        (kern_valid[i]),
                    .out_det          (kern_det[i]),
                    .out_finish       (kern_finish[i]),
                    .core_regfile_pc  (core_pc[i])
                );
            end
        end else begin : g_no_construct
            assign sync_pc     = '0;
            assign kern_det    = '0;
            assign kern_valid  = '0;
            assign kern_finish = '0;
        end
    endgenerate

    // =========================== pass path ============================
    generate
        if (D_IN > 0) begin : g_pass
            DetectorPass #(.D(D_IN)) pass (
                .clk        (clk),
                .rst        (rst),
                .in_valid   (in_det_valid),
                .in_det     (in_det),
                .in_finish  (in_det_finish),
                .out_valid  (pass_valid),
                .out_det    (pass_det),
                .out_finish (pass_finish)
            );
        end else begin : g_no_pass
            assign pass_det    = '0;
            assign pass_valid  = '0;
            assign pass_finish = '0;
        end
    endgenerate

    // =================== gather: d_out bus = pass ++ construct ===================
    always_comb begin
        dbus_det    = '0;
        dbus_valid  = '0;
        dbus_finish = '0;
        for (int i = 0; i < D_IN; i++) begin
            dbus_det[i]    = pass_det[i];
            dbus_valid[i]  = pass_valid[i];
            dbus_finish[i] = pass_finish[i];
        end
        for (int i = 0; i < K; i++) begin
            dbus_det[D_IN + i]    = kern_det[i];
            dbus_valid[D_IN + i]  = kern_valid[i];
            dbus_finish[D_IN + i] = kern_finish[i];
        end
    end

    // ========================== output path ==========================
    generate
        if (HAS_OUTPUT_SYNC) begin : g_out
            RegFileROM #(.DEPTH(NDT), .WIDTH(D_OUT),
                         .INIT_FILE({MEM_DIR, "output_sync.mem"}), .RAM_STYLE(RAM_STYLE_OSYNC)) osync_rom (
                .addr (osync_pc[OSYNC_AW-1:0]),
                .data (osync_mask)
            );

            if (IS_ROOT) begin : g_root
                RegFileROM #(.DEPTH(NDT), .WIDTH(D_OUT*IDX_W),
                             .INIT_FILE({MEM_DIR, "global_index.mem"}), .RAM_STYLE(RAM_STYLE_GIDX)) gidx_rom (
                    .addr (osync_pc[OSYNC_AW-1:0]),
                    .data (gidx_word)
                );
                RegFileROM #(.DEPTH(NDT), .WIDTH(3),
                             .INIT_FILE({MEM_DIR, "round_marker.mem"}), .RAM_STYLE(RAM_STYLE_RM)) rm_rom (
                    .addr (osync_pc[OSYNC_AW-1:0]),
                    .data (rm_word)
                );

                RootOutputSync #(.D(D_OUT), .D_FIFO(OUT_FIFO), .PC_W(PC_W), .INDEX_W(IDX_W),
                                 .HW_WIDTH(HW_WIDTH), .STRIDE(STRIDE), .SENTINEL(SENTINEL), .SAT_PC(OUT_SAT)) rosync (
                    .clk                (clk),
                    .rst                (rst),
                    .in_det             (dbus_det),
                    .in_valid           (dbus_valid),
                    .in_finish          (dbus_finish),
                    .sync_mask          (osync_mask),
                    .global_indexes     (gidx_word),
                    .round_marker       (rm_word),
                    .out_det            (osync_det),     // internal (also feeds postselect)
                    .out_used           (out_used),      // -> block port
                    .out_valid          (osync_valid),   // internal (also feeds postselect)
                    .out_finish         (out_finish),    // -> block port
                    .sync_regfile_pc    (osync_pc),
                    .out_global_indexes (out_global_indexes),
                    .first_normal       (first_normal),
                    .last_normal        (last_normal),
                    .first_wait         (first_wait),
                    .last_wait          (last_wait)
                );

                // drive_output: internal synced det/valid -> block output ports
                assign out_det    = osync_det;
                assign out_valid  = osync_valid;
                assign osync_used = '0;   // unused at root

                // root does not forward detectors up
                assign fwd_det        = '0;
                assign fwd_det_valid  = '0;
                assign fwd_det_finish = 1'b0;
            end else begin : g_nonroot
                OutputSync #(.D(D_OUT), .D_FIFO(OUT_FIFO), .PC_W(PC_W), .SAT_PC(OUT_SAT)) osync (
                    .clk             (clk),
                    .rst             (rst),
                    .in_det          (dbus_det),
                    .in_valid        (dbus_valid),
                    .in_finish       (dbus_finish),
                    .sync_mask       (osync_mask),
                    .out_det         (osync_det),
                    .out_used        (osync_used),
                    .out_valid       (osync_valid),
                    .out_finish      (osync_finish),
                    .sync_regfile_pc (osync_pc)
                );

                // drive_fwd: the OutputSync-synced detectors ARE this board's forwarded output.
                // Collapse (out_valid, out_used[]) back to per-line fwd_det_valid[].
                assign fwd_det = osync_det;
                for (genvar i = 0; i < D_OUT; i++) begin : g_fwd_valid
                    assign fwd_det_valid[i] = osync_valid & osync_used[i];
                end
                assign fwd_det_finish = osync_finish;

                // root-only outputs unused on a non-root board
                assign out_det            = '0;
                assign out_used           = '0;
                assign out_valid          = 1'b0;
                assign out_finish         = 1'b0;
                assign out_global_indexes = '0;
                assign first_normal       = 1'b0;
                assign last_normal        = 1'b0;
                assign first_wait         = 1'b0;
                assign last_wait          = 1'b0;
            end
        end else begin : g_no_out
            // no output sync on this board: tie every output-path port off
            assign osync_pc           = '0;
            assign osync_mask         = '0;
            assign osync_det          = '0;
            assign osync_used         = '0;
            assign osync_valid        = 1'b0;
            assign osync_finish       = 1'b0;
            assign gidx_word          = '0;
            assign rm_word            = '0;
            assign fwd_det            = '0;
            assign fwd_det_valid      = '0;
            assign fwd_det_finish     = 1'b0;
            assign out_det            = '0;
            assign out_used           = '0;
            assign out_valid          = 1'b0;
            assign out_finish         = 1'b0;
            assign out_global_indexes = '0;
            assign first_normal       = 1'b0;
            assign last_normal        = 1'b0;
            assign first_wait         = 1'b0;
            assign last_wait          = 1'b0;
        end
    endgenerate

    // ========================== postselect ==========================
    generate
        if (HAS_PS) begin : g_ps
            RegFileROM #(.DEPTH(NDT), .WIDTH(D_OUT),
                         .INIT_FILE({MEM_DIR, "postselect.mem"}), .RAM_STYLE(RAM_STYLE_PS)) ps_rom (
                .addr (osync_pc[OSYNC_AW-1:0]),   // shared output-sync pc (aligned regfiles)
                .data (ps_mask)
            );
            Postselect #(.D(D_OUT)) ps (
                .clk             (clk),
                .rst             (rst),
                .in_valid        (osync_valid),
                .in_det          (osync_det),
                .postselect_mask (ps_mask),
                .post_select     (post_select)
            );
        end else begin : g_no_ps
            assign ps_mask     = '0;
            assign post_select = 1'b0;
        end
    endgenerate

endmodule
