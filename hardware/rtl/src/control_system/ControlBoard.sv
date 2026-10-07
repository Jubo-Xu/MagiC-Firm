// ControlBoard.sv — the universal per-board wrapper (RTL).
// 1:1 port of emulator/{include,src}/control_system/cultiv_control/control_board.*.
//
// ONE parameterized module for every board kind (leaf / router / mid / stage /
// root / monolithic). It instantiates and wires the three sub-systems so that
// building the whole system is *pure board-to-board wiring* — every piece of
// glue lives inside here:
//
//   DetectorConstructBlock   detector datapath                (always)
//   BoardControl             attempt / event FSM              (always)
//   PhysicalMMIO [P]         command-word generators          (leaf / monolithic)
//
// GLUE ABSORBED HERE:
//   * child concatenation  : in_det / in_meas arrive already concatenated (trivial
//                            {} wiring at the port map); consumed directly.
//   * single-finish broadcast : each child sends ONE det/raw finish; fanned out
//                            across that child's slice using CHILD_DW / CHILD_RAW.
//   * attempt OR-reduce    : data_attempt = OR of children's attempt tags.
//   * stale-valid gating   : DCB input valids masked by set_in_data_to_zero.
//   * event transform      : root turns host start/finish into an event; non-root
//                            relays the parent's event bus.
//   * post-select assembly : root's post_select bus = own DCB ps + stage fast-path
//                            + gap, each stamped with the right attempt.
//   * drain / output status: drain_done = this board's DCB det-finish.
//
// MMIO EVENT SOURCE. Which control event drives the PhysicalMMIOs depends on the
// board kind, because the MMIO must see ABORT to restart its program on a retry:
//   FORWARD  (leaf/mid) : the event ENTERING BoardControl (bc_ev_*) — the parent's
//       relayed START/ABORT/FINISH; the abort reaches the MMIO directly.
//   ORIGINATE(root/mono): BoardControl's GENERATED out_ev (registered, +1 cycle),
//       because an ORIGINATE board's abort is produced onto out_ev, not bc_ev.
//
// DEFERRED (skeleton): leaf in_meas loopback (measured command words fed back to
// in_meas) — for now in_meas/in_meas_valid/in_meas_finish are leaf stimulus ports.
`timescale 1ns / 1ps

module ControlBoard #(
    // ---- board-kind selectors ----
    parameter bit IS_ROOT    = 1'b1,   // final output vs forward-up
    parameter bit IS_LEAF    = 1'b0,   // has PhysicalMMIOs feeding raw measurements
    parameter int EVENT_MODE = 0,      // BoardControl: 0=ORIGINATE (root/mono), 1=FORWARD (mid/leaf)
    parameter int DATA_SRC   = 0,      // BoardControl: 0=EXTERNAL (root/mid), 1=INTERNAL (leaf/mono)
    parameter bit HAS_PS     = 1'b0,   // this board runs a postselect (reject) stage

    // ---- child topology (parent side) ----
    parameter int NCHILD     = 0,      // number of child boards feeding this one
    // CHILD_DW/CHILD_RAW are sized by a FIXED MAX_CHILD (not NCHILD): only indices [0,NCHILD)
    // are read (prefix sums below), so the tail is inert. A size that depends on NCHILD trips a
    // param-ordering limitation in some sim tools when NCHILD + the array are overridden together;
    // a constant size sidesteps it and stays synthesizable (the pad is elaboration-time ints only).
    parameter int MAX_CHILD  = 64,     // upper bound on children per board (pad CHILD_* to this)
    parameter int CHILD_DW  [MAX_CHILD] = '{default: 0},  // det lines per child (= child D_OUT)
    parameter int CHILD_RAW [MAX_CHILD] = '{default: 0},  // raw lines per child

    // ---- root post-select fast path ----
    parameter int NPS        = 0,      // # of stage boards routing post_select straight to this root

    // ---- DetectorConstructBlock pass-through ----
    parameter int M          = 4,      // raw measurement lines on this board (= sum CHILD_RAW for a router)
    parameter int D_IN       = 0,      // input detector lines (= sum CHILD_DW; 0 for leaf/mono)
    parameter int K          = 1,      // construction kernels
    parameter int N          = 2,      // selector width per kernel
    parameter int H          = 2,      // cores per kernel
    parameter int RAW_OUT    = 0,      // raw measurements forwarded up (0 for root)
    parameter int IDX_W      = 10,     // regfile global-index width (root)
    parameter int HW_WIDTH   = 24,     // output-datapath global-index width (root)
    parameter int STRIDE     = 0,      // detectors per wait round (root)
    parameter int SENTINEL   = (1 << IDX_W) - 1,  // all-ones = unused slot (root)
    parameter int T          = 4,      // sync/core regfile rows (meas-times)
    parameter int NDT        = 4,      // output regfile rows (det-times)
    parameter int PC_W       = 16,     // regfile address width
    parameter int SYNC_FIFO  = 4,      // MeasurementSync per-line FIFO depth
    parameter int OUT_FIFO   = 4,      // OutputSync per-line FIFO depth
    parameter bit HAS_OSYNC  = 1'b0,   // carries an output_sync regfile
    parameter bit COPY_LAST  = 1'b0,   // copy-last wait row present (pc saturates)
    parameter     MEM_DIR    = "",     // regfile filename prefix (e.g. "board0_")
    parameter     RAM_STYLE_SYNC  = "distributed",
    parameter     RAM_STYLE_CORE  = "distributed",
    parameter     RAM_STYLE_OSYNC = "distributed",
    parameter     RAM_STYLE_GIDX  = "distributed",
    parameter     RAM_STYLE_RM    = "distributed",
    parameter     RAM_STYLE_PS    = "distributed",

    // ---- BoardControl / PhysicalMMIO misc ----
    parameter int PAYLOAD_W  = 2,      // ev_payload width

    // ---- leaf PhysicalMMIO array ----
    parameter int P           = 0,     // number of physical control cores (leaf only)
    parameter int INSTR_DEPTH = 4,
    parameter int CW_DEPTH    = 16,
    parameter int DATA_W      = 8,
    parameter int WT_W        = 8,

    // ---- derived (do not override) ----
    parameter int D_OUT  = D_IN + K,                     // detector lines out = pass ++ construct
    parameter int DIN_W  = (D_IN    < 1) ? 1 : D_IN,
    parameter int RAW_W  = (RAW_OUT < 1) ? 1 : RAW_OUT,
    parameter int NC_W   = (NCHILD  < 1) ? 1 : NCHILD,
    parameter int NPS_W  = (NPS     < 1) ? 1 : NPS,
    parameter int P_W    = (P       < 1) ? 1 : P,
    parameter int HW_W   = HW_WIDTH,
    // root post_select bus = { own DCB ps (HAS_PS) , NPS stage boards , 1 gap }
    parameter int M_PS   = (HAS_PS ? 1 : 0) + NPS + 1,
    parameter int MPS_W  = (M_PS    < 1) ? 1 : M_PS
) (
    input  logic                       clk,
    input  logic                       rst,

    // ===== event bus in (non-root: from parent) =====
    input  logic                       ev_valid,
    input  logic [1:0]                 ev_type,
    input  logic [PAYLOAD_W-1:0]       ev_payload,
    input  logic                       ev_attempt,

    // ===== host controls (root only) =====
    input  logic                       start,            // -> internal EV_START
    input  logic                       finish,           // -> internal EV_FINISH
    input  logic                       gap_post_select,  // gap-estimator reject, stamped cur_attempt

    // ===== concatenated child data (parent side) =====
    input  logic [DIN_W-1:0]           in_det,
    input  logic [DIN_W-1:0]           in_det_valid,
    input  logic [M-1:0]               in_meas,
    input  logic [M-1:0]               in_meas_valid,
    input  logic [M-1:0]               in_meas_finish,   // leaf stimulus (non-leaf: unused, broadcast used)
    // one finish / attempt per child; broadcast + OR-reduced inside
    input  logic [NC_W-1:0]            in_det_finish_child,
    input  logic [NC_W-1:0]            in_raw_finish_child,
    input  logic [NC_W-1:0]            in_child_attempt,

    // ===== stage-board post-select fast path (root only) =====
    input  logic [NPS_W-1:0]           ps_in,
    input  logic [NPS_W-1:0]           ps_in_attempt,

    // ===== forward-up to parent (non-root) =====
    output logic [D_OUT-1:0]           fwd_det,
    output logic [D_OUT-1:0]           fwd_det_valid,
    output logic                       fwd_det_finish,
    output logic [RAW_W-1:0]           fwd_raw,
    output logic [RAW_W-1:0]           fwd_raw_valid,
    output logic                       fwd_raw_finish,
    output logic                       out_attempt,

    // ===== this board's post-select fast path (stage boards) =====
    output logic                       ps_out,
    output logic                       ps_out_attempt,

    // ===== event bus out to children (non-leaf) =====
    output logic                       out_ev_valid,
    output logic [1:0]                 out_ev_type,
    output logic [PAYLOAD_W-1:0]       out_ev_payload,
    output logic                       out_ev_attempt,

    // ===== root final detector stream =====
    output logic [D_OUT-1:0]           out_det,
    output logic [D_OUT-1:0]           out_used,
    output logic                       out_valid,
    output logic                       out_finish,
    output logic [D_OUT*HW_W-1:0]      out_global_indexes,
    output logic                       first_normal,
    output logic                       last_normal,
    output logic                       first_wait,
    output logic                       last_wait,
    output logic                       discard,

    // ===== leaf outputs to lower control cores =====
    output logic [P_W-1:0]             cw_gen_finish,
    output logic [P_W*DATA_W-1:0]      mmio_out_data,
    output logic [P_W-1:0]             mmio_out_valid
);

    localparam logic [1:0] EV_START = 2'd0, EV_FINISH = 2'd2;

    // ================= prefix-sum helpers (child bus base offsets) =================
    function automatic int det_base(input int c);
        int s = 0;
        for (int i = 0; i < c; i++) s += CHILD_DW[i];
        return s;
    endfunction
    function automatic int raw_base(input int c);
        int s = 0;
        for (int i = 0; i < c; i++) s += CHILD_RAW[i];
        return s;
    endfunction

    // ================= board-local reset + stale-valid gating =================
    logic board_reset;          // BoardControl.reset (folds in rst / abort / finish)
    logic set_in_data_to_zero;

    logic [DIN_W-1:0] dcb_in_det_valid;
    logic [M-1:0]     dcb_in_meas_valid;
    assign dcb_in_det_valid  = in_det_valid  & {DIN_W{~set_in_data_to_zero}};
    assign dcb_in_meas_valid = in_meas_valid & {M{~set_in_data_to_zero}};

    // ================= single-finish broadcast to per-line =================
    // Fan each child's single det/raw finish across that child's slice of the
    // concatenated bus (matches test_dcb_stim's distributed wiring). Whole vector
    // defaults to 0 so unmapped bits (leaf: none; degenerate params) stay clean.
    logic [DIN_W-1:0] dcb_in_det_finish;
    logic [M-1:0]     dcb_in_meas_finish;
    always_comb begin
        dcb_in_det_finish = '0;               // leaf / mono: no children -> stays 0
        for (int c = 0; c < NCHILD; c++)
            for (int j = 0; j < CHILD_DW[c]; j++)
                dcb_in_det_finish[det_base(c) + j] = in_det_finish_child[c];
    end
    always_comb begin
        if (IS_LEAF) begin
            dcb_in_meas_finish = in_meas_finish;   // leaf uses the direct stimulus
        end else begin
            dcb_in_meas_finish = '0;
            for (int c = 0; c < NCHILD; c++)       // router broadcasts child raw-finish
                for (int j = 0; j < CHILD_RAW[c]; j++)
                    dcb_in_meas_finish[raw_base(c) + j] = in_raw_finish_child[c];
        end
    end

    // ================= incoming-data status to BoardControl =================
    logic bc_in_valid, bc_data_attempt;
    assign bc_in_valid     = (|in_det_valid) | (|in_meas_valid);
    assign bc_data_attempt = (NCHILD > 0) ? (|in_child_attempt) : 1'b0;

    // ================= event bus into BoardControl (= MMIO event source) =================
    logic                  bc_ev_valid, bc_ev_attempt;
    logic [1:0]            bc_ev_type;
    logic [PAYLOAD_W-1:0]  bc_ev_payload;
    generate
        if (IS_ROOT) begin : g_root_ev
            assign bc_ev_valid   = start | finish;
            assign bc_ev_type    = finish ? EV_FINISH : EV_START;
            assign bc_ev_payload = '0;
            assign bc_ev_attempt = 1'b0;     // ORIGINATE ignores this
        end else begin : g_fwd_ev
            assign bc_ev_valid   = ev_valid;
            assign bc_ev_type    = ev_type;
            assign bc_ev_payload = ev_payload;
            assign bc_ev_attempt = ev_attempt;
        end
    endgenerate

    // ================= DCB outputs consumed by the glue =================
    logic             dcb_out_finish, dcb_out_valid, dcb_fwd_det_finish, dcb_post_select;
    logic [D_OUT-1:0] dcb_fwd_det_valid;
    logic [RAW_W-1:0] dcb_fwd_raw_valid;
    logic             cur_attempt;

    // drain_done = this board's own DCB det-finish (root: out_finish; else fwd_det_finish)
    logic bc_drain_done, bc_data_valid_output;
    assign bc_drain_done        = IS_ROOT ? dcb_out_finish : dcb_fwd_det_finish;
    assign bc_data_valid_output = IS_ROOT ? dcb_out_valid
                                          : ((|dcb_fwd_det_valid) | (|dcb_fwd_raw_valid));

    // top-port copies of the glue-tapped DCB nets
    assign fwd_det_finish = dcb_fwd_det_finish;
    assign fwd_det_valid  = dcb_fwd_det_valid;
    assign fwd_raw_valid  = dcb_fwd_raw_valid;
    assign out_valid      = dcb_out_valid;
    assign out_finish     = dcb_out_finish;
    assign ps_out         = dcb_post_select;
    assign ps_out_attempt = cur_attempt;

    // ================= root post-select bus assembly =================
    // Layout: [ own DCB ps (HAS_PS) | NPS stage boards | gap ]. Own + gap slots
    // are stamped with cur_attempt so their XNOR-equality always matches (a set
    // bit is a live reject); stage slots keep their own attempt.
    localparam int OWN_BASE   = 0;
    localparam int STAGE_BASE = HAS_PS ? 1 : 0;
    localparam int GAP_IDX    = STAGE_BASE + NPS;
    logic [MPS_W-1:0] bc_ps, bc_ps_attempt;
    always_comb begin
        bc_ps         = '0;
        bc_ps_attempt = '0;
        if (HAS_PS) begin
            bc_ps[OWN_BASE]         = dcb_post_select;
            bc_ps_attempt[OWN_BASE] = cur_attempt;
        end
        for (int i = 0; i < NPS; i++) begin
            bc_ps[STAGE_BASE + i]         = ps_in[i];
            bc_ps_attempt[STAGE_BASE + i] = ps_in_attempt[i];
        end
        bc_ps[GAP_IDX]         = gap_post_select;
        bc_ps_attempt[GAP_IDX] = cur_attempt;
    end

    // ================= BoardControl =================
    BoardControl #(
        .EVENT_MODE(EVENT_MODE), .DATA_SRC(DATA_SRC), .M(MPS_W), .PAYLOAD_W(PAYLOAD_W)
    ) u_ctrl (
        .clk(clk), .rst(rst),
        .in_valid(bc_in_valid), .data_attempt(bc_data_attempt),
        .ev_valid(bc_ev_valid), .ev_type(bc_ev_type),
        .ev_payload(bc_ev_payload), .ev_attempt(bc_ev_attempt),
        .post_select(bc_ps), .post_select_attempt(bc_ps_attempt),
        .drain_done(bc_drain_done), .data_valid_output(bc_data_valid_output),
        .out_attempt(out_attempt), .cur_attempt(cur_attempt), .discard(discard),
        .out_ev_valid(out_ev_valid), .out_ev_type(out_ev_type),
        .out_ev_payload(out_ev_payload), .out_ev_attempt(out_ev_attempt),
        .reset(board_reset), .set_in_data_to_zero(set_in_data_to_zero)
    );

    // ================= DetectorConstructBlock =================
    DetectorConstructBlock #(
        .M(M), .D_IN(D_IN), .K(K), .N(N), .H(H), .RAW_OUT(RAW_OUT),
        .IDX_W(IDX_W), .HW_WIDTH(HW_WIDTH), .STRIDE(STRIDE), .SENTINEL(SENTINEL),
        .T(T), .NDT(NDT), .PC_W(PC_W), .SYNC_FIFO(SYNC_FIFO), .OUT_FIFO(OUT_FIFO),
        .IS_ROOT(IS_ROOT), .HAS_PS(HAS_PS), .HAS_OSYNC(HAS_OSYNC), .COPY_LAST(COPY_LAST),
        .MEM_DIR(MEM_DIR),
        .RAM_STYLE_SYNC(RAM_STYLE_SYNC),   .RAM_STYLE_CORE(RAM_STYLE_CORE),
        .RAM_STYLE_OSYNC(RAM_STYLE_OSYNC), .RAM_STYLE_GIDX(RAM_STYLE_GIDX),
        .RAM_STYLE_RM(RAM_STYLE_RM),       .RAM_STYLE_PS(RAM_STYLE_PS)
    ) u_dcb (
        .clk(clk), .rst(board_reset),
        .in_meas(in_meas), .in_valid(dcb_in_meas_valid), .in_meas_finish(dcb_in_meas_finish),
        .in_det(in_det), .in_det_valid(dcb_in_det_valid), .in_det_finish(dcb_in_det_finish),
        .fwd_det(fwd_det), .fwd_det_valid(dcb_fwd_det_valid), .fwd_det_finish(dcb_fwd_det_finish),
        .fwd_raw(fwd_raw), .fwd_raw_valid(dcb_fwd_raw_valid), .fwd_raw_finish(fwd_raw_finish),
        .out_det(out_det), .out_used(out_used), .out_valid(dcb_out_valid),
        .out_finish(dcb_out_finish), .out_global_indexes(out_global_indexes),
        .first_normal(first_normal), .last_normal(last_normal),
        .first_wait(first_wait), .last_wait(last_wait),
        .post_select(dcb_post_select)
    );

    // ================= leaf PhysicalMMIO array =================
    // Event source: ORIGINATE boards feed the generated out_ev (registered, carries
    // the internal ABORT); FORWARD boards feed the received event (bc_ev). TODO: load
    // each core's instr / command-word memory and close the measurement loopback.
    localparam bit MMIO_FROM_OUT_EV = (EVENT_MODE == 0);   // 0 = ORIGINATE
    logic                 mmio_ev_valid;
    logic [1:0]           mmio_ev_type;
    logic [PAYLOAD_W-1:0] mmio_ev_payload;
    assign mmio_ev_valid   = MMIO_FROM_OUT_EV ? out_ev_valid   : bc_ev_valid;
    assign mmio_ev_type    = MMIO_FROM_OUT_EV ? out_ev_type    : bc_ev_type;
    assign mmio_ev_payload = MMIO_FROM_OUT_EV ? out_ev_payload : bc_ev_payload;
    generate
        if (IS_LEAF && P > 0) begin : g_mmio
            for (genvar p = 0; p < P; p++) begin : g_core
                PhysicalMMIO #(
                    .INSTR_DEPTH(INSTR_DEPTH), .CW_DEPTH(CW_DEPTH), .DATA_W(DATA_W),
                    .WT_W(WT_W), .PAYLOAD_W(PAYLOAD_W),
                    // flat, board-prefixed, per-core mem filenames (same idiom as the DCB
                    // regfiles): "<MEM_DIR>MMIO_instr_<p>.mem" / "<MEM_DIR>MMIO_cw_<p>.mem",
                    // where MEM_DIR is the "board<i>_" prefix and p is this core's index —
                    // the same index as its out_valid / mmio_out_data line.
                    .INSTR_FILE($sformatf("%sMMIO_instr_%0d.mem", MEM_DIR, p)),
                    .CW_FILE   ($sformatf("%sMMIO_cw_%0d.mem",    MEM_DIR, p))
                    // INIT_HEX left at default 0 ($readmemb), matching the DCB flat mems
                ) u_mmio (
                    // System reset ONLY — NOT board_reset. The MMIO handles abort/finish itself
                    // via the event bus (sequencer EXEC restarts on EV_ABORT, drains on EV_FINISH)
                    // and its own rst_out. Folding abort/finish into its top reset would slam the
                    // sequencer to IDLE, where it ignores EV_ABORT, so a gap/abort would never
                    // re-issue the program and the readout would never advance to the next shot.
                    .clk(clk), .rst(rst),
                    .ev_valid(mmio_ev_valid), .ev_type(mmio_ev_type), .ev_payload(mmio_ev_payload),
                    .out_data(mmio_out_data[p*DATA_W +: DATA_W]),
                    .out_valid(mmio_out_valid[p]),
                    .cw_gen_finish(cw_gen_finish[p])
                );
            end
        end else begin : g_no_mmio
            assign cw_gen_finish  = '0;
            assign mmio_out_data  = '0;
            assign mmio_out_valid = '0;
        end
    endgenerate

endmodule
