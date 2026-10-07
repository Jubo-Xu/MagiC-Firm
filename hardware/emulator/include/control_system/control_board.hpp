// control_board.hpp — the universal per-board wrapper.
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
// GLUE ABSORBED HERE (mirrors what test_dcb_stim wired by hand, plus control):
//   * child concatenation  : in_det / in_meas arrive already concatenated (trivial
//                            {} wiring at the port map); we consume them directly.
//   * single-finish broadcast : each child sends ONE det/raw finish; we fan it out
//                            across that child's slice using child_dw / child_raw.
//   * attempt OR-reduce    : data_attempt = OR of children's attempt tags.
//   * stale-valid gating   : DCB input valids are masked by set_in_data_to_zero.
//   * event transform      : root turns host start/finish into an event; non-root
//                            relays the parent's event bus.
//   * post-select assembly : root's post_select bus = own DCB ps + stage fast-path
//                            + gap, each stamped with the right attempt.
//   * drain / output status: drain_done = this board's DCB det-finish; a board is
//                            done when its whole subtree has drained.
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
#pragma once

#include <systemc>

#include <cstdint>
#include <memory>
#include <vector>

#include "control_system/board_config.hpp"
#include "control_system/cultiv_control/board_control.hpp"
#include "control_system/cultiv_control/physical_mmio.hpp"
#include "control_system/detector_construct/detector_construct_block.hpp"
#include "signals.hpp"

namespace emu {

SC_MODULE(ControlBoard) {
    // --- clock / async reset ---
    sc_core::sc_in<bool> clk;
    sc_core::sc_in<bool> rst;

    // --- event bus in (non-root: from parent) ---
    sc_core::sc_in<bool>     ev_valid;
    sc_core::sc_in<uint32_t> ev_type;
    sc_core::sc_in<Bits>     ev_payload;
    sc_core::sc_in<bool>     ev_attempt;

    // --- host controls (root only) ---
    sc_core::sc_in<bool> start;            // -> internal EV_START
    sc_core::sc_in<bool> finish;           // -> internal EV_FINISH
    sc_core::sc_in<bool> gap_post_select;  // gap-estimator reject, stamped cur_attempt

    // --- concatenated child data (parent side) ---
    sc_core::sc_in<Bits> in_det;          // [d_in]
    sc_core::sc_in<Bits> in_det_valid;    // [d_in]
    sc_core::sc_in<Bits> in_meas;         // [m]
    sc_core::sc_in<Bits> in_meas_valid;   // [m]
    sc_core::sc_in<Bits> in_meas_finish;  // [m] leaf stimulus (non-leaf: unused, broadcast used instead)

    // --- one finish / attempt per child (broadcast + OR-reduced inside) ---
    sc_core::sc_in<Bits> in_det_finish_child;  // [nchild]
    sc_core::sc_in<Bits> in_raw_finish_child;  // [nchild]
    sc_core::sc_in<Bits> in_child_attempt;     // [nchild]

    // --- stage-board post-select fast path (root only) ---
    sc_core::sc_in<Bits> ps_in;           // [nps]
    sc_core::sc_in<Bits> ps_in_attempt;   // [nps]

    // --- forward-up to parent (non-root) ---
    sc_core::sc_out<Bits> fwd_det;         // [d_out]
    sc_core::sc_out<Bits> fwd_det_valid;   // [d_out]
    sc_core::sc_out<bool> fwd_det_finish;
    sc_core::sc_out<Bits> fwd_raw;         // [raw_out]
    sc_core::sc_out<Bits> fwd_raw_valid;   // [raw_out]
    sc_core::sc_out<bool> fwd_raw_finish;
    sc_core::sc_out<bool> out_attempt;

    // --- this board's post-select fast path (stage boards) ---
    sc_core::sc_out<bool> ps_out;
    sc_core::sc_out<bool> ps_out_attempt;

    // --- event bus out to children (non-leaf) ---
    sc_core::sc_out<bool>     out_ev_valid;
    sc_core::sc_out<uint32_t> out_ev_type;
    sc_core::sc_out<Bits>     out_ev_payload;
    sc_core::sc_out<bool>     out_ev_attempt;

    // --- root final detector stream ---
    sc_core::sc_out<Bits> out_det;             // [d_out]
    sc_core::sc_out<Bits> out_used;            // [d_out]
    sc_core::sc_out<bool> out_valid;
    sc_core::sc_out<bool> out_finish;
    sc_core::sc_out<Bits> out_global_indexes;  // [d_out*hw_width]
    sc_core::sc_out<bool> first_normal, last_normal, first_wait, last_wait;
    sc_core::sc_out<bool> discard;

    // --- leaf outputs to lower control cores (one per core) ---
    sc_core::sc_vector<sc_core::sc_out<Bits>> mmio_out_data;    // [P] each [data_w]
    sc_core::sc_vector<sc_core::sc_out<bool>> mmio_out_valid;   // [P]
    sc_core::sc_vector<sc_core::sc_out<bool>> cw_gen_finish;     // [P]

    ControlBoard(sc_core::sc_module_name nm, const BoardConfig& cfg);

  private:
    // glue processes
    void drive_gating_broadcast();  // DCB input valids (gated) + finish broadcast
    void drive_bc_status();         // in_valid + data_attempt into BoardControl
    void drive_ev();                // event bus into BoardControl (root transform / relay)
    void drive_ps_bus();            // root post_select + post_select_attempt assembly
    void drive_drain_status();      // drain_done + data_valid_output into BoardControl
    void drive_board_outputs();     // fan the sub-block outputs to the top ports

    BoardConfig cfg_;

    std::unique_ptr<DetectorConstructBlock>    dcb_;
    std::unique_ptr<BoardControl>              ctrl_;
    std::vector<std::unique_ptr<PhysicalMMIO>> mmios_;

    // --- internal nets ---
    sc_core::sc_signal<bool> board_reset_;   // BoardControl.reset -> DCB / MMIO rst
    sc_core::sc_signal<bool> set_zero_;       // BoardControl.set_in_data_to_zero
    sc_core::sc_signal<bool> cur_attempt_;    // BoardControl.cur_attempt (ps stamp + ps_out_attempt)

    // DCB gated inputs
    sc_core::sc_signal<Bits> gin_det_valid_, gin_meas_valid_, gin_det_finish_, gin_meas_finish_;
    // DCB outputs consumed by the glue
    sc_core::sc_signal<bool> dcb_out_finish_, dcb_out_valid_, dcb_fwd_det_finish_, dcb_post_select_;
    sc_core::sc_signal<Bits> dcb_fwd_det_valid_, dcb_fwd_raw_valid_;

    // BoardControl inputs
    sc_core::sc_signal<bool>     bc_in_valid_, bc_data_attempt_;
    // event entering BoardControl — also the FORWARD-board MMIO event source
    sc_core::sc_signal<bool>     bc_ev_valid_, bc_ev_attempt_;
    sc_core::sc_signal<uint32_t> bc_ev_type_;
    sc_core::sc_signal<Bits>     bc_ev_payload_;
    // BoardControl generated event bus (out_ev): to children, and the ORIGINATE MMIO source
    sc_core::sc_signal<bool>     oev_valid_, oev_attempt_;
    sc_core::sc_signal<uint32_t> oev_type_;
    sc_core::sc_signal<Bits>     oev_payload_;
    sc_core::sc_signal<Bits>     bc_ps_, bc_ps_attempt_;
    sc_core::sc_signal<bool>     bc_drain_done_, bc_dvo_;
};

}  // namespace emu
