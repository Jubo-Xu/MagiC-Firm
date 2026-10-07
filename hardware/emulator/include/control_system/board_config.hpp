// board_config.hpp — configuration for the universal ControlBoard wrapper.
//
// A ControlBoard bundles the three per-board sub-systems (DetectorConstructBlock,
// BoardControl, and — on a leaf — an array of PhysicalMMIOs) and all the glue
// that wires them and connects a board to its neighbours. BoardConfig carries
// everything needed to build one: the board's role, its child topology, the
// embedded DcbConfig for the datapath, and (leaf only) one MmioConfig per core.
#pragma once

#include <vector>

#include "control_system/cultiv_control/physical_mmio.hpp"
#include "control_system/detector_construct/detector_construct_block.hpp"

namespace emu {

struct BoardConfig {
    // --- role / kind ---
    bool is_root        = false;  // final output vs forward-up
    bool is_leaf        = false;  // has PhysicalMMIOs (raw measurements originate here)
    bool has_postselect = false;  // this board runs a postselect (reject) stage

    // BoardControl behaviour axes (see board_control.hpp):
    //   event_mode 0=ORIGINATE (root/mono), 1=FORWARD (mid/leaf)
    //   data_src   0=EXTERNAL  (root/mid),  1=INTERNAL (leaf/mono)
    int event_mode = 0;
    int data_src   = 0;
    int payload_w  = 2;   // ev_payload width (shared by BoardControl and MMIOs)

    // --- child topology (parent side) ---
    // One entry per child board; the value is how many det / raw lines that child
    // forwards up. Used to broadcast each child's single finish across its slice
    // of the concatenated bus. Empty on a leaf.
    std::vector<int> child_dw;   // detector lines each child forwards (== child d_out)
    std::vector<int> child_raw;  // raw lines each child forwards

    // --- root post-select fast path ---
    int nps = 0;   // # of stage boards routing post_select straight into this root

    // --- detector datapath ---
    DcbConfig dcb;

    // --- leaf command-word generators (one per physical control core) ---
    std::vector<MmioConfig> mmios;

    // ---- derived ----
    int nchild() const { return static_cast<int>(child_dw.size()); }
    int p()      const { return static_cast<int>(mmios.size()); }
    // root post_select bus = { own DCB ps (has_postselect) , nps stage boards , 1 gap }
    int m_ps()   const { return (has_postselect ? 1 : 0) + nps + 1; }
    // bus index layout inside the m_ps() post_select bus
    int own_base()   const { return 0; }
    int stage_base() const { return has_postselect ? 1 : 0; }
    int gap_index()  const { return stage_base() + nps; }
};

}  // namespace emu
