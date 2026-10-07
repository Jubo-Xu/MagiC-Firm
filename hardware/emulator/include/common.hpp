// common.hpp — project-wide vocabulary and constants.
//
// Design-agnostic. Holds only the things every file needs: the emu namespace,
// generic hardware type aliases, and global simulation constants. No block data
// structures live here (those go next to the block that owns them).
#pragma once

#include <systemc>

#include <cstdint>

namespace emu {

// --- generic hardware value types (not tied to any block) -------------------
using Bit = bool;  // a single 1-bit wire value

// Sized types are pulled in on demand from sc_dt where a real width is known,
// e.g. `sc_dt::sc_bv<N>` for a mask word. We don't alias a fixed WIDTH here
// because widths are per-regfile and only known once a design is loaded.

// --- generic hardware helpers -----------------------------------------------
// Bits needed to index `num_items` lines/entries: ceil(log2 num_items), min 1.
// (Selector index width, regfile address width, etc.)
inline int index_width(int num_items) {
    int iw = 1;
    while ((1u << iw) < static_cast<unsigned>(num_items)) ++iw;
    return iw;
}

// --- global simulation constants --------------------------------------------
// One "cycle" = one tick of the system clock. The period is a modelling knob
// (latency-accurate: what matters is cycle *counts*, not the absolute ns), and
// can be overridden per run via SimConfig.
constexpr double     kDefaultClockPeriodNs = 10.0;
constexpr sc_core::sc_time_unit kTimeUnit  = sc_core::SC_NS;

}  // namespace emu
