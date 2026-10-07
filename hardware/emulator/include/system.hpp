// system.hpp — the top-level design container.
//
// This is the socket the whole micro-architecture plugs into. Right now it is an
// empty clocked module: it takes the system clock and does nothing but announce
// itself. As we build the design, the board tree (leaf/router/root modules) gets
// instantiated and wired here from the compiler regfiles — main.cpp never
// changes, only this container grows.
#pragma once

#include <systemc>

#include "log.hpp"
#include "sim_config.hpp"

namespace emu {

SC_MODULE(System) {
    sc_core::sc_in<bool> clk;

    // Heartbeat so an empty run still shows the clock advancing. Removed once
    // real modules provide their own observable activity.
    void heartbeat() {
        log_info(name(), "cycle " + std::to_string(cycles_++));
    }

    System(sc_core::sc_module_name nm, const SimConfig& cfg)
        : sc_module(nm), cfg_(cfg) {
        SC_METHOD(heartbeat);
        sensitive << clk.pos();
        dont_initialize();
        log_info(name(), "constructed (empty design container)");
    }

  private:
    SimConfig  cfg_;
    uint64_t   cycles_ = 0;
};

}  // namespace emu
