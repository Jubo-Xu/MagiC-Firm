// sim_config.hpp — run-time simulation knobs, parsed from the command line.
//
// Design-agnostic. These are *simulation* parameters (how long to run, clock
// period, seed, tracing), NOT the hardware design (that comes from the compiler
// regfiles, loaded later). Keeping them here means nothing is hardcoded in main.
#pragma once

#include <cstdint>
#include <string>

#include "common.hpp"
#include "log.hpp"

namespace emu {

struct SimConfig {
    double      clock_period_ns = kDefaultClockPeriodNs;
    uint64_t    run_cycles      = 20;         // 0 = run until sc_stop()
    uint64_t    seed            = 1;
    bool        trace           = false;      // emit a VCD waveform (added later)
    std::string trace_file      = "waves.vcd";
    std::string design_dir;                   // compiler results/<example> dir (loaded later)

    // Minimal manual CLI parser: --cycles N  --period NS  --seed S
    //                            --trace[=FILE]  --design DIR
    static SimConfig from_args(int argc, char** argv) {
        SimConfig c;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            auto next = [&](const char* name) -> std::string {
                if (i + 1 >= argc) { log_error("cli", std::string(name) + " needs a value"); return ""; }
                return argv[++i];
            };
            if (a == "--cycles")      c.run_cycles     = std::stoull(next("--cycles"));
            else if (a == "--period") c.clock_period_ns = std::stod(next("--period"));
            else if (a == "--seed")   c.seed           = std::stoull(next("--seed"));
            else if (a == "--design") c.design_dir     = next("--design");
            else if (a == "--trace")  c.trace          = true;
            else if (a.rfind("--trace=", 0) == 0) { c.trace = true; c.trace_file = a.substr(8); }
            else log_warn("cli", "ignoring unknown arg: " + a);
        }
        return c;
    }

    void log() const {
        log_info("sim", "clock_period=" + std::to_string(clock_period_ns) + "ns"
                        + " run_cycles=" + std::to_string(run_cycles)
                        + " seed=" + std::to_string(seed)
                        + " trace=" + (trace ? "on" : "off")
                        + (design_dir.empty() ? "" : " design=" + design_dir));
    }
};

}  // namespace emu
