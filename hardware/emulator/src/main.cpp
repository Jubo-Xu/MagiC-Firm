// main.cpp — simulation entry point.
//
// Fixed harness that does not change as the design grows: parse the run config,
// create the clock, build the (currently empty) System, run for the requested
// number of cycles, and exit with a pass/fail code derived from SystemC's error
// count. All design-specific wiring lives inside System, not here.
#include <systemc>

#include "log.hpp"
#include "sim_config.hpp"
#include "system.hpp"

int sc_main(int argc, char* argv[]) {
    using namespace sc_core;
    emu::SimConfig cfg = emu::SimConfig::from_args(argc, argv);
    cfg.log();

    sc_clock clk("clk", cfg.clock_period_ns, emu::kTimeUnit);
    emu::System system("system", cfg);
    system.clk(clk);

    if (cfg.run_cycles == 0) {
        emu::log_info("main", "running until sc_stop()");
        sc_start();
    } else {
        sc_start(cfg.run_cycles * cfg.clock_period_ns, emu::kTimeUnit);
        emu::log_info("main", "reached run_cycles limit");
    }

    const int errors = sc_report_handler::get_count(SC_ERROR)
                     + sc_report_handler::get_count(SC_FATAL);
    emu::log_info("main", errors ? "FAIL" : "PASS");
    return errors ? 1 : 0;
}
