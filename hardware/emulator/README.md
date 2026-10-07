# Control-System Emulator (SystemC)

A latency-accurate SystemC model of the control-system micro-architecture — the
hardware that the [detector-construction compiler](../compiler/control_system/)
targets. It is **config-driven**: it will load the compiler's per-board regfiles
and replay sampled measurements, checked against stim.

Modelling style: clocked behavioral (`SC_MODULE` + `SC_METHOD`/`SC_CTHREAD` on a
clock) with `sc_fifo` channels between blocks. Latency-accurate (cycle counts and
stalls), not full RTL.

## Status

**M0 — platform skeleton** (done): generic, design-agnostic foundation that
builds, runs a clock, logs, and exits pass/fail. No hardware modelled yet.

Roadmap: M1 config/regfile loader · M2 single-leaf detector construction vs stim ·
M3 board tree + pass/raw-filter · M4 output path (sync/postselect/global index) ·
M5 latency/throughput instrumentation.

## Build & run

Requires the locally-built SystemC 3.0.1 (`~/opt/systemc-3.0.1`). One-time per shell:

```bash
cd hardware/emulator
source env.sh                 # sets SystemCLanguage_DIR + LD_LIBRARY_PATH
cmake -B build -S . && cmake --build build -j
./build/emu --cycles 5
```

CLI knobs (simulation only — the design comes from regfiles later):
`--cycles N` · `--period NS` · `--seed S` · `--trace[=FILE]` · `--design DIR`.

## Layout

```
include/emu/   common.hpp  log.hpp  sim_config.hpp  system.hpp   # platform layer
src/           main.cpp                                          # sc_main harness
third_party/   nlohmann/json.hpp                                 # vendored
tests/         (unit tests added as blocks land)
env.sh         SystemC paths
```
