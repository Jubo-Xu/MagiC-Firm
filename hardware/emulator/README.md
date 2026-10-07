# hardware/emulator/

SystemC model of the control-system micro-architecture. It is the reference
implementation of every block: each SystemVerilog module under `../rtl/src/`
is a 1:1 port of the SystemC module of the same name, and the RTL testbench
generators replay the same compiled examples and expect the same counts.

Modelling style: clocked behavioral (`SC_MODULE` with `SC_METHOD` on a clock),
cycle-accurate for latency and stalls, not RTL. Register files are
`RegFileROM`/`SyncROM` instances loaded from the compiler's `.mem` files;
buses use the runtime-width `Bits` type in `signals.hpp`.

```
include/, src/
  lib/                        RegFileROM, SyncROM, BankedRAM, .mem parsing
  control_system/
    detector_construct/       MeasurementSync, Kernel, RawSelector, DetectorPass,
                              OutputSync, RootOutputSync, Postselect,
                              DetectorConstructBlock (one module for every board kind)
    cultiv_control/           InstrSequencer, InstrUnpack, InstrDecode, PhysicalMMIO,
                              BoardControl, DrainAggregator
    control_board.*           the universal per-board wrapper (datapath + control + MMIO)
    stim_readout.*            per-leaf measurement replay closing the mmio -> readout loop
    *_loader.hpp              build a board's configuration from a compiled directory
tests/                        one unit test per block, plus the stim-driven harnesses below
third_party/nlohmann/         vendored JSON
```

## Build

Requires SystemC 3.0.1, CMake 3.20 and a C++17 compiler. `env.sh` points
CMake at a SystemC built under `~/opt/systemc-3.0.1`; edit it for another
location.

```bash
cd hardware/emulator
source env.sh
cmake -B build -S . && cmake --build build -j
ctest --test-dir build          # 17 unit tests, no inputs needed
```

## Running against a compiled example

Compile an example first (`../compiler/control_system/README.md`, with
`--instr-reg sim --cw-mem sim --sim dcb readout` so the stimuli exist), then
from the repository root:

```bash
E=hardware/compiler/control_system/results/<example>
hardware/emulator/build/test_control_board_loader $E [board_id]   # loaders only
hardware/emulator/build/test_dcb_stim $E [--shots N --input-gap G --shot-gap G]
hardware/emulator/build/test_control_board_stim $E [root_board_id]
```

`test_dcb_stim` instantiates one `DetectorConstructBlock` per board, wires
the tree from `connections.json`, drives the leaves with the sampled
measurements and checks the root's detectors and every stage board's
post-select decision against stim. `test_control_board_stim` does the same
with full `ControlBoard`s and a `StimReadout` per leaf, so the control plane
is exercised too: START from the host, events down, data and attempt tags
up, aborts on post-select rejects, finish and drain. Both pass for the
monolithic and the distributed examples with and without wait rounds.

`build/emu` is a bare clock harness (`--cycles`, `--period`, `--seed`,
`--trace`) with no design in it; the harnesses above are the entry points.
