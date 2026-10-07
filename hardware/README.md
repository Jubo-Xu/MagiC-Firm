# hardware/

The control-system runtime: the offline compiler that turns a circuit and a
board configuration into per-board microprograms, the RTL that executes
them, the synthesis flow, and a SystemC model of the micro-architecture.

```
hardware/
├── compiler/control_system/   stim circuit + board config -> per-board register files; validated against stim
├── rtl/                       SystemVerilog sources, testbench generators, Verilator harness
├── synth/                     Vivado synthesis bundles, a local/remote driver, result collection
└── emulator/                  SystemC reference model of every block (the RTL is its 1:1 port), unit and system tests
```

The boards are designed to be hosted by an existing distributed FPGA
control system rather than to replace it. A leaf board needs only the raw
measurement bits of its qubits and the command words it emits; the boards
above it exchange compact detector and event messages. Each board is one
compiled module with literal parameters, so it can be placed on the FPGA
that already drives those qubits and connected over the system's existing
links.

The paper's timing results come from the Python estimator in `algorithms/`
fed with latency constants measured on the RTL and with the decoder
latencies characterized on micro-blossom. The emulator is not part of that
pipeline.

## Workflow

Every step is a standalone program; `--help` lists its options.

**1. Compile a circuit for a board configuration.** Produces one folder per
board with its register-file images, then checks the constructed detectors
against stim's detector sampler. Details, config format and flags in
[`compiler/control_system/README.md`](compiler/control_system/README.md).

```bash
cd hardware/compiler/control_system
python cli.py <circuit.stim> <config.json> [--wait-rounds 3 --wait-row copy-last] [--instr-reg sim --cw-mem sim --sim dcb readout]
python validate_serialized.py results/<example>        # round trip against stim: PASS
```

For an experiment circuit, `gen_control_config.py <circuit folder> --q 14
--fanout 29` builds the board tree with at most 14 qubits per leaf and 29
children per router, compiles it, and records the per-link message widths
that `algorithms/control_latency.py` turns into a latency profile.

**2. Simulate the RTL.** `gen_cb_tb.py` generates a self-checking testbench
for a whole tree of ControlBoards from a compiled example, with the
register files as `.mem` images; `gen_dcb_tb.py` does the same for the bare
detector-construction block. `vsim.sh` compiles and runs a testbench with
Verilator in a temporary build directory and reports PASS or FAIL.

```bash
python hardware/rtl/scripts/gen_cb_tb.py hardware/compiler/control_system/results/<example> --shots 1000
hardware/rtl/scripts/vsim.sh hardware/rtl/testbench/control_system/generated/<example>/tb_cb.sv \
    hardware/rtl/src/lib/*.sv hardware/rtl/src/control_system/**/*.sv
```

The generator prints the equivalent raw `verilator` command for people who
prefer it. `--latency-trace` makes the testbench print the per-board cycle
counts used by the control-latency model.

**3. Synthesize.** `synth/gen_synth.py` builds a self-contained Vivado bundle
per compiled tree, `synth/run_synth.sh` runs it locally or on a remote
Vivado host, and `synth/collect_synth.py` gathers the reports into
`experiments/result/synthesis/`. See [`synth/README.md`](synth/README.md).

**4. Emulator.** Builds with CMake against SystemC 3.0.1; see
[`emulator/README.md`](emulator/README.md).

## Requirements

| Step | Needs |
|---|---|
| Compiler, testbench generation | the Python environment of the repository (stim, numpy) |
| RTL simulation | Verilator 5 with `--timing` support |
| Synthesis | Vivado 2022.1, on this machine or reachable by ssh |
| Emulator | SystemC 3.0.1, CMake, a C++17 compiler |

Generated outputs (`compiler/control_system/results/`, Verilator build
directories, waveform dumps) are not tracked; the testbench generators and
the compiler recreate them.
