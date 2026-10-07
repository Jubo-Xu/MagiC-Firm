# hardware/synth/

FPGA synthesis of the compiled control boards with Vivado, in three steps
that can run on different machines.

| Step | Script | Runs where | Needs |
|---|---|---|---|
| 1. Build bundles | `gen_synth.py` | anywhere | Python, a compiled tree |
| 2. Synthesize | `run_synth.sh` | wherever Vivado is | Vivado 2022.1 |
| 3. Collect | `collect_synth.py` | anywhere | Python |

## 1. Build a bundle

```bash
python hardware/synth/gen_synth.py <compiled tree dir> --out-dir <bundles>/<name> \
    [--part xcvu19p-fsva3824-2-e] [--fifo 4] [--period-ns 2.0] [--select pareto|all]
```

A bundle is self-contained: `src/` (a wrapper module per board with its
literal parameters), `mem/` (the board's register files), `tcl/` (one
Vivado batch script per board: out-of-context synthesis, placement,
routing, reports, a JSON summary) and `boards.json`.

- `--part` takes any device Vivado supports; synthesis is out of context
  with a single clock constraint (`--period-ns`), so no board file or pin
  constraints are involved.
- `--select pareto` synthesizes every router and the root but only the
  leaves that are maximal in some size parameter, which is enough for the
  per-role maxima the paper reports. `--select all` synthesizes every board.

## 2. Run Vivado

Copy `vivado.env.example` to `vivado.env` and edit it, then:

```bash
hardware/synth/run_synth.sh <bundles> [<name> ...]
```

`VIVADO_MODE=local` runs the TCL scripts on this machine, `VIVADO_JOBS` at a
time. `VIVADO_MODE=remote` copies each bundle to `VIVADO_HOST` with rsync,
runs the same loop there under `VIVADO_REMOTE_DIR`, and copies the reports
back. Boards whose `runs/boardN/summary.json` exists are skipped.

Any machine with Vivado can also run one board without the driver:

```bash
cd <bundle>/runs/boardN && vivado -mode batch -source ../../tcl/boardN.tcl
```

## 3. Collect

```bash
python hardware/synth/collect_synth.py <bundles>     # -> experiments/result/synthesis/
```

Writes `boards.csv` with one row per synthesized board (bundle, role, size
parameters, LUT, FF, BRAM, DSP, Fmax, timing status, critical path),
`results.json` with the same rows plus a per-role summary, and `table.md`
with the summary. The summary is what
`experiments/figures/table_fpga_resources.py` reads, so the FPGA table of
the paper regenerates without Vivado; the per-board rows are in the CSV.

## The paper's runs

Six trees, `d_cultiv` 3 and 5 with `d_escape` 13, 15, 17 at `p = 1e-3`,
part `xcvu19p-fsva3824-2-e`, FIFO depth 4, 2 ns clock target, Pareto
selection. Bundles and run directories live under `<output dir>/synth/`;
only the collected summary is tracked.
