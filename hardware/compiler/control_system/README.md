# Control-System Compiler

Compiles a **stim circuit** + a **hardware config** into per-board register-file
images (the bitstring masks) that a distributed tree of control boards uses to
build detectors (syndrome) from raw physical-qubit measurements in real time.
`cli.py` is the entry point; besides the detector-construction regfiles it emits
the per-board instruction and command-word programs and the simulation stimuli
used by the RTL testbenches.

Given the circuit and config it produces, for every board:

- **sync** masks (which measurement channels fire each measurement-time),
- per-kernel **selector** + **core** masks (which measurements XOR into which
  detector, and when to emit),
- **raw-measurement forwarding** selectors + inter-board **connections**,
- the root's **output** (detector-time sync + global detector index) and
  per-stage **postselect** masks,

as loadable `.json` (emulator) and `.mem` (FPGA `$readmemh/$readmemb`) files,
plus a human-readable utilization `report.txt`.

The pipeline is validated end-to-end against stim: the constructed detectors
match `stim`'s detector sampler exactly, on every board topology.

---

## Requirements

The repository's Python environment (`stim`, `numpy`). Run all commands
**from inside this directory** (`hardware/compiler/control_system/`), since
the scripts import each other and use relative paths:

```bash
cd hardware/compiler/control_system
PY=python
```

---

## Inputs

1. **A stim circuit** (`.stim`). Use a *noisy* circuit so detectors actually
   fire. Example circuits live in
   `../../../magic_state_cultivation/circuits_dump/` (the `noise=uniform` files).

2. **A compiler config** (`.json`) — the board tree + the postselect rule.
   See [`data/config.template.json`](data/config.template.json) for the format
   and field documentation. Two sections:
   - `hardware.layers` — leaf → router → root boards (circuit-agnostic),
   - `postselect` — which detectors are postselected: `"none"`, an explicit
     list of stim detector indices, or `{"from_file": "<json>"}` reading the
     list written by `algorithms/get_desaturated_dem_with_obs_detector.py`.

   Ready-made example configs are in `data/config_<circuit>_<shape>.json`
   (shapes: `mono`, `strip2`, `strip4`, `grid4`, `tree3`). To (re)generate them:
   ```bash
   $PY gen_and_test_configs.py        # writes data/config_*.json and tests each
   ```

   To build a config for a circuit from the physical constraints of a
   deployment instead of writing it by hand:
   ```bash
   $PY gen_control_config.py <circuit folder> --q 14 --fanout 29 [--out-dir DIR] [--no-compile]
   ```
   `--q` is the maximum number of qubits wired to one leaf board, `--fanout`
   the maximum number of children per router or root. Qubits are clustered by
   recursive coordinate bisection into leaves, leaves into routers, until one
   root remains; kernel counts and `raw_out` are sized from a first compile.
   It writes `config_q<q>_f<fanout>.json` and `links_q<q>_f<fanout>.json`
   (per-link message widths for `algorithms/control_latency.py`) under
   `<circuit folder>/control_system/`, and the compiled tree under the output
   directory. The circuit folder is one produced by
   `algorithms/get_desaturated_dem_with_obs_detector.py`, so the postselect
   list is picked up automatically.

---

## 1. Compile → regfiles

```bash
$PY serializer.py <circuit.stim> <config.json> \
    [--mem hex|bin] [--out results] \
    [--output-sync all|output-only] \
    [--wait-rounds W] [--wait-row normal|copy-last] \
    [--postselect-layout flat|per_stage]
```

Example:

```bash
C="../../../magic_state_cultivation/circuits_dump/c=end2end-inplace-distillation,p=0.001,noise=uniform,g=css,q=166,b=Y,r=11,r1=3,d1=3,r2=3,d2=9.stim"
$PY serializer.py "$C" data/config_d3d9_tree3.json --mem bin --wait-rounds 3 --wait-row copy-last
```

This writes `results/<config>__<circuit-key>__<flags>/` — the folder name records
the compiler flags (`osync=…,wr=…,wrow=…,ps=…`) so distinct settings don't collide:

```
results/tree3_sized__d1=3,d2=9,b=Y,p=0.001__osync=all,wr=3,wrow=copy-last,ps=flat/
  <circuit>.stim              # copy of the input
  <config>.json              # copy of the input
  measurement_map.json        # stim record -> (qubit, meas_time)  (emulator input interface)
  manifest.json               # structured index (boards, regfiles, widths, utilization, flags)
  report.txt                  # human-readable utilization + derived params + stage boards
  board<id>/
    json/                     # semantic regfiles (self-contained for the emulator)
      qubit_scope.json        # leaf only: used input-port->qubit + unused (dropped) qubits
      sync.json
      k<i>_selector.json  k<i>_core.json     # one per kernel
      raw_selector.json  raw_selector_obs.json
      output_sync.json                        # EVERY board (with --output-sync all)
      global_index.json  round_marker.json    # root only
      postselect.json                         # stage boards
      connections.json        # non-leaf: inter-board wiring (input port -> child output port)
    mem/                      # same regfiles packed as hex|bin words, one per line, LSB=bit 0
      sync.mem  k<i>_selector.mem  k<i>_core.mem  ...
```

- `--mem hex` (default) or `--mem bin` chooses the `.mem` word format.
- The `report.txt` shows, per board: `m` (measurement channels), kernels
  used/available, selector `n` used/cap, core `h` used/cap, raw-forward
  normal/+observable/cap, output lines, input ports — and the stage-board map.

### Compiler flags

| flag | default | effect |
|---|---|---|
| `--output-sync` | `all` | `all`: emit `output_sync` on **every** board (OutputSync is the output stage everywhere, so each board forwards det-time-synced detectors — one finish wire per link). `output-only`: legacy (root + stage boards only). |
| `--wait-rounds W` | `0` | The **last `W` real det-times** are marked as wait/hold rounds (`is_wait`) in the root's `round_marker`. Set `W = r2` (the circuit's number of trailing hold rounds); these are real stim detectors with real global indexes. |
| `--wait-row` | `normal` | `normal`: no appended row. `copy-last`: also append **one saturating wait row** to `sync`/`core`/`output_sync` (copied from the last round), `postselect` (zeroed — no postselect during hold), and `global_index` (fresh base `ndet..ndet+stride-1`). The hardware pc saturates here for finish arriving beyond the baked-in `W`. ⚠️ the copied row's detector *values* are placeholders (see Notes). |

### Root output extras (wait / finish support)

When compiling with wait support, the **root** gains:

- **`round_marker`** (`ndt[+1] × 3`, one word per det-time): bit0 `is_first_normal`,
  bit1 `is_last_normal`, bit2 `is_wait`. The hardware derives four output signals —
  `first_normal`/`last_normal` directly, `first_wait` (edge of `is_wait`), `last_wait`
  (`is_wait & finish`).
- **`global_index`** now uses an **all-ones sentinel** for unused slots (since `ndet` is
  a valid wait index), a compact `index_width = ceil(log2(ndet+stride+1))` (+1 if the max
  is all-ones), and reports `stride` (detectors per wait round). The manifest also records
  `global_index_hw_width` (the wider hardware datapath width, into which the regfile value
  is zero-extended and the running wait offset `+w·stride` is added).

### Port ordering & wiring

A board has three blocks — **detector-pass** (forwards input detectors),
**detector-construct** (the kernels build new detectors), **raw-measure-filter**
(forwards raw measurements). Ports use a **pass ++ construct** concatenation, so
wiring is contiguous ranges, no reordering:

- **`qubit_scope.json`** (leaf): `used[i]` = physical qubit on measurement input
  port `i`; `unused` = declared-but-never-measured qubits (connected to nothing —
  the leaf's scope filtering). So the used→port map is explicit.
- **Output detector ports** = children's detector outputs first (each child a
  contiguous block, in child order), then local kernels in kernel order. This is
  the `out_lines` order used by `output_sync` / `global_index` line indices.
- **`connections.json`** (non-leaf): `measurement_ports[i]` and
  `detector_ports[j]` give `(child, child_output_port)` for each input port. Because
  of the concatenation order these are **contiguous per-child ranges** — child C's
  outputs occupy one offset block of the parent's inputs — e.g. `child 1000 -> parent
  det ports 0..47`, `child 1001 -> 48..91`. Wire a parent input port straight to the
  named child's output port.

---

## 2. Validate (round-trip against stim)

Reconstructs detectors from the **serialized files only** (no compiler objects)
and checks they match stim's detector sampler:

```bash
$PY validate_serialized.py [<example_dir> ...]     # default: every dir under results/
```

Example:

```bash
$PY validate_serialized.py "results/tree3_sized__d1=3,d2=9,b=Y,p=0.001__osync=all,wr=3,wrow=copy-last,ps=flat"
# -> ... mismatched_detectors=0  root_output_bad_slots=0  PASS
```

`PASS` means the regfiles + `measurement_map` are self-sufficient: an emulator or
FPGA can rebuild the exact syndrome from the files alone. (Any appended `copy-last`
wait row is skipped — it has no stim ground truth; only the real rounds are checked.)

### (optional) Validate the compiler directly

`test_harness.py` runs the same check straight off the in-memory compiler (no
serialization), useful during development:

```bash
$PY test_harness.py                      # defaults to the two noisy cultivation circuits
```

---

## Pipeline modules

| file | role |
|---|---|
| `cli.py` | entry point: runs the pipeline below, then the instruction/command-word and simulation-stimulus generators. |
| `gen_control_config.py` | build a board-tree config for a circuit folder from `--q` and `--fanout`, compile it, record per-link widths. |
| `parser.py` | scan a stim circuit → measurement/detector structures; `refine()` canonicalizes + builds `channels`, `stages`, etc. Handles the terminal MPP. |
| `config.py` | load + validate the config, build the board tree, resolve the postselect predicate. |
| `placement.py` | assign each channel to a board (LCA of its qubits), compute forwarding (normal + observable) and stage boards. |
| `compiler.py` | generate all masks: sync, selector, emission-index core `(select,emit)`, detector-time sync, postselect, root output. |
| `serializer.py` | write per-board, per-regfile `.json`/`.mem` + `manifest.json` + `report.txt` + `measurement_map.json`. |
| `validate_serialized.py` | round-trip: reconstruct from serialized files, compare to stim. |
| `test_harness.py` | validate the in-memory compiler against stim. |
| `gen_and_test_configs.py` | generate a family of example configs and test them. |
| `gen_instr_cw.py` | per-board instruction regfile and command-word memory for the PhysicalMMIO block. |
| `gen_sim.py` | sample ground-truth shots and write measurement stimuli plus expected outputs beside the regfiles. |
| `timing.py` | round completion times of a circuit from the gate-time table (self-contained copy of the estimator's model). |

---

## Notes

- **Well-formedness:** the compiler requires that detectors finishing in the same
  measurement step have *distinct spatial coordinates* (so each channel emits one
  detector per step). Cultivation circuits satisfy this; a circuit that reuses a
  coordinate for its transversal-readout detectors (e.g. the sample BB code as
  annotated) is reported as infeasible with a clear message rather than
  mis-compiled.
- **Observable:** the logical observable is currently only a *reported* forwarding
  requirement (`raw_selector_obs`), not yet a functional construct-and-emit
  channel at the root.
- **Wait-round detector values (`--wait-row copy-last`):** the appended saturating
  row is a *structural placeholder* — it repeats the last round's operations, but
  because the kernel emits with emit-and-clear, the copied "detector" is a raw
  syndrome, **not a correct consecutive-round difference**. `validate_serialized`
  skips it (no stim ground truth). Correct wait detectors come from the
  **`--wait-rounds W`** path instead: generate the circuit with a large `r2` so the
  hold rounds are *real* stim rounds, and mark the last `W` det-times `is_wait` — those
  detectors are correct and stim-validatable. `copy-last` only supplies the
  saturating tail for finish arriving beyond the baked-in `W`.
