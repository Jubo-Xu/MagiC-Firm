# MagiC-Firm architecture

How the repository's parts fit together and how data flows from a circuit
to a figure. The folder READMEs describe each part's scripts.

## Principle

- `algorithms/` holds the method as libraries and command-line tools. It
  contains no experiment-specific logic, no fixed output location, and
  reads nothing produced by the experiment scripts.
- `experiments/` consumes `algorithms/`: its scripts generate data, draw the
  figures, and chain together in `reproduce.py`. Shared code sits in
  `experiments/lib/`; scripts never import each other.
- `hardware/` is the control system: compiler, RTL, synthesis, SystemC model.
  It connects to the experiments through the compiled board trees, from which
  `algorithms/control_latency.py` derives the control-latency profiles.
- Dependencies point one way: experiments → algorithms, experiments →
  hardware outputs. `explorations/` is an archive nothing imports.
- Large generated data lives in an output directory outside git; the
  repository carries inputs and distilled results.

## Data tiers

| Tier | Location | Properties |
|---|---|---|
| Inputs per circuit | `experiments/data/circuits/<name>/` | generated once; the anchor everything keys off |
| Distilled results | `experiments/result/` | small files the figures read; tracked |
| Large derived data | `<output dir>/` (`MAGICFIRM_OUT`, default `out/`) | traces, gap tables, decoder runs, synthesis bundles; regenerable |

A circuit folder holds the noisy circuit, its DEM, the gap DEM (matchable,
clipped, with the observable as a detector), the gap circuit aligned with
it, the postselected detector set, the per-detector scores, one
`partial_<tag>/` subfolder per mask with its contracted DEM, and the
compiled control tree under `control_system/`.

**Join key.** A shot is identified forever by (circuit, master seed, attempt
index): `algorithms/attempt_stream.py` generates attempt i as a pure
function of those, in append-only trace files. Every per-shot measurement
is a table over that index, so decoder latency measured on the RTL
simulator and shots replayed by the estimator refer to the same physical
shot, verified by a per-shot hash.

## Data flow

```
circuit parameters
   │ gen_circuit_folders ── get_desaturated_dem_with_obs_detector
   ▼
circuit folder ── gen_essentials ──▶ logical-ambiguity scores Q(d)
   │                                        │
   │ gen_partial_mask_dem (top-K_Q, closure, contraction)
   ▼
partial_<tag>/ (mask, contracted DEM)
   │
   ├─ run_mb_characterization ──▶ per-shot decoder latency on micro-blossom (result/mb_characterization)
   ├─ gen_control_config + control_latency ──▶ control-latency profile (control_system/latency_*.json)
   ├─ collect_gap_table ──▶ per-shot (g_c, g_p, error) tables (<out>/gap_tables)
   │
   ▼
run_runtime_estimation (trace replay + MB latencies + control profile)
   │   run_estimation_campaign / select_th_from_tables / run_tc_match
   ▼
result/runtime_estimation/summary*.json ──▶ figures/  ──▶ result/figures/
```

The ablations (`run_gap_sensitivity`, `run_mask_heuristics`,
`collect_mask_ablation`, `collect_pymatching_latency`) branch off the circuit
folder and feed their own figures. The hardware flow (compile, RTL
testbench, synthesis) feeds the control-latency and FPGA tables.

## Evaluation backends

| Quantity | Source |
|---|---|
| Preparation time, accept rate | `runtime_estimator.py`: attempt-by-attempt simulation with gate times, control latencies and decoder latencies |
| Decoder latency | micro-blossom RTL simulation per shot (`run_mb_characterization`), used as means (Mode A) or per shot (Mode B) |
| Control latency | link model + per-board cycle counts measured on the RTL (`control_latency.py`) |
| Logical error rate | large per-shot gap tables (`collect_gap_table`), evaluated on a held-out half |
| FPGA resources | Vivado post-route reports (`hardware/synth/`) |

The SystemC emulator under `hardware/emulator/` is the reference model the
RTL was ported from; the evaluation uses the RTL, not the emulator.

## Configuration

`experiments/lib/paths.py` resolves the output directory, the micro-blossom
checkout and the toolchain container name from environment variables
(`MAGICFIRM_OUT`, `MAGICFIRM_MB_ROOT`, `MAGICFIRM_MB_CONTAINER`), then from
`.magicfirm.env` written by `setup_local.sh`, then from in-repo defaults.
The Vivado host is configured separately in `hardware/synth/vivado.env`.
