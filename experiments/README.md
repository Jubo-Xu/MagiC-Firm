# experiments/

Everything used to produce the paper's results: the scripts that generate
data, the scripts that draw the figures and tables, the stored results they
read, and one entry point that chains them.

```
experiments/
├── reproduce.py     entry point: figures | quick | full
├── generate/        scripts that produce data (campaigns, tables, ablations)
├── figures/         one script per paper figure or table; extra/ holds diagnostics
├── tools/           maintenance scripts and the mask viewer notebook
├── lib/             code shared by the scripts (paths, naming, plotting, campaign helpers)
├── data/circuits/   inputs: one folder per circuit (circuit, DEMs, detector scores, masks)
└── result/          stored results the figures read; result/figures/ holds the outputs
```

Each subfolder has its own README describing its scripts. The method itself
lives in [`../algorithms/`](../algorithms/README.md); the scripts here only
call it.

## Reproducing the paper

Three levels, from cheapest to most complete. All need the setup in the
[root README](../README.md).

### 1. Redraw the figures from stored results

```bash
python experiments/reproduce.py figures            # a few minutes, Python only
```

Writes every figure and table to `experiments/result/figures/`. The table
below maps them to the paper.

| Paper item | Output files | Script |
|---|---|---|
| Gap-shift maps and preparation time vs t_c, with the consumption pipeline | `paper_gap_panels_*` | `figures/fig_gap_shift_and_threshold_sweep.py` |
| Mask slice before and after closure | `mask_slice_base.svg`, `mask_slice_closure.svg` | `figures/fig_mask_slices.py` |
| Gap agreement vs mask size | `mask_size.*` | `figures/fig_mask_size_sensitivity.py` |
| DEM-contraction ablation | `contraction_grid.*` | `figures/fig_dem_contraction_ablation.py` |
| Preparation time at matched LER vs d_escape and p | `paper_scaling_*` | `figures/fig_iso_ler_scaling.py` |
| Preparation time vs LER frontiers and their sensitivity | `paper_frontiers_*` | `figures/fig_pareto_frontiers.py` |
| Mask-construction strategies at matched size | `mask_heuristics_table.*` | `figures/table_mask_strategies.py` |
| Control configurations and latencies | `control_latency_table.*` | `figures/table_control_latency.py` |
| FPGA resources per board role | `fpga_resources_table.*` | `figures/table_fpga_resources.py` |

`--only NAME ...` runs a subset; names are listed by `--help`.

### 2. Run the method end to end on one small circuit

```bash
python experiments/reproduce.py quick --workers 8   # about one minute with 8 workers
```

Circuit `d_cultiv = 3, d_escape = 11, p = 1e-3`: circuit folder, detector
scores, mask with closure and contracted DEM, complete-only and two-stage
estimation, a 1M-shot gap table, and a printed summary with the
preparation-time reduction. Decoder latencies come from the stored
micro-blossom characterization of this circuit. Everything is written under
`<output dir>/quick/`; the paper's data is not touched.

### 3. Rerun the campaigns

```bash
python experiments/reproduce.py full --dry-run      # print every command
python experiments/reproduce.py full                # days; resumable
python experiments/reproduce.py full --stage masks  # one stage
```

Stages, in order: `circuits`, `essentials`, `masks`, `control`,
`estimation`, `gap_tables`, `thresholds`, `ablation`, `figures`. Every
stage skips work whose output exists. Requirements: about 70 GB under the
output directory, many cores (the paper used 32 workers), and the
micro-blossom toolchain container for the `masks` stage and the
masked-syndrome part of `ablation`. Without the container those commands
are skipped and later stages use the stored decoder characterizations.

Two things to know before running it on a checkout that already holds the
paper's results:

- The stored results are the subset the paper needed. A full run computes
  the missing sweep points and rewrites the summary files; the `thresholds`
  stage then re-derives the operating points from the gap tables.
- Estimator result files record the worker count, because the trace is
  partitioned across workers. A result produced with another worker count
  is statistically equivalent and is reused when present.

The FPGA synthesis is a separate flow that needs Vivado; see
[`../hardware/synth/`](../hardware/synth/README.md). Its stored summary is
what the FPGA table reads.

## Running your own experiment

The scripts under `generate/` are independent programs with their own
options (`--help`). The chain for one circuit:

```bash
F=experiments/data/circuits/<name>
python experiments/generate/gen_circuit_folders.py --d1 3 --d2 13 --p 0.001
python algorithms/gen_essentials.py --circuit $F/<name>.stim --out $F --type logical_ambiguity --shots 2000000
python algorithms/gen_partial_mask_dem.py --out $F --base-size 200 --q-type exp --closure --contract
python experiments/generate/run_eval_partial_mask.py $F --base-size 300 --closure --shots 200000   # try a mask in memory
python experiments/generate/run_mb_characterization.py $F --partial partial_<tag> --shots 1000 --frequency 62e6
python experiments/generate/run_runtime_estimation.py $F --partial partial_<tag> --mode pcp --tc 35 --tl 0 --th 45 \
    --epochs 1000000 --latency mb --trace
python experiments/generate/collect_gap_table.py $F --partial partial_<tag> --shots 20000000
```

The campaign scripts (`run_mask_campaign.py`, `run_estimation_campaign.py`,
`run_control_sensitivity.py`) take `--only` filters and parameter options,
so a sweep can be restricted or changed without editing them.

## Where data lives

| What | Where | In git |
|---|---|---|
| Circuit folders: circuit, DEMs, detector scores, masks, control trees | `data/circuits/` | yes |
| Distilled results: summaries, characterizations, accuracy tables, figures | `result/` | yes |
| Shot traces, gap tables, decoder runs, latency caches, quick workspaces | `<output dir>/` | no |

The output directory defaults to `out/` in the repository and is set by
`setup_local.sh` or the `MAGICFIRM_OUT` environment variable (see
[`lib/README.md`](lib/README.md)).
