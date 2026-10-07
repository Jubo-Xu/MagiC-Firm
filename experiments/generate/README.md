# experiments/generate/

Scripts that produce data. Each is a standalone program; run it with
`--help` for its options. Outputs go to `experiments/result/` (small,
tracked) or to the output directory (large, not tracked).

## Pipeline, in order

| Script | Does | Reads | Writes |
|---|---|---|---|
| `gen_circuit_folders.py` | Generates the cultivation circuits of a parameter grid and bootstraps their folders (DEMs, gap circuit, postselected detectors) | parameters | `data/circuits/<name>/` |
| `run_eval_partial_mask.py` | Evaluates one mask recipe in memory: partial-gap accuracy and two-stage gating statistics on replayed shots | circuit folder, essentials | nothing, unless `--save` |
| `run_mask_campaign.py` | Per circuit: sweeps the mask size, picks a mask, generates its folder, characterizes both decoders on micro-blossom | circuit folders | `result/mask_campaign/summary.*`, mask folders, characterizations |
| `run_mb_characterization.py` | Per-shot decode latency of one decoder on the RTL simulator | circuit folder, toolchain container | `result/mb_characterization/*.json`, latency caches |
| `run_runtime_estimation.py` | One preparation-time estimation for one circuit and gating mode | circuit folder, characterization | `result/runtime_estimation/<name>/*.json` |
| `run_estimation_campaign.py` | Baseline and two-stage threshold sweep over all circuits with a chosen mask | mask campaign summary | `result/runtime_estimation/summary*.{json,md}` |
| `collect_gap_table.py` | Per-shot gap table: complete gap, partial gap and error for millions of post-selected shots | circuit folder | `<out>/gap_tables/` |
| `select_th_from_tables.py` | Operating point t_h* and LER-matched baseline from the gap tables | gap tables, estimator results | summary files |
| `run_tc_match.py` | Complete-only baselines at a threshold grid, matched to the two-stage LER | summary | summary files |
| `collect_threshold_sweep.py` | Preparation time and LER against t_c, and the two-stage envelope, for one circuit | estimator, gap table | `result/runtime_estimation/<name>/threshold_sweep_*.json` |
| `run_control_sensitivity.py` | Preparation time against the control-system parameters | board trees | `result/runtime_estimation/control_sensitivity.*` |

## Ablations

| Script | Does | Writes |
|---|---|---|
| `run_gap_sensitivity.py` | Per-detector gap shift and error rate when the detector fires | `result/gap_sensitivity/<name>/` |
| `run_mask_heuristics.py` | Four mask heuristics at equal size, compared by partial-gap accuracy | `result/mask_ablation/heuristics_*.json` |
| `collect_mask_ablation.py` | Gap accuracy of the masked syndrome on the complete DEM and on the contracted DEM | `result/mask_ablation/accuracy/`, `<out>/mask_ablation/` |
| `collect_pymatching_latency.py` | Software decode latency of the three decoding setups | `result/mask_ablation/pymatching/` |
| `collect_mask_size_table.py` | The mask-size table the sensitivity figure reads | `result/mask_ablation/mask_size_table.csv` |

## Conventions

- A circuit folder is `data/circuits/end2end_d1=<d1>_d2=<d2>_r1=<r1>_r2=0_p=<p>_inj=<inj>_b=Y/`.
  A mask lives in a `partial_<tag>/` subfolder of it; the paper's masks are
  `partial_qexp<K>_cl_wc15` (top-K logical-ambiguity base, closure, contraction cutoff 15).
- Decoder clock per escape distance: 77 MHz for d2 = 11, 62 MHz for 13, 43 MHz for 15 and above.
- Estimator result names record the gating mode, thresholds, latency source,
  epochs, worker count and control profile, for example
  `partial_qexp250_cl_wc15__pcp_tc35_tl0_th47__lat-mb_trace-s0_e1000000_w32_csq14f29.json`.
- Every campaign script is resumable: existing outputs are validated against
  their inputs and reused.
