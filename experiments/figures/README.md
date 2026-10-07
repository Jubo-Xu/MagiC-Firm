# experiments/figures/

One script per paper figure or table. Each reads stored results under
`experiments/result/` and writes PDF, PNG and CSV (or Markdown) files.
`python experiments/reproduce.py figures` runs all of them with the paper's
arguments into `experiments/result/figures/`; each can also be run alone.

| Script | Shows | Main inputs |
|---|---|---|
| `fig_gap_shift_and_threshold_sweep.py` | Per-detector gap-shift maps of three code slices, preparation time against t_c, and the consumption pipeline drawing | gap sensitivity JSON, threshold sweep JSON, `assets/pipeline.svg` |
| `fig_mask_slices.py` | The logical-ambiguity base mask and its closure on one code slice | the circuit's logical-ambiguity scores |
| `fig_mask_size_sensitivity.py` | Gap agreement against the retained-detector fraction | `result/mask_ablation/mask_size_table.csv` |
| `fig_dem_contraction_ablation.py` | Decode latency and gap agreement for complete, masked and contracted decoding | characterizations, pymatching latencies, accuracy JSONs |
| `fig_iso_ler_scaling.py` | Preparation time at matched LER against d_escape and p | `result/runtime_estimation/summary_csq14f29.json` |
| `fig_pareto_frontiers.py` | Preparation-time against LER frontiers, and their sensitivity to mask size and p | threshold sweep JSONs |
| `table_mask_strategies.py` | Geometric, LAP and LAP + closure masks at matched size | `result/mask_ablation/heuristics_*_N336.json` |
| `table_control_latency.py` | Compiled control trees and worst-case latencies | the circuits' link files |
| `table_fpga_resources.py` | Resources and Fmax per board role | `result/synthesis/results.json` |

Options such as `--size`, `--summary` or `--out` are listed by `--help`.

## extra/

Diagnostic plots that are not in the paper but remain useful when working
on one circuit:

| Script | Shows |
|---|---|
| `plot_lifetime.py` | Survival per stage and accumulated time of one cultivation; runs the estimator itself |
| `plot_threshold_sweep.py` | Preparation-time breakdown against t_c for one circuit |
| `plot_ler_latency.py` | LER against preparation time, complete versus two-stage, for one circuit |
| `plot_mask_heuristics.py` | Bar chart of the mask-strategy comparison |

## assets/

`pipeline.svg`: the consumption-pipeline drawing (draw.io export) embedded in
the gap-shift figure.
