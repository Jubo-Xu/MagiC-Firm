# experiments/lib/

Code shared by the experiment scripts. Scripts import these modules; no
script imports another script.

| Module | Provides |
|---|---|
| `paths.py` | Repository locations and the import setup every script starts with. The output directory, the micro-blossom checkout and the toolchain container name resolve from `MAGICFIRM_OUT`, `MAGICFIRM_MB_ROOT`, `MAGICFIRM_MB_CONTAINER` in the environment, then from `.magicfirm.env` in the repository root (written by `setup_local.sh`), then from in-repo defaults. `python experiments/lib/paths.py` prints the resolved values. |
| `naming.py` | Circuit-name parsing, short labels, the clock per escape distance, the `<f>MHz` tag of characterization files. |
| `circuit_folder.py` | Loads a circuit folder and its mask, compiles the complete and partial decoders, and checks they match the stored DEMs. |
| `campaign.py` | Shared by the campaign drivers: running one estimator job, validating a stored result against its inputs, result naming, summaries, the essentials lookup. |
| `estimator_run.py` | Builds a `RuntimeEstimator` from a circuit folder and the common options; the shard function for parallel runs. |
| `gap_table.py` | Sampling of per-shot gap tables, gating-rule statistics, and the input fingerprints that guard reuse. |
| `mb_toolchain.py` | Runs the micro-blossom toolchain inside its container: builds, RTL simulator launches, worker checkouts for parallel campaigns. |
| `mb_latency.py` | Reads micro-blossom characterization files and hands the estimator plain latency values. |
| `plot_style.py` | Colours and matplotlib settings of the figures. |
| `sweep.py` | Readers of threshold sweeps and campaign summaries: effective LER, Pareto frontier, iso-LER view, the stacked time bars. |
| `contraction_data.py` | Loads the stored results of the DEM-contraction ablation. |

## Script header

Every script locates this folder and sets up imports with the same three lines:

```python
sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()
```

`setup_imports` puts `algorithms/`, the cultivation sources and this folder
on the path, so scripts import by module name from any folder depth.
