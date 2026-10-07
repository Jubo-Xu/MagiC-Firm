# algorithms/

The method: partial-mask construction, gap decoding on a contracted DEM,
and the preparation-time estimator. Nothing here depends on an experiment:
every input and output location is an argument, and no module reads files
produced by the scripts under `experiments/`, which are the consumers of
this folder.

## Concepts

- **Gap decoding.** A cultivation attempt is accepted when the complementary
  gap of the escape-stage syndrome exceeds a threshold. The gap is the weight
  difference between the two minimum-weight explanations of the syndrome
  under opposite logical hypotheses. `partial_desaturation_sampler.py`
  implements it on the gap DEM, a matchable DEM with the observable added as
  a detector.
- **Partial mask.** A subset of detectors on which a cheaper decoder runs.
  The base mask is the top K_Q detectors by logical-ambiguity score Q(d), the
  ambiguity-weighted frequency with which detector d lies on the symmetric
  difference of the two explanations. Structural closure then adds hidden
  detectors whose inclusion lowers the crossing mass S(M), the probability
  mass of error components straddling the mask boundary.
  `partial_mask_builder.py` and `dem_structure.py`.
- **DEM contraction.** The gap DEM is contracted onto the mask: hidden
  vertices are eliminated by shortest paths, synthetic edges above a weight
  cutoff are dropped, and boundary reachability is repaired, so the partial
  decoder works on a much smaller graph. `partial_desaturation_sampler.py`.
- **Two-stage early escape.** The partial gap accepts high-confidence
  attempts early; all others wait for the complete gap. `runtime_estimator.py`
  simulates the whole protocol attempt by attempt with gate, control and
  decoder latencies and reports preparation time and logical error rate.

## Modules

| Module | Purpose |
|---|---|
| `partial_mask_builder.py` | Per-detector scores (canonical, causal, blending, meangap, logical ambiguity) and mask operations: region, top-K, augment, prune, closure. |
| `partial_desaturation_sampler.py` | Complementary-gap decoding; with a mask, the DEM contraction and decoding on the contracted graph. |
| `get_desaturated_dem_with_obs_detector.py` | Circuit to gap DEM, gap circuit and postselected detector set. Library function and CLI. |
| `gen_essentials.py` | CLI: computes the per-detector scores of a circuit, in parallel. |
| `gen_partial_mask_dem.py` | CLI: builds a mask from a recipe and writes it with its contracted DEM. |
| `dem_structure.py` | DEM components, crossing mass S(M), closure. |
| `dem_stats.py` | DEM size and degree statistics. |
| `runtime_estimator.py` | Preparation-time estimator: gating modes, stage timing, decoder and control latency. |
| `gate_times.py` | Gate durations used for timing. |
| `layer_schedule.py` | Round completion times and per-layer syndrome arrival offsets. |
| `control_latency.py` | Control-system feedback and delivery latency from a compiled board tree. Library function and CLI. |
| `attempt_stream.py` | Deterministic, extendable shot traces keyed by (circuit, seed, index). |
| `mb_graph.py` | DEM to micro-blossom graph format, and defect lists. |

## Requirements

- The packages in `../environment.yml` or `../requirements.txt`.
- The `magic_state_cultivation/upstream` submodule with the repository's
  patches applied (`setup_local.sh` does this). Modules add its `src/` to
  `sys.path` themselves.
- Modules import each other by bare name, so library users need
  `PYTHONPATH=algorithms`. The command-line tools set this up themselves.

## Command-line workflow

Each tool writes into the folder given by `--out`. `--help` lists all options.

```bash
# 1. gap DEM, gap circuit, postselected detectors
python algorithms/get_desaturated_dem_with_obs_detector.py \
    --circuit path/to/circuit.stim --out path/to/circuit_folder

# 2. per-detector scores
python algorithms/gen_essentials.py \
    --circuit path/to/circuit.stim --out path/to/circuit_folder \
    --type logical_ambiguity --shots 2000000 --workers 8

# 3. mask and contracted DEM, written to circuit_folder/partial_<tag>/
python algorithms/gen_partial_mask_dem.py --out path/to/circuit_folder \
    --base-size 200 --q-type exp --closure --contract --weight-cutoff 15.0
```

## Library use

```python
import numpy as np, sinter, stim
from partial_desaturation_sampler import PartialDesaturationSampler
from runtime_estimator import RuntimeEstimator

circuit = stim.Circuit.from_file("circuit_folder/circuit.stim")
mask = np.load("circuit_folder/partial_<tag>/mask_bool.npy")
task = sinter.Task(circuit=circuit, detector_error_model=circuit.detector_error_model())
sampler = PartialDesaturationSampler()
complete = sampler.compiled_sampler_for_task(task)
partial = sampler.compiled_sampler_for_task(task, partial_mask_bool=mask, weight_cutoff=15.0)

est = RuntimeEstimator(circuit=circuit, complete_compiled=complete,
                       partial_compiled=partial, epoch=10_000)
est.estimate_runtime(gap_check=True, gap_threshold=35,
                     gap_threshold_low=0, gap_threshold_high=45)
est.print_statistics()
```

Decoder latency defaults to the measured pymatching wall-clock time. Supply
your own with `set_decoder_latency_constants` (means) or
`set_decoder_latency_table` (per-shot values joined by attempt index); the
experiments do this with micro-blossom measurements.

## Tests

```bash
CONDA_ENV=<env> bash algorithms/run_tests.sh   # pytest over test_python/; default env magicfirm, conda on the PATH
```
