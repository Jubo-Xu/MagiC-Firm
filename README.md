# MagiC-Firm

A runtime for fault-tolerant quantum protocols with algorithm–hardware
co-design, instantiated for magic-state cultivation: a configurable control
system that executes a protocol end to end from offline-compiled
microprograms, and a two-stage early-escape scheme in which a partial
decoder on a compact, contracted decoding problem accepts high-confidence
attempts before complete decoding finishes.

The runtime is not tied to cultivation. It provides two things any
fault-tolerant protocol needs:

- **Online streaming syndrome construction.** The compiler maps the
  detectors of any stim circuit onto a tree of control boards, each board
  builds its detectors from the measurements as they arrive and forwards
  only completed detectors and the residual measurements needed above it.
  This works for any QEC circuit, memory or otherwise.
- **Control flow for post-selective protocols.** Boards evaluate
  post-selection conditions locally, and the root turns decisions, including
  decoding-based acceptance, into global events that abort, retry or finish
  an attempt across every board with deterministic latency.

Cultivation is the first protocol we support; others are planned.

```
MagiC-Firm/
├── algorithms/                 the method: mask construction, DEM contraction, gap decoding, estimator
├── experiments/                data generation, paper figures, stored results, reproduce.py
├── hardware/                   control-system compiler, RTL, synthesis flow, SystemC model
├── magic_state_cultivation/    cultivation circuits (upstream submodule + patches)
├── micro-blossom/              hardware decoder used for latency characterization (submodule + patches)
├── arch_figures/               architecture drawings (draw.io sources and SVG)
├── explorations/               archived early studies; nothing depends on them
└── docs/                       architecture and data flow
```

Each folder has a README. [`experiments/README.md`](experiments/README.md)
explains how to reproduce the paper; [`algorithms/README.md`](algorithms/README.md)
explains the method and its tools; [`hardware/README.md`](hardware/README.md)
the control system; [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) how the
pieces fit together.

## Setup

```bash
git clone --recurse-submodules <this repository>
cd MagiC-Firm
conda env create -f environment.yml && conda activate magicfirm   # or: pip install -r requirements.txt
./setup_local.sh                 # applies the submodule patches; outputs stay in the repository
./setup_local.sh /data/$USER     # same, with large outputs and micro-blossom builds on a bigger disk
```

The second form records the chosen locations in `.magicfirm.env`, which the
scripts read; `MAGICFIRM_OUT` and the other variables in that file can also
be set in the environment. Decoder latency characterization needs Docker and
the micro-blossom toolchain image (`docker build -t micro-blossom micro-blossom/`);
everything else runs with the Python environment alone.

## Artifact evaluation

```bash
python experiments/reproduce.py figures   # every figure and table from stored results, minutes
python experiments/reproduce.py quick     # the method end to end on one small circuit, about a minute
python experiments/reproduce.py full      # the campaigns, days; --dry-run prints the commands
```

[`experiments/README.md`](experiments/README.md) explains what each level
needs and produces, which figure each script generates, and how to run
your own experiments with the same scripts.

## Tests

```bash
bash algorithms/run_tests.sh
```

## Citing MagiCFirm

If you use the MagiCFirm runtime, its compiler, or the two-stage early-escape scheme in your research, please cite:

```bibtex
@misc{xu2026magicfirm,
      title={MagiCFirm: A Runtime for Magic-State Cultivation with Algorithm-Hardware Co-Design},
      author={Jubo Xu and Abbas B. Ziad and Prakash Murali and Hongxiang Fan},
      year={2026},
      eprint={2609.29267},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2609.29267},
}
```

## Contact

For any questions or concerns, please [email me](mailto:jx1820@ic.ac.uk).

## License

Copyright 2026 Jubo Xu

Licensed under the [Apache License 2.0](LICENSE). The upstream cultivation
code (Apache 2.0) and micro-blossom (MIT) keep their own licenses; the
repository's patches to them are in `magic_state_cultivation/patches/` and
`micro-blossom-patches/`.
