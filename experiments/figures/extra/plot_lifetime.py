#!/usr/bin/env python3
"""Lifetime plot ("life of a cultivation": survival proportion per stage plus
the accumulated gate / feedback / decoder time), straight from the
RuntimeEstimator library: build the estimator for a circuit folder, run the
estimation, draw RuntimeEstimator.plot_lifetime. Nothing is stored except the
figure.

    python experiments/figures/extra/plot_lifetime.py experiments/data/circuits/<config> \
        [--partial partial_<tag>] [--mode single|pcp|cpc|no_gating --tc 35 --tl 0 --th 60 --tp 40] \
        [--feedback-ns 3100] [--latency mb|constant --mb-freq 43e6 --mb-shots 1000 --decoder-ns ...] \
        [--epochs 1000000 --workers 32] [--record until_success|first_reach] [--no-title]
    -> experiments/result/runtime_estimation/figures/lifetime_<config>_<variant>_<mode>_fb<ns>_<record>.{pdf,png}
"""
import argparse
import json
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

from estimator_run import build, gating, shard                             # noqa: E402
from naming import short_name                                              # noqa: E402



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", default=None, help="partial subfolder tag (partial_...)")
    ap.add_argument("--mode", choices=["single", "pcp", "cpc", "no_gating"], default="single")
    ap.add_argument("--tc", type=float, default=35.0)
    ap.add_argument("--tl", type=float, default=None)
    ap.add_argument("--th", type=float, default=None)
    ap.add_argument("--tp", type=float, default=None)
    ap.add_argument("--measure", choices=["serial", "parallel"], default="parallel")
    ap.add_argument("--feedback-ns", type=float, default=1000.0)
    ap.add_argument("--control-latency", type=pathlib.Path, default=None,
                    help="control-system latency profile (algorithms/control_latency.py --save); replaces --feedback-ns")
    ap.add_argument("--wait-rounds", type=int, default=2)
    ap.add_argument("--latency", choices=["mb", "constant"], default="mb")
    ap.add_argument("--mb-seed", type=int, default=0)
    ap.add_argument("--mb-shots", type=int, default=1000)
    ap.add_argument("--mb-freq", type=float, default=43e6)
    ap.add_argument("--decoder-ns", type=float, default=10_000.0)
    ap.add_argument("--partial-decoder-ns", type=float, default=500.0)
    ap.add_argument("--trace-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--record", choices=["until_success", "first_reach"], default="until_success",
                    help="until_success: wall clock including failed retries (default); first_reach: first pass")
    ap.add_argument("--title", default="", help="figure title (default none; the estimator's own title is dropped)")
    ap.add_argument("--legend", choices=["above", "inside"], default="above",
                    help="legend above the axes (paper layout, default) or the estimator's inside placement")
    ap.add_argument("--fig-dir", type=pathlib.Path,
                    default=paths.RESULT / "runtime_estimation" / "figures")
    ap.add_argument("--size", type=float, nargs=2, default=(7.0, 3.6), help="figure size (in)")
    args = ap.parse_args()
    if args.mode in ("pcp", "cpc") and args.partial is None:
        sys.exit("error: pcp/cpc need --partial")

    est, cf = build(args, args.epochs, start=0, step=1)
    K = args.workers
    if K > 1:
        shares = [args.epochs // K + (1 if k < args.epochs % K else 0) for k in range(K)]
        with ProcessPoolExecutor(max_workers=K, mp_context=get_context("fork")) as ex:
            est.merge_raw(list(ex.map(shard, [args] * K, range(K), [K] * K, shares)))
    else:
        est.estimate_runtime(decoder_time_measure_mode=args.measure, **gating(args))
    est.collect_statistics()
    ready = est.stage_names.index("ready")
    print(f"{cf.folder.name}: {est.attempt_count} attempts for {args.epochs} accepted shots, "
          f"accept {est.statistics['accept_rate']:.3f}, prep {est.statistics['records_until_success']['total']['mean'][ready] / 1000:.2f} us")

    fig, ax = plt.subplots(figsize=tuple(args.size), dpi=150)
    est.plot_lifetime(record_type=args.record, ax=ax)
    ax.set_title(args.title)
    if args.legend == "above":
        leg = ax.get_legend()
        handles, labels = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
        leg.remove()
        ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=3,
                  frameon=False, fontsize=8, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    thr = "".join(f"_{k}{getattr(args, k):g}" for k in ("tc", "tl", "th", "tp") if getattr(args, k) is not None)
    stem = f"lifetime_{short_name(cf.folder.name)}_{args.partial or 'complete'}_{args.mode}{thr}_fb{args.feedback_ns:g}_{args.record}"
    for ext in ("pdf", "png"):
        fig.savefig(args.fig_dir / f"{stem}.{ext}", bbox_inches="tight")
    print(f"saved {args.fig_dir / stem}.{{pdf,png}}")


if __name__ == "__main__":
    main()
