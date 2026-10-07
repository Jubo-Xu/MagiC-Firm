#!/usr/bin/env python3
"""Folder-driven runtime estimation.

    python experiments/generate/run_runtime_estimation.py <circuit folder> [--partial TAG]
        --mode {no_gating,single,pcp,cpc} [--tc T] [--tl T] [--th T] [--tp T]
        --epochs N [--latency {mb,mb_trace,measured,constant}] [--trace] [--save-raw]

Everything about the circuit comes from the folder (circuit, postselect set,
mask, contraction parameters — see experiments/lib/circuit_folder.py); decoder
latencies come from the MB characterization of the same folder (Mode A:
result JSON means; Mode B: per-shot cache, needs --trace); shots come from
the shared attempt trace (--trace) or fresh simulation.

Outputs:
  experiments/result/runtime_estimation/<folder>/<variant>__<mode>..._e<N>.json
      statistics summary + `config` block recording every input (small)
  out/runtime_estimation/<folder>/<same basename>.npz        (--save-raw)
      raw per-epoch / per-shot arrays (RuntimeEstimator.export_raw), on /data

Example (paper first data, flagship folder):
  python experiments/generate/run_runtime_estimation.py \\
      experiments/data/circuits/end2end_d1=3_d2=15_r1=3_r2=0_p=0.001_inj=unitary_b=Y \\
      --partial partial_l1-14_t1-9_ts1-7_aug-canonical50_pr-causal50_wc15 \\
      --mode pcp --tc 35 --tl 20 --th 60 --epochs 100000 --latency mb --trace --save-raw
"""
import argparse
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702

from circuit_folder import load_circuit_folder, compile_samplers           # noqa: E402
from mb_latency import load_mb_constants, load_mb_latency_table            # noqa: E402
from runtime_estimator import RuntimeEstimator                             # noqa: E402

REPO = paths.REPO
MB_RESULT_DIR = paths.RESULT / "mb_characterization"
RESULT_DIR = paths.RESULT / "runtime_estimation"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", default=None, help="partial subfolder tag (partial_...)")
    g = ap.add_argument_group("gating")
    g.add_argument("--mode", required=True, choices=["no_gating", "single", "pcp", "cpc"])
    g.add_argument("--tc", type=float, help="complete gap threshold (single/pcp/cpc)")
    g.add_argument("--tl", type=float, help="partial low threshold (pcp/cpc)")
    g.add_argument("--th", type=float, help="partial high threshold (pcp/cpc)")
    g.add_argument("--tp", type=float, help="partial middle-band threshold (cpc)")
    e = ap.add_argument_group("estimator")
    e.add_argument("--epochs", type=int, default=10_000, help="accepted shots to simulate")
    e.add_argument("--workers", type=int, default=1,
                   help="parallel worker processes; worker k of K replays the strided "
                        "trace partition k, k+K, ... (a different but equivalent shot "
                        "sample from --workers 1; reproducible for fixed seed/K)")
    e.add_argument("--wait-rounds", type=int, default=2,
                   help="plot-only [wait for gap] stages + 1 (zero time)")
    e.add_argument("--feedback-ns", type=float, default=1000.0)
    e.add_argument("--control-latency", type=pathlib.Path, default=None, metavar="PROFILE.json",
                   help="control-system latency profile (algorithms/control_latency.py --save): "
                        "per-round post-select feedback, gap-decision feedback and syndrome "
                        "delivery from the compiled board tree replace --feedback-ns; the "
                        "result name gains _cs<q>f<fanout>")
    e.add_argument("--measure", choices=["serial", "parallel"], default="parallel",
                   help="ON/OFF gap decodes back to back (serial) or concurrent (parallel)")
    e.add_argument("--no-mpp-policy", action="store_true",
                   help="price the terminal MPP round raw (500ns) instead of like "
                        "the previous round")
    lat = ap.add_argument_group("decoder latency")
    lat.add_argument("--latency", choices=["mb", "mb_trace", "measured", "constant"],
                     default="mb",
                     help="mb: constants from MB result JSON means (Mode A); mb_trace: "
                          "per-shot MB cache lookup (Mode B, implies --trace); measured: "
                          "pymatching wall-clock; constant: --decoder-ns/--partial-decoder-ns")
    lat.add_argument("--mb-seed", type=int, default=0)
    lat.add_argument("--mb-shots", type=int, default=1000,
                     help="shots in the MB result JSON name (Mode A)")
    lat.add_argument("--mb-freq", type=float, default=43e6)
    lat.add_argument("--decoder-ns", type=float, default=10_000.0)
    lat.add_argument("--partial-decoder-ns", type=float, default=500.0)
    tr = ap.add_argument_group("attempt source")
    tr.add_argument("--trace", action="store_true", help="replay the shared attempt trace")
    tr.add_argument("--trace-seed", type=int, default=0)
    tr.add_argument("--trace-start", type=int, default=0, help="first attempt index to replay")
    o = ap.add_argument_group("output")
    o.add_argument("--out-root", type=pathlib.Path, default=paths.OUT,
                   help="traces / MB caches / raw outputs root (symlink to /data)")
    o.add_argument("--result-dir", type=pathlib.Path, default=RESULT_DIR)
    o.add_argument("--save-raw", action="store_true",
                   help="also write export_raw() arrays to <out-root>/runtime_estimation/")
    o.add_argument("--tag", default="", help="extra suffix for the result filename")
    return ap


def gating_kwargs(args) -> dict:
    """CLI --mode/--t* -> RuntimeEstimator.estimate_runtime threshold kwargs."""
    need = {"no_gating": [], "single": ["tc"], "pcp": ["tc", "tl", "th"],
            "cpc": ["tc", "tl", "th", "tp"]}[args.mode]
    missing = [f"--{k}" for k in need if getattr(args, k) is None]
    if missing:
        sys.exit(f"error: --mode {args.mode} requires {' '.join(missing)}")
    if args.mode == "no_gating":
        return dict(gap_check=False)
    kw = dict(gap_check=True, gap_threshold=args.tc)
    if args.mode in ("pcp", "cpc"):
        kw.update(gap_threshold_low=args.tl, gap_threshold_high=args.th)
    if args.mode == "cpc":
        kw["gap_threshold_partial"] = args.tp
    return kw


def result_basename(args) -> str:
    thr = "".join(f"_{k}{getattr(args, k):g}" for k in ("tc", "tl", "th", "tp")
                  if getattr(args, k) is not None)
    variant = args.partial or "complete"
    src = f"_trace-s{args.trace_seed}" if args.trace else "_sim"
    w = f"_w{args.workers}" if args.workers > 1 else ""
    return f"{variant}__{args.mode}{thr}__lat-{args.latency}{src}_e{args.epochs}{w}{control_tag(args)}{args.tag}"


def build_estimator(args, cf, complete, partial, *, epoch: int,
                    start: int = 0, step: int = 1) -> RuntimeEstimator:
    """Estimator for this run's configuration: stage clock from the folder
    circuit, attempt source (trace partition or simulation), latency source."""
    est = RuntimeEstimator(
        circuit=cf.circuit, complete_compiled=complete, partial_compiled=partial,
        postselected_detectors=cf.postselected, epoch=epoch,
        feedback_time_ns=args.feedback_ns, wait_rounds=args.wait_rounds,
        use_measured_decode_time=(args.latency == "measured"),
        decoder_latency_ns=args.decoder_ns, partial_decoder_latency_ns=args.partial_decoder_ns,
        boundary_round_as_previous=not args.no_mpp_policy,
    )
    if args.trace:
        est.attach_trace(cf.trace_dir(args.out_root), cf.gap_circuit,
                         master_seed=args.trace_seed, start=start, step=step)
    mb = dict(seed=args.mb_seed, freq_hz=args.mb_freq)
    if args.latency == "mb":
        load_mb_constants(
            est,
            complete_json=cf.mb_result_json(MB_RESULT_DIR, shots=args.mb_shots,
                                            complete=True, **mb),
            partial_json=(cf.mb_result_json(MB_RESULT_DIR, shots=args.mb_shots, **mb)
                          if partial is not None else None))
    elif args.latency == "mb_trace":
        load_mb_latency_table(
            est,
            complete_cache=cf.mb_latency_cache(args.out_root, complete=True, **mb),
            partial_cache=(cf.mb_latency_cache(args.out_root, **mb)
                           if partial is not None else None))
    if args.control_latency is not None:
        est.set_control_latency(json.loads(args.control_latency.read_text()))
    return est


def control_tag(args) -> str:
    """Result-name suffix for a control-latency profile (empty without one)."""
    if args.control_latency is None:
        return ""
    p = json.loads(args.control_latency.read_text())
    return "_cs" + p.get("tag", f"q{p['q']}f{p['fanout']}")


def run_shard(args, k: int, K: int, epoch: int) -> dict:
    """Worker k of K: own folder load + sampler compile, strided trace
    partition (start=trace_start+k, step=K) or independent simulation,
    returns export_raw() for the driver to merge."""
    cf = load_circuit_folder(args.folder, args.partial)
    complete, partial = compile_samplers(cf)
    est = build_estimator(args, cf, complete, partial, epoch=epoch,
                          start=args.trace_start + k, step=K)
    est.estimate_runtime(decoder_time_measure_mode=args.measure, **gating_kwargs(args))
    return est.export_raw()


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode in ("pcp", "cpc") and args.partial is None:
        sys.exit("error: pcp/cpc need --partial")
    if args.latency == "mb_trace":
        args.trace = True
    if args.workers < 1 or args.workers > args.epochs:
        sys.exit("error: --workers must be in [1, epochs]")
    gating_kwargs(args)                       # validate thresholds before any work

    t0 = time.time()
    cf = load_circuit_folder(args.folder, args.partial)
    print(f"folder: {cf.folder.name}  variant: {args.partial or 'complete'}  "
          f"({cf.circuit.num_detectors} dets, {len(cf.postselected)} postselected"
          f"{f', mask {int(cf.mask_bool.sum())}' if cf.mask_bool is not None else ''})")
    complete, partial = compile_samplers(cf)
    print(f"samplers compiled, DEMs verified against folder ({time.time() - t0:.0f}s)")

    est = build_estimator(args, cf, complete, partial, epoch=args.epochs,
                          start=args.trace_start, step=1)
    print(f"stages: {est.stage_names}")
    print(f"round gate ns: {est.layer_gate_times_ns.astype(int).tolist()} "
          f"(total {est.gate_to_complete_ns:.0f})")
    print(f"attempt source: {est.attempt_source}   latency source: {est.latency_source}")

    t1 = time.time()
    K = args.workers
    shares = [args.epochs // K + (1 if k < args.epochs % K else 0) for k in range(K)]
    if K > 1:
        print(f"running {K} workers, epochs per worker {shares[0]}..{shares[-1]}")
        with ProcessPoolExecutor(max_workers=K, mp_context=get_context("fork")) as ex:
            raws = list(ex.map(run_shard, [args] * K, range(K), [K] * K, shares))
        est.merge_raw(raws)
    else:
        est.estimate_runtime(decoder_time_measure_mode=args.measure, **gating_kwargs(args))
    print(f"simulated {est.attempt_count} attempts for {args.epochs} accepted shots "
          f"({time.time() - t1:.0f}s)")

    est.collect_statistics()
    base = result_basename(args)
    raw_path = args.out_root / "runtime_estimation" / cf.folder.name / f"{base}.npz"
    from gap_table import result_fingerprint
    inputs = result_fingerprint(cf.folder, args.partial, mb_shots=args.mb_shots, mb_seed=args.mb_seed,
                                mb_freq_hz=args.mb_freq, control_latency=args.control_latency)
    est.statistics["config"] = {
        "folder": cf.folder.name, "partial": args.partial, "decoder_key": cf.decoder_key,
        "mask_sha": inputs["mask_sha"], "inputs": inputs,      # DEM/mask, MB characterization, control profile
        "circuit_file": f"{cf.stem}.stim",
        "gating": dict(mode=args.mode, tc=args.tc, tl=args.tl, th=args.th, tp=args.tp,
                       measure=args.measure),
        "estimator": dict(epochs=args.epochs, wait_rounds=args.wait_rounds,
                          feedback_ns=args.feedback_ns,
                          control_latency=(str(args.control_latency) if args.control_latency else None),
                          boundary_round_as_previous=not args.no_mpp_policy,
                          stage_names=est.stage_names,
                          round_gate_ns=est.layer_gate_times_ns.tolist(),
                          last_kept_layer=est.last_kept_layer),
        "latency": dict(source=args.latency, mb_seed=args.mb_seed, mb_shots=args.mb_shots,
                        mb_freq_hz=args.mb_freq, decoder_ns=args.decoder_ns,
                        partial_decoder_ns=args.partial_decoder_ns),
        "attempts": dict(trace=args.trace, trace_seed=args.trace_seed,
                         trace_start=args.trace_start, attempts_simulated=est.attempt_count,
                         workers=K, epochs_per_worker=shares),
        "raw_npz": str(raw_path) if args.save_raw else None,
        "argv": sys.argv[1:],
    }
    out = args.result_dir / cf.folder.name / f"{base}.json"
    saved = est.save_statistics_to_json(out)
    print(f"saved {paths.rel(saved)} "
          f"({saved.stat().st_size:,} bytes)")
    if args.save_raw:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(raw_path, **est.export_raw())
        print(f"saved raw {raw_path} ({raw_path.stat().st_size:,} bytes)")
    est.print_statistics()


if __name__ == "__main__":
    main()
