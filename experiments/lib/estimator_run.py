# estimator_run.py — build and run a RuntimeEstimator from a circuit folder and
# the common command-line options (--mode/--tc/--tl/--th/--tp, --latency,
# --feedback-ns, --control-latency, trace seed), shared by the estimator-based
# scripts and their forked shards.

import json
import pathlib

import paths
from circuit_folder import load_circuit_folder, compile_samplers
from mb_latency import load_mb_constants
from runtime_estimator import RuntimeEstimator

MB_RESULT_DIR = paths.RESULT / "mb_characterization"


def build(args, epoch, start, step):
    """Estimator for one shard: folder, samplers, trace partition (start=k,
    step=K for worker k of K), latency source and control profile.
    Returns (estimator, circuit_folder)."""
    cf = load_circuit_folder(args.folder, args.partial)
    complete, partial = compile_samplers(cf)
    est = RuntimeEstimator(
        circuit=cf.circuit, complete_compiled=complete, partial_compiled=partial,
        postselected_detectors=cf.postselected, epoch=epoch,
        feedback_time_ns=args.feedback_ns, wait_rounds=args.wait_rounds,
        use_measured_decode_time=False,
        decoder_latency_ns=args.decoder_ns, partial_decoder_latency_ns=args.partial_decoder_ns,
    )
    est.attach_trace(cf.trace_dir(paths.OUT), cf.gap_circuit, master_seed=args.trace_seed,
                     start=start, step=step)
    if args.latency == "mb":
        mb = dict(seed=args.mb_seed, freq_hz=args.mb_freq, shots=args.mb_shots)
        load_mb_constants(
            est,
            complete_json=cf.mb_result_json(MB_RESULT_DIR, complete=True, **mb),
            partial_json=cf.mb_result_json(MB_RESULT_DIR, **mb) if partial is not None else None)
    # applied here so that every forked shard, not just the driver, charges the
    # modelled control time
    if getattr(args, "control_latency", None):
        est.set_control_latency(json.loads(pathlib.Path(args.control_latency).read_text()))
    return est, cf


def gating(args):
    """--mode/--t* -> RuntimeEstimator.estimate_runtime threshold kwargs."""
    if args.mode == "no_gating":
        return dict(gap_check=False)
    kw = dict(gap_check=True, gap_threshold=args.tc)
    if args.mode in ("pcp", "cpc"):
        kw.update(gap_threshold_low=args.tl, gap_threshold_high=args.th)
    if args.mode == "cpc":
        kw["gap_threshold_partial"] = args.tp
    return kw


def shard(args, k, K, epoch):
    """Worker k of K: run the estimation on its trace partition, return raw arrays."""
    est, _ = build(args, epoch, start=k, step=K)
    est.estimate_runtime(decoder_time_measure_mode=args.measure, **gating(args))
    return est.export_raw()
