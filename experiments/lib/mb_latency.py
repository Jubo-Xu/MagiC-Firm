# mb_latency.py — feed micro-blossom characterization results to the runtime
# estimator. This module owns the file formats written by
# run_mb_characterization; the estimator itself only receives plain numbers.

import json
import pathlib

import numpy as np

_MODES = ("serial", "parallel")


def load_mb_constants(est, *, complete_json, partial_json=None) -> None:
    """Mode A: constant decoder latencies from run_mb_characterization result
    JSONs (mean over the characterized shots, both measure modes). Checks each
    JSON was measured on the same round clock as the estimator: the MB arrival
    schedule spans [first_needed_round, last_needed_round] and must land on
    the same round-completion times (same gate times + MPP policy).
    [used: run_runtime_estimation and plot_lifetime with --latency mb]"""
    consts, prov = {}, {}
    for which, path in (("complete", complete_json), ("partial", partial_json)):
        if path is None:
            continue
        doc = json.loads(pathlib.Path(path).read_text())
        cfg = doc["config"]
        nr, sched = cfg.get("needed_rounds"), cfg["identity"].get("layer_schedule")
        if nr and sched:
            est.check_round_clock(first_round=nr["first_needed_round"],
                                  last_round=nr["last_needed_round"],
                                  span_ns=sched[-1], label=f"{which} MB result")
        consts[which] = {m: float(doc["latency"][m]["mean_us"]) * 1000.0 for m in _MODES}
        prov[which] = {
            "path": str(path), "decoder_key": cfg["decoder_key"],
            "shots": cfg["shots"], "seed": cfg["seed"],
            "frequency_hz": cfg["identity"]["frequency_hz"],
            "layer_schedule": cfg["identity"]["layer_schedule"],
            "needed_rounds": cfg.get("needed_rounds"),
            "mean_us": {m: doc["latency"][m]["mean_us"] for m in _MODES},
        }
    est.set_decoder_latency_constants(complete=consts["complete"], partial=consts.get("partial"),
                                      provenance=prov, source="mb_constant")


def load_mb_latency_table(est, *, complete_cache, partial_cache=None) -> None:
    """Mode B: per-attempt decoder latency from the append-only MB latency
    caches (out/mb/<key>/latency_cache_s<seed>_<f>MHz_realsched.npz, written
    by run_mb_characterization). Requires est.attach_trace() on the same
    trace: the cache's identity.trace (circuit sha, seed, batch size) must
    match. Latencies are converted us -> ns here (float64).
    [used: run_runtime_estimation with --latency mb_trace]"""
    ident = est.trace_identity
    if ident is None:
        raise RuntimeError("attach_trace() first: Mode B joins on attempt index")
    tables, prov = {}, {}
    for which, path in (("complete", complete_cache), ("partial", partial_cache)):
        if path is None:
            continue
        path = pathlib.Path(path)
        meta = json.loads(path.with_name(path.name.replace(".npz", ".meta.json")).read_text())
        if meta["identity"]["trace"] != ident:
            raise ValueError(f"{which} latency cache was measured on a different trace")
        with np.load(path) as z:
            tables[which] = {
                "attempt_indices": z["attempt_indices"],
                "on_ns": z["on_us"].astype(np.float64) * 1000.0,
                "off_ns": z["off_us"].astype(np.float64) * 1000.0,
                "defect_hashes": z["defect_hashes"]}
        prov[which] = {
            "cache": str(path), "n_measured": int(meta["n_measured"]),
            "frequency_hz": meta["identity"]["frequency_hz"],
            "layer_schedule": meta["identity"]["layer_schedule"]}
    est.set_decoder_latency_table(complete=tables["complete"], partial=tables.get("partial"),
                                  provenance=prov, source="mb_trace")
