#!/usr/bin/env python3
"""Characterize a micro-blossom decoder for ONE circuit configuration.

Measures per-shot cycle-accurate decode latency (serial and parallel on/off
gap-decode combinations) on trace shots, using the RTL benchmark_decoding
flow. Latency is the product; accuracy/gap statistics come from software
decoding elsewhere (see docs/ARCHITECTURE.md).

Usage:
    python experiments/generate/run_mb_characterization.py \\
        experiments/data/circuits/<config>/ [--partial partial_<tag>] \\
        [--shots 1000] [--seed 0] [--frequency 100e6] [--keep-run]

Outputs:
    out/mb/<decoder_key>/latency_cache_s<seed>_f<MHz>.npz   (append-only per-shot cache)
    experiments/result/mb_characterization/<decoder_key>__s<seed>_n<shots>_f<MHz>.json

Re-runs are incremental: already-measured shots are never re-simulated
(per-shot latency is deterministic and position-independent — verified by
the parallel-shard gate); growing --shots appends only the missing range.
"""
import os
import argparse
import json
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702

import numpy as np
import stim

import layer_schedule
import mb_graph
import mb_toolchain
from attempt_stream import AttemptTrace
from naming import freq_tag

# Scan granularity of collect_accepted_shots: how many stored attempts to
# load+filter per iteration. Memory/vectorization tuning ONLY — results are
# bit-identical for any value. Unrelated to attempt_stream.BATCH_SIZE
# (generation identity) and CHUNK_ATTEMPTS (file layout).
SCAN_BLOCK = 100_000


def resolve_config(folder: pathlib.Path, partial: str | None) -> dict:
    """Load everything the run needs from the circuit folder (+ partial
    subfolder): gap circuit, decoder DEM, postselected detectors, and for
    partial mode the kept-index mapping + contracted obs-detector index.
    decoder_key = <folder>__complete or <folder>__<partial>.
    [used: once at script start]"""
    folder = folder.resolve()
    gap_circuit = stim.Circuit.from_file(folder / "gap_circuit.stim")
    ps_doc = json.loads((folder / "postselected_detectors.json").read_text())
    if isinstance(ps_doc, dict):                     # self-describing schema
        postselected = set(ps_doc["postselected_detectors"])
        obs_index = int(ps_doc["obs_detector_index"])
        assert int(ps_doc["n_gap_dets"]) == gap_circuit.num_detectors, \
            "postselected_detectors.json disagrees with gap_circuit detector count"
    else:                                            # legacy plain list
        postselected = set(ps_doc)
        obs_index = gap_circuit.num_detectors - 1
    if partial is None:
        dem = stim.DetectorErrorModel.from_file(folder / "desaturated_with_obs.dem")
        kept = None
        obs_vertex = obs_index
        key = f"{folder.name}__complete"
    else:
        pdir = folder / partial
        dem = stim.DetectorErrorModel.from_file(pdir / "contracted.dem")
        kept = np.load(pdir / "kept.npy")           # original det ids, contracted order
        stats_doc = json.loads((pdir / "stats.json").read_text())
        obs_vertex = int(stats_doc["contraction"]["obs_detector_new_index"])
        assert int(stats_doc["contraction"]["n_kept"]) == len(kept)
        o2n_path = pdir / "old_to_new.json"
        if o2n_path.exists():                        # safety: the two mappings must agree
            o2n = {int(k): int(v) for k, v in json.loads(o2n_path.read_text()).items()}
            assert o2n == {int(o): n for n, o in enumerate(kept)}, \
                "kept.npy order disagrees with old_to_new.json"
        key = f"{folder.name}__{partial}"
    return dict(folder=folder, gap_circuit=gap_circuit, postselected=postselected,
                dem=dem, kept=kept, obs_vertex=obs_vertex, key=key)


def collect_accepted_shots(cfg: dict, trace: AttemptTrace, shots: int, *,
                           start_accepted: int = 0,
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float]:
    """Walk the trace from index 0, apply detector postselection (any
    postselected detector fired -> discard; identical semantics to the
    sampler's keep mask), and return accepted shots with accepted-index in
    [start_accepted, start_accepted + shots) — extending the trace
    automatically if needed. Returns (attempt_indices, dets_rows, obs_bits,
    attempts_scanned, acceptance). The from-zero ordering is the contract
    that makes "accepted shot j" mean the same shot to every consumer;
    start_accepted lets the latency cache resume mid-stream (the prefix is
    rescanned, which is cheap and keeps the contract trivially correct).
    [used: once per run; deterministic given (trace seed, range)]"""
    num_dets = cfg["gap_circuit"].num_detectors
    discard_mask = np.packbits(
        np.array([k in cfg["postselected"] for k in range(num_dets)], dtype=np.bool_),
        bitorder="little")
    want = start_accepted + shots
    idx_parts, det_parts, obs_parts = [], [], []
    n_accepted = 0
    scan = 0
    while n_accepted < want:
        if scan >= trace.meta.num_attempts:
            trace.ensure(trace.meta.num_attempts + trace.meta.chunk_attempts)
        end = min(scan + SCAN_BLOCK, trace.meta.num_attempts)
        dets, obs = trace.get_range(scan, end)
        kidx = np.nonzero(~np.any(dets & discard_mask, axis=1))[0]
        idx_parts.append((kidx + scan).astype(np.int64))
        det_parts.append(dets[kidx])
        obs_parts.append(obs[kidx])
        n_accepted += len(kidx)
        scan = end
    lo, hi = start_accepted, want
    indices = np.concatenate(idx_parts)[lo:hi]
    dets_rows = np.concatenate(det_parts)[lo:hi]
    obs_bits = np.concatenate(obs_parts)[lo:hi]
    acceptance = n_accepted / scan       # true rate over everything scanned
    return indices, dets_rows, obs_bits, scan, acceptance


def open_latency_cache(d: pathlib.Path, seed: int, ftag: str, identity: dict,
                       suffix: str = "") -> tuple[pathlib.Path, dict]:
    """Load (or initialize) the append-only per-shot latency cache for one
    (decoder, trace seed, frequency). The identity block (graph hash, layer
    count, timing knobs, trace identity, stim version) must match exactly —
    a changed timing model or graph errors loudly instead of mixing epochs.
    Returns (cache_path, arrays) where arrays holds the measured prefix
    (empty arrays when no cache exists yet).
    [used: at the start of every characterization run]"""
    cache = d / f"latency_cache_s{seed}_{ftag}{suffix}.npz"
    meta_path = d / f"latency_cache_s{seed}_{ftag}{suffix}.meta.json"
    if not cache.exists():
        return cache, {}
    meta = json.loads(meta_path.read_text())
    # compat: caches written before the layer_schedule identity field are
    # uniform-mode measurements
    meta["identity"].setdefault("layer_schedule", None)
    if meta["identity"] != identity:
        raise SystemExit(
            f"latency cache {cache.name} was measured under a different "
            f"identity (graph/timing/trace changed):\ncached: {meta['identity']}\n"
            f"now:    {identity}\nDelete the cache to remeasure, or restore the config.")
    with np.load(cache) as z:
        arrays = {k: z[k] for k in z.files}
    n = len(arrays["on_us"])
    assert all(len(v) == n for v in arrays.values()), "corrupt cache: ragged arrays"
    assert n == meta["n_measured"], "corrupt cache: n_measured mismatch"
    return cache, arrays


def append_latency_cache(cache: pathlib.Path, identity: dict,
                         old: dict, new: dict) -> dict:
    """Append newly measured rows (accepted-shot order) to the cache and
    rewrite it atomically. Returns the merged arrays.
    [used: after each run that measured new shots]"""
    merged = {k: (np.concatenate([old[k], new[k]]) if old else new[k]) for k in new}
    tmp = cache.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **merged)
    tmp.replace(cache)
    meta_path = cache.parent / (cache.name.replace(".npz", ".meta.json"))
    meta_path.write_text(json.dumps(
        {"identity": identity, "n_measured": int(len(merged["on_us"]))}, indent=2))
    return merged


def build_paired_defects(cfg: dict, dets_rows: np.ndarray) -> list[list[int]]:
    """For each accepted shot produce TWO defect lists in decoder vertex
    space — obs vertex forced IN (entry 2j) and forced OUT (entry 2j+1) —
    mirroring the sampler's |= / ^= byte trick. Complete mode: defect ids
    are detector ids. Partial mode: project the full syndrome onto the
    kept detectors and renumber by contracted order (kept.npy).
    [used: once per run, before packing the defects file]"""
    num_dets_full = cfg["gap_circuit"].num_detectors
    fired_lists = mb_graph.bit_packed_to_defect_lists(dets_rows, num_dets_full)
    kept = cfg["kept"]
    if kept is not None:
        old_to_new = {int(o): n for n, o in enumerate(kept)}
        fired_lists = [[old_to_new[d] for d in row if d in old_to_new]
                       for row in fired_lists]
    elif cfg.get("mask") is not None:
        # --mask-only: complete decoder graph, but only the masked detectors are
        # observed (unmasked ones forced to 0); padding / obs detectors untouched
        mask = cfg["mask"]; n_circ = len(mask)
        fired_lists = [[d for d in row if d >= n_circ or mask[d]] for row in fired_lists]
    obs_v = cfg["obs_vertex"]
    paired = []
    for row in fired_lists:
        base = [d for d in row if d != obs_v]
        paired.append(sorted(base + [obs_v]))    # ON: obs vertex forced in
        paired.append(base)                      # OFF: obs vertex forced out
    return paired


def main() -> None:
    import dataclasses

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folder", type=pathlib.Path)
    ap.add_argument("--partial", default=None)
    ap.add_argument("--mask-only", default=None, metavar="PARTIAL_TAG",
                    help="DEM-contraction ablation: run the COMPLETE decoder (complete DEM, same compiled "
                         "design, full layer schedule) on syndromes restricted to this partial folder's mask "
                         "(unmasked detectors forced to 0). Separate latency cache and result file "
                         "(suffix _mask-<tag>).")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frequency", type=float, default=100e6)
    ap.add_argument("--measurement-cycle-ns", type=int, default=1000)
    ap.add_argument("--schedule", choices=["real", "uniform"], default="real",
                    help="Layer arrival timing: 'real' derives per-layer arrival "
                         "offsets from the circuit's round structure (Google gate "
                         "times); 'uniform' is the legacy fixed measurement-cycle "
                         "spacing. Separate cache namespaces.")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel simulator workers (per-worker micro-blossom "
                         "checkouts on /data; created+built on first use). "
                         "Sharding never changes results, only wall-clock.")
    ap.add_argument("--keep-run", action="store_true")
    ap.add_argument("--shard-retries", type=int, default=2,
                    help="re-launch a shard whose simulator host died before starting "
                         "(OpenJDK-11 GC segfault during elaboration) up to N times (default 2)")
    ap.add_argument("--prune-workers", action="store_true",
                    help="Delete the persistent worker checkout pool "
                         "(micro-blossom-workers/, ~3GB/worker) after a "
                         "successful run. Default keeps it: the pool is "
                         "circuit-AGNOSTIC and reused by every campaign; "
                         "pruning re-pays cargo+sbt builds next time.")
    args = ap.parse_args()
    # Cap the per-worker RTL compile parallelism (SpinalHDL runs make -j<cpus
    # visible to the JVM>) so W workers use at most ~all cores together instead
    # of W x all cores. Respected by mb_toolchain._exec via JAVA_TOOL_OPTIONS.
    os.environ.setdefault("MB_COMPILE_CPUS",
                          str(max(2, (os.cpu_count() or 8) // max(1, args.workers))))

    if args.partial and args.mask_only:
        ap.error("--partial and --mask-only are mutually exclusive")
    cfg = resolve_config(args.folder, args.partial)
    mask_suffix = ""
    if args.mask_only:
        mask = np.load(cfg["folder"] / args.mask_only / "mask_bool.npy")
        assert mask.dtype == np.bool_ and mask.ndim == 1
        cfg["mask"] = mask
        mask_suffix = f"_mask-{args.mask_only}"
        print(f"mask-only: complete decoder on {int(mask.sum())}/{len(mask)} observed detectors ({args.mask_only})")
    out_root = paths.OUT
    ftag = freq_tag(args.frequency)

    # decoder identity
    graph, graph_stats = mb_graph.dem_to_graph(cfg["dem"])
    graph_json = mb_toolchain.prepare_decoder(
        cfg["key"], graph, mb_graph.get_positions(cfg["dem"]))
    graph_doc = json.loads(graph_json.read_text())
    num_layer_fusion = graph_doc["layer_fusion"]["num_layers"]

    if args.schedule == "real":
        offsets = layer_schedule.fusion_layer_offsets_ns(
            graph_doc, cfg["gap_circuit"], kept=cfg["kept"])
        print(f"real layer schedule ({len(offsets)} layers, ns): {offsets}")
    else:
        offsets = None
    variant_dets = ([int(k) for k in cfg["kept"]] if cfg["kept"] is not None
                    else list(range(cfg["gap_circuit"].num_detectors)))
    rounds_info = layer_schedule.needed_rounds(
        cfg["gap_circuit"], variant_dets, obs_detector=cfg["obs_vertex"]
        if cfg["kept"] is None else int(cfg["kept"][cfg["obs_vertex"]]))
    print(f"needed rounds: [{rounds_info['first_needed_round']}, "
          f"{rounds_info['last_needed_round']}] of {rounds_info['num_circuit_rounds']} "
          f"(saved at end: {rounds_info['rounds_saved_at_end']})")

    # shot trace (keyed by gap circuit + seed) and measurement identity
    trace = AttemptTrace.create_or_open(
        out_root / "traces" / cfg["folder"].name, cfg["gap_circuit"], args.seed)
    identity = {
        "graph_hash": mb_graph.graph_hash(graph),
        "num_layer_fusion": num_layer_fusion,
        "measurement_cycle_ns": args.measurement_cycle_ns,
        "frequency_hz": args.frequency,
        "stim_version": stim.__version__,
        "trace": {"circuit_sha": trace.meta.circuit_sha,
                  "master_seed": trace.meta.master_seed,
                  "batch_size": trace.meta.batch_size},
        "layer_schedule": offsets,
    }
    if args.mask_only:
        import hashlib
        identity["mask_only"] = {"tag": args.mask_only,
                                 "mask_sha": hashlib.sha256(np.packbits(cfg["mask"]).tobytes()).hexdigest()[:16]}
    d = mb_toolchain.decoder_dir(cfg["key"])
    sched_suffix = ("" if args.schedule == "uniform" else "_realsched") + mask_suffix
    cache, cached = open_latency_cache(d, args.seed, ftag, identity,
                                       suffix=sched_suffix)
    n_cached = len(cached["on_us"]) if cached else 0
    acceptance = None

    if n_cached >= args.shots:
        print(f"latency cache hit: {n_cached} shots measured >= {args.shots} "
              f"requested — no simulation needed")
        merged = cached
    else:
        # measure ONLY the missing accepted-shot range [n_cached, shots)
        need = args.shots - n_cached
        indices, dets_rows, obs_bits, scanned, acceptance = collect_accepted_shots(
            cfg, trace, need, start_accepted=n_cached)
        print(f"measuring accepted shots [{n_cached}, {args.shots}) — {need} new "
              f"(cache has {n_cached}; detector-postselect acceptance "
              f"{acceptance:.3f} over {scanned} attempts)")
        if n_cached:
            assert indices[0] > int(cached["attempt_indices"][-1]), \
                "cache continuity violated: new shots do not follow cached ones"

        positions = mb_graph.get_positions(cfg["dem"])
        run_tag = f"s{args.seed}_{ftag}{sched_suffix}_a{n_cached}-{args.shots}"

        shard_files: list[pathlib.Path] = []

        def run_shard(shard_idx: int, rows: np.ndarray, root: pathlib.Path):
            paired = build_paired_defects(cfg, rows)
            tag = f"{run_tag}_shard{shard_idx}"
            syn = d / f"characterization_{tag}.syndromes"
            mb_graph.write_syndrome_file(syn, graph, positions, paired)
            defects = d / f"characterization_{tag}.defects"
            shard_files.extend([syn, defects])
            mb_toolchain.pack_defects(syn, defects, root=root)
            # The SpinalHDL simulation host is an OpenJDK-11 JVM that occasionally
            # segfaults in the collector during elaboration (heap corruption, seen
            # with G1 and ParallelGC alike; ~1 in 8 JVMs on >2000-vertex designs).
            # It dies before the simulation starts, so retrying just this shard is
            # safe and loses nothing; the other shards are unaffected.
            for attempt in range(1, args.shard_retries + 2):
                try:
                    stdout, _ = mb_toolchain.run_benchmark_decoding(
                        cfg["key"], defects, num_layer_fusion=num_layer_fusion,
                        measurement_cycle_ns=args.measurement_cycle_ns,
                        frequency_hz=args.frequency, layer_schedule_ns=offsets,
                        root=root, keep_run=args.keep_run)
                    break
                except mb_toolchain.ToolchainError as e:
                    if attempt > args.shard_retries:
                        raise
                    print(f"shard {shard_idx}: simulator launch failed (attempt {attempt}), "
                          f"retrying on {root.name}: {str(e).splitlines()[0][:100]}", flush=True)
            lat = mb_toolchain.parse_benchmark_latencies(stdout)[1:]  # 1-based [n]
            assert len(lat) == 2 * len(rows) and not np.isnan(lat).any(), \
                f"shard {shard_idx}: expected {2 * len(rows)} dense entries, got " \
                f"{len(lat)} (NaNs: {int(np.isnan(lat).sum())})"
            counts = np.array([len(b) for b in paired[1::2]])
            return lat, counts

        if args.workers <= 1:
            shard_results = [run_shard(0, dets_rows, mb_toolchain.MB_ROOT)]
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                roots = list(pool.map(mb_toolchain.ensure_worker, range(args.workers)))
                row_shards = [r for r in np.array_split(dets_rows, args.workers) if len(r)]
                shard_results = list(pool.map(
                    lambda t: run_shard(t[0], t[1], roots[t[0]]), enumerate(row_shards)))
        lat_us = np.concatenate([lat for lat, _ in shard_results])
        new = {
            "attempt_indices": indices,
            "on_us": lat_us[0::2],
            "off_us": lat_us[1::2],
            "obs_bits": obs_bits,
            "defect_hashes": np.array([AttemptTrace.defect_hash(r) for r in dets_rows]),
            "defect_counts": np.concatenate([c for _, c in shard_results]),
        }
        merged = append_latency_cache(cache, identity, cached, new)
        print(f"cache now holds {len(merged['on_us'])} measured shots")
        for f in shard_files:                 # intermediates: cache is the record
            f.unlink(missing_ok=True)

    # ---- distilled result over the first `shots` cached rows ----
    on_us = merged["on_us"][:args.shots]
    off_us = merged["off_us"][:args.shots]
    serial_us, parallel_us = on_us + off_us, np.maximum(on_us, off_us)

    def stats_of(a: np.ndarray) -> dict:
        return {"mean_us": float(a.mean()), "p50_us": float(np.percentile(a, 50)),
                "p99_us": float(np.percentile(a, 99)), "max_us": float(a.max()),
                "mean_cycles": float(a.mean() * 1e-6 * args.frequency)}

    result = {
        "config": {
            "folder": cfg["folder"].name, "partial": args.partial, "mask_only": args.mask_only,
            "decoder_key": cfg["key"], "seed": args.seed, "shots": args.shots,
            "acceptance": acceptance, "graph_stats": graph_stats,
            "needed_rounds": rounds_info,
            "workers": args.workers, "identity": identity,
            "trace": dataclasses.asdict(trace.meta),
        },
        "latency": {"serial": stats_of(serial_us), "parallel": stats_of(parallel_us),
                    "on": stats_of(on_us), "off": stats_of(off_us)},
        "raw": {"cache": str(cache), "rows_used": int(args.shots),
                "rows_available": int(len(merged["on_us"]))},
    }
    out_json = (paths.RESULT / "mb_characterization"
                / f"{cfg['key']}__s{args.seed}_n{args.shots}_{ftag}{sched_suffix}.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2))
    print("wrote", out_json)

    if args.prune_workers and mb_toolchain.WORKERS_DIR.exists():
        import shutil
        shutil.rmtree(mb_toolchain.WORKERS_DIR)
        print(f"pruned worker pool {mb_toolchain.WORKERS_DIR} "
              f"(recreated+rebuilt automatically on next --workers>1 run)")


if __name__ == "__main__":
    main()
