#!/usr/bin/env python3
"""Generate essentials orderings (per-detector scores) for a cultivation
circuit and save them as JSON for PartialMaskBuilder.load_essentials_from_json.

The scores describe the complete decoder on the full circuit, independent of
any partial mask, so one run per circuit serves every mask configuration:
    canonical          P(d fires | gap < th) - P(d fires | gap >= th)
    causal             mean |gap - gap(d cleared)| over shots where d fired
    logical_ambiguity  Q: how often d lies on Z = E_on xor E_off, the difference
                       between the complete decoder's two minimum-weight explanations
                       (Q_all unweighted, Q_exp weighted by exp(-|w_on - w_off|))

--workers forks over detector ranges (causal) or shot ranges (logical_ambiguity).
Same results as --workers 1 (logical_ambiguity up to float summation order).

Usage:
    python algorithms/gen_essentials.py \\
        --circuit path/to/circuit.stim \\
        --out experiments/data/circuits/<circuit_name> \\
        --type all --shots 10000000 --workers 8

Output files (flat, in --out):
    <stem>_canonical_th<G>_shots<N>.json
    <stem>_causal_shots<N>[_sub<S>].json
    <stem>_logical_ambiguity_shots<N>.json
"""
import argparse
import multiprocessing as mp
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in [REPO / "algorithms", REPO / "magic_state_cultivation" / "upstream" / "src"]:
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

import stim

from partial_mask_builder import PartialMaskBuilder

# Scoring context inherited by forked workers (copy-on-write, not pickled).
# The builder is single-process; all parallelism is in this script.
_CTX: dict = {}


def circuit_stem(path: pathlib.Path) -> str:
    """Output-name stem: the basename up to the first '.stim'."""
    name = path.name
    if ".stim" in name:
        return name[: name.index(".stim")]
    return path.stem


def fmt_g(x: float) -> str:
    return f"{x:g}"


def _ranges(n: int, workers: int) -> list[tuple[int, int]]:
    """Up to workers*4 contiguous non-empty ranges covering [0, n)."""
    n_chunks = max(1, min(n, workers * 4))
    edges = np.linspace(0, n, n_chunks + 1, dtype=np.int64)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_chunks) if edges[i] < edges[i + 1]]


def _causal_chunk(bounds):
    return PartialMaskBuilder.score_causal_range(_CTX["ctx"], *bounds)


def _ambiguity_chunk(bounds):
    return PartialMaskBuilder.score_logical_ambiguity_range(_CTX["ctx"], *bounds, _CTX["num_detectors"])


def _pool_map(fn, items, workers):
    if workers <= 1:
        return [fn(it) for it in items]
    with mp.get_context("fork").Pool(workers) as pool:
        return pool.map(fn, items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--circuit", type=pathlib.Path, required=True,
                        help="Path to the noisy stim circuit file.")
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Output directory (the circuit's data folder); "
                             "created if missing. Essentials JSONs are written "
                             "directly into it, no subfolder.")
    parser.add_argument("--type", choices=["canonical", "causal", "logical_ambiguity", "both", "all"],
                        default="all", help="Which ordering(s) to compute "
                                            "('both' = canonical + causal, kept for compatibility).")
    parser.add_argument("--shots", type=int, required=True,
                        help="Shots to sample per ordering (e.g. 10000000).")
    parser.add_argument("--gap-threshold", type=float, default=50.0,
                        help="Reject/accept split for the canonical score "
                             "(default: 50.0).")
    parser.add_argument("--subsample-size", type=int, default=None,
                        help="Causal only: score on this many randomly chosen "
                             "kept shots instead of all of them.")
    parser.add_argument("--rng-seed", type=int, default=0,
                        help="RNG seed for the causal subsample draw.")
    parser.add_argument("--q-type", choices=["all", "exp"], default="all",
                        help="logical_ambiguity: which Q the saved ordering follows "
                             "(both score arrays are always stored).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for the causal and logical_ambiguity "
                             "scoring loops (fork-based).")
    args = parser.parse_args()

    if not args.circuit.is_file():
        parser.error(f"circuit file not found: {args.circuit}")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)

    circuit = stim.Circuit.from_file(args.circuit)
    stem = circuit_stem(args.circuit)
    builder = PartialMaskBuilder(circuit=circuit)
    print(f"circuit: {args.circuit}  ({circuit.num_detectors} detectors)")
    want = {"canonical": args.type in ("canonical", "both", "all"),
            "causal": args.type in ("causal", "both", "all"),
            "logical_ambiguity": args.type in ("logical_ambiguity", "all")}

    if want["canonical"]:
        out = args.out / f"{stem}_canonical_th{fmt_g(args.gap_threshold)}_shots{args.shots}.json"
        print(f"[canonical] computing (shots={args.shots}, "
              f"gap_threshold={args.gap_threshold}) ...")
        builder.compute_canonical_essentials_order(
            gap_threshold=args.gap_threshold, shots=args.shots)
        meta = builder.canonical_essentials_meta
        print(f"[canonical] kept {meta['kept_shots']}/{args.shots} shots, "
              f"accept/reject = {meta.get('complete_accept_count')}/"
              f"{meta.get('complete_reject_count')}")
        builder.save_essentials_to_json(out, type="canonical")
        print(f"[canonical] wrote {out}")

    if want["causal"]:
        sub_tag = f"_sub{args.subsample_size}" if args.subsample_size else ""
        out = args.out / f"{stem}_causal_shots{args.shots}{sub_tag}.json"
        print(f"[causal] computing (shots={args.shots}, "
              f"subsample={args.subsample_size}, workers={args.workers}) ...")
        ctx = builder.prepare_causal_essentials(
            shots=args.shots, subsample_size=args.subsample_size, rng_seed=args.rng_seed)
        nd = builder.num_detectors
        if ctx["kept_shots"] == 0:
            parts = [(np.zeros(nd), np.zeros(nd), np.zeros(nd, np.int64))]
            bounds = [(0, nd)]
        else:
            _CTX["ctx"] = ctx
            bounds = _ranges(nd, args.workers)
            parts = _pool_map(_causal_chunk, bounds, args.workers)
            _CTX.clear()
        sum_abs = np.zeros(nd); sum_signed = np.zeros(nd); count = np.zeros(nd, np.int64)
        for (lo, hi), (sa, ss, c) in zip(bounds, parts):
            sum_abs[lo:hi] = sa; sum_signed[lo:hi] = ss; count[lo:hi] = c
        builder.finalize_causal_essentials(ctx, sum_abs, sum_signed, count)
        meta = builder.causal_essentials_meta
        print(f"[causal] kept {meta['kept_shots']}/{args.shots} shots, "
              f"scored on {meta['subsample_size']}")
        builder.save_essentials_to_json(out, type="causal")
        print(f"[causal] wrote {out}")

    if want["logical_ambiguity"]:
        out = args.out / f"{stem}_logical_ambiguity_shots{args.shots}.json"
        print(f"[logical_ambiguity] computing (shots={args.shots}, q_type={args.q_type}, "
              f"workers={args.workers}) ...")
        ctx = builder.prepare_logical_ambiguity_essentials(shots=args.shots)
        nd = builder.num_detectors
        _CTX.update(ctx=ctx, num_detectors=nd)
        parts = _pool_map(_ambiguity_chunk, _ranges(ctx["kept_shots"], args.workers), args.workers)
        _CTX.clear()
        count_all = sum(p[0] for p in parts); wexp = sum(p[1] for p in parts)
        n = sum(p[2] for p in parts); wsum = sum(p[3] for p in parts)
        builder.finalize_logical_ambiguity_essentials(ctx, count_all, wexp, n, wsum, q_type=args.q_type)
        meta = builder.logical_ambiguity_essentials_meta
        print(f"[logical_ambiguity] kept {meta['kept_shots']}/{args.shots} shots, scored {meta['n_scored']}")
        builder.save_essentials_to_json(out, type="logical_ambiguity")
        print(f"[logical_ambiguity] wrote {out}")


if __name__ == "__main__":
    main()
