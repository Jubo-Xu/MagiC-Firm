#!/usr/bin/env python3
"""Generate a partial-mask configuration folder inside a circuit data folder.

Loads the circuit from --out (written by get_desaturated_dem_with_obs_detector.py),
builds the mask, optionally runs the chain-aware DEM contraction (--contract),
and writes the subfolder <out>/partial_<paramtag>/.

Base mask M0 (--base):
    logical_ambiguity  TopK(Q) over the logical-ambiguity essentials
                       (default; --base-size K_Q, --q-type all|exp)
    region             geometric spacetime region (--left-len/--top-len/--t)
    list               explicit detector list (--detectors FILE)
Refinement ops, applied in command-line order:
    --augment TYPE:K   add the K highest-scoring hidden detectors of an essentials list
    --closure [TAU]    add hidden detectors that lower the DEM crossing mass S(M)
    --prune TYPE:K     remove the K lowest-scoring selected detectors

Usage (final recipe):
    python algorithms/gen_partial_mask_dem.py \
        --out experiments/data/circuits/<circuit_name> \
        --base-size 288 --augment canonical:50 --closure --prune causal:50 \
        --contract --weight-cutoff 15.0
Geometric base:
    python algorithms/gen_partial_mask_dem.py --out ... --base region \
        --left-len 1 14 --top-len 1 9 --t 1 7 --augment canonical:50 --prune causal:50 --contract
"""
import argparse
import hashlib
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
for p in [REPO / "algorithms", REPO / "magic_state_cultivation" / "upstream" / "src"]:
    s = str(p)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)

import numpy as np
import stim

from partial_mask_builder import PartialMaskBuilder
from partial_desaturation_sampler import PartialDesaturationSampler
from dem_stats import dem_stats, print_dem_stats

_ESSENTIALS_TYPES = ("canonical", "causal", "blending", "meangap", "logical_ambiguity")


class _OrderedOp(argparse.Action):
    """Collect --augment/--prune/--closure into namespace.ops in command-line
    order, which separate argparse lists would lose. Entries are
    ("augment"|"prune", TYPE, K) or ("closure", None, TAU)."""
    def __call__(self, parser, namespace, value, option_string=None):
        ops = getattr(namespace, "ops", None) or []
        if option_string == "--closure":
            tau = 0.0 if value is None else float(value)
            if tau < 0:
                parser.error(f"--closure TAU must be >= 0; got {tau}")
            ops.append(("closure", None, tau))
            namespace.ops = ops
            return
        op = "augment" if option_string == "--augment" else "prune"
        try:
            type_, k_str = value.split(":")
            k = int(k_str)
        except ValueError:
            parser.error(f"{option_string} expects TYPE:K (e.g. canonical:50); got {value!r}")
        if type_ not in _ESSENTIALS_TYPES:
            parser.error(f"{option_string} type must be one of {_ESSENTIALS_TYPES}; got {type_!r}")
        if k < 0:
            parser.error(f"{option_string} K must be >= 0; got {k}")
        ops.append((op, type_, k))
        namespace.ops = ops


def add_mask_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the mask flags: base, ordered ops, essentials files, contraction.
    [used: main, experiments/run_eval_partial_mask.py]"""
    # --- base mask ---
    parser.add_argument("--base", choices=["logical_ambiguity", "region", "list"],
                        default="logical_ambiguity",
                        help="Base mask M0: TopK(Q) of the logical-ambiguity essentials "
                             "(default; needs --base-size), a geometric region "
                             "(--left-len/--top-len/--t) or an explicit detector list "
                             "(--detectors).")
    parser.add_argument("--base-size", type=int, default=None, metavar="K_Q",
                        help="logical_ambiguity base: number of top-Q detectors.")
    parser.add_argument("--q-type", choices=["all", "exp"], default="exp",
                        help="logical_ambiguity base: Q_exp (default; exp(-|w_on-w_off|)-"
                             "weighted) or Q_all (unweighted) ranking. The two were found "
                             "equivalent end-to-end.")
    parser.add_argument("--detectors", type=pathlib.Path, default=None,
                        help="list base: JSON list of detector ids, or .npy (bool mask or int ids).")
    # --- geometric base region (--base region) ---
    parser.add_argument("--left-len", nargs=2, type=int, default=None,
                        metavar=("START", "END"), help="Row-rank interval [START, END).")
    parser.add_argument("--top-len", nargs=2, type=int, default=None,
                        metavar=("START", "END"), help="Column-rank interval [START, END).")
    parser.add_argument("--t", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Time-slice interval [START, END), 1-indexed from the "
                             "first hybrid slice.")
    parser.add_argument("--mode", choices=["rectangle", "triangle"], default="rectangle")
    parser.add_argument("--triangle-half", choices=["left", "right"], default="left")
    parser.add_argument("--stage-name", default="escape")
    # --- ordered essentials ops ---
    parser.add_argument("--augment", action=_OrderedOp, metavar="TYPE:K", default=None,
                        help="Add the top-K TYPE-essentials detectors not already in "
                             "the mask. Repeatable; applied in command-line order "
                             "together with --prune. Omit for no augmentation.")
    parser.add_argument("--prune", action=_OrderedOp, metavar="TYPE:K", default=None,
                        help="Remove the K lowest-scoring TYPE-essentials detectors "
                             "from the mask. Repeatable, order-preserving.")
    parser.add_argument("--closure", action=_OrderedOp, nargs="?", metavar="TAU", default=None,
                        help="Add every hidden detector whose inclusion lowers the DEM "
                             "crossing mass S(M) by more than TAU (default 0), repeated to "
                             "a fixed point. Repeatable, order-preserving with "
                             "--augment/--prune.")
    parser.add_argument("--essentials-dir", type=pathlib.Path, default=None,
                        help="Where to look for essentials JSONs (default: --out).")
    parser.add_argument("--essentials-canonical", type=pathlib.Path, default=None,
                        help="Explicit canonical essentials file (overrides discovery).")
    parser.add_argument("--essentials-causal", type=pathlib.Path, default=None)
    parser.add_argument("--essentials-meangap", type=pathlib.Path, default=None)
    parser.add_argument("--essentials-logical-ambiguity", type=pathlib.Path, default=None)
    parser.add_argument("--blending-alpha", type=float, default=None,
                        help="Alpha for TYPE=blending ops (requires canonical and "
                             "causal essentials; computed on the fly, no file).")
    # --- contraction ---
    parser.add_argument("--contract", action="store_true",
                        help="Run the chain-aware DEM contraction for this mask.")
    parser.add_argument("--weight-cutoff", type=float, default=15.0)
    parser.add_argument("--connectivity-fix", choices=["per_detector", "union_find"],
                        default="per_detector")


def validate_mask_args(parser: argparse.ArgumentParser, args) -> None:
    if args.base == "region" and any(v is None for v in (args.left_len, args.top_len, args.t)):
        parser.error("--base region requires --left-len, --top-len and --t")
    if args.base == "logical_ambiguity" and (args.base_size is None or args.base_size <= 0):
        parser.error("--base logical_ambiguity (the default) requires --base-size K_Q > 0")
    if args.base == "list" and args.detectors is None:
        parser.error("--base list requires --detectors FILE")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="Circuit data folder (must contain the artifacts "
                             "written by get_desaturated_dem_with_obs_detector.py); "
                             "the partial_<tag>/ subfolder is written into it.")
    add_mask_arguments(parser)
    return parser


def resolve_circuit(out_dir: pathlib.Path) -> tuple[stim.Circuit, str, dict]:
    """Load the folder's circuit via postselected_detectors.json -> circuit_file.
    Returns (circuit, stem, folder_meta)."""
    meta_path = out_dir / "postselected_detectors.json"
    if not meta_path.is_file():
        sys.exit(f"error: {meta_path} not found - run "
                 f"get_desaturated_dem_with_obs_detector.py --circuit ... --out {out_dir} first.")
    folder_meta = json.loads(meta_path.read_text())
    circuit_path = out_dir / folder_meta["circuit_file"]
    if not circuit_path.is_file():
        sys.exit(f"error: circuit file {circuit_path} named by "
                 f"postselected_detectors.json is missing.")
    name = circuit_path.name
    stem = name[: name.index(".stim")] if ".stim" in name else circuit_path.stem
    return stim.Circuit.from_file(circuit_path), stem, folder_meta


def discover_essentials(kind: str, ess_dir: pathlib.Path,
                        explicit: "pathlib.Path | None") -> pathlib.Path:
    """Explicit path if given, else the unique '*_<kind>_*.json' in ess_dir."""
    if explicit is not None:
        if not explicit.is_file():
            sys.exit(f"error: essentials file not found: {explicit}")
        return explicit
    matches = sorted(ess_dir.glob(f"*_{kind}_*.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        sys.exit(f"error: no {kind} essentials file (*_{kind}_*.json) in {ess_dir}; "
                 f"generate one with gen_essentials.py or pass --essentials-{kind}.")
    sys.exit(f"error: multiple {kind} essentials files in {ess_dir}: "
             f"{[m.name for m in matches]}; disambiguate with --essentials-{kind}.")


def _load_detector_list(path: pathlib.Path, num_detectors: int) -> np.ndarray:
    """Detector ids from a JSON list, a .npy bool mask, or a .npy int array."""
    if not path.is_file():
        sys.exit(f"error: detector list file not found: {path}")
    if path.suffix == ".npy":
        arr = np.load(path)
        ids = np.flatnonzero(arr) if arr.dtype == np.bool_ else np.asarray(arr, dtype=np.int64)
    else:
        ids = np.asarray(json.loads(path.read_text()), dtype=np.int64)
    ids = np.unique(ids)
    if ids.size == 0 or ids.min() < 0 or ids.max() >= num_detectors:
        sys.exit(f"error: detector ids in {path} must be non-empty and within [0, {num_detectors}).")
    return ids


def _ensure_essentials(builder: PartialMaskBuilder, kind: str, args, loaded: set[str],
                       ess_dir: pathlib.Path) -> pathlib.Path:
    """Load essentials `kind` into the builder once. Returns the file path."""
    path = discover_essentials(kind, ess_dir, getattr(args, f"essentials_{kind}", None))
    if kind not in loaded:
        builder.load_essentials_from_json(path, type=kind)
        print(f"[mask] loaded {kind} essentials: {path.name}")
        loaded.add(kind)
    return path


def build_base(builder: PartialMaskBuilder, args, loaded: set[str],
               ess_dir: pathlib.Path) -> tuple[np.ndarray, dict, dict]:
    """Base mask M0 by --base. Returns (mask_bool, svg_meta, base_spec)."""
    if args.base == "region":
        mask, _, meta = builder.build_partial_region_mask(
            left_len_start=args.left_len[0], left_len_end=args.left_len[1],
            top_len_start=args.top_len[0], top_len_end=args.top_len[1],
            t_start=args.t[0], t_end=args.t[1],
            mode=args.mode, triangle_half=args.triangle_half,
            stage_name=args.stage_name,
        )
        spec = {"type": "region", "left_len": args.left_len, "top_len": args.top_len, "t": args.t,
                "mode": args.mode, "triangle_half": args.triangle_half, "stage_name": args.stage_name}
    elif args.base == "logical_ambiguity":
        path = _ensure_essentials(builder, "logical_ambiguity", args, loaded, ess_dir)
        mask, _, _ = builder.logical_ambiguity_base_mask(args.base_size, q_type=args.q_type)
        _, _, meta = builder.build_mask_from_indices(np.flatnonzero(mask))
        spec = {"type": "logical_ambiguity", "K_Q": int(args.base_size), "q_type": args.q_type,
                "essentials_file": path.name,
                "essentials_shots": int(builder.logical_ambiguity_essentials_meta.get("shots", 0))}
    else:
        ids = _load_detector_list(args.detectors, builder.num_detectors)
        mask, _, meta = builder.build_mask_from_indices(ids)
        spec = {"type": "list", "detectors_file": str(args.detectors), "n": int(ids.size)}
    print(f"[mask] base {spec['type']}: {int(mask.sum())}/{builder.num_detectors} detectors")
    return mask, meta, spec


def build_mask(builder: PartialMaskBuilder, args, ess_dir: pathlib.Path,
               on_stage=None) -> tuple[np.ndarray, dict, dict, list[dict]]:
    """Build the base mask, then apply the ops in order. Returns
    (mask_bool, svg_meta, base_spec, op_log). `on_stage(label, mask)`, if
    given, is called after the base and after each op."""
    loaded: set[str] = set()
    mask, base_meta, base_spec = build_base(builder, args, loaded, ess_dir)
    if on_stage is not None:
        on_stage(f"base:{base_spec['type']}", mask)

    ops = getattr(args, "ops", None) or []
    op_log: list[dict] = []
    for op, type_, k in ops:
        if op == "closure":
            mask, _, n, info = builder.closure_mask(mask, tau=k, return_info=True)
            print(f"[mask] closure(tau={k:g}) -> +{n} in {info['iterations']} iterations, "
                  f"S {info['S_before']:.3f} -> {info['S_after']:.3f}, "
                  f"mask = {int(mask.sum())}/{builder.num_detectors}")
            op_log.append({"op": "closure", "tau": float(k), "applied": int(n),
                           "iterations": int(info["iterations"]),
                           "S_before": float(info["S_before"]), "S_after": float(info["S_after"]),
                           "converged": bool(info["converged"]),
                           "mask_sum_after": int(mask.sum())})
            if on_stage is not None:
                on_stage(f"closure(tau={k:g})", mask)
            continue
        need = ("canonical", "causal") if type_ == "blending" else (type_,)
        for t in need:
            _ensure_essentials(builder, t, args, loaded, ess_dir)
        if type_ == "blending" and "blending" not in loaded:
            if args.blending_alpha is None:
                sys.exit("error: TYPE=blending requires --blending-alpha.")
            builder.compute_blending_essentials_order(alpha=args.blending_alpha)
            loaded.add("blending")

        if op == "augment":
            mask, _, n = builder.augment_mask_with_essentials(mask, K=k, type=type_)
        else:
            mask, _, n = builder.prune_mask_with_essentials(mask, K=k, type=type_)
        print(f"[mask] {op} {type_}:{k} -> {'+' if op == 'augment' else '-'}{n}, "
              f"mask = {int(mask.sum())}/{builder.num_detectors}")
        op_log.append({"op": op, "type": type_, "K": k, "applied": int(n),
                       "mask_sum_after": int(mask.sum())})
        if on_stage is not None:
            on_stage(f"{op} {type_}:{k}", mask)
    return mask, base_meta, base_spec, op_log


def param_tag(args, base_spec: dict) -> str:
    if base_spec["type"] == "region":
        tag = (f"l{args.left_len[0]}-{args.left_len[1]}"
               f"_t{args.top_len[0]}-{args.top_len[1]}"
               f"_ts{args.t[0]}-{args.t[1]}")
        if args.mode == "triangle":
            tag += f"_tri{args.triangle_half}"
    elif base_spec["type"] == "logical_ambiguity":
        tag = f"q{base_spec['q_type']}{base_spec['K_Q']}"
    else:
        tag = f"list{base_spec['n']}-{base_spec['ids_sha']}"
    for op, type_, k in (getattr(args, "ops", None) or []):
        if op == "closure":
            tag += "_cl" if k == 0 else f"_cl{k:g}"
            continue
        short = {"augment": "aug", "prune": "pr"}[op]
        tag += f"_{short}-{type_}{k}"
        if type_ == "blending" and args.blending_alpha is not None:
            tag += f"a{args.blending_alpha:g}"
    if args.contract:
        tag += f"_wc{args.weight_cutoff:g}"
        if args.connectivity_fix != "per_detector":
            tag += "_uf"
    return tag


def _needed_rounds_info(out_dir: pathlib.Path, compiled) -> dict:
    """First and last measurement round the kept detectors depend on, without
    the obs detector (gap decoding forces its bit). rounds_saved_at_end > 0
    means the mask needs no data from that many final rounds."""
    import layer_schedule
    gap = stim.Circuit.from_file(pathlib.Path(out_dir) / "gap_circuit.stim")
    import json as _json
    ps = _json.loads((pathlib.Path(out_dir) / "postselected_detectors.json").read_text())
    obs_orig = int(ps["obs_detector_index"]) if isinstance(ps, dict) else None
    return layer_schedule.needed_rounds(gap, [int(k) for k in compiled.kept],
                                        obs_detector=obs_orig)


def contract(circuit: stim.Circuit, mask: np.ndarray, args):
    """Compile and return the partial sampler. This runs the contraction and
    builds the pymatching decoder, which checks that the contracted DEM loads."""
    import sinter
    task = sinter.Task(circuit=circuit, detector_error_model=circuit.detector_error_model())
    return PartialDesaturationSampler().compiled_sampler_for_task(
        task,
        partial_mask_bool=mask,
        weight_cutoff=args.weight_cutoff,
        connectivity_fix_type=args.connectivity_fix,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_mask_args(parser, args)
    circuit, stem, folder_meta = resolve_circuit(args.out)
    print(f"circuit: {folder_meta['circuit_file']}  "
          f"({circuit.num_detectors} detectors)")

    builder = PartialMaskBuilder(circuit=circuit)
    mask, base_meta, base_spec, op_log = build_mask(builder, args, args.essentials_dir or args.out)
    if base_spec["type"] == "list":
        base_spec["ids_sha"] = hashlib.sha256(
            json.dumps(np.flatnonzero(mask).tolist()).encode()).hexdigest()[:6]

    tag = param_tag(args, base_spec)
    pdir = args.out / f"partial_{tag}"
    pdir.mkdir(parents=True, exist_ok=True)
    print(f"[out] {pdir}")

    # --- mask artifacts ---
    mask_packed = np.packbits(mask, bitorder="little")
    np.save(pdir / "mask_bool.npy", mask)
    np.save(pdir / "mask_packed.npy", mask_packed)
    (pdir / "mask_meta.json").write_text(json.dumps(base_meta, indent=2))

    # Metadata from the final mask, not the base: augmented detectors can lie
    # in time slices outside the base region, and the SVG must render them.
    _, _, final_meta = builder.build_mask_from_indices(np.flatnonzero(mask))
    builder.visualize_partial_region_mask_svg(
        pdir / "mask.svg", mask_bool=mask, metadata=final_meta)

    # --- DEM statistics per pipeline stage ---
    stats: dict = {
        "mask": {
            "num_detectors": int(builder.num_detectors),
            "mask_sum": int(mask.sum()),
            "mask_ratio": float(mask.sum() / builder.num_detectors),
            "base": base_spec,
            "ops": op_log,
            # probability-weighted crossing mass S(M), see dem_structure.py
            "S_crossing_mass": float(builder.mask_crossing_mass(mask)),
        },
    }
    if builder.logical_ambiguity_essentials_meta is not None:
        stats["mask"]["q_coverage"] = {
            q: float(builder.logical_ambiguity_coverage(mask, q_type=q)) for q in ("all", "exp")}
    original_dem = circuit.detector_error_model()
    stats["original"] = dem_stats(original_dem)
    print_dem_stats(stats["original"], label="original")

    if args.contract:
        compiled = contract(circuit, mask, args)
        stats["desaturated_with_obs"] = dem_stats(compiled.gap_dem_complete)
        print_dem_stats(stats["desaturated_with_obs"], label="desaturated+obs")
        stats["contracted"] = dem_stats(compiled.gap_dem)
        print_dem_stats(stats["contracted"], label="contracted")

        compiled.gap_dem.to_file(pdir / "contracted.dem")
        np.save(pdir / "kept.npy", np.asarray(compiled.kept, dtype=np.int64))
        (pdir / "old_to_new.json").write_text(json.dumps(
            {str(k): int(v) for k, v in compiled.old_to_new.items()}, indent=2))
        stats["contraction"] = {
            "weight_cutoff": float(args.weight_cutoff),
            "connectivity_fix_type": args.connectivity_fix,
            "n_kept": len(compiled.kept),
            "n_clipped": int(circuit.num_detectors - int(mask.sum())),
            "needed_rounds": _needed_rounds_info(args.out, compiled),
            "obs_detector_new_index": int(compiled.old_to_new[
                compiled.gap_dem_complete.num_detectors - 1]),
        }
        print(f"[contract] kept {len(compiled.kept)} detectors "
              f"(mask {int(mask.sum())} + "
              f"{len(compiled.kept) - int(mask.sum())} virtual/obs)")

    (pdir / "stats.json").write_text(json.dumps(stats, indent=2))

    # --- config: the recipe that reproduces this folder ---
    (pdir / "config.json").write_text(json.dumps({
        "circuit_file": folder_meta["circuit_file"],
        "circuit_stem": stem,
        "n_circuit_dets": int(circuit.num_detectors),
        "base": base_spec,
        # only meaningful for --base region; existing readers expect the key
        "region": {
            "left_len": args.left_len, "top_len": args.top_len, "t": args.t,
            "mode": args.mode, "triangle_half": args.triangle_half,
            "stage_name": args.stage_name,
        },
        "ops": op_log,
        "blending_alpha": args.blending_alpha,
        "essentials_dir": str(args.essentials_dir or args.out),
        "contract": bool(args.contract),
        "weight_cutoff": float(args.weight_cutoff),
        "connectivity_fix_type": args.connectivity_fix,
        "param_tag": tag,
    }, indent=2))

    written = ["mask_bool.npy", "mask_packed.npy", "mask_meta.json", "mask.svg",
               "stats.json", "config.json"]
    if args.contract:
        written[4:4] = ["contracted.dem", "kept.npy", "old_to_new.json"]
    print("wrote:")
    for f in written:
        print(f"  {f}")


if __name__ == "__main__":
    main()
