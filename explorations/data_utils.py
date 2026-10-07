"""Module-level data utilities for partial-decoder experiments.

Self-contained, class-free counterparts to the GapSensitivityCollect methods
of the same names. The compiled sampler `dec` is taken as input rather than
recreated from a circuit — caller is responsible for building it (typically
via `cultiv.DesaturationSampler().compiled_sampler_for_task(task)` or the
new `PartialDesaturationSampler().compiled_sampler_for_task(task)`).

Workflow:
    raw      = generate_raw_samples(dec, shots=N)
    complete = compute_complete_decode(dec, raw)
    samples  = {**raw, **complete, "dec": dec}
    # or, equivalently:
    samples  = generate_samples_for_comparison(dec, shots=N)

    # Cache to disk:
    save_samples_npz(samples, "samples.npz")
    samples_reloaded = load_samples_npz("samples.npz", dec=dec)

    # Run the 2-stage protocol against any partial mask:
    summary = compare_partial_gap_with_complete_gap_2stage(
        samples,
        partial_mask=mask_bool,
        gap_threshold=50.0,
        gap_threshold_low=20.0,
        gap_threshold_high=50.0,
        partial_decoder=fpd,    # optional fast path
    )
    print_2stage_summary(summary)

    # Persist the comparison summary:
    save_2stage_summary_json(summary, "summary.json")
    summary_reloaded = load_2stage_summary_json("summary.json")
"""
import json
import pathlib
from typing import Any, Optional

import numpy as np


# ====================================================================
# Sample generation
# ====================================================================

def generate_raw_samples(
    dec: Any,
    *,
    shots: int,
) -> dict[str, Any]:
    """Sample `shots` shots from `dec.gap_circuit_sampler`, postselect via
    `dec._discard_mask`, and return raw outputs without running the complete
    decoder.

    Returns a dict with keys: `shots_sampled`, `kept_shots`, `dets`,
    `actual_flip`. No `dec` field — caller already has it.
    """
    if shots <= 0:
        raise ValueError("shots must be positive.")

    dets, actual_obs = dec.gap_circuit_sampler.sample(
        shots, separate_observables=True, bit_packed=True,
    )
    keep = ~np.any(dets & dec._discard_mask, axis=1)
    dets = dets[keep]
    actual_obs = actual_obs[keep]
    n_kept = int(dets.shape[0])

    if n_kept == 0:
        return {
            "shots_sampled": int(shots),
            "kept_shots": 0,
            "dets": dets,
            "actual_flip": np.zeros(0, dtype=np.bool_),
        }

    actual_flip = actual_obs[:, 0].astype(np.bool_)
    return {
        "shots_sampled": int(shots),
        "kept_shots": n_kept,
        "dets": dets,
        "actual_flip": actual_flip,
    }


def compute_complete_decode(
    dec: Any,
    raw_samples: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete decoder on the raw samples and return a dict with
    keys: `complete_preds`, `complete_gaps`, `complete_errs`. The output is
    NOT merged with the input — caller chooses to merge.

    `dec` must be a *full-mode* compiled sampler (cultiv's class or the new
    `CompiledPartialDesaturationSampler` with `kept is None`); a partial-mode
    sampler would silently produce partial-decoder outputs instead of
    complete-decoder outputs.
    """
    required = {"shots_sampled", "kept_shots", "dets", "actual_flip"}
    missing = required - set(raw_samples.keys())
    if missing:
        raise ValueError(
            f"raw_samples missing keys {sorted(missing)}; "
            "build it via generate_raw_samples(...)."
        )

    n_kept = int(raw_samples["kept_shots"])
    if n_kept == 0:
        return {
            "complete_preds": np.zeros(0, dtype=np.bool_),
            "complete_gaps":  np.zeros(0, dtype=np.float64),
            "complete_errs":  np.zeros(0, dtype=np.bool_),
        }

    dets = raw_samples["dets"]
    actual_flip = np.asarray(raw_samples["actual_flip"], dtype=np.bool_)
    preds_c, gaps_c = dec._decode_batch_overwrite_last_byte(dets.copy())
    preds_c = preds_c.astype(np.bool_)
    gaps_c = gaps_c.astype(np.float64)
    err_c = preds_c ^ actual_flip
    return {
        "complete_preds": preds_c,
        "complete_gaps": gaps_c,
        "complete_errs": err_c,
    }


def generate_samples_for_comparison(
    dec: Any,
    *,
    shots: int,
) -> dict[str, Any]:
    """Wrapper: generate_raw_samples + compute_complete_decode merged into a
    single dict, plus the `dec` field. Output matches the legacy
    GapSensitivityCollect.generate_samples_for_comparison schema.
    """
    raw = generate_raw_samples(dec, shots=shots)
    complete = compute_complete_decode(dec, raw)
    return {**raw, **complete, "dec": dec}


# ====================================================================
# NPZ persistence
# ====================================================================

_RAW_KEYS = ("shots_sampled", "kept_shots", "dets", "actual_flip")
_FULL_KEYS = _RAW_KEYS + ("complete_preds", "complete_gaps", "complete_errs")


def save_raw_samples_npz(
    raw_samples: dict[str, Any],
    filepath,
) -> pathlib.Path:
    """Persist raw samples (output of `generate_raw_samples`) to a compressed
    .npz file. Auto-creates parent directories. Returns the resolved path
    written to.
    """
    missing = set(_RAW_KEYS) - set(raw_samples.keys())
    if missing:
        raise ValueError(
            f"raw_samples missing keys {sorted(missing)}; "
            "build it via generate_raw_samples(...)."
        )
    path = pathlib.Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shots_sampled=np.int64(raw_samples["shots_sampled"]),
        kept_shots=np.int64(raw_samples["kept_shots"]),
        dets=np.asarray(raw_samples["dets"], dtype=np.uint8),
        actual_flip=np.asarray(raw_samples["actual_flip"], dtype=np.bool_),
    )
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    return path


def load_raw_samples_npz(filepath) -> dict[str, Any]:
    """Load a raw-samples dict previously written by `save_raw_samples_npz`.
    No `dec` is required — caller supplies it later if needed.
    """
    path = pathlib.Path(filepath)
    z = np.load(path, allow_pickle=False)
    missing = set(_RAW_KEYS) - set(z.files)
    if missing:
        raise ValueError(
            f"npz at {path} is missing raw-samples keys {sorted(missing)}."
        )
    return {
        "shots_sampled": int(z["shots_sampled"]),
        "kept_shots": int(z["kept_shots"]),
        "dets": np.asarray(z["dets"], dtype=np.uint8),
        "actual_flip": np.asarray(z["actual_flip"], dtype=np.bool_),
    }


def save_samples_npz(
    samples: dict[str, Any],
    filepath,
) -> pathlib.Path:
    """Persist a full samples dict (raw + complete decoder outputs) to a
    compressed .npz file. The `dec` field is dropped (not serializable).
    Auto-creates parent directories. Returns the resolved path written to.
    """
    missing = set(_FULL_KEYS) - set(samples.keys())
    if missing:
        raise ValueError(
            f"samples dict missing keys {sorted(missing)}; "
            "build it via generate_samples_for_comparison(...)."
        )
    path = pathlib.Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        shots_sampled=np.int64(samples["shots_sampled"]),
        kept_shots=np.int64(samples["kept_shots"]),
        dets=np.asarray(samples["dets"], dtype=np.uint8),
        actual_flip=np.asarray(samples["actual_flip"], dtype=np.bool_),
        complete_preds=np.asarray(samples["complete_preds"], dtype=np.bool_),
        complete_gaps=np.asarray(samples["complete_gaps"], dtype=np.float64),
        complete_errs=np.asarray(samples["complete_errs"], dtype=np.bool_),
    )
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    return path


def load_samples_npz(
    filepath,
    *,
    dec: Any,
) -> dict[str, Any]:
    """Load a full samples dict previously written by `save_samples_npz`,
    re-injecting the supplied `dec` (not serialized). Caller is responsible
    for ensuring `dec` was built from the same circuit as the saved file —
    no automatic check, since the circuit isn't serialized.
    """
    path = pathlib.Path(filepath)
    z = np.load(path, allow_pickle=False)
    missing = set(_FULL_KEYS) - set(z.files)
    if missing:
        raise ValueError(
            f"npz at {path} is missing samples keys {sorted(missing)}."
        )
    return {
        "shots_sampled": int(z["shots_sampled"]),
        "kept_shots": int(z["kept_shots"]),
        "dets": np.asarray(z["dets"], dtype=np.uint8),
        "actual_flip": np.asarray(z["actual_flip"], dtype=np.bool_),
        "complete_preds": np.asarray(z["complete_preds"], dtype=np.bool_),
        "complete_gaps": np.asarray(z["complete_gaps"], dtype=np.float64),
        "complete_errs": np.asarray(z["complete_errs"], dtype=np.bool_),
        "dec": dec,
    }


# ====================================================================
# 2-stage gating protocol
# ====================================================================

def _normalize_mask_to_packed(
    mask: np.ndarray,
    *,
    num_detectors: int,
    width: int,
) -> np.ndarray:
    """Coerce `mask` (bool array of length num_detectors, or already-packed
    uint8 array) into a packed-uint8 array of length `width`.
    """
    arr = np.asarray(mask)
    if arr.dtype == np.bool_ or arr.dtype == bool:
        if arr.shape[0] != num_detectors:
            raise ValueError(
                f"bool mask must have length num_detectors={num_detectors}, "
                f"got {arr.shape[0]}."
            )
        packed = np.packbits(arr.astype(np.uint8), bitorder="little")
    else:
        packed = np.asarray(arr, dtype=np.uint8)
    out = np.zeros(width, dtype=np.uint8)
    copy_n = min(len(packed), len(out))
    out[:copy_n] = packed[:copy_n]
    return out


def _packed_mask_size(packed_mask: np.ndarray, num_detectors: int) -> int:
    unpacked = np.unpackbits(packed_mask, bitorder="little")
    return int(np.count_nonzero(unpacked[:num_detectors]))


def compare_partial_gap_with_complete_gap_2stage(
    samples: dict[str, Any],
    *,
    partial_mask: np.ndarray,
    gap_threshold: float,
    gap_threshold_low: float,
    gap_threshold_high: float,
    partial_decoder: Optional[Any] = None,
) -> dict[str, Any]:
    """2-stage gating protocol over pre-generated samples.

    For each kept shot, the partial gap routes it to one of three tiers:
        gap_p < T_low                  -> tier 1: cheap reject (complete NEVER runs)
        T_low <= gap_p <= T_high       -> tier 2: rescue band; run complete and
                                          accept iff gap_c >= T_complete
        gap_p > T_high                 -> tier 3: direct accept (complete runs ONLY
                                          for prediction)
    All accepted shots use the complete decoder's prediction.

    If `partial_decoder` is provided (e.g. a `CompiledPartialDesaturationSampler`
    in partial mode, or the legacy `FastPartialDecoder`), the partial decode
    uses its `_decode_batch_overwrite_last_byte` (or `decode_batch`) method
    instead of the slow syndrome-suppression path. The `partial_mask` is
    still required for mask-coverage statistics in the summary.

    Returns the summary dict.
    """
    required = {"shots_sampled", "kept_shots", "dets", "actual_flip",
                "complete_preds", "complete_gaps", "complete_errs", "dec"}
    missing = required - set(samples.keys())
    if missing:
        raise ValueError(
            f"samples dict missing keys {sorted(missing)}; "
            "build it via generate_samples_for_comparison(...)."
        )
    if not (gap_threshold_low <= gap_threshold_high):
        raise ValueError(
            f"gap_threshold_low ({gap_threshold_low}) must be <= "
            f"gap_threshold_high ({gap_threshold_high})."
        )

    thr_c = float(gap_threshold)
    t_lo = float(gap_threshold_low)
    t_hi = float(gap_threshold_high)

    n_shots = int(samples["shots_sampled"])
    n_kept = int(samples["kept_shots"])
    dets = samples["dets"]
    gaps_c = samples["complete_gaps"]
    err_c = samples["complete_errs"]
    dec = samples["dec"]
    n_total_dets = int(dec.gap_circuit.num_detectors) - 1
    # ^ subtract 1 because gap_circuit has the obs detector appended;
    # the partial mask is over original circuit detectors. For cultiv-style
    # samplers num_dets = n_circuit_dets + n_virtual_pairs + 1; the original
    # circuit's detectors are at indices [0, n_circuit_dets). The mask shape
    # must match the original circuit count.
    if isinstance(partial_mask, np.ndarray) and partial_mask.dtype == np.bool_:
        n_circuit_dets = partial_mask.shape[0]
    else:
        n_circuit_dets = n_total_dets   # assume packed-mask follows the gap_circuit width
    width = dets.shape[1] if n_kept > 0 else (-(-dec.gap_circuit.num_detectors // 8))

    full_mask = _normalize_mask_to_packed(
        partial_mask, num_detectors=n_circuit_dets, width=width,
    )
    m = _packed_mask_size(full_mask, n_circuit_dets)
    ratio = (m / n_circuit_dets) if n_circuit_dets > 0 else 0.0

    def _safe_div(num: int, den: int) -> float:
        return float(num) / float(den) if den > 0 else float("nan")

    if n_kept == 0:
        return {
            "shots_sampled": n_shots,
            "kept_shots": 0,
            "complete_total_detectors": n_circuit_dets,
            "partial_mask_size": m,
            "partial_mask_ratio": ratio,
            "gap_threshold": thr_c,
            "gap_threshold_low": t_lo,
            "gap_threshold_high": t_hi,
            "tier1_count": 0, "tier1_rate": float("nan"),
            "tier2_count": 0, "tier2_rate": float("nan"),
            "tier2_accept_count": 0, "tier2_accept_rate": float("nan"),
            "tier2_reject_count": 0, "tier2_reject_rate": float("nan"),
            "tier3_count": 0, "tier3_rate": float("nan"),
            "accept_count": 0, "accept_rate": float("nan"),
            "reject_count": 0, "reject_rate": float("nan"),
            "complete_for_gap_count": 0, "complete_for_gap_rate": float("nan"),
            "complete_for_pred_only_count": 0, "complete_for_pred_only_rate": float("nan"),
            "complete_decode_count": 0, "complete_decode_rate": float("nan"),
            "complete_decode_savings": float("nan"),
            "gating_decision_count": 0, "gating_decision_rate": float("nan"),
            "gating_decision_savings": float("nan"),
            "errors": 0,
            "LER_over_accept": float("nan"),
            "LER_over_kept": float("nan"),
            "LER_over_shots": float("nan"),
            "complete_accept_count": 0, "complete_accept_rate": float("nan"),
            "complete_reject_count": 0, "complete_reject_rate": float("nan"),
            "complete": {
                "errors": 0,
                "LER_over_accept": float("nan"),
                "LER_over_kept": float("nan"),
                "LER_over_shots": float("nan"),
            },
            "two_stage": {
                "errors": 0,
                "LER_over_accept": float("nan"),
                "LER_over_kept": float("nan"),
                "LER_over_shots": float("nan"),
            },
            "warning": "no shots survived postselect",
        }

    # Partial decode: either via partial_decoder (fast path) or via mask-then-
    # decode through the full DEM (slow syndrome-suppression).
    if partial_decoder is None:
        d_p = dets.copy()
        d_p &= full_mask.reshape(1, -1)
        _, gaps_p = dec._decode_batch_overwrite_last_byte(d_p.copy())
    else:
        # Duck-typed: try _decode_batch_overwrite_last_byte (CompiledPartialDesaturationSampler
        # convention) then decode_batch (legacy FastPartialDecoder convention).
        if hasattr(partial_decoder, "_decode_batch_overwrite_last_byte"):
            _, gaps_p = partial_decoder._decode_batch_overwrite_last_byte(dets.copy())
        elif hasattr(partial_decoder, "decode_batch"):
            _, gaps_p = partial_decoder.decode_batch(dets)
        else:
            raise TypeError(
                "partial_decoder must expose either _decode_batch_overwrite_last_byte "
                "or decode_batch."
            )
    gaps_p = gaps_p.astype(np.float64)

    # Tier masks
    tier1_m = gaps_p < t_lo
    tier2_m = (gaps_p >= t_lo) & (gaps_p <= t_hi)
    tier3_m = gaps_p > t_hi

    tier2_accept_m = tier2_m & (gaps_c >= thr_c)
    tier2_reject_m = tier2_m & (gaps_c <  thr_c)

    accept_m = tier2_accept_m | tier3_m

    tier1 = int(tier1_m.sum())
    tier2 = int(tier2_m.sum())
    tier2_accept = int(tier2_accept_m.sum())
    tier2_reject = int(tier2_reject_m.sum())
    tier3 = int(tier3_m.sum())

    accept = int(accept_m.sum())
    reject = n_kept - accept

    complete_for_gap = tier2          # tier 2 uses complete gap to gate
    complete_for_pred_only = tier3    # tier 3 uses complete only for prediction
    complete_decode_count = complete_for_gap + complete_for_pred_only

    # All accepted shots use complete prediction; errors come from err_c.
    errors = int(err_c[accept_m].sum())

    # Complete-only reference (gap_c >= T_complete): for printing alongside 2-stage.
    ca_m = gaps_c >= thr_c
    complete_accept = int(ca_m.sum())
    complete_reject = n_kept - complete_accept
    complete_errors = int(err_c[ca_m].sum())

    return {
        "shots_sampled": n_shots,
        "kept_shots": n_kept,
        "complete_total_detectors": n_circuit_dets,
        "partial_mask_size": m,
        "partial_mask_ratio": ratio,
        "gap_threshold": thr_c,
        "gap_threshold_low": t_lo,
        "gap_threshold_high": t_hi,

        "tier1_count": tier1,
        "tier1_rate":  tier1 / n_kept,
        "tier2_count": tier2,
        "tier2_rate":  tier2 / n_kept,
        "tier2_accept_count": tier2_accept,
        "tier2_accept_rate":  tier2_accept / n_kept,
        "tier2_reject_count": tier2_reject,
        "tier2_reject_rate":  tier2_reject / n_kept,
        "tier3_count": tier3,
        "tier3_rate":  tier3 / n_kept,

        "accept_count": accept,
        "accept_rate":  accept / n_kept,
        "reject_count": reject,
        "reject_rate":  reject / n_kept,

        "complete_for_gap_count": complete_for_gap,
        "complete_for_gap_rate":  complete_for_gap / n_kept,
        "complete_for_pred_only_count": complete_for_pred_only,
        "complete_for_pred_only_rate":  complete_for_pred_only / n_kept,
        "complete_decode_count": complete_decode_count,
        "complete_decode_rate":  complete_decode_count / n_kept,
        "complete_decode_savings": 1.0 - complete_decode_count / n_kept,
        # Fraction of kept shots whose accept/reject decision is reached via
        # partial gap alone (tier 1 reject + tier 3 direct accept).
        "gating_decision_count":   tier1 + tier3,
        "gating_decision_rate":    (tier1 + tier3) / n_kept,
        "gating_decision_savings": (tier1 + tier3) / n_kept,

        "errors": errors,
        "LER_over_accept": _safe_div(errors, accept),
        "LER_over_kept":   _safe_div(errors, n_kept),
        "LER_over_shots":  _safe_div(errors, n_shots),

        "complete_accept_count": complete_accept,
        "complete_accept_rate":  complete_accept / n_kept,
        "complete_reject_count": complete_reject,
        "complete_reject_rate":  complete_reject / n_kept,
        "complete": {
            "errors": complete_errors,
            "LER_over_accept": _safe_div(complete_errors, complete_accept),
            "LER_over_kept":   _safe_div(complete_errors, n_kept),
            "LER_over_shots":  _safe_div(complete_errors, n_shots),
        },
        "two_stage": {
            "errors": errors,
            "LER_over_accept": _safe_div(errors, accept),
            "LER_over_kept":   _safe_div(errors, n_kept),
            "LER_over_shots":  _safe_div(errors, n_shots),
        },
    }


def print_2stage_summary(summary: dict[str, Any]) -> None:
    """Pretty-print the dict returned by `compare_partial_gap_with_complete_gap_2stage`."""
    d = summary
    n_total = int(d["complete_total_detectors"])
    m = int(d["partial_mask_size"])
    r = float(d["partial_mask_ratio"])
    n_kept = int(d["kept_shots"])
    n_shots = int(d["shots_sampled"])

    print(f"shots_sampled = {n_shots}")
    print(f"kept_shots    = {n_kept}")
    print(f"partial_mask_size: {m}/{n_total} = {r:.6f}")
    print(
        f"thresholds: T_low = {d['gap_threshold_low']}, "
        f"T_high = {d['gap_threshold_high']}, "
        f"T_complete = {d['gap_threshold']}"
    )

    if n_kept == 0:
        print("\n(no shots survived postselect)")
        return

    print(f"\nTier breakdown:")
    print(f"  tier 1 (gap_p < T_low, cheap reject):                 "
          f"{d['tier1_count']}/{n_kept} = {d['tier1_rate']:.6f}")
    print(f"  tier 2 (T_low <= gap_p <= T_high, rescue):            "
          f"{d['tier2_count']}/{n_kept} = {d['tier2_rate']:.6f}")
    print(f"      tier 2 accept (gap_c >= T_complete):              "
          f"{d['tier2_accept_count']}/{n_kept} = {d['tier2_accept_rate']:.6f}")
    print(f"      tier 2 reject (gap_c <  T_complete):              "
          f"{d['tier2_reject_count']}/{n_kept} = {d['tier2_reject_rate']:.6f}")
    print(f"  tier 3 (gap_p > T_high, direct accept):               "
          f"{d['tier3_count']}/{n_kept} = {d['tier3_rate']:.6f}")

    print(f"\nComplete-decoder usage:")
    print(f"  used for gap (tier 2):                                "
          f"{d['complete_for_gap_count']}/{n_kept} = {d['complete_for_gap_rate']:.6f}")
    print(f"  used only for prediction (tier 3):                    "
          f"{d['complete_for_pred_only_count']}/{n_kept} = {d['complete_for_pred_only_rate']:.6f}")
    print(f"  total complete-decoder calls (tier 2 + tier 3):       "
          f"{d['complete_decode_count']}/{n_kept} = {d['complete_decode_rate']:.6f}")
    print(f"  full savings (= tier 1, complete never runs):         "
          f"{d['tier1_count']}/{n_kept} = {d['complete_decode_savings']:.6f}")
    print(f"  gating-decision savings (= tier 1 + tier 3):          "
          f"{d['gating_decision_count']}/{n_kept} = {d['gating_decision_savings']:.6f}")

    ca = int(d["complete_accept_count"]); cr = int(d["complete_reject_count"])
    ta = int(d["accept_count"]);          tr = int(d["reject_count"])

    print(
        f"\nComplete-gating accept: {ca}/{n_kept} ({d['complete_accept_rate']:.6f}); "
        f"reject: {cr}/{n_kept} ({d['complete_reject_rate']:.6f})"
    )
    print(
        f"2-stage gating  accept: {ta}/{n_kept} ({d['accept_rate']:.6f}); "
        f"reject: {tr}/{n_kept} ({d['reject_rate']:.6f})"
    )

    def _print_scheme(title: str, accepts: int, block: dict[str, Any]) -> None:
        e = int(block["errors"])
        print(f"\n--- {title} ---")
        print(f"errors        = {e}")
        print(f"LER over accept = {e}/{accepts} = {block['LER_over_accept']:.6f}")
        print(f"LER over kept   = {e}/{n_kept} = {block['LER_over_kept']:.6f}")
        print(f"LER over shots  = {e}/{n_shots} = {block['LER_over_shots']:.6f}")

    _print_scheme(
        "Complete gating, complete prediction (reference)",
        ca, d["complete"],
    )
    _print_scheme(
        "2-stage gating, COMPLETE prediction (the protocol)",
        ta, d["two_stage"],
    )


def save_2stage_summary_json(
    summary: dict[str, Any],
    filepath,
) -> pathlib.Path:
    """Persist a 2-stage comparison summary to a JSON file. Auto-creates
    parent directories. Returns the resolved path written to.
    """
    path = pathlib.Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=2)
    return path


def load_2stage_summary_json(filepath) -> dict[str, Any]:
    """Load a 2-stage comparison summary from a JSON file."""
    path = pathlib.Path(filepath)
    with path.open("r") as f:
        return json.load(f)
