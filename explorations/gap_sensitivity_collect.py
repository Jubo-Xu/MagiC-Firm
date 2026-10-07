import collections
import heapq
import math
import pathlib
import sys
import tempfile
from typing import Any

from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import pymatching
import sinter
import stim

src_path = pathlib.Path(__file__).parent.parent / 'magic_state_cultivation' / 'upstream' / 'src'
assert src_path.exists()
sys.path.append(str(src_path))

import gen
import cultiv
from cultiv._decoding._desaturation_sampler import clipped_matchable_dem

# The class of the Gap collect
class GapSensitivityCollect:
    def __init__(
        self, 
        *,
        circuit: Optional[stim.Circuit] = None,
        circuit_generator: Optional[Any] = None
    ):
        if circuit_generator is not None:
            if getattr(circuit_generator, "ideal_circuit", None) is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = getattr(circuit_generator, "noisy_circuit", None) or circuit_generator.ideal_circuit
        if circuit is None:
            raise ValueError("Provide either circuit or circuit_generator.")
        
        self.circuit = circuit
        
        codes = gen.circuit_to_cycle_code_slices(circuit)
        self.ticks = codes.keys()
        self.codes = [code.with_transformed_coords(lambda e: e * (1 + 1j)) for code in codes.values()]

        self.single_dec_gap_scores = []
        self.overall_gap_scores = {
            "count": [],
            "sum_gap": [],
            "sum_err": [],
            "mean_gap_when_active": [],
            "error_rate_when_active": [],
        }
        # Causal-gap-sensitivity score: per detector d, average of |gap_with_d - gap_without_d|
        # across shots where d fired. Asks "how much of the gap signal does d's firing carry?"
        # Populated by collect_causal_gap_sens(...).
        self.causal_gap_scores = {
            "count": [],                    # shots where d fired (== n_samples_per_det)
            "sum_abs_delta": [],
            "sum_signed_delta": [],
            "causal_score_abs": [],         # mean(|delta|) when active
            "causal_score_signed": [],      # mean(delta) when active; positive => d firing usually lowers gap
            "subsample_size": 0,
        }
        self.partial_gap_sens_data: dict[str, Any] = {
            "remained_shots": 0,
            "complete_gaps": [],
            "partial_gaps": [],
            "closest_matches": [],
            "combos": [],
            "left_lengths": None,
            "top_lengths": None,
            "t_lengths": None,
            "mode": None,
            "stage_name": None,
            "complete_accept_and_reject": [],
            "closest_partial_accept_and_reject": [],
            "weighted_multi_partial_accept_and_reject": [],
            "pure_avg_accept_and_reject": [],
            "pure_avg_adjust_threshold_accept_and_reject": [],
            "gap_average_parameters": [],
        }
        self.best_combo_distribution: Optional[dict[str, Any]] = None
        # Flex pipeline (input-mask driven, no best_combo, no selector). Independent state from
        # the legacy partial_gap_sens_data / best_combo_distribution above.
        self.flex_partial_gap_sens_data: Optional[dict[str, Any]] = None
        self.flex_distribution: Optional[dict[str, Any]] = None
        # Summary from compare_partial_gap_with_complete_gap (separate from the flex pipeline).
        self.compare_partial_complete_summary: Optional[dict[str, Any]] = None
        # Summary from the 2-stage variant compare_partial_gap_with_complete_gap_2stage_from_samples.
        self.compare_partial_complete_summary_2stage: Optional[dict[str, Any]] = None
        # Per-shot (gap, is_error) arrays for shot_gap_1D_scatter plotting,
        # populated by collect_shot_gap_1D_scatter_from_samples.
        self.shot_gap_1D_scatter_data: Optional[dict[str, Any]] = None

    # Helper: unpack one bit-packed detector row into 0/1 array up to num_detectors.
    def _unpack_det_row(self, row_packed: np.ndarray) -> np.ndarray:
        unpacked = np.unpackbits(np.asarray(row_packed, dtype=np.uint8), bitorder="little")
        return unpacked[: self.circuit.num_detectors].astype(np.uint8, copy=False)

    # Helper: detector density = number of non-trivial detectors in one packed row.
    def _det_density_from_packed_list(self, packed_list: list[int]) -> int:
        return int(np.count_nonzero(self._unpack_det_row(np.asarray(packed_list, dtype=np.uint8))))

    # NEW: total number of detectors selected by a packed mask (popcount of mask within num_detectors).
    def _mask_size(self, full_mask: np.ndarray) -> int:
        unpacked = np.unpackbits(full_mask, bitorder="little")
        return int(np.count_nonzero(unpacked[: self.circuit.num_detectors]))

    # Helper: container for false-shot diagnostics (accept-false / reject-false).
    def _empty_false_data_block(self, *, valid_best_combo: bool) -> dict[str, Any]:
        return {
            "best_combo": {
                "valid": bool(valid_best_combo),
                "det_density": [],
                "complete_det_density": [],
                "gap_value": [],
                "complete_gap_value": [],
                # --- NEW: pre-computed ratios (one value per false shot) ---
                "ratio_over_complete_nontrivial": [],   # det_density / complete_det_density
                "ratio_over_region_all": [],            # det_density / region_total_detectors
                "ratio_over_complete_all": [],          # det_density / circuit.num_detectors
            },
            "weighted_regions": {
                "det_density": [[], [], [], [], []],
                "complete_det_density": [],
                "gap_value": [[], [], [], [], [], []],  # 5 region gaps + 1 weighted average
                "complete_gap_value": [],
                # --- NEW: pre-computed ratios (list of 5 lists, one per region per false shot) ---
                "ratio_over_complete_nontrivial": [[], [], [], [], []],
                "ratio_over_region_all": [[], [], [], [], []],
                "ratio_over_complete_all": [[], [], [], [], []],
            },
            "pure_avg": {
                "det_density": [],
                "complete_det_density": [],
                "gap_value": [],
                "complete_gap_value": [],
                # --- NEW: pre-computed ratios (one value per false shot) ---
                "ratio_over_complete_nontrivial": [],
                "ratio_over_region_all": [],
                "ratio_over_complete_all": [],
            },
        }

    # Helper: container for the flex pipeline (no best_combo; weighted_regions / min_regions
    # sized to N). gap_value carries N region gaps + 1 weighted_avg + 1 min_gap = N+2 entries.
    def _empty_flex_data_block(self, *, num_weighted_regions: int) -> dict[str, Any]:
        n = int(num_weighted_regions)
        def _region_block():
            return {
                "det_density":                    [[] for _ in range(n)],
                "complete_det_density":           [],
                "gap_value":                      [[] for _ in range(n + 2)],   # N + weighted_avg + min
                "complete_gap_value":             [],
                "ratio_over_complete_nontrivial": [[] for _ in range(n)],
                "ratio_over_region_all":          [[] for _ in range(n)],
                "ratio_over_complete_all":        [[] for _ in range(n)],
            }
        return {
            "weighted_regions": _region_block(),
            "min_regions":      _region_block(),
            "pure_avg": {
                "det_density":                    [],
                "complete_det_density":           [],
                "gap_value":                      [],
                "complete_gap_value":             [],
                "ratio_over_complete_nontrivial": [],
                "ratio_over_region_all":          [],
                "ratio_over_complete_all":        [],
            },
        }

    # Helper: take a user mask (bool length num_detectors OR packed uint8) and produce a packed
    # mask padded to `width` bytes (matches dets.shape[1]).
    def _normalize_mask_to_packed(self, mask: np.ndarray, width: int) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.dtype == np.bool_ or arr.dtype == bool:
            if arr.shape[0] != self.circuit.num_detectors:
                raise ValueError(
                    f"bool mask must have length num_detectors={self.circuit.num_detectors}, got {arr.shape[0]}."
                )
            packed = np.packbits(arr.astype(np.uint8), bitorder="little")
        else:
            packed = np.asarray(arr, dtype=np.uint8)
        out = np.zeros(width, dtype=np.uint8)
        copy_n = min(len(packed), len(out))
        out[:copy_n] = packed[:copy_n]
        return out

    # Define the functions to collect the gap sensitivity data and generate the SVG plots.
    def collect_single_detector_gap_sens(self, sampler = cultiv.DesaturationSampler()):
        dec = sampler.compiled_sampler_for_task(sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model()))
        for d in range(self.circuit.num_detectors):
            self.single_dec_gap_scores.append(dec.decode_det_set({d}))
    
    def single_detector_gap_sens_svg(self, path: pathlib.Path):
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=lambda tile: GapSensitivityCollect.tile_coloring(tile, self.single_dec_gap_scores),
            show_coords=False,
            show_obs=False,
        )

    def single_detector_gap_sens_plot(self):
        """Return an inline-displayable SVG object for Jupyter notebooks."""
        if not self.single_dec_gap_scores:
            raise ValueError("No single-detector gap data collected. Call collect_single_detector_gap_sens() first.")
        return self._plot_from_svg_writer(lambda path: self.single_detector_gap_sens_svg(path))

    # Define the functions to collect the overall gap sensitivity data and generate the SVG plots.
    def collect_overall_gap_sens(self, shots: int, sampler = cultiv.DesaturationSampler()):
        if shots <= 0:
            raise ValueError("shots must be positive.")
        dec = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        dets, actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
        keep_mask = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]

        if dets.shape[0] == 0:
            num_dets = self.circuit.num_detectors
            self.overall_gap_scores = {
                "count": [0] * num_dets,
                "sum_gap": [0.0] * num_dets,
                "sum_err": [0.0] * num_dets,
                "mean_gap_when_active": [0.0] * num_dets,
                "error_rate_when_active": [0.0] * num_dets,
            }
            return

        predictions, gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        if actual_obs.shape[1] != 1:
            raise ValueError(f"Expected exactly one observable, got shape={actual_obs.shape}.")
        actual_obs_bits = (actual_obs[:, 0] & 1).astype(np.bool_)
        errs = (predictions ^ actual_obs_bits).astype(np.float64)

        det_bits = np.unpackbits(dets, axis=1, bitorder='little')[:, :self.circuit.num_detectors].astype(np.float64)
        count = det_bits.sum(axis=0)
        sum_gap = det_bits.T @ gaps.astype(np.float64)
        sum_err = det_bits.T @ errs

        mean_gap_when_active = np.divide(sum_gap, count, out=np.zeros_like(sum_gap), where=count > 0)
        error_rate_when_active = np.divide(sum_err, count, out=np.zeros_like(sum_err), where=count > 0)

        self.overall_gap_scores = {
            "count": count.tolist(),
            "sum_gap": sum_gap.tolist(),
            "sum_err": sum_err.tolist(),
            "mean_gap_when_active": mean_gap_when_active.tolist(),
            "error_rate_when_active": error_rate_when_active.tolist(),
        }
    

    def overall_detector_gap_sens_svg(self, path: pathlib.Path):
        if not self.overall_gap_scores["mean_gap_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        gap_tile_color = self._make_tile_color_func(
            values=self.overall_gap_scores["mean_gap_when_active"],
            counts=self.overall_gap_scores["count"],
            invert=False,
            low_q=0.05,
            high_q=0.95,
            gamma=1.6,
        )
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=gap_tile_color,
            show_coords=False,
            show_obs=False,
        )
        self._append_svg_legend(
            path,
            title="Overall Gap Sensitivity",
            low_label="black = low gap (bad)",
            high_label="red = high gap (good)",
            no_data_label="gray = no data/postselected",
            red_is_high=True,
        )

    def overall_detector_gap_sens_plot(self):
        if not self.overall_gap_scores["mean_gap_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        return self._plot_from_svg_writer(lambda path: self.overall_detector_gap_sens_svg(path))

    def overall_detector_err_sens_svg(self, path: pathlib.Path):
        if not self.overall_gap_scores["error_rate_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        err_tile_color = self._make_tile_color_func(
            values=self.overall_gap_scores["error_rate_when_active"],
            counts=self.overall_gap_scores["count"],
            invert=True,
            low_q=0.00,
            high_q=0.99,
            gamma=1.0,
        )
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=err_tile_color,
            show_coords=False,
            show_obs=False,
        )
        self._append_svg_legend(
            path,
            title="Overall Error Sensitivity",
            low_label="red = low error rate (good)",
            high_label="black = high error rate (bad)",
            no_data_label="gray = no data/postselected",
            red_is_high=False,
        )

    def overall_detector_err_sens_plot(self):
        if not self.overall_gap_scores["error_rate_when_active"]:
            raise ValueError("No overall gap data collected. Call collect_overall_gap_sens(shots=...) first.")
        return self._plot_from_svg_writer(lambda path: self.overall_detector_err_sens_svg(path))

    # Define the functions to collect the causal gap-sensitivity data and generate the SVG plots.
    # Per detector d, score = mean(|gap_with_d - gap_without_d|) over shots where d fired.
    # Unlike `mean_gap_when_active` (a correlational marginal: "average gap when d fires"),
    # this is a causal/mechanistic statistic: "how much of the gap signal does d's firing carry?".
    def collect_causal_gap_sens(
        self,
        shots: int,
        sampler = cultiv.DesaturationSampler(),
        *,
        subsample_size: Optional[int] = None,
        rng_seed: int = 0,
    ):
        """Collect causal gap-sensitivity scores for every detector.

        For each detector d, score[d] is averaged over shots where d fired:
            causal_score_abs[d]    = mean(|gap_orig - gap_with_d_unfired|)
            causal_score_signed[d] = mean(gap_orig - gap_with_d_unfired)

        Cost is dominated by ~num_detectors batched decode calls; total work
        scales roughly linearly with the number of (shot, fired-detector) pairs.

        Args:
            shots: raw shots to sample. Postselect via the sampler's discard mask.
            sampler: desaturation sampler.
            subsample_size: if None (default), use all kept shots after postselect.
                If an int, randomly subsample this many kept shots before scoring.
                Smaller subsamples produce noisier scores but are much faster.
            rng_seed: subsample RNG seed (ignored if subsample_size is None).
        """
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if subsample_size is not None and subsample_size <= 0:
            raise ValueError("subsample_size must be positive (or None).")

        dec = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        dets, _actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
        keep = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep]
        n_kept = int(dets.shape[0])
        num_dets = self.circuit.num_detectors

        if n_kept == 0:
            self.causal_gap_scores = {
                "count": [0] * num_dets,
                "sum_abs_delta": [0.0] * num_dets,
                "sum_signed_delta": [0.0] * num_dets,
                "causal_score_abs": [0.0] * num_dets,
                "causal_score_signed": [0.0] * num_dets,
                "subsample_size": 0,
            }
            return

        if subsample_size is None or subsample_size >= n_kept:
            sub_dets = dets
            sub_n = n_kept
        else:
            rng = np.random.default_rng(rng_seed)
            sub_idx = rng.choice(n_kept, size=int(subsample_size), replace=False)
            sub_dets = dets[sub_idx].copy()
            sub_n = int(subsample_size)

        # Baseline gaps (full-DEM complete decode), and the unpacked syndromes for fast lookup.
        _, sub_gaps_orig = dec._decode_batch_overwrite_last_byte(sub_dets.copy())
        sub_gaps_orig = sub_gaps_orig.astype(np.float64)
        sub_unpacked = np.unpackbits(sub_dets, axis=1, bitorder='little')[:, :num_dets].astype(np.bool_)

        sum_abs = np.zeros(num_dets, dtype=np.float64)
        sum_signed = np.zeros(num_dets, dtype=np.float64)
        count = np.zeros(num_dets, dtype=np.int64)

        # For each detector d: pick the shots that fired d, clear that bit in their bit-packed
        # syndromes, decode the batch, and aggregate gap shift. One batched decode per detector
        # is much faster than one decode per (shot, fired d).
        for d in range(num_dets):
            shots_with_d = np.where(sub_unpacked[:, d])[0]
            if shots_with_d.size == 0:
                continue
            flip_dets = sub_dets[shots_with_d].copy()
            byte_idx = d // 8
            bit_mask = np.uint8(1 << (d % 8))
            flip_dets[:, byte_idx] &= np.uint8(~bit_mask)
            _, gaps_alt = dec._decode_batch_overwrite_last_byte(flip_dets.copy())
            gaps_alt = gaps_alt.astype(np.float64)
            delta = sub_gaps_orig[shots_with_d] - gaps_alt
            sum_abs[d] = float(np.sum(np.abs(delta)))
            sum_signed[d] = float(np.sum(delta))
            count[d] = int(shots_with_d.size)

        causal_abs = np.divide(sum_abs, count, out=np.zeros_like(sum_abs), where=count > 0)
        causal_signed = np.divide(sum_signed, count, out=np.zeros_like(sum_signed), where=count > 0)

        self.causal_gap_scores = {
            "count": count.tolist(),
            "sum_abs_delta": sum_abs.tolist(),
            "sum_signed_delta": sum_signed.tolist(),
            "causal_score_abs": causal_abs.tolist(),
            "causal_score_signed": causal_signed.tolist(),
            "subsample_size": sub_n,
        }

    def causal_detector_gap_sens_svg(self, path: pathlib.Path, *, score_type: str = "abs"):
        if score_type not in {"abs", "signed"}:
            raise ValueError("score_type must be 'abs' or 'signed'.")
        key = "causal_score_abs" if score_type == "abs" else "causal_score_signed"
        if not self.causal_gap_scores[key]:
            raise ValueError("No causal gap data collected. Call collect_causal_gap_sens(shots=...) first.")
        causal_tile_color = self._make_tile_color_func(
            values=self.causal_gap_scores[key],
            counts=self.causal_gap_scores["count"],
            invert=False,
            low_q=0.05,
            high_q=0.95,
            gamma=1.6,
        )
        self.codes[0].write_svg(
            path,
            canvas_height=1000,
            other=self.codes[1:],
            title=[f'tick={e}' for e in sorted(self.ticks)],
            tile_color_func=causal_tile_color,
            show_coords=False,
            show_obs=False,
        )
        if score_type == "abs":
            title = "Causal Gap Sensitivity (mean |Δgap| when active)"
            low_label = "black = low |Δgap| (follower / redundant)"
            high_label = "red = high |Δgap| (causally informative)"
        else:
            title = "Causal Gap Sensitivity (mean signed Δgap when active)"
            low_label = "black = small Δgap"
            high_label = "red = large positive Δgap (firing lowers the gap)"
        self._append_svg_legend(
            path,
            title=title,
            low_label=low_label,
            high_label=high_label,
            no_data_label="gray = no data/postselected",
            red_is_high=True,
        )

    def causal_detector_gap_sens_plot(self, *, score_type: str = "abs"):
        if score_type not in {"abs", "signed"}:
            raise ValueError("score_type must be 'abs' or 'signed'.")
        key = "causal_score_abs" if score_type == "abs" else "causal_score_signed"
        if not self.causal_gap_scores[key]:
            raise ValueError("No causal gap data collected. Call collect_causal_gap_sens(shots=...) first.")
        return self._plot_from_svg_writer(lambda path: self.causal_detector_gap_sens_svg(path, score_type=score_type))


    ## Collection and Visualization for Partial Detector Gap Sensitivity
    def collect_partial_detector_gap_sens(
        self,
        shots: int,
        sampler = cultiv.DesaturationSampler(),
        selector = None,
        left_lengths: tuple[int, int] = (1, 1),
        top_lengths: tuple[int, int] = (1, 1),
        t_lengths: tuple[int, int] = (1, 1),
        mode: str = "rectangle",
        stage_name: str = "escape",
        check_bestcombo: bool = True,
        gap_average_parameters: Optional[list[float]] = None,
        left_len_avg: int = 7,
        top_len_avg: int = 8,
        t_avg: int = 5,
        mode_avg: str = "rectangle",
    ):
        if selector is None:
            raise ValueError("selector is required and should be an instance of DetectorSelection3D.")
        if shots <= 0:
            raise ValueError("shots must be positive.")
        l0, l1 = left_lengths
        u0, u1 = top_lengths
        t0, t1 = t_lengths
        if not (l1 >= l0 and u1 >= u0 and t1 >= t0):
            raise ValueError("Each *_lengths tuple must be (start, end) with end >= start.")
        
        if gap_average_parameters is None:
            gap_average_parameters = [1.0, 0.0, 0.0, 0.0, 0.0]
        if len(gap_average_parameters) != 5:
            raise ValueError("gap_average_parameters must have length 5.")
        gap_average_parameters = [float(x) for x in gap_average_parameters]

        dec = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        dets, actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
        keep_mask = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        remained_shots = int(dets.shape[0])

        if remained_shots == 0:
            self.partial_gap_sens_data = {
                "remained_shots": 0,
                "complete_gaps": [],
                "partial_gaps": [],
                "closest_matches": [],
                "combos": [],
                "left_lengths": left_lengths,
                "top_lengths": top_lengths,
                "t_lengths": t_lengths,
                "mode": mode,
                "stage_name": stage_name,
                "complete_accept_and_reject": [],
                "closest_partial_accept_and_reject": [],
                "weighted_multi_partial_accept_and_reject": [],
                "pure_avg_accept_and_reject": [],
                "pure_avg_adjust_threshold_accept_and_reject": [],
                "gap_average_parameters": gap_average_parameters,
                "check_bestcombo": bool(check_bestcombo),
            }
            self.best_combo_distribution = None
            return

        _, complete_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        complete_gaps = complete_gaps.astype(np.float64)
        
        
        # Calculate the gaps for best combo and multi masks
        combos = [
            (left_len, top_len, t)
            for left_len in range(l0, l1 + 1)
            for top_len in range(u0, u1 + 1)
            for t in range(t0, t1 + 1)
        ]

        combo_to_gaps: dict[tuple[int, int, int], np.ndarray] = {}
        combo_to_full_mask: dict[tuple[int, int, int], np.ndarray] = {}
        if check_bestcombo:
            for left_len, top_len, t in combos:
                _, mask_packed = selector.build_color_region_mask(
                    left_len=left_len,
                    top_len=top_len,
                    t=t,
                    mode=mode,
                    stage_name=stage_name,
                )
                full_mask = np.zeros(dets.shape[1], dtype=np.uint8)
                copy_n = min(len(mask_packed), len(full_mask))
                full_mask[:copy_n] = mask_packed[:copy_n]
                combo_to_full_mask[(left_len, top_len, t)] = full_mask.copy()

                dets_partial = dets.copy()
                dets_partial &= full_mask.reshape(1, -1)
                _, partial_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial)
                combo_to_gaps[(left_len, top_len, t)] = partial_gaps.astype(np.float64)

        # NEW: popcount of each combo mask = total detectors selected by that region.
        combo_to_region_size: dict[tuple[int, int, int], int] = {}
        if check_bestcombo:
            for combo, mask in combo_to_full_mask.items():
                combo_to_region_size[combo] = self._mask_size(mask)

        # Build reusable masks once for weighted-region and pure-avg paths.
        _, avg_masks_packed, _ = selector.build_partial_region_multi_mask(
            left_len=left_len_avg,
            top_len=top_len_avg,
            t=t_avg,
            mode=mode_avg,
            stage_name=stage_name,
        )
        avg_masks_full: list[np.ndarray] = []
        for mask_packed_avg in avg_masks_packed:
            full_mask = np.zeros(dets.shape[1], dtype=np.uint8)
            copy_n = min(len(mask_packed_avg), len(full_mask))
            full_mask[:copy_n] = mask_packed_avg[:copy_n]
            avg_masks_full.append(full_mask)
        # NEW: total detectors selected by each of the 5 weighted regions (constant across shots).
        avg_region_sizes: list[int] = [self._mask_size(m) for m in avg_masks_full]

        _, mask_packed_pure_avg = selector.build_color_region_mask(
            left_len=left_len_avg,
            top_len=top_len_avg,
            t=t_avg,
            mode=mode_avg,
            stage_name=stage_name,
        )
        pure_avg_full_mask = np.zeros(dets.shape[1], dtype=np.uint8)
        copy_n_pure_avg = min(len(mask_packed_pure_avg), len(pure_avg_full_mask))
        pure_avg_full_mask[:copy_n_pure_avg] = mask_packed_pure_avg[:copy_n_pure_avg]
        # NEW: total detectors selected by the pure-avg region (constant across shots).
        pure_avg_region_size: int = self._mask_size(pure_avg_full_mask)

        partial_gaps_per_shot: list[dict[tuple[int, int, int], float]] = []
        closest_matches: list[dict[str, Any]] = []
        
        for shot_idx in range(remained_shots):
            shot_det = dets[shot_idx:shot_idx + 1].copy()
            if check_bestcombo:
                gap_map: dict[tuple[int, int, int], float] = {}
                for combo in combos:
                    gap_map[combo] = float(combo_to_gaps[combo][shot_idx])
                partial_gaps_per_shot.append(gap_map)
            else:
                gap_map = {}
                partial_gaps_per_shot.append({})
            complete_gap = float(complete_gaps[shot_idx])
            if check_bestcombo:
                best_combo = min(combos, key=lambda c: abs(gap_map[c] - complete_gap))
                best_partial = float(gap_map[best_combo])
                best_abs_diff = abs(best_partial - complete_gap)
            else:
                best_combo = (np.nan, np.nan, np.nan)
                best_partial = np.nan
                best_abs_diff = np.nan
            complete_det_packed = shot_det[0].astype(np.uint8).tolist()

            # Best-combo selected detectors (invalid/all-zero when check_bestcombo=False).
            if check_bestcombo:
                best_mask_full = combo_to_full_mask[tuple(best_combo)]
                dets_best_partial = shot_det.copy()
                dets_best_partial &= best_mask_full.reshape(1, -1)
                best_partial_det_packed = dets_best_partial[0].astype(np.uint8).tolist()
            else:
                best_partial_det_packed = [0 for _ in range(shot_det.shape[1])]

            region_gaps_vals: list[float] = []
            region_det_packed: list[list[int]] = []
            for full_mask in avg_masks_full:
                dets_partial = shot_det.copy()
                dets_partial &= full_mask.reshape(1, -1)
                _, rg = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial)
                region_gaps_vals.append(float(rg[0]))
                region_det_packed.append(dets_partial[0].astype(np.uint8).tolist())

            weighted_components = np.asarray(region_gaps_vals, dtype=np.float64)
            weighted_avg_gap = float(np.dot(np.asarray(gap_average_parameters, dtype=np.float64), weighted_components))
            weighted_abs_diff = abs(weighted_avg_gap - complete_gap)
            
            # Calculate the gap for pure-average selected region.
            dets_partial_pure_avg = shot_det.copy()
            dets_partial_pure_avg &= pure_avg_full_mask.reshape(1, -1)
            _, partial_gaps_pure_avg = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets_partial_pure_avg)
            
            closest_matches.append({
                "shot_idx": int(shot_idx),
                "combo": tuple(best_combo),
                "partial_gap": best_partial,
                "complete_gap": complete_gap,
                "abs_diff": best_abs_diff,
                "weighted_avg_gap": weighted_avg_gap,
                "weighted_abs_diff": weighted_abs_diff,
                "weighted_region_gaps": weighted_components.tolist(),
                "pure_avg_gap": partial_gaps_pure_avg[0],
                "pure_avg_abs_diff": abs(partial_gaps_pure_avg[0] - complete_gap),
                # Newly added detector snapshots for downstream false-shot analysis.
                "complete_det_packed": complete_det_packed,
                "best_combo_det_packed": best_partial_det_packed,
                "weighted_region_det_packed": region_det_packed,
                "pure_avg_det_packed": dets_partial_pure_avg[0].astype(np.uint8).tolist(),
                # NEW: total detectors in the best-combo region for this shot (varies per shot).
                "best_combo_region_size": combo_to_region_size.get(tuple(best_combo), 0) if check_bestcombo else 0,
            })
        self.partial_gap_sens_data = {
            "remained_shots": remained_shots,
            "complete_gaps": complete_gaps.tolist(),
            "partial_gaps": partial_gaps_per_shot,
            "closest_matches": closest_matches,
            "combos": combos,
            "left_lengths": left_lengths,
            "top_lengths": top_lengths,
            "t_lengths": t_lengths,
            "mode": mode,
            "stage_name": stage_name,
            "complete_accept_and_reject": [],
            "closest_partial_accept_and_reject": [],
            "weighted_multi_partial_accept_and_reject": [],
            "pure_avg_accept_and_reject": [],
            "pure_avg_adjust_threshold_accept_and_reject": [],
            "gap_average_parameters": gap_average_parameters,
            "check_bestcombo": bool(check_bestcombo),
            # NEW: region sizes (mask popcounts) for ratio_over_region_all computation.
            "avg_region_sizes": avg_region_sizes,
            "pure_avg_region_size": pure_avg_region_size,
        }
        self.best_combo_distribution = None

    def collect_partial_accpet_and_reject(
        self,
        *,
        gap_threshold: Optional[float] = None,
        gap_threshold_pure_avg: Optional[float] = None,
    ) -> None:
        if not self.partial_gap_sens_data.get("closest_matches"):
            self.partial_gap_sens_data["complete_accept_and_reject"] = []
            self.partial_gap_sens_data["closest_partial_accept_and_reject"] = []
            self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"] = []
            self.partial_gap_sens_data["pure_avg_accept_and_reject"] = []
            self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"] = []
            if gap_threshold is not None:
                self.partial_gap_sens_data["gap_threshold"] = float(gap_threshold)
            if gap_threshold_pure_avg is not None:
                self.partial_gap_sens_data["gap_threshold_pure_avg"] = float(gap_threshold_pure_avg)
            return

        if gap_threshold is None:
            gap_threshold = float(self.partial_gap_sens_data.get("gap_threshold", 0.0))
        if gap_threshold_pure_avg is None:
            gap_threshold_pure_avg = float(self.partial_gap_sens_data.get("gap_threshold_pure_avg", gap_threshold))

        complete_accept_and_reject: list[bool] = []
        closest_partial_accept_and_reject: list[bool] = []
        weighted_multi_partial_accept_and_reject: list[bool] = []
        pure_avg_accept_and_reject: list[bool] = []
        pure_avg_adjust_threshold_accept_and_reject: list[bool] = []
        for rec in self.partial_gap_sens_data["closest_matches"]:
            complete_gap = float(rec["complete_gap"])
            partial_gap = float(rec["partial_gap"])
            weighted_avg_gap = float(rec["weighted_avg_gap"])
            pure_avg_gap = float(rec["pure_avg_gap"])

            complete_accept_and_reject.append(complete_gap >= gap_threshold)
            closest_partial_accept_and_reject.append(partial_gap >= gap_threshold)
            weighted_multi_partial_accept_and_reject.append(weighted_avg_gap >= gap_threshold)
            pure_avg_accept_and_reject.append(pure_avg_gap >= gap_threshold)
            pure_avg_adjust_threshold_accept_and_reject.append(pure_avg_gap >= gap_threshold_pure_avg)

        self.partial_gap_sens_data["complete_accept_and_reject"] = complete_accept_and_reject
        self.partial_gap_sens_data["closest_partial_accept_and_reject"] = closest_partial_accept_and_reject
        self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"] = weighted_multi_partial_accept_and_reject
        self.partial_gap_sens_data["pure_avg_accept_and_reject"] = pure_avg_accept_and_reject
        self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"] = pure_avg_adjust_threshold_accept_and_reject
        self.partial_gap_sens_data["gap_threshold"] = float(gap_threshold)
        self.partial_gap_sens_data["gap_threshold_pure_avg"] = float(gap_threshold_pure_avg)
        self.best_combo_distribution = None

    def paritial_detector_gap_sens_svg(self, shot_idx: int, out: Optional[pathlib.Path] = None, interactive: bool = True):
        if not self.partial_gap_sens_data["partial_gaps"]:
            raise ValueError("No partial gap data found. Call collect_partial_detector_gap_sens(...) first.")
        remained = self.partial_gap_sens_data["remained_shots"]
        if shot_idx < 0 or shot_idx >= remained:
            raise ValueError(f"shot_idx out of range: 0 <= shot_idx < {remained}.")

        shot_partial: dict[tuple[int, int, int], float] = self.partial_gap_sens_data["partial_gaps"][shot_idx]
        complete_gap = float(self.partial_gap_sens_data["complete_gaps"][shot_idx])

        combos = list(shot_partial.keys())
        x = [c[0] for c in combos]
        y = [c[1] for c in combos]
        z = [c[2] for c in combos]
        partial_vals = [float(shot_partial[c]) for c in combos]

        all_vals = np.asarray(partial_vals + [complete_gap], dtype=np.float64)
        vmin = float(np.min(all_vals))
        vmax = float(np.max(all_vals))
        if vmax <= vmin:
            vmax = vmin + 1e-9

        def scale_size(v: float, lo: float = 4.0, hi: float = 24.0) -> float:
            norm = (v - vmin) / (vmax - vmin)
            return float(lo + (hi - lo) * norm)

        partial_sizes = [scale_size(v) for v in partial_vals]
        complete_size = scale_size(complete_gap)
        complete_sizes = [complete_size] * len(combos)

        hover_black = [
            f"(left,top,t)=({a},{b},{c})<br>partial_gap={g:.4f}"
            for (a, b, c), g in zip(combos, partial_vals)
        ]
        hover_red = [
            f"(left,top,t)=({a},{b},{c})<br>complete_gap={complete_gap:.4f}"
            for (a, b, c) in combos
        ]
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="Partial Gap",
                marker=dict(size=partial_sizes, color="black", symbol="circle", opacity=0.92),
                text=hover_black,
                hoverinfo="text",
            ))
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                name="Complete Gap (Reference)",
                marker=dict(size=complete_sizes, color="rgba(0,0,0,0)", symbol="circle-open", line=dict(color="red", width=2)),
                text=hover_red,
                hoverinfo="text",
            ))
            fig.update_layout(
                title=f"Shot {shot_idx}: Partial vs Complete Gap",
                scene=dict(
                    xaxis_title="left_len",
                    yaxis_title="top_len",
                    zaxis_title="t",
                ),
                legend=dict(x=0.01, y=0.99),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            if out is not None:
                if out.suffix.lower() == ".html":
                    fig.write_html(str(out))
                elif out.suffix.lower() == ".svg":
                    fig.write_image(str(out))
                else:
                    fig.write_html(str(out.with_suffix(".html")))
            return fig
        except ImportError as ex:
            if interactive:
                raise ImportError(
                    "Interactive 3D rotation requires plotly. Install it (e.g. `pip install plotly`) "
                    "or call with interactive=False for static matplotlib output."
                ) from ex
            import matplotlib.pyplot as plt

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(x, y, z, s=partial_sizes, c="black", alpha=0.92, label="Partial Gap")
            ax.scatter(x, y, z, s=complete_sizes, facecolors="none", edgecolors="red", linewidths=1.5, label="Complete Gap (Reference)")
            ax.set_xlabel("left_len")
            ax.set_ylabel("top_len")
            ax.set_zlabel("t")
            ax.set_title(f"Shot {shot_idx}: Partial vs Complete Gap")
            ax.legend(loc="upper left")
            if out is not None:
                if out.suffix.lower() == ".svg":
                    fig.savefig(out, format="svg", bbox_inches="tight")
                else:
                    fig.savefig(out.with_suffix(".svg"), format="svg", bbox_inches="tight")
            return fig

    def partial_detector_gap_sens_svg(self, shot_idx: int, out: Optional[pathlib.Path] = None):
        # Alias with corrected spelling.
        return self.paritial_detector_gap_sens_svg(shot_idx=shot_idx, out=out)

    def print_partial_gap_match_for_shot(self, shot_idx: int) -> dict[str, Any]:
        if not self.partial_gap_sens_data["closest_matches"]:
            raise ValueError("No partial gap match data found. Call collect_partial_detector_gap_sens(...) first.")
        remained = self.partial_gap_sens_data["remained_shots"]
        if shot_idx < 0 or shot_idx >= remained:
            raise ValueError(f"shot_idx out of range: 0 <= shot_idx < {remained}.")
        rec = self.partial_gap_sens_data["closest_matches"][shot_idx]
        left_len, top_len, t = rec["combo"]
        print(
            f"shot_idx={rec['shot_idx']}, "
            f"closest_combo=(left_len={left_len}, top_len={top_len}, t={t}), "
            f"partial_gap={rec['partial_gap']:.6f}, "
            f"complete_gap={rec['complete_gap']:.6f}, "
            f"abs_diff={rec['abs_diff']:.6f}"
        )
        return rec


    def calculate_partial_gap_best_combo_distribution(
        self,
        *,
        max_correct_shots: Optional[int] = None,
    ) -> None:
        if not self.partial_gap_sens_data["closest_matches"]:
            raise ValueError("No partial gap data found. Call collect_partial_detector_gap_sens(...) first.")

        best = self.partial_gap_sens_data["closest_matches"]
        remained = int(self.partial_gap_sens_data["remained_shots"])
        check_bestcombo = bool(self.partial_gap_sens_data.get("check_bestcombo", True))
        # Per-scheme cap for correct-shot data collection (memory bound; correct shots dominate).
        # None = no cap. Cap applies independently to each scheme's accept_correct / reject_correct lists.
        cap_correct = int(max_correct_shots) if max_correct_shots is not None else None
        if remained == 0:
            self.best_combo_distribution = {
                "remained_shots": 0,
                "mean_combo": {"left_len": 0.0, "top_len": 0.0, "t": 0.0},
                "var_combo": {"left_len": 0.0, "top_len": 0.0, "t": 0.0},
                "std_combo": {"left_len": 0.0, "top_len": 0.0, "t": 0.0},
                "mean_abs_diff": 0.0,
                "var_abs_diff": 0.0,
                "std_abs_diff": 0.0,
                "mean_complete_gap": 0.0,
                "best_combo_samples": [],
                "abs_diff_samples": [],
                "complete_accept_rate": 1.0,
                "complete_reject_rate": 0.0,
                "closest_partial_accept_rate": 1.0,
                "closest_partial_reject_rate": 0.0,
                "partial_accept_false_num": 0.0,
                "partial_reject_false_num": 0.0,
                "partial_accept_false_rate(over_remained_shots)": 0.0,
                "partial_reject_false_rate(over_remained_shots)": 0.0,
                "partial_accept_false_rate(over_complete_accept_num)": 0.0,
                "partial_reject_false_rate(over_complete_reject_num)": 0.0,
                "weighted_mean_abs_diff": 0.0,
                "weighted_var_abs_diff": 0.0,
                "weighted_std_abs_diff": 0.0,
                "weighted_multi_partial_accept_rate": 1.0,
                "weighted_multi_partial_reject_rate": 0.0,
                "weighted_accept_false_num": 0.0,
                "weighted_reject_false_num": 0.0,
                "weighted_accept_false_rate(over_remained_shots)": 0.0,
                "weighted_reject_false_rate(over_remained_shots)": 0.0,
                "weighted_accept_false_rate(over_complete_accept_num)": 0.0,
                "weighted_reject_false_rate(over_complete_reject_num)": 0.0,
                "pure_avg_mean_abs_diff": 0.0,
                "pure_avg_var_abs_diff": 0.0,
                "pure_avg_std_abs_diff": 0.0,
                "pure_avg_accept_rate": 1.0,
                "pure_avg_reject_rate": 0.0,
                "pure_avg_accept_false_num": 0.0,
                "pure_avg_reject_false_num": 0.0,
                "pure_avg_accept_false_rate(over_remained_shots)": 0.0,
                "pure_avg_reject_false_rate(over_remained_shots)": 0.0,
                "pure_avg_accept_false_rate(over_complete_accept_num)": 0.0,
                "pure_avg_reject_false_rate(over_complete_reject_num)": 0.0,
                "pure_avg_adjust_threshold_accept_rate": 1.0,
                "pure_avg_adjust_threshold_reject_rate": 0.0,
                "pure_avg_adjust_threshold_accept_false_num": 0.0,
                "pure_avg_adjust_threshold_reject_false_num": 0.0,
                "pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)": 0.0,
                "pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)": 0.0,
                "pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)": 0.0,
                "pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)": 0.0,
                "accept_false_data": self._empty_false_data_block(valid_best_combo=check_bestcombo),
                "reject_false_data": self._empty_false_data_block(valid_best_combo=check_bestcombo),
                "accept_correct_data": self._empty_false_data_block(valid_best_combo=check_bestcombo),
                "reject_correct_data": self._empty_false_data_block(valid_best_combo=check_bestcombo),
                "check_bestcombo": check_bestcombo,
            }
            return

        if check_bestcombo:
            left_vals = np.asarray([float(rec["combo"][0]) for rec in best], dtype=np.float64)
            top_vals = np.asarray([float(rec["combo"][1]) for rec in best], dtype=np.float64)
            t_vals = np.asarray([float(rec["combo"][2]) for rec in best], dtype=np.float64)
            abs_diff_vals = np.asarray([float(rec["abs_diff"]) for rec in best], dtype=np.float64)
        else:
            left_vals = np.asarray([], dtype=np.float64)
            top_vals = np.asarray([], dtype=np.float64)
            t_vals = np.asarray([], dtype=np.float64)
            abs_diff_vals = np.asarray([], dtype=np.float64)
        weighted_abs_diff_vals = np.asarray([float(rec["weighted_abs_diff"]) for rec in best], dtype=np.float64)
        pure_avg_abs_diff_vals = np.asarray([float(rec["pure_avg_abs_diff"]) for rec in best], dtype=np.float64)
        complete = np.asarray(self.partial_gap_sens_data["complete_gaps"], dtype=np.float64)
        
        assert len(self.partial_gap_sens_data["complete_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["closest_partial_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["pure_avg_accept_and_reject"]) == remained
        assert len(self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"]) == remained
        
        complete_accept_num = 0
        complete_reject_num = 0
        partial_accept_num = 0
        partial_reject_num = 0
        partial_accept_false_num = 0
        partial_reject_false_num = 0
        weighted_accept_num = 0
        weighted_reject_num = 0
        weighted_accept_false_num = 0
        weighted_reject_false_num = 0
        pure_avg_accept_num = 0
        pure_avg_reject_num = 0
        pure_avg_accept_false_num = 0
        pure_avg_reject_false_num = 0
        pure_avg_adjust_threshold_accept_num = 0
        pure_avg_adjust_threshold_reject_num = 0
        pure_avg_adjust_threshold_accept_false_num = 0
        pure_avg_adjust_threshold_reject_false_num = 0
        for i in range(remained):
            if self.partial_gap_sens_data["complete_accept_and_reject"][i]:
                complete_accept_num += 1
            else:
                complete_reject_num += 1
            if check_bestcombo:
                if self.partial_gap_sens_data["closest_partial_accept_and_reject"][i]:
                    partial_accept_num += 1
                else:
                    partial_reject_num += 1
                if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["closest_partial_accept_and_reject"][i]):
                    partial_reject_false_num += 1
                if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["closest_partial_accept_and_reject"][i]:
                    partial_accept_false_num += 1

            if self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i]:
                weighted_accept_num += 1
            else:
                weighted_reject_num += 1
            if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i]):
                weighted_reject_false_num += 1
            if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i]:
                weighted_accept_false_num += 1
            
            if self.partial_gap_sens_data["pure_avg_accept_and_reject"][i]:
                pure_avg_accept_num += 1
            else:
                pure_avg_reject_num += 1
            if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["pure_avg_accept_and_reject"][i]):
                pure_avg_reject_false_num += 1
            if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["pure_avg_accept_and_reject"][i]:
                pure_avg_accept_false_num += 1
            
            if self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"][i]:
                pure_avg_adjust_threshold_accept_num += 1
            else:
                pure_avg_adjust_threshold_reject_num += 1
            if self.partial_gap_sens_data["complete_accept_and_reject"][i] and (not self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"][i]):
                pure_avg_adjust_threshold_reject_false_num += 1
            if (not self.partial_gap_sens_data["complete_accept_and_reject"][i]) and self.partial_gap_sens_data["pure_avg_adjust_threshold_accept_and_reject"][i]:
                pure_avg_adjust_threshold_accept_false_num += 1

        # Newly added: gather false-shot detector-density/gap diagnostics by selection type.
        # NEW: read region sizes for ratio_over_region_all; get total detector count for ratio_over_complete_all.
        complete_total_detectors: int = self.circuit.num_detectors
        avg_region_sizes: list[int] = self.partial_gap_sens_data.get("avg_region_sizes", [0, 0, 0, 0, 0])
        pure_avg_region_size: int = int(self.partial_gap_sens_data.get("pure_avg_region_size", 0))

        accept_false_data = self._empty_false_data_block(valid_best_combo=check_bestcombo)
        reject_false_data = self._empty_false_data_block(valid_best_combo=check_bestcombo)
        accept_correct_data = self._empty_false_data_block(valid_best_combo=check_bestcombo)
        reject_correct_data = self._empty_false_data_block(valid_best_combo=check_bestcombo)

        def _under_cap(target: dict, scheme: str) -> bool:
            if cap_correct is None:
                return True
            if scheme == "weighted_regions":
                return len(target[scheme]["det_density"][0]) < cap_correct
            return len(target[scheme]["det_density"]) < cap_correct
        for i, rec in enumerate(best):
            complete_accept = bool(self.partial_gap_sens_data["complete_accept_and_reject"][i])
            partial_accept = bool(self.partial_gap_sens_data["closest_partial_accept_and_reject"][i])
            weighted_accept = bool(self.partial_gap_sens_data["weighted_multi_partial_accept_and_reject"][i])
            pure_accept = bool(self.partial_gap_sens_data["pure_avg_accept_and_reject"][i])

            complete_density = self._det_density_from_packed_list(rec["complete_det_packed"])
            complete_gap_value = float(rec["complete_gap"])

            # Best-combo false data (invalid when check_bestcombo=False).
            if check_bestcombo:
                best_density = self._det_density_from_packed_list(rec["best_combo_det_packed"])
                best_gap_value = float(rec["partial_gap"])
                # NEW: denominators for the three ratios for this shot's best-combo region.
                best_region_size = int(rec.get("best_combo_region_size", 0))
                best_r1 = best_density / complete_density if complete_density > 0 else 0.0
                best_r2 = best_density / best_region_size if best_region_size > 0 else 0.0
                best_r3 = best_density / complete_total_detectors if complete_total_detectors > 0 else 0.0
                if partial_accept and (not complete_accept):
                    accept_false_data["best_combo"]["det_density"].append(best_density)
                    accept_false_data["best_combo"]["complete_det_density"].append(complete_density)
                    accept_false_data["best_combo"]["gap_value"].append(best_gap_value)
                    accept_false_data["best_combo"]["complete_gap_value"].append(complete_gap_value)
                    # NEW
                    accept_false_data["best_combo"]["ratio_over_complete_nontrivial"].append(best_r1)
                    accept_false_data["best_combo"]["ratio_over_region_all"].append(best_r2)
                    accept_false_data["best_combo"]["ratio_over_complete_all"].append(best_r3)
                if (not partial_accept) and complete_accept:
                    reject_false_data["best_combo"]["det_density"].append(best_density)
                    reject_false_data["best_combo"]["complete_det_density"].append(complete_density)
                    reject_false_data["best_combo"]["gap_value"].append(best_gap_value)
                    reject_false_data["best_combo"]["complete_gap_value"].append(complete_gap_value)
                    # NEW
                    reject_false_data["best_combo"]["ratio_over_complete_nontrivial"].append(best_r1)
                    reject_false_data["best_combo"]["ratio_over_region_all"].append(best_r2)
                    reject_false_data["best_combo"]["ratio_over_complete_all"].append(best_r3)
                # Correct-shot mirrors (capped via cap_correct).
                if partial_accept and complete_accept and _under_cap(accept_correct_data, "best_combo"):
                    accept_correct_data["best_combo"]["det_density"].append(best_density)
                    accept_correct_data["best_combo"]["complete_det_density"].append(complete_density)
                    accept_correct_data["best_combo"]["gap_value"].append(best_gap_value)
                    accept_correct_data["best_combo"]["complete_gap_value"].append(complete_gap_value)
                    accept_correct_data["best_combo"]["ratio_over_complete_nontrivial"].append(best_r1)
                    accept_correct_data["best_combo"]["ratio_over_region_all"].append(best_r2)
                    accept_correct_data["best_combo"]["ratio_over_complete_all"].append(best_r3)
                if (not partial_accept) and (not complete_accept) and _under_cap(reject_correct_data, "best_combo"):
                    reject_correct_data["best_combo"]["det_density"].append(best_density)
                    reject_correct_data["best_combo"]["complete_det_density"].append(complete_density)
                    reject_correct_data["best_combo"]["gap_value"].append(best_gap_value)
                    reject_correct_data["best_combo"]["complete_gap_value"].append(complete_gap_value)
                    reject_correct_data["best_combo"]["ratio_over_complete_nontrivial"].append(best_r1)
                    reject_correct_data["best_combo"]["ratio_over_region_all"].append(best_r2)
                    reject_correct_data["best_combo"]["ratio_over_complete_all"].append(best_r3)

            # Weighted-regions false data.
            weighted_region_density = [
                self._det_density_from_packed_list(packed) for packed in rec["weighted_region_det_packed"]
            ]
            weighted_region_gaps = [float(v) for v in rec["weighted_region_gaps"]]
            weighted_all_gaps = weighted_region_gaps + [float(rec["weighted_avg_gap"])]
            # NEW: compute three ratios per region.
            weighted_r1 = [
                (weighted_region_density[ri] / complete_density if complete_density > 0 else 0.0)
                for ri in range(5)
            ]
            weighted_r2 = [
                (weighted_region_density[ri] / avg_region_sizes[ri] if avg_region_sizes[ri] > 0 else 0.0)
                for ri in range(5)
            ]
            weighted_r3 = [
                (weighted_region_density[ri] / complete_total_detectors if complete_total_detectors > 0 else 0.0)
                for ri in range(5)
            ]
            if weighted_accept and (not complete_accept):
                for region_i in range(5):
                    accept_false_data["weighted_regions"]["det_density"][region_i].append(weighted_region_density[region_i])
                    # NEW
                    accept_false_data["weighted_regions"]["ratio_over_complete_nontrivial"][region_i].append(weighted_r1[region_i])
                    accept_false_data["weighted_regions"]["ratio_over_region_all"][region_i].append(weighted_r2[region_i])
                    accept_false_data["weighted_regions"]["ratio_over_complete_all"][region_i].append(weighted_r3[region_i])
                for gap_i in range(6):
                    accept_false_data["weighted_regions"]["gap_value"][gap_i].append(weighted_all_gaps[gap_i])
                accept_false_data["weighted_regions"]["complete_det_density"].append(complete_density)
                accept_false_data["weighted_regions"]["complete_gap_value"].append(complete_gap_value)
            if (not weighted_accept) and complete_accept:
                for region_i in range(5):
                    reject_false_data["weighted_regions"]["det_density"][region_i].append(weighted_region_density[region_i])
                    # NEW
                    reject_false_data["weighted_regions"]["ratio_over_complete_nontrivial"][region_i].append(weighted_r1[region_i])
                    reject_false_data["weighted_regions"]["ratio_over_region_all"][region_i].append(weighted_r2[region_i])
                    reject_false_data["weighted_regions"]["ratio_over_complete_all"][region_i].append(weighted_r3[region_i])
                for gap_i in range(6):
                    reject_false_data["weighted_regions"]["gap_value"][gap_i].append(weighted_all_gaps[gap_i])
                reject_false_data["weighted_regions"]["complete_det_density"].append(complete_density)
                reject_false_data["weighted_regions"]["complete_gap_value"].append(complete_gap_value)
            # Correct-shot mirrors (capped via cap_correct).
            if weighted_accept and complete_accept and _under_cap(accept_correct_data, "weighted_regions"):
                for region_i in range(5):
                    accept_correct_data["weighted_regions"]["det_density"][region_i].append(weighted_region_density[region_i])
                    accept_correct_data["weighted_regions"]["ratio_over_complete_nontrivial"][region_i].append(weighted_r1[region_i])
                    accept_correct_data["weighted_regions"]["ratio_over_region_all"][region_i].append(weighted_r2[region_i])
                    accept_correct_data["weighted_regions"]["ratio_over_complete_all"][region_i].append(weighted_r3[region_i])
                for gap_i in range(6):
                    accept_correct_data["weighted_regions"]["gap_value"][gap_i].append(weighted_all_gaps[gap_i])
                accept_correct_data["weighted_regions"]["complete_det_density"].append(complete_density)
                accept_correct_data["weighted_regions"]["complete_gap_value"].append(complete_gap_value)
            if (not weighted_accept) and (not complete_accept) and _under_cap(reject_correct_data, "weighted_regions"):
                for region_i in range(5):
                    reject_correct_data["weighted_regions"]["det_density"][region_i].append(weighted_region_density[region_i])
                    reject_correct_data["weighted_regions"]["ratio_over_complete_nontrivial"][region_i].append(weighted_r1[region_i])
                    reject_correct_data["weighted_regions"]["ratio_over_region_all"][region_i].append(weighted_r2[region_i])
                    reject_correct_data["weighted_regions"]["ratio_over_complete_all"][region_i].append(weighted_r3[region_i])
                for gap_i in range(6):
                    reject_correct_data["weighted_regions"]["gap_value"][gap_i].append(weighted_all_gaps[gap_i])
                reject_correct_data["weighted_regions"]["complete_det_density"].append(complete_density)
                reject_correct_data["weighted_regions"]["complete_gap_value"].append(complete_gap_value)

            # Pure-avg false data.
            pure_density = self._det_density_from_packed_list(rec["pure_avg_det_packed"])
            pure_gap_value = float(rec["pure_avg_gap"])
            # NEW: compute three ratios for pure-avg region.
            pure_r1 = pure_density / complete_density if complete_density > 0 else 0.0
            pure_r2 = pure_density / pure_avg_region_size if pure_avg_region_size > 0 else 0.0
            pure_r3 = pure_density / complete_total_detectors if complete_total_detectors > 0 else 0.0
            if pure_accept and (not complete_accept):
                accept_false_data["pure_avg"]["det_density"].append(pure_density)
                accept_false_data["pure_avg"]["complete_det_density"].append(complete_density)
                accept_false_data["pure_avg"]["gap_value"].append(pure_gap_value)
                accept_false_data["pure_avg"]["complete_gap_value"].append(complete_gap_value)
                # NEW
                accept_false_data["pure_avg"]["ratio_over_complete_nontrivial"].append(pure_r1)
                accept_false_data["pure_avg"]["ratio_over_region_all"].append(pure_r2)
                accept_false_data["pure_avg"]["ratio_over_complete_all"].append(pure_r3)
            if (not pure_accept) and complete_accept:
                reject_false_data["pure_avg"]["det_density"].append(pure_density)
                reject_false_data["pure_avg"]["complete_det_density"].append(complete_density)
                reject_false_data["pure_avg"]["gap_value"].append(pure_gap_value)
                reject_false_data["pure_avg"]["complete_gap_value"].append(complete_gap_value)
                # NEW
                reject_false_data["pure_avg"]["ratio_over_complete_nontrivial"].append(pure_r1)
                reject_false_data["pure_avg"]["ratio_over_region_all"].append(pure_r2)
                reject_false_data["pure_avg"]["ratio_over_complete_all"].append(pure_r3)
            # Correct-shot mirrors (capped via cap_correct).
            if pure_accept and complete_accept and _under_cap(accept_correct_data, "pure_avg"):
                accept_correct_data["pure_avg"]["det_density"].append(pure_density)
                accept_correct_data["pure_avg"]["complete_det_density"].append(complete_density)
                accept_correct_data["pure_avg"]["gap_value"].append(pure_gap_value)
                accept_correct_data["pure_avg"]["complete_gap_value"].append(complete_gap_value)
                accept_correct_data["pure_avg"]["ratio_over_complete_nontrivial"].append(pure_r1)
                accept_correct_data["pure_avg"]["ratio_over_region_all"].append(pure_r2)
                accept_correct_data["pure_avg"]["ratio_over_complete_all"].append(pure_r3)
            if (not pure_accept) and (not complete_accept) and _under_cap(reject_correct_data, "pure_avg"):
                reject_correct_data["pure_avg"]["det_density"].append(pure_density)
                reject_correct_data["pure_avg"]["complete_det_density"].append(complete_density)
                reject_correct_data["pure_avg"]["gap_value"].append(pure_gap_value)
                reject_correct_data["pure_avg"]["complete_gap_value"].append(complete_gap_value)
                reject_correct_data["pure_avg"]["ratio_over_complete_nontrivial"].append(pure_r1)
                reject_correct_data["pure_avg"]["ratio_over_region_all"].append(pure_r2)
                reject_correct_data["pure_avg"]["ratio_over_complete_all"].append(pure_r3)

        self.best_combo_distribution = {
            "remained_shots": remained,
            "mean_combo": {
                "left_len": float(np.mean(left_vals)) if check_bestcombo else None,
                "top_len": float(np.mean(top_vals)) if check_bestcombo else None,
                "t": float(np.mean(t_vals)) if check_bestcombo else None,
            },
            "var_combo": {
                "left_len": float(np.var(left_vals)) if check_bestcombo else None,
                "top_len": float(np.var(top_vals)) if check_bestcombo else None,
                "t": float(np.var(t_vals)) if check_bestcombo else None,
            },
            "std_combo": {
                "left_len": float(np.std(left_vals)) if check_bestcombo else None,
                "top_len": float(np.std(top_vals)) if check_bestcombo else None,
                "t": float(np.std(t_vals)) if check_bestcombo else None,
            },
            "mean_abs_diff": float(np.mean(abs_diff_vals)) if check_bestcombo else None,
            "var_abs_diff": float(np.var(abs_diff_vals)) if check_bestcombo else None,
            "std_abs_diff": float(np.std(abs_diff_vals)) if check_bestcombo else None,
            "mean_complete_gap": float(np.mean(complete)) if complete.size else 0.0,
            "best_combo_samples": [tuple(rec["combo"]) for rec in best] if check_bestcombo else [],
            "abs_diff_samples": abs_diff_vals.tolist() if check_bestcombo else [],
            "complete_accept_rate": float(complete_accept_num) / remained,
            "complete_reject_rate": float(complete_reject_num) / remained,
            "closest_partial_accept_rate": (float(partial_accept_num) / remained) if check_bestcombo else None,
            "closest_partial_reject_rate": (float(partial_reject_num) / remained) if check_bestcombo else None,
            "partial_accept_false_num": partial_accept_false_num if check_bestcombo else None,
            "partial_reject_false_num": partial_reject_false_num if check_bestcombo else None,
            "partial_accept_false_rate(over_remained_shots)": (float(partial_accept_false_num) / remained) if check_bestcombo else None,
            "partial_reject_false_rate(over_remained_shots)": (float(partial_reject_false_num) / remained) if check_bestcombo else None,
            "partial_accept_false_rate(over_complete_accept_num)": (float(partial_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0) if check_bestcombo else None,
            "partial_reject_false_rate(over_complete_reject_num)": (float(partial_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0) if check_bestcombo else None,
            "weighted_mean_abs_diff": float(np.mean(weighted_abs_diff_vals)),
            "weighted_var_abs_diff": float(np.var(weighted_abs_diff_vals)),
            "weighted_std_abs_diff": float(np.std(weighted_abs_diff_vals)),
            "weighted_multi_partial_accept_rate": float(weighted_accept_num) / remained,
            "weighted_multi_partial_reject_rate": float(weighted_reject_num) / remained,
            "weighted_accept_false_num": weighted_accept_false_num,
            "weighted_reject_false_num": weighted_reject_false_num,
            "weighted_accept_false_rate(over_remained_shots)": float(weighted_accept_false_num) / remained,
            "weighted_reject_false_rate(over_remained_shots)": float(weighted_reject_false_num) / remained,
            "weighted_accept_false_rate(over_complete_accept_num)": float(weighted_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0,
            "weighted_reject_false_rate(over_complete_reject_num)": float(weighted_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0,
            "pure_avg_mean_abs_diff": float(np.mean(pure_avg_abs_diff_vals)),
            "pure_avg_var_abs_diff": float(np.var(pure_avg_abs_diff_vals)),
            "pure_avg_std_abs_diff": float(np.std(pure_avg_abs_diff_vals)),
            "pure_avg_accept_rate": float(pure_avg_accept_num) / remained,
            "pure_avg_reject_rate": float(pure_avg_reject_num) / remained,
            "pure_avg_accept_false_num": pure_avg_accept_false_num,
            "pure_avg_reject_false_num": pure_avg_reject_false_num,
            "pure_avg_accept_false_rate(over_remained_shots)": float(pure_avg_accept_false_num) / remained,
            "pure_avg_reject_false_rate(over_remained_shots)": float(pure_avg_reject_false_num) / remained,
            "pure_avg_accept_false_rate(over_complete_accept_num)": float(pure_avg_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0,
            "pure_avg_reject_false_rate(over_complete_reject_num)": float(pure_avg_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0,
            "pure_avg_adjust_threshold_accept_rate": float(pure_avg_adjust_threshold_accept_num) / remained,
            "pure_avg_adjust_threshold_reject_rate": float(pure_avg_adjust_threshold_reject_num) / remained,
            "pure_avg_adjust_threshold_accept_false_num": pure_avg_adjust_threshold_accept_false_num,
            "pure_avg_adjust_threshold_reject_false_num": pure_avg_adjust_threshold_reject_false_num,
            "pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)": float(pure_avg_adjust_threshold_accept_false_num) / remained,
            "pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)": float(pure_avg_adjust_threshold_reject_false_num) / remained,
            "pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)": float(pure_avg_adjust_threshold_accept_false_num) / complete_accept_num if complete_accept_num > 0 else 0.0,
            "pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)": float(pure_avg_adjust_threshold_reject_false_num) / complete_reject_num if complete_reject_num > 0 else 0.0,
            "accept_false_data": accept_false_data,
            "reject_false_data": reject_false_data,
            "accept_correct_data": accept_correct_data,
            "reject_correct_data": reject_correct_data,
            "check_bestcombo": check_bestcombo,
            # NEW: total detector count used as denominator for ratio_over_complete_all.
            "complete_total_detectors": complete_total_detectors,
        }

    def print_partial_gap_best_combo_distribution(self) -> dict[str, Any]:
        if self.best_combo_distribution is None:
            raise ValueError(
                "No best combo distribution found. Call calculate_partial_gap_best_combo_distribution() first."
            )
        d = self.best_combo_distribution
        check_bestcombo = bool(d.get("check_bestcombo", True))
        print(f"remained_shots={d['remained_shots']}")
        if check_bestcombo:
            print(
                "mean_combo="
                f"(left_len={d['mean_combo']['left_len']:.6f}, "
                f"top_len={d['mean_combo']['top_len']:.6f}, "
                f"t={d['mean_combo']['t']:.6f})"
            )
            print(
                "var_combo="
                f"(left_len={d['var_combo']['left_len']:.6f}, "
                f"top_len={d['var_combo']['top_len']:.6f}, "
                f"t={d['var_combo']['t']:.6f})"
            )
            print(
                "std_combo="
                f"(left_len={d['std_combo']['left_len']:.6f}, "
                f"top_len={d['std_combo']['top_len']:.6f}, "
                f"t={d['std_combo']['t']:.6f})"
            )
            print(f"mean_abs_diff={d['mean_abs_diff']:.6f}")
            print(f"var_abs_diff={d['var_abs_diff']:.6f}")
            print(f"std_abs_diff={d['std_abs_diff']:.6f}")
        else:
            print("best_combo_metrics=SKIPPED (check_bestcombo=False)")
        print(f"mean_complete_gap={d['mean_complete_gap']:.6f}")
        print(f"complete_accept_rate={d['complete_accept_rate']:.4f}")
        print(f"complete_reject_rate={d['complete_reject_rate']:.4f}")
        if check_bestcombo:
            print(f"closest_partial_accept_rate={d['closest_partial_accept_rate']:.4f}")
            print(f"closest_partial_reject_rate={d['closest_partial_reject_rate']:.4f}")
            print(f"partial_accept_false_num={d['partial_accept_false_num']}")
            print(f"partial_reject_false_num={d['partial_reject_false_num']}")
        else:
            print("closest_partial_metrics=SKIPPED (check_bestcombo=False)")
        # print(f"partial_accept_false_rate(over_remained_shots)={d['partial_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"partial_reject_false_rate(over_remained_shots)={d['partial_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"partial_accept_false_rate(over_complete_accept_num)={d['partial_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"partial_reject_false_rate(over_complete_reject_num)={d['partial_reject_false_rate(over_complete_reject_num)']:.4f}")
        print(f"weighted_mean_abs_diff={d['weighted_mean_abs_diff']:.6f}")
        print(f"weighted_var_abs_diff={d['weighted_var_abs_diff']:.6f}")
        print(f"weighted_std_abs_diff={d['weighted_std_abs_diff']:.6f}")
        print(f"weighted_multi_partial_accept_rate={d['weighted_multi_partial_accept_rate']:.4f}")
        print(f"weighted_multi_partial_reject_rate={d['weighted_multi_partial_reject_rate']:.4f}")
        print(f"weighted_accept_false_num={d['weighted_accept_false_num']}")
        print(f"weighted_reject_false_num={d['weighted_reject_false_num']}")
        # print(f"weighted_accept_false_rate(over_remained_shots)={d['weighted_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"weighted_reject_false_rate(over_remained_shots)={d['weighted_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"weighted_accept_false_rate(over_complete_accept_num)={d['weighted_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"weighted_reject_false_rate(over_complete_reject_num)={d['weighted_reject_false_rate(over_complete_reject_num)']:.4f}")
        print(f"pure_avg_mean_abs_diff={d['pure_avg_mean_abs_diff']:.6f}")
        print(f"pure_avg_var_abs_diff={d['pure_avg_var_abs_diff']:.6f}")
        print(f"pure_avg_std_abs_diff={d['pure_avg_std_abs_diff']:.6f}")
        print(f"pure_avg_accept_rate={d['pure_avg_accept_rate']:.4f}")
        print(f"pure_avg_reject_rate={d['pure_avg_reject_rate']:.4f}")
        print(f"pure_avg_accept_false_num={d['pure_avg_accept_false_num']}")
        print(f"pure_avg_reject_false_num={d['pure_avg_reject_false_num']}")
        print(f"pure_avg_adjust_threshold_accept_rate={d['pure_avg_adjust_threshold_accept_rate']:.4f}")
        print(f"pure_avg_adjust_threshold_reject_rate={d['pure_avg_adjust_threshold_reject_rate']:.4f}")
        print(f"pure_avg_adjust_threshold_accept_false_num={d['pure_avg_adjust_threshold_accept_false_num']}")
        print(f"pure_avg_adjust_threshold_reject_false_num={d['pure_avg_adjust_threshold_reject_false_num']}")
        # print(f"pure_avg_accept_false_rate(over_remained_shots)={d['pure_avg_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_reject_false_rate(over_remained_shots)={d['pure_avg_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_accept_false_rate(over_complete_accept_num)={d['pure_avg_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"pure_avg_reject_false_rate(over_complete_reject_num)={d['pure_avg_reject_false_rate(over_complete_reject_num)']:.4f}")
        # print(f"pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)={d['pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)={d['pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)']:.4f}")
        # print(f"pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)={d['pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)']:.4f}")
        # print(f"pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)={d['pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)']:.4f}")

        return {
            "remained_shots": d["remained_shots"],
            "mean_combo": d["mean_combo"],
            "var_combo": d["var_combo"],
            "std_combo": d["std_combo"],
            "mean_abs_diff": d["mean_abs_diff"],
            "var_abs_diff": d["var_abs_diff"],
            "std_abs_diff": d["std_abs_diff"],
            "mean_complete_gap": d["mean_complete_gap"],
            "complete_accept_rate": d["complete_accept_rate"],
            "complete_reject_rate": d["complete_reject_rate"],
            "closest_partial_accept_rate": d["closest_partial_accept_rate"],
            "closest_partial_reject_rate": d["closest_partial_reject_rate"],
            "partial_accept_false_num": d["partial_accept_false_num"],
            "partial_reject_false_num": d["partial_reject_false_num"],
            "partial_accept_false_rate(over_remained_shots)": d["partial_accept_false_rate(over_remained_shots)"],
            "partial_reject_false_rate(over_remained_shots)": d["partial_reject_false_rate(over_remained_shots)"],
            "partial_accept_false_rate(over_complete_accept_num)": d["partial_accept_false_rate(over_complete_accept_num)"],
            "partial_reject_false_rate(over_complete_reject_num)": d["partial_reject_false_rate(over_complete_reject_num)"],
            "weighted_mean_abs_diff": d["weighted_mean_abs_diff"],
            "weighted_var_abs_diff": d["weighted_var_abs_diff"],
            "weighted_std_abs_diff": d["weighted_std_abs_diff"],
            "weighted_multi_partial_accept_rate": d["weighted_multi_partial_accept_rate"],
            "weighted_multi_partial_reject_rate": d["weighted_multi_partial_reject_rate"],
            "weighted_accept_false_num": d["weighted_accept_false_num"],
            "weighted_reject_false_num": d["weighted_reject_false_num"],
            "weighted_accept_false_rate(over_remained_shots)": d["weighted_accept_false_rate(over_remained_shots)"],
            "weighted_reject_false_rate(over_remained_shots)": d["weighted_reject_false_rate(over_remained_shots)"],
            "weighted_accept_false_rate(over_complete_accept_num)": d["weighted_accept_false_rate(over_complete_accept_num)"],
            "weighted_reject_false_rate(over_complete_reject_num)": d["weighted_reject_false_rate(over_complete_reject_num)"],
            "pure_avg_mean_abs_diff": d["pure_avg_mean_abs_diff"],
            "pure_avg_var_abs_diff": d["pure_avg_var_abs_diff"],
            "pure_avg_std_abs_diff": d["pure_avg_std_abs_diff"],
            "pure_avg_accept_rate": d["pure_avg_accept_rate"],
            "pure_avg_reject_rate": d["pure_avg_reject_rate"],
            "pure_avg_accept_false_num": d["pure_avg_accept_false_num"],
            "pure_avg_reject_false_num": d["pure_avg_reject_false_num"],
            "pure_avg_accept_false_rate(over_remained_shots)": d["pure_avg_accept_false_rate(over_remained_shots)"],
            "pure_avg_reject_false_rate(over_remained_shots)": d["pure_avg_reject_false_rate(over_remained_shots)"],
            "pure_avg_accept_false_rate(over_complete_accept_num)": d["pure_avg_accept_false_rate(over_complete_accept_num)"],
            "pure_avg_reject_false_rate(over_complete_reject_num)": d["pure_avg_reject_false_rate(over_complete_reject_num)"],
            "pure_avg_adjust_threshold_accept_rate": d["pure_avg_adjust_threshold_accept_rate"],
            "pure_avg_adjust_threshold_reject_rate": d["pure_avg_adjust_threshold_reject_rate"],
            "pure_avg_adjust_threshold_accept_false_num": d["pure_avg_adjust_threshold_accept_false_num"],
            "pure_avg_adjust_threshold_reject_false_num": d["pure_avg_adjust_threshold_reject_false_num"],
            "pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)": d["pure_avg_adjust_threshold_accept_false_rate(over_remained_shots)"],
            "pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)": d["pure_avg_adjust_threshold_reject_false_rate(over_remained_shots)"],
            "pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)": d["pure_avg_adjust_threshold_accept_false_rate(over_complete_accept_num)"],
            "pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)": d["pure_avg_adjust_threshold_reject_false_rate(over_complete_reject_num)"],
            "accept_false_data": d["accept_false_data"],
            "reject_false_data": d["reject_false_data"],
        }

    def false_accept_reject_detail_plot(
        self,
        *,
        select_type: str = "best_combo",
        first_shots_num: Optional[int] = None,
        path: Optional[pathlib.Path] = None,
        # NEW: if not None, replaces the detector-density panels with the chosen pre-computed ratio.
        # Options: "over_complete_nontrivial", "over_region_all", "over_complete_all".
        ratio_type: Optional[str] = None,
        # NEW: which shot category to plot. "false" = misclassified (default, unchanged behavior),
        # "correct" = agreed accept/reject between complete and partial schemes.
        category: str = "false",
    ):
        if self.best_combo_distribution is None:
            raise ValueError("No best combo distribution found. Call calculate_partial_gap_best_combo_distribution() first.")
        if select_type not in {"best_combo", "weighted_regions", "pure_avg"}:
            raise ValueError("select_type must be one of: 'best_combo', 'weighted_regions', 'pure_avg'.")
        if category not in {"false", "correct"}:
            raise ValueError("category must be one of: 'false', 'correct'.")
        # NEW: validate ratio_type.
        _valid_ratio_types = {"over_complete_nontrivial", "over_region_all", "over_complete_all"}
        if ratio_type is not None and ratio_type not in _valid_ratio_types:
            raise ValueError(f"ratio_type must be one of {_valid_ratio_types} or None.")

        data_accept = self.best_combo_distribution[f"accept_{category}_data"][select_type]
        data_reject = self.best_combo_distribution[f"reject_{category}_data"][select_type]
        cat_label = category.capitalize()
        if first_shots_num is not None and first_shots_num <= 0:
            raise ValueError("first_shots_num must be positive when provided.")

        # NEW: determine which data field and y-axis label to use for the density panels.
        # ratio_type is e.g. "over_complete_nontrivial"; the stored key has a "ratio_" prefix.
        density_field = f"ratio_{ratio_type}" if ratio_type is not None else "det_density"
        density_ylabel = ratio_type if ratio_type is not None else "Detector density"

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=120)
        if select_type == "weighted_regions":
            from matplotlib.patches import Patch

            density_colors = ["#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2"]
            gap_colors = ["#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#EDC948"]
            shot_sep = 2.0  # Extra separation between false shots.

            # Newly added weighted plot mode: no spacing inside one shot-group, only between shots.
            def plot_weighted(ax, data: dict[str, Any], *, is_gap: bool, title: str) -> None:
                n_total = len(data["complete_gap_value"] if is_gap else data["complete_det_density"])
                n = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                cat_n = 6 if is_gap else 5
                if n == 0:
                    ax.set_title(f"{select_type} | {title} (no data)")
                    ax.grid(True, axis="y", alpha=0.25)
                    return

                xs: list[float] = []
                masked_vals: list[float] = []
                complete_vals: list[float] = []
                color_vals: list[str] = []
                centers: list[float] = []
                for i in range(n):
                    base = i * (cat_n + shot_sep)
                    centers.append(base + (cat_n - 1) / 2.0)
                    for c in range(cat_n):
                        x = base + c
                        xs.append(x)
                        if is_gap:
                            masked_vals.append(float(data["gap_value"][c][i]))
                            complete_vals.append(float(data["complete_gap_value"][i]))
                            color_vals.append(gap_colors[c])
                        else:
                            # NEW: use density_field to select det_density or a pre-computed ratio.
                            masked_vals.append(float(data[density_field][c][i]))
                            complete_vals.append(float(data["complete_det_density"][i]))
                            color_vals.append(density_colors[c])

                # NEW: only draw the complete reference bar when showing raw density (ratio plots omit it
                # because the complete bar would use a different denominator and be misleading).
                if not is_gap and ratio_type is None:
                    ax.bar(xs, complete_vals, width=0.95, color="#4c78a8", alpha=0.35, label="Complete")
                elif is_gap:
                    ax.bar(xs, complete_vals, width=0.95, color="#4c78a8", alpha=0.35, label="Complete")
                ax.bar(xs, masked_vals, width=0.72, color=color_vals, alpha=0.85)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(centers)
                ax.set_xticklabels([f"s{i}" for i in range(n)], rotation=0, fontsize=8)
                ax.grid(True, axis="y", alpha=0.25)

                if is_gap:
                    handles = [
                        Patch(color="#4c78a8", alpha=0.35, label="Complete"),
                        Patch(color=gap_colors[0], label="Region1 gap"),
                        Patch(color=gap_colors[1], label="Region2 gap"),
                        Patch(color=gap_colors[2], label="Region3 gap"),
                        Patch(color=gap_colors[3], label="Region4 gap"),
                        Patch(color=gap_colors[4], label="Region5 gap"),
                        Patch(color=gap_colors[5], label="Weighted avg gap"),
                    ]
                else:
                    # NEW: legend label changes based on whether we're showing density or a ratio.
                    region_label = "density" if ratio_type is None else ratio_type
                    handles_list = [] if ratio_type is not None else [Patch(color="#4c78a8", alpha=0.35, label="Complete")]
                    handles_list += [
                        Patch(color=density_colors[0], label=f"Region1 {region_label}"),
                        Patch(color=density_colors[1], label=f"Region2 {region_label}"),
                        Patch(color=density_colors[2], label=f"Region3 {region_label}"),
                        Patch(color=density_colors[3], label=f"Region4 {region_label}"),
                        Patch(color=density_colors[4], label=f"Region5 {region_label}"),
                    ]
                    handles = handles_list
                ax.legend(handles=handles, loc="upper right", fontsize=8)

            plot_weighted(axes[0, 0], data_accept, is_gap=False, title=f"Accept {cat_label}: {density_ylabel}")
            plot_weighted(axes[0, 1], data_accept, is_gap=True, title=f"Accept {cat_label}: Gap")
            plot_weighted(axes[1, 0], data_reject, is_gap=False, title=f"Reject {cat_label}: {density_ylabel}")
            plot_weighted(axes[1, 1], data_reject, is_gap=True, title=f"Reject {cat_label}: Gap")
        else:
            # Build plotting vectors (x labels + masked/complete values) for non-weighted select types.
            # NEW: build_density_points reads from density_field instead of always "det_density".
            def build_density_points(data: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
                labels: list[str] = []
                masked: list[float] = []
                complete: list[float] = []
                n_total = len(data["complete_det_density"])
                n = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                for i in range(n):
                    labels.append(f"s{i}")
                    masked.append(float(data[density_field][i]))
                    complete.append(float(data["complete_det_density"][i]))
                return labels, masked, complete

            def build_gap_points(data: dict[str, Any]) -> tuple[list[str], list[float], list[float]]:
                labels: list[str] = []
                masked: list[float] = []
                complete: list[float] = []
                n_total = len(data["complete_gap_value"])
                n = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                for i in range(n):
                    labels.append(f"s{i}")
                    masked.append(float(data["gap_value"][i]))
                    complete.append(float(data["complete_gap_value"][i]))
                return labels, masked, complete

            a_dx, a_dm, a_dc = build_density_points(data_accept)
            a_gx, a_gm, a_gc = build_gap_points(data_accept)
            r_dx, r_dm, r_dc = build_density_points(data_reject)
            r_gx, r_gm, r_gc = build_gap_points(data_reject)

            density_panels = [
                (axes[0, 0], a_dx, a_dm, a_dc, f"Accept {cat_label}: {density_ylabel}"),
                (axes[1, 0], r_dx, r_dm, r_dc, f"Reject {cat_label}: {density_ylabel}"),
            ]
            gap_panels = [
                (axes[0, 1], a_gx, a_gm, a_gc, f"Accept {cat_label}: Gap"),
                (axes[1, 1], r_gx, r_gm, r_gc, f"Reject {cat_label}: Gap"),
            ]
            for ax, labels, masked_vals, complete_vals, title in density_panels:
                x = np.arange(len(labels), dtype=np.float64)
                # NEW: only draw complete reference bar when showing raw density.
                if ratio_type is None:
                    ax.bar(x, complete_vals, width=0.80, color="#4c78a8", alpha=0.40, label="Complete")
                ax.bar(x, masked_vals, width=0.58, color="#f28e2b", alpha=0.75, label=select_type)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(x)
                if len(labels) <= 60:
                    ax.set_xticklabels(labels, rotation=90, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
            for ax, labels, masked_vals, complete_vals, title in gap_panels:
                x = np.arange(len(labels), dtype=np.float64)
                ax.bar(x, complete_vals, width=0.80, color="#4c78a8", alpha=0.40, label="Complete")
                ax.bar(x, masked_vals, width=0.58, color="#f28e2b", alpha=0.75, label=select_type)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(x)
                if len(labels) <= 60:
                    ax.set_xticklabels(labels, rotation=90, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
        axes[0, 0].set_ylabel(density_ylabel)
        axes[1, 0].set_ylabel(density_ylabel)
        axes[0, 1].set_ylabel("Gap value")
        axes[1, 1].set_ylabel("Gap value")
        fig.tight_layout()

        if path is not None:
            if path.suffix.lower() == ".svg":
                fig.savefig(path, format="svg")
            else:
                fig.savefig(path.with_suffix(".svg"), format="svg")
        return fig

    # ------------------------------------------------------------------
    # Flex pipeline: input-mask driven (no best_combo, no selector args).
    # State is kept in self.flex_partial_gap_sens_data and self.flex_distribution.
    # ------------------------------------------------------------------

    def collect_partial_gap_sens_with_input_masks(
        self,
        *,
        shots: int,
        sampler = cultiv.DesaturationSampler(),
        weighted_masks: list[np.ndarray],
        gap_average_parameters: Optional[list[float]] = None,
        pure_avg_mask: np.ndarray,
    ) -> None:
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if not weighted_masks:
            raise ValueError("weighted_masks must be a non-empty list of masks.")
        n = len(weighted_masks)
        if gap_average_parameters is None:
            gap_average_parameters = [1.0] + [0.0] * (n - 1)
        if len(gap_average_parameters) != n:
            raise ValueError(
                f"gap_average_parameters length ({len(gap_average_parameters)}) "
                f"must equal number of weighted_masks ({n})."
            )
        gap_average_parameters = [float(x) for x in gap_average_parameters]

        dec = sampler.compiled_sampler_for_task(
            sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        )
        dets, actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
        keep_mask = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        remained_shots = int(dets.shape[0])

        if remained_shots == 0:
            self.flex_partial_gap_sens_data = {
                "remained_shots": 0,
                "num_weighted_regions": n,
                "complete_gaps": [],
                "weighted_region_gaps_per_shot": [],
                "weighted_avg_gap_per_shot": [],
                "min_gap_per_shot": [],
                "pure_avg_gap_per_shot": [],
                "complete_det_packed_per_shot": [],
                "weighted_region_det_packed_per_shot": [],
                "pure_avg_det_packed_per_shot": [],
                "weighted_region_sizes": [0] * n,
                "pure_avg_region_size": 0,
                "gap_average_parameters": gap_average_parameters,
            }
            self.flex_distribution = None
            return

        width = dets.shape[1]
        weighted_full_masks = [self._normalize_mask_to_packed(m, width) for m in weighted_masks]
        pure_full_mask = self._normalize_mask_to_packed(pure_avg_mask, width)
        weighted_region_sizes = [self._mask_size(m) for m in weighted_full_masks]
        pure_avg_region_size = self._mask_size(pure_full_mask)

        # Complete gap once for all surviving shots.
        _, complete_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        complete_gaps = complete_gaps.astype(np.float64)

        # Per-region gaps for all N weighted masks (vectorized over shots, per mask).
        region_gaps_per_mask: list[np.ndarray] = []
        region_dets_per_mask: list[np.ndarray] = []   # packed dets after AND with mask
        for full_mask in weighted_full_masks:
            d_partial = dets.copy()
            d_partial &= full_mask.reshape(1, -1)
            _, rg = dec._decode_batch_overwrite_last_byte(bit_packed_dets=d_partial.copy())
            region_gaps_per_mask.append(rg.astype(np.float64))
            region_dets_per_mask.append(d_partial)

        # Pure-avg gap.
        d_pure = dets.copy()
        d_pure &= pure_full_mask.reshape(1, -1)
        _, pure_gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=d_pure.copy())
        pure_gaps = pure_gaps.astype(np.float64)

        # Weighted average per shot using user-provided weights.
        weights = np.asarray(gap_average_parameters, dtype=np.float64)
        stacked = np.vstack(region_gaps_per_mask)            # shape (N, S)
        weighted_avg_per_shot = (weights[:, None] * stacked).sum(axis=0)
        # Min gap per shot across all N masks (equal-weight worst-case-confidence aggregator).
        min_gap_per_shot = stacked.min(axis=0)

        # Per-shot snapshots for downstream density / ratio computation.
        weighted_region_gaps_per_shot: list[list[float]] = []
        weighted_region_det_packed_per_shot: list[list[list[int]]] = []
        for s in range(remained_shots):
            weighted_region_gaps_per_shot.append([float(region_gaps_per_mask[r][s]) for r in range(n)])
            weighted_region_det_packed_per_shot.append(
                [region_dets_per_mask[r][s].astype(np.uint8).tolist() for r in range(n)]
            )

        self.flex_partial_gap_sens_data = {
            "remained_shots": remained_shots,
            "num_weighted_regions": n,
            "complete_gaps": complete_gaps.tolist(),
            "weighted_region_gaps_per_shot": weighted_region_gaps_per_shot,
            "weighted_avg_gap_per_shot": weighted_avg_per_shot.tolist(),
            "min_gap_per_shot": min_gap_per_shot.tolist(),
            "pure_avg_gap_per_shot": pure_gaps.tolist(),
            "complete_det_packed_per_shot": [dets[s].astype(np.uint8).tolist() for s in range(remained_shots)],
            "weighted_region_det_packed_per_shot": weighted_region_det_packed_per_shot,
            "pure_avg_det_packed_per_shot": [d_pure[s].astype(np.uint8).tolist() for s in range(remained_shots)],
            "weighted_region_sizes": weighted_region_sizes,
            "pure_avg_region_size": pure_avg_region_size,
            "gap_average_parameters": gap_average_parameters,
        }
        self.flex_distribution = None

    def calculate_partial_gap_distribution_with_input_masks(
        self,
        *,
        gap_threshold: float,
        gap_threshold_pure_avg: Optional[float] = None,
        max_correct_shots: Optional[int] = None,
    ) -> None:
        data = self.flex_partial_gap_sens_data
        if data is None:
            raise ValueError("No flex partial gap data found. Call collect_partial_gap_sens_with_input_masks(...) first.")

        n = int(data["num_weighted_regions"])
        remained = int(data["remained_shots"])
        thr = float(gap_threshold)
        thr_pure = float(gap_threshold_pure_avg) if gap_threshold_pure_avg is not None else thr
        cap_correct = int(max_correct_shots) if max_correct_shots is not None else None

        if remained == 0:
            self.flex_distribution = {
                "remained_shots": 0,
                "num_weighted_regions": n,
                "gap_threshold": thr,
                "gap_threshold_pure_avg": thr_pure,
                "complete_accept_rate": 1.0, "complete_reject_rate": 0.0,
                "weighted_accept_rate": 1.0, "weighted_reject_rate": 0.0,
                "weighted_accept_false_num": 0, "weighted_reject_false_num": 0,
                "weighted_accept_false_rate(over_remained_shots)": 0.0,
                "weighted_reject_false_rate(over_remained_shots)": 0.0,
                "weighted_accept_false_rate(over_complete_accept_num)": 0.0,
                "weighted_reject_false_rate(over_complete_reject_num)": 0.0,
                "min_accept_rate": 1.0, "min_reject_rate": 0.0,
                "min_accept_false_num": 0, "min_reject_false_num": 0,
                "min_accept_false_rate(over_remained_shots)": 0.0,
                "min_reject_false_rate(over_remained_shots)": 0.0,
                "min_accept_false_rate(over_complete_accept_num)": 0.0,
                "min_reject_false_rate(over_complete_reject_num)": 0.0,
                "pure_avg_accept_rate": 1.0, "pure_avg_reject_rate": 0.0,
                "pure_avg_accept_false_num": 0, "pure_avg_reject_false_num": 0,
                "pure_avg_accept_false_rate(over_remained_shots)": 0.0,
                "pure_avg_reject_false_rate(over_remained_shots)": 0.0,
                "pure_avg_accept_false_rate(over_complete_accept_num)": 0.0,
                "pure_avg_reject_false_rate(over_complete_reject_num)": 0.0,
                "pure_avg_adjust_threshold_accept_rate": 1.0,
                "pure_avg_adjust_threshold_reject_rate": 0.0,
                "pure_avg_adjust_threshold_accept_false_num": 0,
                "pure_avg_adjust_threshold_reject_false_num": 0,
                "accept_false_data": self._empty_flex_data_block(num_weighted_regions=n),
                "reject_false_data": self._empty_flex_data_block(num_weighted_regions=n),
                "accept_correct_data": self._empty_flex_data_block(num_weighted_regions=n),
                "reject_correct_data": self._empty_flex_data_block(num_weighted_regions=n),
                "complete_total_detectors": int(self.circuit.num_detectors),
                "weighted_region_sizes": list(data.get("weighted_region_sizes", [0] * n)),
                "pure_avg_region_size": int(data.get("pure_avg_region_size", 0)),
            }
            return

        complete_gaps = np.asarray(data["complete_gaps"], dtype=np.float64)
        weighted_avg = np.asarray(data["weighted_avg_gap_per_shot"], dtype=np.float64)
        min_gap = np.asarray(data["min_gap_per_shot"], dtype=np.float64)
        pure_gaps = np.asarray(data["pure_avg_gap_per_shot"], dtype=np.float64)

        complete_accept = complete_gaps >= thr
        weighted_accept = weighted_avg >= thr
        min_accept = min_gap >= thr
        pure_accept = pure_gaps >= thr
        pure_accept_adj = pure_gaps >= thr_pure

        complete_accept_num = int(complete_accept.sum())
        complete_reject_num = remained - complete_accept_num
        weighted_accept_num = int(weighted_accept.sum())
        weighted_reject_num = remained - weighted_accept_num
        min_accept_num = int(min_accept.sum())
        min_reject_num = remained - min_accept_num
        pure_accept_num = int(pure_accept.sum())
        pure_reject_num = remained - pure_accept_num
        pure_adj_accept_num = int(pure_accept_adj.sum())
        pure_adj_reject_num = remained - pure_adj_accept_num

        weighted_accept_false_num = int((weighted_accept & ~complete_accept).sum())
        weighted_reject_false_num = int((~weighted_accept & complete_accept).sum())
        min_accept_false_num = int((min_accept & ~complete_accept).sum())
        min_reject_false_num = int((~min_accept & complete_accept).sum())
        pure_accept_false_num = int((pure_accept & ~complete_accept).sum())
        pure_reject_false_num = int((~pure_accept & complete_accept).sum())
        pure_adj_accept_false_num = int((pure_accept_adj & ~complete_accept).sum())
        pure_adj_reject_false_num = int((~pure_accept_adj & complete_accept).sum())

        accept_false = self._empty_flex_data_block(num_weighted_regions=n)
        reject_false = self._empty_flex_data_block(num_weighted_regions=n)
        accept_correct = self._empty_flex_data_block(num_weighted_regions=n)
        reject_correct = self._empty_flex_data_block(num_weighted_regions=n)

        def _under_cap(target: dict, scheme: str) -> bool:
            if cap_correct is None:
                return True
            if scheme in ("weighted_regions", "min_regions"):
                return len(target[scheme]["det_density"][0]) < cap_correct
            return len(target[scheme]["det_density"]) < cap_correct

        complete_total_dets = self.circuit.num_detectors
        weighted_region_sizes = data["weighted_region_sizes"]
        pure_size = int(data["pure_avg_region_size"])
        weighted_region_gaps_per_shot = data["weighted_region_gaps_per_shot"]
        weighted_region_det_packed_per_shot = data["weighted_region_det_packed_per_shot"]
        complete_det_packed_per_shot = data["complete_det_packed_per_shot"]
        pure_avg_det_packed_per_shot = data["pure_avg_det_packed_per_shot"]

        for i in range(remained):
            ca = bool(complete_accept[i])
            wa = bool(weighted_accept[i])
            ma = bool(min_accept[i])
            pa = bool(pure_accept[i])

            cd = self._det_density_from_packed_list(complete_det_packed_per_shot[i])
            cg = float(complete_gaps[i])

            # ----- region-aggregated per shot (shared across weighted_regions and min_regions) -----
            wr_dens = [
                self._det_density_from_packed_list(weighted_region_det_packed_per_shot[i][r])
                for r in range(n)
            ]
            wr_gaps = list(weighted_region_gaps_per_shot[i])
            # gap_value layout: [region_1, ..., region_N, weighted_avg, min_gap]
            wr_all_gaps = wr_gaps + [float(weighted_avg[i]), float(min_gap[i])]
            wr_r1 = [(wr_dens[r] / cd) if cd > 0 else 0.0 for r in range(n)]
            wr_r2 = [
                (wr_dens[r] / weighted_region_sizes[r]) if weighted_region_sizes[r] > 0 else 0.0
                for r in range(n)
            ]
            wr_r3 = [(wr_dens[r] / complete_total_dets) if complete_total_dets > 0 else 0.0 for r in range(n)]

            def _push_region_block(target: dict, scheme: str) -> None:
                for r in range(n):
                    target[scheme]["det_density"][r].append(wr_dens[r])
                    target[scheme]["ratio_over_complete_nontrivial"][r].append(wr_r1[r])
                    target[scheme]["ratio_over_region_all"][r].append(wr_r2[r])
                    target[scheme]["ratio_over_complete_all"][r].append(wr_r3[r])
                for g in range(n + 2):
                    target[scheme]["gap_value"][g].append(wr_all_gaps[g])
                target[scheme]["complete_det_density"].append(cd)
                target[scheme]["complete_gap_value"].append(cg)

            # weighted_regions block: classified by weighted-rule
            if wa and (not ca):
                _push_region_block(accept_false, "weighted_regions")
            if (not wa) and ca:
                _push_region_block(reject_false, "weighted_regions")
            if wa and ca and _under_cap(accept_correct, "weighted_regions"):
                _push_region_block(accept_correct, "weighted_regions")
            if (not wa) and (not ca) and _under_cap(reject_correct, "weighted_regions"):
                _push_region_block(reject_correct, "weighted_regions")

            # min_regions block: classified by min-rule (same per-shot gap/density values, different routing)
            if ma and (not ca):
                _push_region_block(accept_false, "min_regions")
            if (not ma) and ca:
                _push_region_block(reject_false, "min_regions")
            if ma and ca and _under_cap(accept_correct, "min_regions"):
                _push_region_block(accept_correct, "min_regions")
            if (not ma) and (not ca) and _under_cap(reject_correct, "min_regions"):
                _push_region_block(reject_correct, "min_regions")

            # ----- pure_avg per shot -----
            pd = self._det_density_from_packed_list(pure_avg_det_packed_per_shot[i])
            pg = float(pure_gaps[i])
            pr_r1 = (pd / cd) if cd > 0 else 0.0
            pr_r2 = (pd / pure_size) if pure_size > 0 else 0.0
            pr_r3 = (pd / complete_total_dets) if complete_total_dets > 0 else 0.0

            def _push_pure(target: dict) -> None:
                target["pure_avg"]["det_density"].append(pd)
                target["pure_avg"]["complete_det_density"].append(cd)
                target["pure_avg"]["gap_value"].append(pg)
                target["pure_avg"]["complete_gap_value"].append(cg)
                target["pure_avg"]["ratio_over_complete_nontrivial"].append(pr_r1)
                target["pure_avg"]["ratio_over_region_all"].append(pr_r2)
                target["pure_avg"]["ratio_over_complete_all"].append(pr_r3)

            if pa and (not ca):
                _push_pure(accept_false)
            if (not pa) and ca:
                _push_pure(reject_false)
            if pa and ca and _under_cap(accept_correct, "pure_avg"):
                _push_pure(accept_correct)
            if (not pa) and (not ca) and _under_cap(reject_correct, "pure_avg"):
                _push_pure(reject_correct)

        self.flex_distribution = {
            "remained_shots": remained,
            "num_weighted_regions": n,
            "gap_threshold": thr,
            "gap_threshold_pure_avg": thr_pure,
            "gap_average_parameters": list(data["gap_average_parameters"]),

            "complete_accept_rate": complete_accept_num / remained,
            "complete_reject_rate": complete_reject_num / remained,

            "weighted_accept_rate": weighted_accept_num / remained,
            "weighted_reject_rate": weighted_reject_num / remained,
            "weighted_accept_false_num": weighted_accept_false_num,
            "weighted_reject_false_num": weighted_reject_false_num,
            "weighted_accept_false_rate(over_remained_shots)": weighted_accept_false_num / remained,
            "weighted_reject_false_rate(over_remained_shots)": weighted_reject_false_num / remained,
            "weighted_accept_false_rate(over_complete_accept_num)":
                (weighted_accept_false_num / complete_accept_num) if complete_accept_num > 0 else 0.0,
            "weighted_reject_false_rate(over_complete_reject_num)":
                (weighted_reject_false_num / complete_reject_num) if complete_reject_num > 0 else 0.0,

            "min_accept_rate": min_accept_num / remained,
            "min_reject_rate": min_reject_num / remained,
            "min_accept_false_num": min_accept_false_num,
            "min_reject_false_num": min_reject_false_num,
            "min_accept_false_rate(over_remained_shots)": min_accept_false_num / remained,
            "min_reject_false_rate(over_remained_shots)": min_reject_false_num / remained,
            "min_accept_false_rate(over_complete_accept_num)":
                (min_accept_false_num / complete_accept_num) if complete_accept_num > 0 else 0.0,
            "min_reject_false_rate(over_complete_reject_num)":
                (min_reject_false_num / complete_reject_num) if complete_reject_num > 0 else 0.0,

            "pure_avg_accept_rate": pure_accept_num / remained,
            "pure_avg_reject_rate": pure_reject_num / remained,
            "pure_avg_accept_false_num": pure_accept_false_num,
            "pure_avg_reject_false_num": pure_reject_false_num,
            "pure_avg_accept_false_rate(over_remained_shots)": pure_accept_false_num / remained,
            "pure_avg_reject_false_rate(over_remained_shots)": pure_reject_false_num / remained,
            "pure_avg_accept_false_rate(over_complete_accept_num)":
                (pure_accept_false_num / complete_accept_num) if complete_accept_num > 0 else 0.0,
            "pure_avg_reject_false_rate(over_complete_reject_num)":
                (pure_reject_false_num / complete_reject_num) if complete_reject_num > 0 else 0.0,

            "pure_avg_adjust_threshold_accept_rate": pure_adj_accept_num / remained,
            "pure_avg_adjust_threshold_reject_rate": pure_adj_reject_num / remained,
            "pure_avg_adjust_threshold_accept_false_num": pure_adj_accept_false_num,
            "pure_avg_adjust_threshold_reject_false_num": pure_adj_reject_false_num,

            "accept_false_data": accept_false,
            "reject_false_data": reject_false,
            "accept_correct_data": accept_correct,
            "reject_correct_data": reject_correct,

            # Mask-coverage metadata (passed through from collect step).
            "complete_total_detectors": int(self.circuit.num_detectors),
            "weighted_region_sizes": list(weighted_region_sizes),
            "pure_avg_region_size": int(pure_size),
        }

    def print_partial_gap_distribution_with_input_masks(self) -> dict[str, Any]:
        if self.flex_distribution is None:
            raise ValueError(
                "No flex distribution found. Call calculate_partial_gap_distribution_with_input_masks(...) first."
            )
        d = self.flex_distribution
        print(f"remained_shots={d['remained_shots']}")
        print(f"num_weighted_regions={d['num_weighted_regions']}")
        print(f"gap_threshold={d['gap_threshold']}")
        print(f"gap_threshold_pure_avg={d['gap_threshold_pure_avg']}")
        print(f"gap_average_parameters={d.get('gap_average_parameters', [])}")
        # Mask coverage ratios — m/n = r (m = mask popcount, n = total circuit detectors).
        n_total = int(d.get("complete_total_detectors", 0))
        m_pure = int(d.get("pure_avg_region_size", 0))
        r_pure = (m_pure / n_total) if n_total > 0 else 0.0
        print(f"pure_avg_mask_size: {m_pure}/{n_total} = {r_pure:.6f}")
        wr_sizes = d.get("weighted_region_sizes", [])
        for i, m_w in enumerate(wr_sizes):
            r_w = (int(m_w) / n_total) if n_total > 0 else 0.0
            print(f"weighted_region[{i}]_mask_size: {int(m_w)}/{n_total} = {r_w:.6f}")
        print(f"complete_accept_rate={d['complete_accept_rate']:.4f}")
        print(f"complete_reject_rate={d['complete_reject_rate']:.4f}")

        print(f"weighted_accept_rate={d['weighted_accept_rate']:.4f}")
        print(f"weighted_reject_rate={d['weighted_reject_rate']:.4f}")
        print(f"weighted_accept_false_num={d['weighted_accept_false_num']}")
        print(f"weighted_reject_false_num={d['weighted_reject_false_num']}")

        print(f"min_accept_rate={d['min_accept_rate']:.4f}")
        print(f"min_reject_rate={d['min_reject_rate']:.4f}")
        print(f"min_accept_false_num={d['min_accept_false_num']}")
        print(f"min_reject_false_num={d['min_reject_false_num']}")

        print(f"pure_avg_accept_rate={d['pure_avg_accept_rate']:.4f}")
        print(f"pure_avg_reject_rate={d['pure_avg_reject_rate']:.4f}")
        print(f"pure_avg_accept_false_num={d['pure_avg_accept_false_num']}")
        print(f"pure_avg_reject_false_num={d['pure_avg_reject_false_num']}")

        print(f"pure_avg_adjust_threshold_accept_rate={d['pure_avg_adjust_threshold_accept_rate']:.4f}")
        print(f"pure_avg_adjust_threshold_reject_rate={d['pure_avg_adjust_threshold_reject_rate']:.4f}")
        print(f"pure_avg_adjust_threshold_accept_false_num={d['pure_avg_adjust_threshold_accept_false_num']}")
        print(f"pure_avg_adjust_threshold_reject_false_num={d['pure_avg_adjust_threshold_reject_false_num']}")

        return {
            "remained_shots": d["remained_shots"],
            "num_weighted_regions": d["num_weighted_regions"],
            "gap_threshold": d["gap_threshold"],
            "gap_threshold_pure_avg": d["gap_threshold_pure_avg"],
            "gap_average_parameters": d.get("gap_average_parameters", []),
            "complete_accept_rate": d["complete_accept_rate"],
            "complete_reject_rate": d["complete_reject_rate"],
            "weighted_accept_rate": d["weighted_accept_rate"],
            "weighted_reject_rate": d["weighted_reject_rate"],
            "weighted_accept_false_num": d["weighted_accept_false_num"],
            "weighted_reject_false_num": d["weighted_reject_false_num"],
            "weighted_accept_false_rate(over_remained_shots)": d["weighted_accept_false_rate(over_remained_shots)"],
            "weighted_reject_false_rate(over_remained_shots)": d["weighted_reject_false_rate(over_remained_shots)"],
            "weighted_accept_false_rate(over_complete_accept_num)": d["weighted_accept_false_rate(over_complete_accept_num)"],
            "weighted_reject_false_rate(over_complete_reject_num)": d["weighted_reject_false_rate(over_complete_reject_num)"],
            "min_accept_rate": d["min_accept_rate"],
            "min_reject_rate": d["min_reject_rate"],
            "min_accept_false_num": d["min_accept_false_num"],
            "min_reject_false_num": d["min_reject_false_num"],
            "min_accept_false_rate(over_remained_shots)": d["min_accept_false_rate(over_remained_shots)"],
            "min_reject_false_rate(over_remained_shots)": d["min_reject_false_rate(over_remained_shots)"],
            "min_accept_false_rate(over_complete_accept_num)": d["min_accept_false_rate(over_complete_accept_num)"],
            "min_reject_false_rate(over_complete_reject_num)": d["min_reject_false_rate(over_complete_reject_num)"],
            "pure_avg_accept_rate": d["pure_avg_accept_rate"],
            "pure_avg_reject_rate": d["pure_avg_reject_rate"],
            "pure_avg_accept_false_num": d["pure_avg_accept_false_num"],
            "pure_avg_reject_false_num": d["pure_avg_reject_false_num"],
            "pure_avg_accept_false_rate(over_remained_shots)": d["pure_avg_accept_false_rate(over_remained_shots)"],
            "pure_avg_reject_false_rate(over_remained_shots)": d["pure_avg_reject_false_rate(over_remained_shots)"],
            "pure_avg_accept_false_rate(over_complete_accept_num)": d["pure_avg_accept_false_rate(over_complete_accept_num)"],
            "pure_avg_reject_false_rate(over_complete_reject_num)": d["pure_avg_reject_false_rate(over_complete_reject_num)"],
            "pure_avg_adjust_threshold_accept_rate": d["pure_avg_adjust_threshold_accept_rate"],
            "pure_avg_adjust_threshold_reject_rate": d["pure_avg_adjust_threshold_reject_rate"],
            "pure_avg_adjust_threshold_accept_false_num": d["pure_avg_adjust_threshold_accept_false_num"],
            "pure_avg_adjust_threshold_reject_false_num": d["pure_avg_adjust_threshold_reject_false_num"],
            "accept_false_data": d["accept_false_data"],
            "reject_false_data": d["reject_false_data"],
            "accept_correct_data": d["accept_correct_data"],
            "reject_correct_data": d["reject_correct_data"],
            "complete_total_detectors": d.get("complete_total_detectors", 0),
            "pure_avg_region_size": d.get("pure_avg_region_size", 0),
            "weighted_region_sizes": d.get("weighted_region_sizes", []),
            "pure_avg_region_ratio": (
                d.get("pure_avg_region_size", 0) / d["complete_total_detectors"]
                if d.get("complete_total_detectors", 0) > 0 else 0.0
            ),
        }

    def partial_gap_detail_plot_with_input_masks(
        self,
        *,
        select_type: str,
        category: str = "false",
        first_shots_num: Optional[int] = None,
        ratio_type: Optional[str] = None,
        path: Optional[pathlib.Path] = None,
    ):
        if self.flex_distribution is None:
            raise ValueError(
                "No flex distribution found. Call calculate_partial_gap_distribution_with_input_masks(...) first."
            )
        if select_type not in {"weighted_regions", "min_regions", "pure_avg"}:
            raise ValueError("select_type must be one of: 'weighted_regions', 'min_regions', 'pure_avg'.")
        if category not in {"false", "correct"}:
            raise ValueError("category must be one of: 'false', 'correct'.")
        _valid_ratio_types = {"over_complete_nontrivial", "over_region_all", "over_complete_all"}
        if ratio_type is not None and ratio_type not in _valid_ratio_types:
            raise ValueError(f"ratio_type must be one of {_valid_ratio_types} or None.")
        if first_shots_num is not None and first_shots_num <= 0:
            raise ValueError("first_shots_num must be positive when provided.")

        data_accept = self.flex_distribution[f"accept_{category}_data"][select_type]
        data_reject = self.flex_distribution[f"reject_{category}_data"][select_type]
        n = int(self.flex_distribution["num_weighted_regions"])
        cat_label = category.capitalize()

        density_field = f"ratio_{ratio_type}" if ratio_type is not None else "det_density"
        density_ylabel = ratio_type if ratio_type is not None else "Detector density"

        import matplotlib.cm as _cm
        density_colors = [_cm.tab10(i % 10) for i in range(n)]
        # gap layout: N region gaps + weighted_avg + min_gap → N+2 entries.
        gap_colors = [_cm.tab10(i % 10) for i in range(n + 2)]

        fig, axes = plt.subplots(2, 2, figsize=(16, 9), dpi=120)

        if select_type in ("weighted_regions", "min_regions"):
            from matplotlib.patches import Patch
            shot_sep = 2.0

            def plot_weighted(ax, data: dict[str, Any], *, is_gap: bool, title: str) -> None:
                n_total = len(data["complete_gap_value"] if is_gap else data["complete_det_density"])
                m = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                cat_n = (n + 2) if is_gap else n
                if m == 0:
                    ax.set_title(f"{select_type} | {title} (no data)")
                    ax.grid(True, axis="y", alpha=0.25)
                    return

                xs: list[float] = []
                masked_vals: list[float] = []
                complete_vals: list[float] = []
                color_vals: list[Any] = []
                centers: list[float] = []
                for i in range(m):
                    base = i * (cat_n + shot_sep)
                    centers.append(base + (cat_n - 1) / 2.0)
                    for c in range(cat_n):
                        xs.append(base + c)
                        if is_gap:
                            masked_vals.append(float(data["gap_value"][c][i]))
                            complete_vals.append(float(data["complete_gap_value"][i]))
                            color_vals.append(gap_colors[c])
                        else:
                            masked_vals.append(float(data[density_field][c][i]))
                            complete_vals.append(float(data["complete_det_density"][i]))
                            color_vals.append(density_colors[c])

                if is_gap or ratio_type is None:
                    ax.bar(xs, complete_vals, width=0.95, color="#4c78a8", alpha=0.35, label="Complete")
                ax.bar(xs, masked_vals, width=0.72, color=color_vals, alpha=0.85)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(centers)
                ax.set_xticklabels([f"s{i}" for i in range(m)], rotation=0, fontsize=8)
                ax.grid(True, axis="y", alpha=0.25)

                if is_gap:
                    handles = [Patch(color="#4c78a8", alpha=0.35, label="Complete")]
                    handles += [Patch(color=gap_colors[r], label=f"Region{r+1} gap") for r in range(n)]
                    handles += [Patch(color=gap_colors[n],     label="Weighted avg gap")]
                    handles += [Patch(color=gap_colors[n + 1], label="Min gap")]
                else:
                    region_label = "density" if ratio_type is None else ratio_type
                    handles = [] if ratio_type is not None else [Patch(color="#4c78a8", alpha=0.35, label="Complete")]
                    handles += [Patch(color=density_colors[r], label=f"Region{r+1} {region_label}") for r in range(n)]
                ax.legend(handles=handles, loc="upper right", fontsize=8)

            plot_weighted(axes[0, 0], data_accept, is_gap=False, title=f"Accept {cat_label}: {density_ylabel}")
            plot_weighted(axes[0, 1], data_accept, is_gap=True, title=f"Accept {cat_label}: Gap")
            plot_weighted(axes[1, 0], data_reject, is_gap=False, title=f"Reject {cat_label}: {density_ylabel}")
            plot_weighted(axes[1, 1], data_reject, is_gap=True, title=f"Reject {cat_label}: Gap")
        else:
            def build_density_points(data: dict[str, Any]):
                labels: list[str] = []
                masked: list[float] = []
                complete: list[float] = []
                n_total = len(data["complete_det_density"])
                m = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                for i in range(m):
                    labels.append(f"s{i}")
                    masked.append(float(data[density_field][i]))
                    complete.append(float(data["complete_det_density"][i]))
                return labels, masked, complete

            def build_gap_points(data: dict[str, Any]):
                labels: list[str] = []
                masked: list[float] = []
                complete: list[float] = []
                n_total = len(data["complete_gap_value"])
                m = min(n_total, int(first_shots_num)) if first_shots_num is not None else n_total
                for i in range(m):
                    labels.append(f"s{i}")
                    masked.append(float(data["gap_value"][i]))
                    complete.append(float(data["complete_gap_value"][i]))
                return labels, masked, complete

            a_dx, a_dm, a_dc = build_density_points(data_accept)
            a_gx, a_gm, a_gc = build_gap_points(data_accept)
            r_dx, r_dm, r_dc = build_density_points(data_reject)
            r_gx, r_gm, r_gc = build_gap_points(data_reject)

            density_panels = [
                (axes[0, 0], a_dx, a_dm, a_dc, f"Accept {cat_label}: {density_ylabel}"),
                (axes[1, 0], r_dx, r_dm, r_dc, f"Reject {cat_label}: {density_ylabel}"),
            ]
            gap_panels = [
                (axes[0, 1], a_gx, a_gm, a_gc, f"Accept {cat_label}: Gap"),
                (axes[1, 1], r_gx, r_gm, r_gc, f"Reject {cat_label}: Gap"),
            ]
            for ax, labels, masked_vals, complete_vals, title in density_panels:
                x = np.arange(len(labels), dtype=np.float64)
                if ratio_type is None:
                    ax.bar(x, complete_vals, width=0.80, color="#4c78a8", alpha=0.40, label="Complete")
                ax.bar(x, masked_vals, width=0.58, color="#f28e2b", alpha=0.75, label=select_type)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(x)
                if len(labels) <= 60:
                    ax.set_xticklabels(labels, rotation=90, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)
            for ax, labels, masked_vals, complete_vals, title in gap_panels:
                x = np.arange(len(labels), dtype=np.float64)
                ax.bar(x, complete_vals, width=0.80, color="#4c78a8", alpha=0.40, label="Complete")
                ax.bar(x, masked_vals, width=0.58, color="#f28e2b", alpha=0.75, label=select_type)
                ax.set_title(f"{select_type} | {title}")
                ax.set_xticks(x)
                if len(labels) <= 60:
                    ax.set_xticklabels(labels, rotation=90, fontsize=7)
                else:
                    ax.set_xticklabels([])
                ax.grid(True, axis="y", alpha=0.25)
                ax.legend(loc="upper right", fontsize=8)

        axes[0, 0].set_ylabel(density_ylabel)
        axes[1, 0].set_ylabel(density_ylabel)
        axes[0, 1].set_ylabel("Gap value")
        axes[1, 1].set_ylabel("Gap value")
        fig.tight_layout()

        if path is not None:
            if path.suffix.lower() == ".svg":
                fig.savefig(path, format="svg")
            else:
                fig.savefig(path.with_suffix(".svg"), format="svg")
        return fig

    # ------------------------------------------------------------------
    # Direct partial-vs-complete gap comparison (single mask, three schemes,
    # three LER normalizations each).
    # ------------------------------------------------------------------

    def generate_samples_for_comparison(
        self,
        *,
        shots: int = 100_000,
        sampler: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Sample `shots` shots from self.circuit, postselect via the sampler's
        discard mask, and run the complete decoder once. Returns a dict that can
        be passed to compare_partial_gap_with_complete_gap_from_samples(...) for
        one or more mask comparisons without re-sampling.

        The complete decoder's per-shot output (preds, gaps, err) is invariant
        across mask choice and is computed here so downstream comparisons only
        need the partial-mask-dependent decode step.
        """
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if sampler is None:
            sampler = cultiv.DesaturationSampler()

        task = sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        dec = sampler.compiled_sampler_for_task(task)
        dets, actual_obs = dec.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)
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
                "complete_preds": np.zeros(0, dtype=np.bool_),
                "complete_gaps":  np.zeros(0, dtype=np.float64),
                "complete_errs":  np.zeros(0, dtype=np.bool_),
                "dec": dec,
            }

        actual_flip = actual_obs[:, 0].astype(np.bool_)
        preds_c, gaps_c = dec._decode_batch_overwrite_last_byte(dets.copy())
        preds_c = preds_c.astype(np.bool_)
        gaps_c = gaps_c.astype(np.float64)
        err_c = preds_c ^ actual_flip

        return {
            "shots_sampled": int(shots),
            "kept_shots": n_kept,
            "dets": dets,
            "actual_flip": actual_flip,
            "complete_preds": preds_c,
            "complete_gaps": gaps_c,
            "complete_errs": err_c,
            "dec": dec,
        }

    @staticmethod
    def samples_to_npz(samples: dict[str, Any], filepath) -> pathlib.Path:
        """Persist a samples dict produced by generate_samples_for_comparison(...)
        to a compressed `.npz` file. Everything except the `dec` (compiled
        decoder) is saved — `dec` is recreated on load from the circuit since
        it is not serializable.

        Static because the `samples` dict is self-contained for save.
        Auto-creates parent directories. Returns the resolved path written to.
        """
        required = {"shots_sampled", "kept_shots", "dets", "actual_flip",
                    "complete_preds", "complete_gaps", "complete_errs"}
        missing = required - set(samples.keys())
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
        # np.savez_compressed appends `.npz` if the path lacks it; resolve once
        # so the returned path matches what was actually written.
        if path.suffix != ".npz":
            path = path.with_suffix(path.suffix + ".npz")
        return path

    def from_npz_to_samples(
        self,
        filepath,
        *,
        sampler: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Load a samples dict previously written by samples_to_npz(...).

        Recreates the `dec` (compiled decoder) entry from `self.circuit` and the
        supplied `sampler`, since it cannot be serialized. The reloaded dict is
        shape-compatible with what generate_samples_for_comparison(...) returns
        and can be passed straight into compare_partial_gap_with_complete_gap_from_samples(...).

        Caller's responsibility: the file must have been produced from the same
        circuit (same DEM) as this instance — there's no automatic check, since
        circuits aren't serialized into the npz.

        Args:
            filepath: path to the .npz produced by samples_to_npz(...).
            sampler: optional decoder sampler (defaults to cultiv.DesaturationSampler()).
        """
        if sampler is None:
            sampler = cultiv.DesaturationSampler()
        path = pathlib.Path(filepath)
        z = np.load(path, allow_pickle=False)
        required = {"shots_sampled", "kept_shots", "dets", "actual_flip",
                    "complete_preds", "complete_gaps", "complete_errs"}
        missing = required - set(z.files)
        if missing:
            raise ValueError(
                f"npz at {path} is missing samples keys {sorted(missing)}."
            )

        task = sinter.Task(circuit=self.circuit, detector_error_model=self.circuit.detector_error_model())
        dec = sampler.compiled_sampler_for_task(task)

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

    def build_fast_partial_decoder(
        self,
        samples: dict[str, Any],
        partial_mask: np.ndarray,
        *,
        weight_cutoff: float = 15.0,
        connectivity_fix_type: str = "per_detector",
    ) -> "FastPartialDecoder":
        """Construct a FastPartialDecoder from a samples dict + partial mask.

        Returns an object whose `.decode_batch(packed_dets)` is a fast drop-in
        for `dec._decode_batch_overwrite_last_byte`. Pass to the optional
        `fast_decoder` parameter of `compare_partial_gap_with_complete_gap_2stage_from_samples`
        or `collect_shot_gap_1D_scatter_from_samples` for ~3-4x faster partial
        decoding via DEM contraction.

        `connectivity_fix_type` selects the algorithm for the post-Step-4
        correctness repair: 'per_detector' (default; BFS per detector with
        on-the-fly updates) or 'union_find' (component-wise via Union-Find).
        Both produce equivalent results for typical sizes; see
        `_apply_connectivity_fix(...)` for details.

        See FastPartialDecoder's docstring for the calibration caveat (slight
        gap-distribution shift vs syndrome suppression).
        """
        if 'dec' not in samples:
            raise ValueError(
                "samples dict missing 'dec' key; "
                "build it via generate_samples_for_comparison(...) or from_npz_to_samples(...)."
            )
        if partial_mask.shape != (self.circuit.num_detectors,):
            raise ValueError(
                f"partial_mask must have shape ({self.circuit.num_detectors},); "
                f"got {partial_mask.shape}."
            )
        dec = samples['dec']
        return FastPartialDecoder(
            dec.gap_dem,
            partial_mask,
            decibels_per_w=dec.decibels_per_w,
            weight_cutoff=weight_cutoff,
            connectivity_fix_type=connectivity_fix_type,
        )

    def compare_partial_gap_with_complete_gap_from_samples(
        self,
        samples: dict[str, Any],
        *,
        partial_mask: np.ndarray,
        gap_threshold: float,
        gap_threshold_partial: Optional[float] = None,
    ) -> dict[str, Any]:
        """Same comparison as compare_partial_gap_with_complete_gap, but uses
        pre-generated samples from generate_samples_for_comparison(...). Avoids
        re-sampling and re-running the complete decoder when comparing multiple
        masks (or thresholds) against the same shot set.

        Stores summary on self.compare_partial_complete_summary; returns it.
        """
        required = {"shots_sampled", "kept_shots", "dets", "actual_flip",
                    "complete_preds", "complete_gaps", "complete_errs", "dec"}
        missing = required - set(samples.keys())
        if missing:
            raise ValueError(
                f"samples dict missing keys {sorted(missing)}; "
                "build it via generate_samples_for_comparison(...)."
            )
        if partial_mask.shape != (self.circuit.num_detectors,):
            raise ValueError(
                f"partial_mask must have shape ({self.circuit.num_detectors},); got {partial_mask.shape}."
            )

        thr_c = float(gap_threshold)
        thr_p = float(gap_threshold_partial) if gap_threshold_partial is not None else thr_c

        n_shots = int(samples["shots_sampled"])
        n_kept = int(samples["kept_shots"])
        dets = samples["dets"]
        actual_flip = samples["actual_flip"]
        gaps_c = samples["complete_gaps"]
        err_c = samples["complete_errs"]                        # = preds_c ^ actual_flip from samples
        dec = samples["dec"]
        n_total_dets = int(self.circuit.num_detectors)

        # Mask coverage (independent of n_kept).
        width = dets.shape[1] if n_kept > 0 else (-(-n_total_dets // 8))
        full_mask = self._normalize_mask_to_packed(partial_mask, width)
        m = self._mask_size(full_mask)
        ratio = (m / n_total_dets) if n_total_dets > 0 else 0.0

        def _safe_div(num: int, den: int) -> float:
            return float(num) / float(den) if den > 0 else float("nan")

        if n_kept == 0:
            zero_block = {
                "errors": 0,
                "LER_over_accept": float("nan"),
                "LER_over_kept": float("nan"),
                "LER_over_shots": float("nan"),
            }
            summary = {
                "shots_sampled": n_shots,
                "kept_shots": 0,
                "complete_total_detectors": n_total_dets,
                "partial_mask_size": m,
                "partial_mask_ratio": ratio,
                "gap_threshold": thr_c,
                "gap_threshold_partial": thr_p,
                "complete_accept_count": 0, "complete_reject_count": 0,
                "complete_accept_rate": float("nan"), "complete_reject_rate": float("nan"),
                "complete":               dict(zero_block),
                "partial_accept_count": 0, "partial_reject_count": 0,
                "partial_accept_rate": float("nan"), "partial_reject_rate": float("nan"),
                "partial_complete_pred":  dict(zero_block),
                "partial_partial_pred":   dict(zero_block),
                "warning": "no shots survived postselect",
            }
            self.compare_partial_complete_summary = summary
            return summary

        # Partial decode (only mask-dependent step).
        d_p = dets.copy()
        d_p &= full_mask.reshape(1, -1)
        preds_p, gaps_p = dec._decode_batch_overwrite_last_byte(d_p.copy())
        preds_p = preds_p.astype(np.bool_)
        gaps_p = gaps_p.astype(np.float64)
        err_p = preds_p ^ actual_flip

        ca = gaps_c >= thr_c
        pa = gaps_p >= thr_p

        complete_accept = int(ca.sum()); complete_reject = n_kept - complete_accept
        partial_accept  = int(pa.sum()); partial_reject  = n_kept - partial_accept

        e_complete   = int(err_c[ca].sum())
        e_partial_cp = int(err_c[pa].sum())
        e_partial_pp = int(err_p[pa].sum())

        complete_block = {
            "errors": e_complete,
            "LER_over_accept": _safe_div(e_complete, complete_accept),
            "LER_over_kept":   _safe_div(e_complete, n_kept),
            "LER_over_shots":  _safe_div(e_complete, n_shots),
        }
        partial_complete_pred_block = {
            "errors": e_partial_cp,
            "LER_over_accept": _safe_div(e_partial_cp, partial_accept),
            "LER_over_kept":   _safe_div(e_partial_cp, n_kept),
            "LER_over_shots":  _safe_div(e_partial_cp, n_shots),
        }
        partial_partial_pred_block = {
            "errors": e_partial_pp,
            "LER_over_accept": _safe_div(e_partial_pp, partial_accept),
            "LER_over_kept":   _safe_div(e_partial_pp, n_kept),
            "LER_over_shots":  _safe_div(e_partial_pp, n_shots),
        }

        summary = {
            "shots_sampled": n_shots,
            "kept_shots": n_kept,
            "complete_total_detectors": n_total_dets,
            "partial_mask_size": m,
            "partial_mask_ratio": ratio,
            "gap_threshold": thr_c,
            "gap_threshold_partial": thr_p,

            "complete_accept_count": complete_accept,
            "complete_reject_count": complete_reject,
            "complete_accept_rate": complete_accept / n_kept,
            "complete_reject_rate": complete_reject / n_kept,
            "complete": complete_block,

            "partial_accept_count": partial_accept,
            "partial_reject_count": partial_reject,
            "partial_accept_rate": partial_accept / n_kept,
            "partial_reject_rate": partial_reject / n_kept,

            "partial_complete_pred": partial_complete_pred_block,
            "partial_partial_pred":  partial_partial_pred_block,
        }
        self.compare_partial_complete_summary = summary
        return summary

    def compare_partial_gap_with_complete_gap(
        self,
        *,
        partial_mask: np.ndarray,
        gap_threshold: float,
        gap_threshold_partial: Optional[float] = None,
        shots: int = 100_000,
        sampler: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Sample shots and run a single mask comparison. Thin wrapper around
        generate_samples_for_comparison(...) + compare_partial_gap_with_complete_gap_from_samples(...).
        See those for full semantics. Use the underlying pair directly when comparing
        multiple masks or thresholds against the same sample set."""
        samples = self.generate_samples_for_comparison(shots=shots, sampler=sampler)
        return self.compare_partial_gap_with_complete_gap_from_samples(
            samples,
            partial_mask=partial_mask,
            gap_threshold=gap_threshold,
            gap_threshold_partial=gap_threshold_partial,
        )

    def print_compare_partial_gap_with_complete_gap(self) -> dict[str, Any]:
        if self.compare_partial_complete_summary is None:
            raise ValueError(
                "No comparison summary found. Call compare_partial_gap_with_complete_gap(...) first."
            )
        d = self.compare_partial_complete_summary
        n_total = int(d["complete_total_detectors"])
        m = int(d["partial_mask_size"])
        r = float(d["partial_mask_ratio"])
        n_kept = int(d["kept_shots"])
        n_shots = int(d["shots_sampled"])

        print(f"shots_sampled = {n_shots}")
        print(f"kept_shots    = {n_kept}")
        print(f"partial_mask_size: {m}/{n_total} = {r:.6f}")
        print(f"gap_threshold = {d['gap_threshold']} (partial: {d['gap_threshold_partial']})")

        if n_kept == 0:
            print("\n(no shots survived postselect; LER metrics undefined)")
            return d

        def _print_scheme(title: str, accepts: int, block: dict[str, Any]) -> None:
            e = int(block["errors"])
            print(f"\n--- {title} ---")
            print(f"errors        = {e}")
            print(f"LER over accept = {e}/{accepts} = {block['LER_over_accept']:.6f}")
            print(f"LER over kept   = {e}/{n_kept} = {block['LER_over_kept']:.6f}")
            print(f"LER over shots  = {e}/{n_shots} = {block['LER_over_shots']:.6f}")

        ca = int(d["complete_accept_count"]); cr = int(d["complete_reject_count"])
        pa = int(d["partial_accept_count"]);  pr = int(d["partial_reject_count"])

        print(f"\nComplete-gating accept: {ca}/{n_kept} ({d['complete_accept_rate']:.6f}); "
              f"reject: {cr}/{n_kept} ({d['complete_reject_rate']:.6f})")
        print(f"Partial-gating  accept: {pa}/{n_kept} ({d['partial_accept_rate']:.6f}); "
              f"reject: {pr}/{n_kept} ({d['partial_reject_rate']:.6f})")

        _print_scheme(
            "Complete gating, complete prediction (reference)",
            ca, d["complete"],
        )
        _print_scheme(
            "Partial gating, COMPLETE prediction (the protocol)",
            pa, d["partial_complete_pred"],
        )
        _print_scheme(
            "Partial gating, PARTIAL prediction (diagnostic; expected near-random)",
            pa, d["partial_partial_pred"],
        )
        return d

    def compare_partial_gap_with_complete_gap_2stage_from_samples(
        self,
        samples: dict[str, Any],
        *,
        partial_mask: np.ndarray,
        gap_threshold: float,
        gap_threshold_low: float,
        gap_threshold_high: float,
        fast_decoder: Optional["FastPartialDecoder"] = None,
    ) -> dict[str, Any]:
        """2-stage gating protocol over pre-generated samples.

        For each kept shot, the partial gap routes it to one of three tiers:
            gap_p < T_low                  -> tier 1: cheap reject (complete NEVER runs)
            T_low <= gap_p <= T_high       -> tier 2: rescue band; run complete and
                                              accept iff gap_c >= T_complete (complete
                                              runs to obtain BOTH gap and prediction)
            gap_p > T_high                 -> tier 3: direct accept (complete runs ONLY
                                              for prediction)
        All accepted shots use the complete decoder's prediction.

        If `fast_decoder` is provided (built via build_fast_partial_decoder(...)),
        the partial decode uses the reduced-DEM matcher instead of the slow
        `dec._decode_batch_overwrite_last_byte` path. Same partial_mask must be
        passed for mask-coverage statistics.

        Stores summary on self.compare_partial_complete_summary_2stage; returns it.
        """
        required = {"shots_sampled", "kept_shots", "dets", "actual_flip",
                    "complete_preds", "complete_gaps", "complete_errs", "dec"}
        missing = required - set(samples.keys())
        if missing:
            raise ValueError(
                f"samples dict missing keys {sorted(missing)}; "
                "build it via generate_samples_for_comparison(...)."
            )
        if partial_mask.shape != (self.circuit.num_detectors,):
            raise ValueError(
                f"partial_mask must have shape ({self.circuit.num_detectors},); got {partial_mask.shape}."
            )
        if not (gap_threshold_low <= gap_threshold_high):
            raise ValueError(
                f"gap_threshold_low ({gap_threshold_low}) must be <= gap_threshold_high ({gap_threshold_high})."
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
        n_total_dets = int(self.circuit.num_detectors)

        width = dets.shape[1] if n_kept > 0 else (-(-n_total_dets // 8))
        full_mask = self._normalize_mask_to_packed(partial_mask, width)
        m = self._mask_size(full_mask)
        ratio = (m / n_total_dets) if n_total_dets > 0 else 0.0

        def _safe_div(num: int, den: int) -> float:
            return float(num) / float(den) if den > 0 else float("nan")

        if n_kept == 0:
            summary = {
                "shots_sampled": n_shots,
                "kept_shots": 0,
                "complete_total_detectors": n_total_dets,
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
            self.compare_partial_complete_summary_2stage = summary
            return summary

        # Partial decode (only mask-dependent step).
        if fast_decoder is None:
            d_p = dets.copy()
            d_p &= full_mask.reshape(1, -1)
            _, gaps_p = dec._decode_batch_overwrite_last_byte(d_p.copy())
        else:
            _, gaps_p = fast_decoder.decode_batch(dets)
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

        summary = {
            "shots_sampled": n_shots,
            "kept_shots": n_kept,
            "complete_total_detectors": n_total_dets,
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
            # partial gap alone (tier 1 reject + tier 3 direct accept). For tier 3,
            # the complete decoder still runs in parallel for prediction, but the
            # gating decision does not wait on it — relevant for decision-latency
            # to the "ready" stage.
            "gating_decision_count":   tier1 + tier3,
            "gating_decision_rate":    (tier1 + tier3) / n_kept,
            "gating_decision_savings": (tier1 + tier3) / n_kept,

            "errors": errors,
            "LER_over_accept": _safe_div(errors, accept),
            "LER_over_kept":   _safe_div(errors, n_kept),
            "LER_over_shots":  _safe_div(errors, n_shots),

            # Complete-only reference (gap_c >= T_complete) for side-by-side comparison.
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
        self.compare_partial_complete_summary_2stage = summary
        return summary

    def print_compare_partial_gap_with_complete_gap_2stage(self) -> dict[str, Any]:
        if self.compare_partial_complete_summary_2stage is None:
            raise ValueError(
                "No 2-stage comparison summary found. "
                "Call compare_partial_gap_with_complete_gap_2stage_from_samples(...) first."
            )
        d = self.compare_partial_complete_summary_2stage
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
            return d

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
        return d

    # ------------------------------------------------------------------
    # Per-shot 1D dot scatter of gap values, with logical-error shots
    # highlighted. One dot per kept shot on a gap-value x-axis. Useful for
    # visually checking where error shots concentrate on the gap line.
    # ------------------------------------------------------------------
    def collect_shot_gap_1D_scatter_from_samples(
        self,
        samples: dict[str, Any],
        *,
        partial_mask: Optional[np.ndarray] = None,
        fast_decoder: Optional["FastPartialDecoder"] = None,
    ) -> dict[str, Any]:
        """Collect per-shot (gap, is_error) arrays for the 1D-scatter plot.

        Always stores complete-gating data (gaps_c + err_c) from `samples`. If
        `partial_mask` is provided, additionally runs partial decode once and
        stores partial_gaps. The error marker for both plots is the operational
        error indicator (complete-decoder prediction wrong), since that is what
        determines whether the cultivated state is corrupted.

        If `fast_decoder` is provided (built via build_fast_partial_decoder(...)),
        the partial decode uses the reduced-DEM matcher instead of the slow
        `dec._decode_batch_overwrite_last_byte` path.

        Stores result on self.shot_gap_1D_scatter_data; returns it.
        """
        required = {"shots_sampled", "kept_shots", "dets",
                    "complete_gaps", "complete_errs", "dec"}
        missing = required - set(samples.keys())
        if missing:
            raise ValueError(
                f"samples dict missing keys {sorted(missing)}; "
                "build it via generate_samples_for_comparison(...)."
            )

        n_shots = int(samples["shots_sampled"])
        n_kept = int(samples["kept_shots"])
        gaps_c = np.asarray(samples["complete_gaps"], dtype=np.float64)
        err_c = np.asarray(samples["complete_errs"], dtype=np.bool_)

        partial_gaps = None
        partial_mask_size = None
        partial_mask_ratio = None
        if partial_mask is not None:
            if partial_mask.shape != (self.circuit.num_detectors,):
                raise ValueError(
                    f"partial_mask must have shape ({self.circuit.num_detectors},); "
                    f"got {partial_mask.shape}."
                )
            n_total_dets = int(self.circuit.num_detectors)
            dets = samples["dets"]
            dec = samples["dec"]
            width = dets.shape[1] if n_kept > 0 else (-(-n_total_dets // 8))
            full_mask = self._normalize_mask_to_packed(partial_mask, width)
            partial_mask_size = self._mask_size(full_mask)
            partial_mask_ratio = (partial_mask_size / n_total_dets) if n_total_dets > 0 else 0.0

            if n_kept > 0:
                if fast_decoder is None:
                    d_p = dets.copy()
                    d_p &= full_mask.reshape(1, -1)
                    _, gaps_p = dec._decode_batch_overwrite_last_byte(d_p.copy())
                else:
                    _, gaps_p = fast_decoder.decode_batch(dets)
                partial_gaps = np.asarray(gaps_p, dtype=np.float64)
            else:
                partial_gaps = np.zeros(0, dtype=np.float64)

        self.shot_gap_1D_scatter_data = {
            "shots_sampled": n_shots,
            "kept_shots": n_kept,
            "complete_total_detectors": int(self.circuit.num_detectors),
            "complete_gaps":   gaps_c,
            "complete_errors": err_c,
            "partial_gaps":    partial_gaps,
            "partial_errors":  err_c if partial_gaps is not None else None,
            "partial_mask_size":  partial_mask_size,
            "partial_mask_ratio": partial_mask_ratio,
        }
        return self.shot_gap_1D_scatter_data

    def plot_shot_gap_1D_scatter(
        self,
        *,
        gap_type: str = "complete",
        threshold: Optional[float] = None,
        subsample_size: int = 20_000,
        figsize: tuple = (12.0, 1.8),
        normal_color: str = "#222222",
        error_color: str = "#d62728",
        normal_alpha: float = 0.30,
        error_alpha: float = 0.95,
        normal_marker_size: float = 2.0,
        error_marker_size: float = 14.0,
        jitter: float = 1.0,
        rng_seed: int = 0,
        save_path: Optional[pathlib.Path] = None,
        title: Optional[str] = None,
    ):
        """Plot a 1D scatter of per-shot gap values with logical-error shots in red.

        Args:
            gap_type: "complete" or "partial" — selects which gap array to plot.
            threshold: if set, draws a vertical dashed line at this x value and
                shades the reject region to its left.
            subsample_size: max number of non-error shots to plot (errors always
                fully shown). Set to None or <=0 to plot all non-errors.
            figsize, *_color, *_alpha, *_marker_size, jitter, rng_seed: cosmetic.
            save_path: if given, writes the figure to this path before returning.
            title: optional override for the auto-generated title.

        Returns the matplotlib Figure for inline display.
        """
        if gap_type not in {"complete", "partial"}:
            raise ValueError(f"gap_type must be 'complete' or 'partial'; got {gap_type!r}.")
        if self.shot_gap_1D_scatter_data is None:
            raise ValueError(
                "No 1D-scatter data found. "
                "Call collect_shot_gap_1D_scatter_from_samples(...) first."
            )
        d = self.shot_gap_1D_scatter_data

        gaps = d[f"{gap_type}_gaps"]
        errs = d[f"{gap_type}_errors"]
        if gaps is None or errs is None:
            raise ValueError(
                f"No {gap_type} data collected. "
                "Pass `partial_mask` to collect_shot_gap_1D_scatter_from_samples(...)."
            )

        n_kept = int(d["kept_shots"])

        # Auto title.
        if title is None:
            n_err = int(errs.sum())
            n_total_dets = int(d["complete_total_detectors"])
            if gap_type == "complete":
                head = f"Complete gap | {n_kept} kept shots, {n_err} errors"
            else:
                m = d["partial_mask_size"]; r = d["partial_mask_ratio"]
                if m is not None and r is not None:
                    head = (f"Partial gap (mask {m}/{n_total_dets} = {r:.4f}) | "
                            f"{n_kept} kept shots, {n_err} errors")
                else:
                    head = f"Partial gap | {n_kept} kept shots, {n_err} errors"
            title = head

        rng = np.random.default_rng(rng_seed)

        # Split error / non-error indices.
        err_idx = np.where(errs)[0]
        ok_idx  = np.where(~errs)[0]

        # Subsample non-errors only (errors are rare and always shown).
        if subsample_size is not None and subsample_size > 0 and ok_idx.size > subsample_size:
            ok_idx = rng.choice(ok_idx, size=int(subsample_size), replace=False)

        ok_x = gaps[ok_idx]
        er_x = gaps[err_idx]
        ok_y = rng.uniform(-jitter, jitter, size=ok_x.size)
        er_y = rng.uniform(-jitter, jitter, size=er_x.size)

        fig, ax = plt.subplots(figsize=figsize, dpi=120)

        # Optional reject shading + threshold line.
        if threshold is not None:
            x_min = float(min(gaps.min() if gaps.size else 0.0, threshold)) - 1.0
            ax.axvspan(x_min, float(threshold), color="#cccccc", alpha=0.18, zorder=0)
            ax.axvline(float(threshold), color="#666666", linestyle="--",
                       linewidth=1.0, zorder=1)

        # Non-errors first (so errors land on top).
        ax.scatter(
            ok_x, ok_y,
            s=normal_marker_size, c=normal_color, alpha=normal_alpha,
            linewidths=0, zorder=2,
        )
        ax.scatter(
            er_x, er_y,
            s=error_marker_size, c=error_color, alpha=error_alpha,
            linewidths=0, zorder=3,
        )

        ax.set_xlabel("gap")
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_ylim(-jitter * 1.4, jitter * 1.4)
        ax.set_title(title, fontsize=10)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # Legend (single dummy entry per category).
        from matplotlib.lines import Line2D
        legend_handles = [
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=normal_color, markersize=4, alpha=0.7,
                   label=f"correct (sub-sampled to {ok_x.size:d})" if subsample_size else "correct"),
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=error_color, markersize=6, alpha=0.95,
                   label=f"error ({er_x.size:d})"),
        ]
        if threshold is not None:
            legend_handles.append(
                Line2D([0], [0], color="#666666", linestyle="--",
                       linewidth=1.0, label=f"threshold = {threshold}")
            )
        ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8)
        fig.tight_layout()

        if save_path is not None:
            save_path = pathlib.Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=160, bbox_inches="tight")
        return fig

    def complete_gap_vs_pure_avg_gap_plot(
        self,
        *,
        path: Optional[pathlib.Path] = None,
        bins: int = 40,
        gap_threshold: Optional[float] = None,
    ):
        if not self.partial_gap_sens_data.get("closest_matches"):
            raise ValueError("No partial gap data found. Call collect_partial_detector_gap_sens(...) first.")

        complete = np.asarray(self.partial_gap_sens_data["complete_gaps"], dtype=np.float64)
        pure_avg = np.asarray(
            [float(rec["pure_avg_gap"]) for rec in self.partial_gap_sens_data["closest_matches"]],
            dtype=np.float64,
        )
        if complete.size == 0 or pure_avg.size == 0:
            raise ValueError("No values available for complete/pure-avg distribution plotting.")

        if gap_threshold is None:
            gap_threshold = self.partial_gap_sens_data.get("gap_threshold", 0.0)

        all_vals = np.concatenate([complete, pure_avg])
        x_min = float(np.min(all_vals))
        x_max = float(np.max(all_vals))
        if x_max <= x_min:
            x_max = x_min + 1e-9

        c_mean = float(np.mean(complete))
        c_std = float(np.std(complete))
        p_mean = float(np.mean(pure_avg))
        p_std = float(np.std(pure_avg))

        fig, ax = plt.subplots(1, 1, figsize=(9.0, 5.2), dpi=120)

        # Histogram overlays.
        ax.hist(
            complete,
            bins=bins,
            density=True,
            alpha=0.25,
            color="#4c78a8",
            edgecolor="#2b5d9a",
            linewidth=0.8,
            label="Complete gap histogram",
        )
        ax.hist(
            pure_avg,
            bins=bins,
            density=True,
            alpha=0.25,
            color="#f28e2b",
            edgecolor="#c96a08",
            linewidth=0.8,
            label="Pure-avg gap histogram",
        )

        # Smoothed envelopes from histogram densities.
        c_hist, c_edges = np.histogram(complete, bins=bins, density=True)
        p_hist, p_edges = np.histogram(pure_avg, bins=bins, density=True)
        c_x = 0.5 * (c_edges[:-1] + c_edges[1:])
        p_x = 0.5 * (p_edges[:-1] + p_edges[1:])
        if len(c_hist) >= 3 and len(p_hist) >= 3:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= np.sum(kernel)
            c_smooth = np.convolve(c_hist, kernel, mode="same")
            p_smooth = np.convolve(p_hist, kernel, mode="same")
            ax.plot(c_x, c_smooth, color="#1f4f99", linewidth=2.3, label="Complete gap envelope")
            ax.plot(p_x, p_smooth, color="#b85a00", linewidth=2.3, label="Pure-avg gap envelope")

        # Mean lines.
        ax.axvline(c_mean, color="#1f4f99", linewidth=2.0, linestyle="-", label="Complete mean")
        ax.axvline(p_mean, color="#b85a00", linewidth=2.0, linestyle="-", label="Pure-avg mean")

        # Std shaded bands.
        ax.axvspan(c_mean - c_std, c_mean + c_std, color="#4c78a8", alpha=0.12, label="Complete ±1σ")
        ax.axvspan(p_mean - p_std, p_mean + p_std, color="#f28e2b", alpha=0.12, label="Pure-avg ±1σ")

        # Threshold line.
        if gap_threshold is not None:
            ax.axvline(float(gap_threshold), color="#d62728", linewidth=2.2, linestyle="--", label="Gap threshold")

        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("Gap")
        ax.set_ylabel("Density")
        ax.set_title("Complete Gap vs Pure-Avg Gap Distribution")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()

        if path is not None:
            if path.suffix.lower() == ".svg":
                fig.savefig(path, format="svg")
            else:
                fig.savefig(path.with_suffix(".svg"), format="svg")
        return fig

    def partial_gap_best_combo_distribution_svg(
        self,
        *,
        path: pathlib.Path,
        value_type: str = "abs_diff",
        bins: int = 40,
    ) -> None:
        import matplotlib.pyplot as plt

        if self.best_combo_distribution is None:
            raise ValueError(
                "No best combo distribution found. Call calculate_partial_gap_best_combo_distribution() first."
            )
        if value_type not in {"left_len", "top_len", "t", "abs_diff"}:
            raise ValueError("value_type must be one of {'left_len', 'top_len', 't', 'abs_diff'}.")

        d = self.best_combo_distribution
        if value_type == "left_len":
            values = np.asarray([float(v[0]) for v in d["best_combo_samples"]], dtype=np.float64)
        elif value_type == "top_len":
            values = np.asarray([float(v[1]) for v in d["best_combo_samples"]], dtype=np.float64)
        elif value_type == "t":
            values = np.asarray([float(v[2]) for v in d["best_combo_samples"]], dtype=np.float64)
        else:
            values = np.asarray(d["abs_diff_samples"], dtype=np.float64)

        if len(values) == 0:
            raise ValueError("No values available for the selected distribution.")

        mean = float(np.mean(values))
        std = float(np.std(values))
        var = float(np.var(values))

        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=120)
        ax.hist(values, bins=bins, density=True, alpha=0.35, color="#4c78a8", edgecolor="white", linewidth=0.8)
        hist_y, hist_edges = np.histogram(values, bins=bins, density=True)
        hist_x = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        if len(hist_y) >= 3:
            kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
            kernel /= np.sum(kernel)
            smooth_y = np.convolve(hist_y, kernel, mode="same")
            ax.plot(hist_x, smooth_y, color="#1f2a44", linewidth=2.0, label="Smoothed density")

        ax.axvline(mean, color="#d62728", linewidth=2.0, linestyle="-", label="Mean")
        ax.axvline(mean - std, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9, label="Mean ± 1σ")
        ax.axvline(mean + std, color="#d62728", linewidth=1.5, linestyle="--", alpha=0.9)

        ax.set_title(f"Best Combo Distribution ({value_type})")
        ax.set_xlabel(value_type)
        ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        stats_text = f"n={len(values)}\nmean={mean:.4f}\nstd={std:.4f}\nvar={var:.4f}"
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.9),
        )
        ax.legend(loc="upper left")
        fig.tight_layout()
        fig.savefig(path, format="svg")
        plt.close(fig)

    def partial_gap_best_combo_distribution_notebook(
        self,
        *,
        value_type: str = "abs_diff",
        bins: int = 40,
    ):
        return self._plot_from_svg_writer(
            lambda path: self.partial_gap_best_combo_distribution_svg(
                path=path,
                value_type=value_type,
                bins=bins,
            )
        )


    # Helper function to create an SVG plot from a writing function, suitable for Jupyter notebook display.
    @staticmethod
    def _plot_from_svg_writer(write_svg_func):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            write_svg_func(tmp_path)
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    @staticmethod
    def tile_coloring(tile: gen.Tile, gap_scores: list[tuple[Any, float]]) -> tuple[float, float, float] | None:
        dr, = tile.flags
        dr = int(dr)
        prediction, gap = gap_scores[dr]
        if gap == 0:
            return 0.5, 0.5, 0.5
        max_gap = max(e for _, e in gap_scores)
        return min(gap / max_gap, 1), 0, 0

    @staticmethod
    def tile_coloring_with_count(tile: gen.Tile, values: list[float], counts: list[float]) -> tuple[float, float, float] | None:
        dr, = tile.flags
        dr = int(dr)
        if dr >= len(values) or dr >= len(counts) or counts[dr] <= 0:
            return 0.5, 0.5, 0.5
        value = values[dr]
        max_value = max(values) if values else 0.0
        if max_value <= 0:
            return 0.5, 0.5, 0.5
        return min(value / max_value, 1), 0, 0

    @staticmethod
    def _make_tile_color_func(
        *,
        values: list[float],
        counts: list[float],
        invert: bool,
        low_q: float,
        high_q: float,
        gamma: float,
    ):
        if not values:
            low = 0.0
            high = 1.0
        else:
            arr = np.asarray(values, dtype=np.float64)
            cnt = np.asarray(counts, dtype=np.float64)
            active = arr[cnt > 0]
            if active.size == 0:
                low = 0.0
                high = 1.0
            else:
                low = float(np.quantile(active, low_q))
                high = float(np.quantile(active, high_q))
                if high <= low:
                    high = low + 1e-9

        def tile_color(tile: gen.Tile) -> tuple[float, float, float] | None:
            dr, = tile.flags
            dr = int(dr)
            if dr >= len(values) or dr >= len(counts) or counts[dr] <= 0:
                return 0.5, 0.5, 0.5
            v = float(values[dr])
            norm = (v - low) / (high - low)
            norm = min(max(norm, 0.0), 1.0)
            norm = norm ** gamma
            if invert:
                norm = 1.0 - norm
            return norm, 0, 0

        return tile_color

    @staticmethod
    def _append_svg_legend(
        path: pathlib.Path,
        *,
        title: str,
        low_label: str,
        high_label: str,
        no_data_label: str,
        red_is_high: bool,
    ) -> None:
        text = path.read_text()
        grad_colors = [
            "rgb(0,0,0)",
            "rgb(51,0,0)",
            "rgb(102,0,0)",
            "rgb(153,0,0)",
            "rgb(204,0,0)",
            "rgb(255,0,0)",
        ]
        if not red_is_high:
            grad_colors = list(reversed(grad_colors))
        legend = [
            '<g id="legend" transform="translate(10,10)">',
            '<rect x="0" y="0" width="260" height="66" fill="white" fill-opacity="0.82" stroke="#444" stroke-width="0.6"/>',
            f'<text x="8" y="14" font-size="10" font-weight="700" fill="#111">{title}</text>',
            f'<rect x="8" y="22" width="14" height="10" fill="{grad_colors[0]}" stroke="none"/>',
            f'<rect x="22" y="22" width="14" height="10" fill="{grad_colors[1]}" stroke="none"/>',
            f'<rect x="36" y="22" width="14" height="10" fill="{grad_colors[2]}" stroke="none"/>',
            f'<rect x="50" y="22" width="14" height="10" fill="{grad_colors[3]}" stroke="none"/>',
            f'<rect x="64" y="22" width="14" height="10" fill="{grad_colors[4]}" stroke="none"/>',
            f'<rect x="78" y="22" width="14" height="10" fill="{grad_colors[5]}" stroke="none"/>',
            f'<text x="98" y="30" font-size="9" fill="#111">{low_label} -> {high_label}</text>',
            '<rect x="8" y="40" width="14" height="10" fill="rgb(128,128,128)" stroke="none"/>',
            f'<text x="28" y="48" font-size="9" fill="#111">{no_data_label}</text>',
            '</g>',
        ]
        legend_block = "\n".join(legend)
        if "</svg>" in text:
            text = text.replace("</svg>", f"{legend_block}\n</svg>", 1)
        else:
            text += "\n" + legend_block
        path.write_text(text)


# ======================================================================
# Fast partial decoder via DEM clipping + chain-aware contraction.
#
# Replaces the slow `dec._decode_batch_overwrite_last_byte` path with a
# matcher built on a renumbered, smaller DEM that physically removes
# non-mask detectors (instead of just zeroing their syndrome bits). The
# kept-detector graph receives chain shortcut edges that capture matching
# paths through chains of clipped detectors, with a weight cutoff that
# drops cosmically-improbable paths the matcher would never use.
#
# Tuned in ideas 17–20: cutoff = 15 gives ~3-4x decode speedup with
# Pearson(reference, fast) ≈ 0.97. The fast gap distribution is highly
# correlated with syndrome-suppression but slightly shifted upward, so
# users may need to recalibrate their gating thresholds when switching.
# ======================================================================


def _apply_connectivity_fix_per_detector(
    selective_kk: dict[tuple[int, int], tuple[float, int]],
    selective_bdy: dict[int, tuple[float, int]],
    chain_kb: dict[int, list[tuple[float, int]]],
    direct_kk_w: dict[tuple[int, int], float],
    direct_bdy_w: dict[int, float],
    n_kept: int,
) -> int:
    """Per-detector connectivity fix with on-the-fly graph updates.

    For each kept detector (in renumbered order), check via BFS through the
    reduced kept-kept graph whether it can reach any node that already has
    boundary access (via direct_bdy_w or selective_bdy). If not, force-add the
    cheapest chain_kb candidate for that detector to selective_bdy, overriding
    the cutoff. The has_boundary set is updated each time, so each affected
    component receives exactly one fix-up edge.

    Mutates selective_bdy in place. Returns number of edges added.
    """
    red_adj: dict[int, set[int]] = collections.defaultdict(set)
    for (a, b) in direct_kk_w.keys():
        red_adj[a].add(b); red_adj[b].add(a)
    for (a, b) in selective_kk.keys():
        red_adj[a].add(b); red_adj[b].add(a)

    has_boundary: set[int] = set(direct_bdy_w) | set(selective_bdy)
    fix_added = 0

    for d in range(n_kept):
        if d in has_boundary:
            continue
        seen = {d}
        stack = [d]
        reached = False
        while stack and not reached:
            cur = stack.pop()
            for nbr in red_adj[cur]:
                if nbr in seen:
                    continue
                if nbr in has_boundary:
                    reached = True
                    break
                seen.add(nbr)
                stack.append(nbr)
        if reached:
            continue
        cands = chain_kb.get(d, [])
        if not cands:
            continue   # truly isolated from boundary even in original DEM
        w_min, obs_min = min(cands, key=lambda x: x[0])
        selective_bdy[d] = (w_min, obs_min)
        has_boundary.add(d)
        fix_added += 1

    return fix_added


def _apply_connectivity_fix_union_find(
    selective_kk: dict[tuple[int, int], tuple[float, int]],
    selective_bdy: dict[int, tuple[float, int]],
    chain_kb: dict[int, list[tuple[float, int]]],
    direct_kk_w: dict[tuple[int, int], float],
    direct_bdy_w: dict[int, float],
    n_kept: int,
) -> int:
    """Union-Find connectivity fix.

    Identify connected components of the reduced kept-kept graph in one pass
    via Union-Find with path compression. For each component without any
    boundary-accessible node, pick the globally cheapest chain_kb candidate
    among ALL detectors in the component and add it to selective_bdy.

    Same correctness guarantee as per-detector, but the picked chain weight
    is minimal across the component rather than first-encountered.

    Mutates selective_bdy in place. Returns number of edges added.
    """
    parent = list(range(n_kept))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b) in direct_kk_w.keys():
        union(a, b)
    for (a, b) in selective_kk.keys():
        union(a, b)

    has_boundary_root: set[int] = set()
    for n in (set(direct_bdy_w) | set(selective_bdy)):
        has_boundary_root.add(find(n))

    comps_to_fix: dict[int, list[int]] = collections.defaultdict(list)
    for d in range(n_kept):
        r = find(d)
        if r in has_boundary_root:
            continue
        comps_to_fix[r].append(d)

    fix_added = 0
    for _r, members in comps_to_fix.items():
        best_d: Optional[int] = None
        best_w = float("inf")
        best_obs = 0
        for d in members:
            for (w, obs) in chain_kb.get(d, []):
                if w < best_w:
                    best_d, best_w, best_obs = d, w, obs
        if best_d is None:
            continue   # truly isolated component
        selective_bdy[best_d] = (best_w, best_obs)
        fix_added += 1

    return fix_added


def _apply_connectivity_fix(
    selective_kk: dict[tuple[int, int], tuple[float, int]],
    selective_bdy: dict[int, tuple[float, int]],
    chain_kb: dict[int, list[tuple[float, int]]],
    direct_kk_w: dict[tuple[int, int], float],
    direct_bdy_w: dict[int, float],
    n_kept: int,
    *,
    type: str = "per_detector",
) -> int:
    """Dispatcher for chain-aware-reduction connectivity fixes.

    Without a fix, aggressive `weight_cutoff` values can leave kept detectors
    in connected components without boundary access, causing pymatching to
    fail on odd-parity syndromes ("No perfect matching could be found"). This
    dispatcher applies one of two correctness fixes after Step 4's selective
    filter:

      - "per_detector": BFS check per detector, with on-the-fly updates.
        Simple and adds one fix-up edge per stranded component.
      - "union_find": Union-Find component identification, with per-component
        globally-cheapest chain pick. Slightly cheaper fix-up edge weights.

    For typical (n_kept ~ few hundred, edges ~ thousand) graphs both run in
    sub-millisecond and produce identical fix-up edge counts.

    Mutates selective_bdy in place. Returns number of fix-up edges added.
    """
    if type == "per_detector":
        return _apply_connectivity_fix_per_detector(
            selective_kk, selective_bdy, chain_kb,
            direct_kk_w, direct_bdy_w, n_kept,
        )
    if type == "union_find":
        return _apply_connectivity_fix_union_find(
            selective_kk, selective_bdy, chain_kb,
            direct_kk_w, direct_bdy_w, n_kept,
        )
    raise ValueError(
        f"connectivity_fix_type must be 'per_detector' or 'union_find'; got {type!r}."
    )


def _reduce_dem_chain_aware_with_cutoff(
    dem: stim.DetectorErrorModel,
    clip_set: set,
    *,
    weight_cutoff: float = 15.0,
    connectivity_fix_type: str = "per_detector",
) -> tuple[stim.DetectorErrorModel, list[int], dict[int, int]]:
    """Build a reduced DEM by physically removing detectors in `clip_set`.

    The reduction:
      - All-kept errors are renumbered and preserved with original probabilities.
      - Cross-edges and multi-hop chain paths through the clipped subgraph are
        absorbed into kept-kept and kept-boundary edges via per-source Dijkstra.
      - Selective filter: chain edges are kept only if strictly cheaper than any
        existing direct alternative AND below `weight_cutoff` (~3 × 10^-7 prob
        at cutoff=15, far below any practical error rate).
      - Connectivity fix (Step 4.5): when the cutoff strands kept detectors in
        components with no boundary access, force-add the cheapest chain to
        boundary so MWPM never fails on odd-parity syndromes. See
        `_apply_connectivity_fix(...)` for the two algorithm choices.

    Returns (new_dem, kept_indices, old_to_new_map).
    """
    n_orig = dem.num_detectors
    kept = sorted(d for d in range(n_orig) if d not in clip_set)
    old_to_new = {d_old: d_new for d_new, d_old in enumerate(kept)}
    kept_set = set(kept)

    # Step 1: parse DEM into adjacency
    adj: list[list[tuple[int, float, int]]] = [[] for _ in range(n_orig)]
    bdy_edges: list[list[tuple[float, int]]] = [[] for _ in range(n_orig)]
    direct_errors: list[tuple[list[int], int, float, float]] = []

    for inst in dem:
        if inst.type != 'error':
            continue
        targets = inst.targets_copy()
        det_targets = [t.val for t in targets if t.is_relative_detector_id()]
        obs_mask = 0
        for t in targets:
            if t.is_logical_observable_id():
                obs_mask ^= (1 << t.val)
        p = float(inst.args_copy()[0])
        if p <= 0 or p >= 1:
            continue
        w = -math.log(p / (1 - p))

        if all(d in old_to_new for d in det_targets):
            direct_errors.append((det_targets, obs_mask, p, w))

        if len(det_targets) == 1:
            d = det_targets[0]
            bdy_edges[d].append((w, obs_mask))
        elif len(det_targets) == 2:
            a, b = det_targets
            adj[a].append((b, w, obs_mask))
            adj[b].append((a, w, obs_mask))
        # 0- or 3+-target errors: ignored for matching graph

    # Step 2: per-pair direct edge weights for filtering
    direct_kk_w: dict[tuple[int, int], float] = {}
    direct_bdy_w: dict[int, float] = {}
    for det_targets, _obs, _p, w in direct_errors:
        if len(det_targets) == 1:
            a = old_to_new[det_targets[0]]
            if a not in direct_bdy_w or direct_bdy_w[a] > w:
                direct_bdy_w[a] = w
        elif len(det_targets) == 2:
            a, b = sorted([old_to_new[d] for d in det_targets])
            if (a, b) not in direct_kk_w or direct_kk_w[(a, b)] > w:
                direct_kk_w[(a, b)] = w

    # Step 3: Dijkstra per kept source through the clipped subgraph
    chain_kk: dict[tuple[int, int], tuple[float, int]] = {}
    chain_kb: dict[int, list[tuple[float, int]]] = {}

    for src in kept:
        dist: dict[int, tuple[float, int]] = {}
        heap: list[tuple[float, int, int]] = []
        for (nbr, w, obs) in adj[src]:
            if nbr in clip_set:
                if nbr not in dist or dist[nbr][0] > w:
                    dist[nbr] = (w, obs)
                    heapq.heappush(heap, (w, nbr, obs))

        while heap:
            cur_w, cur, cur_obs = heapq.heappop(heap)
            if cur_w > dist.get(cur, (float('inf'),))[0]:
                continue

            # Boundary edges at this clipped node → src→boundary chain
            for (w_b, obs_b) in bdy_edges[cur]:
                src_new = old_to_new[src]
                chain_kb.setdefault(src_new, []).append((cur_w + w_b, cur_obs ^ obs_b))

            # Relax to neighbors
            for (nbr, w_e, obs_e) in adj[cur]:
                new_w = cur_w + w_e
                new_obs = cur_obs ^ obs_e
                if nbr == src:
                    continue
                if nbr in kept_set:
                    a, b = sorted([old_to_new[src], old_to_new[nbr]])
                    cur_best = chain_kk.get((a, b))
                    if cur_best is None or cur_best[0] > new_w:
                        chain_kk[(a, b)] = (new_w, new_obs)
                    continue   # don't relax through other kept nodes
                if nbr not in dist or dist[nbr][0] > new_w:
                    dist[nbr] = (new_w, new_obs)
                    heapq.heappush(heap, (new_w, nbr, new_obs))

    # Step 4: selective filter (strictly cheaper than direct AND below cutoff)
    selective_kk: dict[tuple[int, int], tuple[float, int]] = {}
    for (a, b), (w, obs) in chain_kk.items():
        direct_w = direct_kk_w.get((a, b), float('inf'))
        if w >= direct_w or w >= weight_cutoff:
            continue
        selective_kk[(a, b)] = (w, obs)

    selective_bdy: dict[int, tuple[float, int]] = {}
    for a, edges in chain_kb.items():
        min_chain_w, min_chain_obs = min(edges, key=lambda x: x[0])
        direct_w = direct_bdy_w.get(a, float('inf'))
        if min_chain_w >= direct_w or min_chain_w >= weight_cutoff:
            continue
        selective_bdy[a] = (min_chain_w, min_chain_obs)

    # Step 4.5: connectivity fix — guarantee every kept detector has a path
    # to boundary in the reduced matching graph (otherwise pymatching crashes
    # on odd-parity syndromes in stranded components).
    _apply_connectivity_fix(
        selective_kk, selective_bdy, chain_kb,
        direct_kk_w, direct_bdy_w,
        n_kept=len(kept),
        type=connectivity_fix_type,
    )

    # Step 5: assemble new DEM
    new_dem = stim.DetectorErrorModel()
    coords = dem.get_detector_coordinates()
    for d_old in kept:
        d_new = old_to_new[d_old]
        c = coords.get(d_old, [])
        new_dem.append('detector', list(c), [stim.target_relative_detector_id(d_new)])

    # 5a: original all-kept errors (renumbered, original probabilities)
    for det_targets, obs_mask, p, _w in direct_errors:
        new_targets = [stim.target_relative_detector_id(old_to_new[d]) for d in det_targets]
        for o in range(64):
            if (obs_mask >> o) & 1:
                new_targets.append(stim.target_logical_observable_id(o))
        new_dem.append('error', [p], new_targets)

    # 5b: selective kept-kept chain edges
    for (a, b), (w, obs_mask) in selective_kk.items():
        p = math.exp(-w) / (1 + math.exp(-w))
        if p <= 0 or p >= 1:
            continue
        new_targets = [
            stim.target_relative_detector_id(a),
            stim.target_relative_detector_id(b),
        ]
        for o in range(64):
            if (obs_mask >> o) & 1:
                new_targets.append(stim.target_logical_observable_id(o))
        new_dem.append('error', [p], new_targets)

    # 5c: selective boundary chain edges
    for src_new, (w, obs_mask) in selective_bdy.items():
        p = math.exp(-w) / (1 + math.exp(-w))
        if p <= 0 or p >= 1:
            continue
        new_targets = [stim.target_relative_detector_id(src_new)]
        for o in range(64):
            if (obs_mask >> o) & 1:
                new_targets.append(stim.target_logical_observable_id(o))
        new_dem.append('error', [p], new_targets)

    return new_dem, kept, old_to_new


def _repack_syndromes_to_kept(
    packed_dets: np.ndarray,
    kept: list[int],
    n_orig_dets: int,
) -> np.ndarray:
    """Take bit-packed (n_shots, ceil(n_orig/8)) syndrome bytes and produce
    bit-packed (n_shots, ceil(len(kept)/8)) bytes containing only the bits at
    the kept detector indices, in `kept`'s order."""
    unpacked = np.unpackbits(packed_dets, axis=1, bitorder='little')[:, :n_orig_dets]
    return np.packbits(unpacked[:, kept], axis=1, bitorder='little')


class FastPartialDecoder:
    """Fast partial decoder backed by a renumbered, chain-aware-contracted DEM.

    Build once for a given (gap_dem, partial_mask). Then call .decode_batch on
    bit-packed syndrome arrays to get partial predictions and gaps. Provides
    ~3-4x speedup over the slow path `dec._decode_batch_overwrite_last_byte`
    on sparse partial masks.

    Quality: Pearson(slow, fast) ≈ 0.97 with weight_cutoff=15. The fast gap
    distribution is shifted slightly upward, so when switching from slow to
    fast you may need to bump T_p by ~2-3 to recover the same accept rate.

    Typical usage:
        fpd = FastPartialDecoder(dec.gap_dem, mask_bool,
                                 decibels_per_w=dec.decibels_per_w)
        preds, gaps = fpd.decode_batch(samples['dets'])
    """

    def __init__(
        self,
        gap_dem: stim.DetectorErrorModel,
        mask_bool: np.ndarray,
        *,
        decibels_per_w: float,
        weight_cutoff: float = 15.0,
        connectivity_fix_type: str = "per_detector",
    ):
        if mask_bool.ndim != 1:
            raise ValueError(f"mask_bool must be 1-D; got shape {mask_bool.shape}.")
        self.n_circuit_dets = int(mask_bool.shape[0])
        self.n_gap_dets = int(gap_dem.num_detectors)
        self.decibels_per_w = float(decibels_per_w)
        self.weight_cutoff = float(weight_cutoff)
        self.connectivity_fix_type = str(connectivity_fix_type)

        # Clip set: non-mask circuit detectors. Virtual pair nodes (indices
        # n_circuit..n_gap-2) and the obs-detector (index n_gap-1) are NOT in
        # clip_set, so they're kept automatically.
        clip_set = {d for d in range(self.n_circuit_dets) if not bool(mask_bool[d])}

        new_dem, kept, old_to_new = _reduce_dem_chain_aware_with_cutoff(
            gap_dem, clip_set,
            weight_cutoff=self.weight_cutoff,
            connectivity_fix_type=self.connectivity_fix_type,
        )
        self.kept = kept
        self.old_to_new = old_to_new
        self.matcher = pymatching.Matching.from_detector_error_model(new_dem)

        # Obs-detector mechanics in the new (renumbered) index space.
        obs_old_idx = self.n_gap_dets - 1
        if obs_old_idx not in old_to_new:
            raise RuntimeError("obs-detector was unexpectedly clipped; cannot run gap trick.")
        new_obs_idx = old_to_new[obs_old_idx]
        n_kept = len(kept)
        if new_obs_idx != n_kept - 1:
            raise RuntimeError(
                f"obs-detector new index ({new_obs_idx}) is not the last index "
                f"(n_kept-1 = {n_kept - 1}); the bit-flip trick assumes it is."
            )
        self.new_width = (n_kept + 7) // 8
        last_byte_bit = new_obs_idx % 8
        self.obs_byte_mask = np.uint8(1 << last_byte_bit)

    def decode_batch(self, packed_dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode bit-packed syndromes through the reduced DEM.

        Mirrors the return shape of `dec._decode_batch_overwrite_last_byte`:
        returns (predictions, gaps), where predictions is an (n_shots,) bool
        array and gaps is an (n_shots,) float64 array.
        """
        new_packed = _repack_syndromes_to_kept(
            packed_dets, self.kept, self.n_gap_dets,
        )

        on_pkt = new_packed.copy()
        on_pkt[:, -1] |= self.obs_byte_mask
        _, on_w = self.matcher.decode_batch(
            on_pkt, return_weights=True,
            bit_packed_shots=True, bit_packed_predictions=True,
        )

        off_pkt = new_packed.copy()
        off_pkt[:, -1] &= np.uint8(~self.obs_byte_mask)
        _, off_w = self.matcher.decode_batch(
            off_pkt, return_weights=True,
            bit_packed_shots=True, bit_packed_predictions=True,
        )

        gaps = np.abs((on_w - off_w) * self.decibels_per_w).astype(np.float64)
        preds = (on_w < off_w).astype(np.bool_)
        return preds, gaps
