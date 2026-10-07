import json
import pathlib
import sys
import tempfile
from typing import Any, Iterable, Optional

import numpy as np
import sinter
import stim

src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))

import cultiv
import gen


class PartialMaskBuilder:
    """Build detector masks for the partial decoder of a gap-decoded circuit.

    A mask starts from a geometric region (build_partial_region_mask) or an
    explicit index set, and is then edited by augment / prune, which use a
    per-detector "essentials" score (_ESSENTIALS_TYPES), and by closure.
    Masks are bool arrays of shape (num_detectors,); packed forms use little
    bit order.
    """

    def __init__(
        self,
        *,
        circuit: Optional[stim.Circuit] = None,
        circuit_generator: Optional[Any] = None,
        stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ):
        if circuit_generator is not None:
            if getattr(circuit_generator, "ideal_circuit", None) is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = getattr(circuit_generator, "noisy_circuit", None) or circuit_generator.ideal_circuit
            if stage_timeline_map is None and getattr(circuit_generator, "params", None) is not None:
                stage_timeline_map = getattr(circuit_generator.params, "StageTimelineMap", None)
        if circuit is None:
            raise ValueError("Provide either circuit or circuit_generator.")

        self.circuit = circuit
        self.stage_timeline_map = stage_timeline_map or {}
        self.dem = self.circuit.detector_error_model()
        self.det_coords = self.dem.get_detector_coordinates()
        self.num_detectors = self.dem.num_detectors

        # Essentials: per-detector scores, independent of any mask.
        #   canonical          P(d fires | reject) - P(d fires | accept)
        #   causal             mean |gap - gap(d cleared)| over shots where d fired
        #   blending           (1 - alpha) * canonical_n + alpha * causal_n
        #   meangap            max_d' mean_gap(d') - mean_gap(d)
        #   logical_ambiguity  Q(d): how often d is touched by E_on xor E_off
        # <type>_essentials_order ranks detectors by descending score;
        # <type>_essentials_meta holds the score arrays.
        self.canonical_essentials_order: Optional[list[int]] = None
        self.canonical_essentials_meta: Optional[dict[str, Any]] = None
        self.causal_essentials_order: Optional[list[int]] = None
        self.causal_essentials_meta: Optional[dict[str, Any]] = None
        self.blending_essentials_order: Optional[list[int]] = None
        self.blending_essentials_meta: Optional[dict[str, Any]] = None
        self.meangap_essentials_order: Optional[list[int]] = None
        self.meangap_essentials_meta: Optional[dict[str, Any]] = None
        self.logical_ambiguity_essentials_order: Optional[list[int]] = None
        self.logical_ambiguity_essentials_meta: Optional[dict[str, Any]] = None

    # ---------------- geometry: spacetime region masks ----------------

    def _cycle_slices(self):
        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        return ticks, codes

    def _find_first_hybrid_slice_index(
        self,
        *,
        growth_ratio_threshold: float = 1.20,
        growth_abs_threshold: int = 8,
    ) -> tuple[int, list[int], list[int]]:
        ticks, codes = self._cycle_slices()
        counts = [len(codes[t].used_set) for t in ticks]
        if len(ticks) <= 1:
            return 0, ticks, counts
        for i in range(1, len(ticks)):
            prev = max(counts[i - 1], 1)
            cur = counts[i]
            if cur - counts[i - 1] >= growth_abs_threshold or cur / prev >= growth_ratio_threshold:
                return i, ticks, counts
        return 0, ticks, counts

    def _detector_ids_for_tick(self, code: Any) -> np.ndarray:
        ids: list[int] = []
        for tile in code.tiles:
            for f in tile.flags:
                try:
                    d = int(f)
                except ValueError:
                    continue
                if 0 <= d < self.num_detectors:
                    ids.append(d)
        if not ids:
            return np.empty(0, dtype=np.int64)
        return np.asarray(sorted(set(ids)), dtype=np.int64)

    # All detectors with coord[2] == coord_t, including those in no tile flag
    # (e.g. stabilizers at the cultivation -> escape boundary slice).
    def _detector_ids_for_coord_t(self, coord_t: int) -> np.ndarray:
        if not hasattr(self, "_coord_t_index_cache"):
            cache: dict[int, list[int]] = {}
            for d in range(self.num_detectors):
                c = self.det_coords.get(d, [])
                if len(c) < 3:
                    continue
                k = int(round(float(c[2])))
                cache.setdefault(k, []).append(d)
            self._coord_t_index_cache = {k: np.asarray(sorted(v), dtype=np.int64) for k, v in cache.items()}
        return self._coord_t_index_cache.get(int(coord_t), np.empty(0, dtype=np.int64))

    # Distinct coord[2] levels of the detectors flagged in the given ticks.
    def _coord_ts_from_ticks(self, ticks_to_use: list[int], codes: dict) -> list[int]:
        seen: set[int] = set()
        for tk in ticks_to_use:
            for d in self._detector_ids_for_tick(codes[tk]):
                c = self.det_coords.get(int(d), [])
                if len(c) >= 3:
                    seen.add(int(round(float(c[2]))))
        return sorted(seen)

    def build_partial_region_mask(
        self,
        *,
        left_len_start: int,
        left_len_end: int,
        top_len_start: int,
        top_len_end: int,
        t_start: int,
        t_end: int,
        mode: str = "rectangle",
        triangle_half: str = "left",
        stage_name: str = "escape",
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Mask of a spacetime sub-rectangle of the patch.

        All ranges are 1-indexed and half-open. t counts slices from the first
        surface-code (hybrid) slice: t_start=1, t_end=2 selects that slice.
        left_len spans row rank (top to bottom), top_len spans column rank
        (left to right). mode="triangle" keeps the half of the rectangle on
        one side of its anti-diagonal, chosen by triangle_half.

        Returns (mask_bool, mask_packed, metadata).
        """
        if left_len_start <= 0 or top_len_start <= 0 or t_start <= 0:
            raise ValueError("left_len_start, top_len_start, and t_start must be positive.")
        if left_len_end <= left_len_start or top_len_end <= top_len_start or t_end <= t_start:
            raise ValueError("Require *_end > *_start for left_len, top_len, and t.")
        if mode not in {"rectangle", "triangle"}:
            raise ValueError("mode must be 'rectangle' or 'triangle'.")
        if triangle_half not in {"left", "right"}:
            raise ValueError("triangle_half must be 'left' or 'right'.")

        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        if not ticks:
            raise ValueError("No cycle slices found.")

        t_lo = hybrid_idx + int(t_start) - 1
        t_hi = hybrid_idx + int(t_end) - 1
        sel_ticks = ticks[t_lo:t_hi]
        if not sel_ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        # Iterate per coord_t, not per tick, so detectors at cycle boundaries
        # are included.
        sel_coord_ts = self._coord_ts_from_ticks(sel_ticks, codes)
        if not sel_coord_ts:
            raise ValueError("No detector coord[2] levels resolved from the selected ticks.")

        mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        z_selected_all: list[int] = []
        eps = 1e-9

        width = top_len_end - top_len_start
        height = left_len_end - left_len_start

        for coord_t in sel_coord_ts:
            tick_ids = self._detector_ids_for_coord_t(coord_t)
            if len(tick_ids) == 0:
                continue

            x_tick = np.asarray([float(self.det_coords[d][0]) for d in tick_ids], dtype=np.float64)
            y_tick = np.asarray([float(self.det_coords[d][1]) for d in tick_ids], dtype=np.float64)
            z_tick = np.full(len(tick_ids), int(coord_t), dtype=np.int64)
            z_selected_all.extend(z_tick.tolist())

            u_tick = x_tick + y_tick
            v_tick = y_tick - x_tick
            u_levels = np.unique(np.sort(u_tick))
            u_inner = float(u_levels[1]) if len(u_levels) > 1 else float(u_levels[0])

            top_inner_idx = np.flatnonzero(np.abs(u_tick - u_inner) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.flatnonzero(np.abs(u_tick - float(u_levels[0])) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.arange(len(tick_ids), dtype=np.int64)

            top_order = top_inner_idx[np.argsort(-v_tick[top_inner_idx], kind="mergesort")]
            top_v_values = v_tick[top_order]
            if len(top_v_values) == 0:
                continue

            # Ranges covering every row and column select the whole slice.
            saturate = (
                int(top_len_start) <= 1
                and int(top_len_end) >= len(top_v_values) + 1
                and int(left_len_start) <= 1
                and int(left_len_end) >= len(u_levels) + 1
            )
            if saturate:
                mask_bool[tick_ids] = True
                continue

            col_start = min(max(top_len_start - 1, 0), len(top_v_values) - 1)
            col_end_excl = min(max(top_len_end - 1, 1), len(top_v_values))
            if col_end_excl <= col_start:
                continue

            # Column rank, left to right along the top-inner row.
            col_rank = np.argmin(np.abs(v_tick[:, None] - top_v_values[None, :]), axis=1) + 1

            # Row rank, top to bottom.
            row_rank = np.argmin(np.abs(u_tick[:, None] - u_levels[None, :]), axis=1) + 1

            in_rect = (
                (col_rank >= top_len_start)
                & (col_rank < top_len_end)
                & (row_rank >= left_len_start)
                & (row_rank < left_len_end)
            )
            keep = in_rect

            if mode == "triangle" and width > 1 and height > 1:
                c_local = (col_rank.astype(np.float64) - float(top_len_start)) / float(width - 1)
                r_local = (row_rank.astype(np.float64) - float(left_len_start)) / float(height - 1)
                if triangle_half == "left":
                    keep = keep & ((r_local + c_local) <= 1.0 + 1e-9)
                else:
                    keep = keep & ((r_local + c_local) >= 1.0 - 1e-9)

            mask_bool[tick_ids[keep]] = True

        if z_selected_all:
            z_levels_selected = sorted(set(int(e) for e in z_selected_all))
            z0 = int(z_levels_selected[0])
            z1 = int(z_levels_selected[-1])
        else:
            z_levels_selected = []
            z0 = 0
            z1 = 0

        mask_packed = np.packbits(mask_bool, bitorder="little")
        metadata = {
            "mode": mode,
            "triangle_half": triangle_half,
            "stage_name": stage_name,
            "left_len_start": int(left_len_start),
            "left_len_end": int(left_len_end),
            "top_len_start": int(top_len_start),
            "top_len_end": int(top_len_end),
            "t_start": int(t_start),
            "t_end": int(t_end),
            "hybrid_slice_index": int(hybrid_idx),
            "hybrid_slice_tick": int(slice_ticks[hybrid_idx]) if slice_ticks else 0,
            "slice_ticks": [int(e) for e in slice_ticks],
            "slice_qubit_counts": [int(e) for e in slice_counts],
            "selected_slice_ticks": [int(e) for e in sel_ticks],
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in z_levels_selected],
            "selected_detector_count": int(np.count_nonzero(mask_bool)),
        }
        return mask_bool, mask_packed, metadata

    def build_mask_from_indices(
        self,
        indices: Iterable[int],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Mask from explicit detector indices. Duplicates are merged;
        out-of-range values raise ValueError.

        The metadata is derived from the selection, so the result works with
        the visualize_* methods. Its geometric fields are None.

        Returns (mask_bool, mask_packed, metadata).
        """
        idx_arr = np.asarray(list(indices), dtype=np.int64)
        if idx_arr.size > 0:
            if int(idx_arr.min()) < 0 or int(idx_arr.max()) >= self.num_detectors:
                raise ValueError(
                    f"indices out of range [0, {self.num_detectors}); "
                    f"got min={int(idx_arr.min())}, max={int(idx_arr.max())}."
                )
        mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        if idx_arr.size > 0:
            mask_bool[idx_arr] = True
        mask_packed = np.packbits(mask_bool, bitorder="little")

        # coord_t levels of the selected detectors
        sel_coord_ts: list[int] = []
        for d in idx_arr.tolist():
            c = self.det_coords.get(int(d), [])
            if len(c) >= 3:
                sel_coord_ts.append(int(round(float(c[2]))))
        sel_coord_ts = sorted(set(sel_coord_ts))

        # ticks that flag a detector at a selected coord_t
        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        sel_ticks: list[int] = []
        if sel_coord_ts:
            sel_set = set(sel_coord_ts)
            for tk in ticks:
                det_ids = self._detector_ids_for_tick(codes[tk])
                for d in det_ids:
                    c = self.det_coords.get(int(d), [])
                    if len(c) >= 3 and int(round(float(c[2]))) in sel_set:
                        sel_ticks.append(tk)
                        break

        z0 = sel_coord_ts[0] if sel_coord_ts else 0
        z1 = sel_coord_ts[-1] if sel_coord_ts else 0

        metadata = {
            "mode": "custom",
            "triangle_half": None,
            "stage_name": "custom",
            "left_len_start": None,
            "left_len_end": None,
            "top_len_start": None,
            "top_len_end": None,
            "t_start": None,
            "t_end": None,
            "hybrid_slice_index": int(hybrid_idx),
            "hybrid_slice_tick": int(slice_ticks[hybrid_idx]) if slice_ticks else 0,
            "slice_ticks": [int(e) for e in slice_ticks],
            "slice_qubit_counts": [int(e) for e in slice_counts],
            "selected_slice_ticks": [int(e) for e in sel_ticks],
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in sel_coord_ts],
            "selected_detector_count": int(np.count_nonzero(mask_bool)),
        }
        return mask_bool, mask_packed, metadata

    def visualize_partial_region_mask_svg(
        self,
        path: pathlib.Path,
        *,
        mask_bool: np.ndarray,
        metadata: dict[str, Any],
        canvas_height: int = 900,
    ) -> None:
        if mask_bool is None or not np.any(mask_bool):
            raise ValueError("No selected region mask found in mask_bool.")

        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        sel_ticks = [int(e) for e in metadata.get("selected_slice_ticks", [])]
        if not sel_ticks:
            hybrid_idx = int(metadata.get("hybrid_slice_index", 0))
            t_start = int(metadata.get("t_start", 1))
            t_end = int(metadata.get("t_end", t_start + 1))
            t_lo = hybrid_idx + t_start - 1
            t_hi = hybrid_idx + t_end - 1
            sel_ticks = ticks[t_lo:t_hi]
            if not sel_ticks and ticks:
                sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        panels = [codes[k].with_transformed_coords(lambda e: e * (1 + 1j)) for k in sel_ticks]

        def tile_color(tile: gen.Tile):
            dr, = tile.flags
            d = int(dr)
            if d >= len(mask_bool):
                return 0.6, 0.6, 0.6
            if mask_bool[d]:
                return 0.95, 0.1, 0.1
            return 0.82, 0.82, 0.82

        titles = [f"tick={k}" for k in sel_ticks]
        mode = metadata.get("mode", "rectangle")
        if mode == "custom":
            title_prefix = (
                f"Custom mask: {metadata.get('selected_detector_count', '?')} detectors, "
                f"z=[{metadata.get('z_start')}..{metadata.get('z_end')}]"
            )
        else:
            title_prefix = (
                f"Partial region ({mode}, half={metadata.get('triangle_half', 'left')}): "
                f"left=[{metadata.get('left_len_start')}..{metadata.get('left_len_end')}), "
                f"top=[{metadata.get('top_len_start')}..{metadata.get('top_len_end')}), "
                f"t=[{metadata.get('t_start')}..{metadata.get('t_end')}), "
                f"z=[{metadata.get('z_start')}..{metadata.get('z_end')}]"
            )
        if titles:
            titles[0] = f"{title_prefix} | {titles[0]}"

        panels[0].write_svg(
            path,
            canvas_height=canvas_height,
            other=panels[1:],
            title=titles,
            tile_color_func=tile_color,
            show_coords=False,
            show_obs=False,
        )

    def visualize_partial_region_mask_plot(
        self,
        *,
        mask_bool: np.ndarray,
        metadata: dict[str, Any],
    ):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.visualize_partial_region_mask_svg(
                tmp_path,
                mask_bool=mask_bool,
                metadata=metadata,
            )
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    # ---------------- essentials: detector scores ----------------

    _ESSENTIALS_TYPES = ("canonical", "causal", "blending", "meangap", "logical_ambiguity")

    def compute_canonical_essentials_order(
        self,
        *,
        gap_threshold: float,
        sampler: Optional[Any] = None,
        shots: int = 200_000,
    ) -> list[int]:
        """Rank detectors by
            score(d) = P(d fired | gap_c < gap_threshold) - P(d fired | gap_c >= gap_threshold)
        over postselected shots, with gap_c the complete decoder's gap.
        Stores canonical_essentials_order / _meta and returns the order.
        """
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if sampler is None:
            sampler = cultiv.DesaturationSampler()

        task = sinter.Task(circuit=self.circuit, detector_error_model=self.dem)
        dec = sampler.compiled_sampler_for_task(task)
        dets, _actual_obs = dec.gap_circuit_sampler.sample(
            shots, separate_observables=True, bit_packed=True,
        )
        keep = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep]
        n_kept = int(dets.shape[0])

        if n_kept == 0:
            self.canonical_essentials_order = list(range(self.num_detectors))
            self.canonical_essentials_meta = {
                "gap_threshold": float(gap_threshold),
                "shots": int(shots),
                "kept_shots": 0,
                "score_values": [0.0] * self.num_detectors,
                "warning": "no shots survived postselect; ordering is degenerate",
            }
            return list(self.canonical_essentials_order)

        _, complete_gaps = dec._decode_batch_overwrite_last_byte(dets.copy())
        complete_gaps = complete_gaps.astype(np.float64)
        ca = complete_gaps >= float(gap_threshold)

        det_full_bool = (
            np.unpackbits(dets.astype(np.uint8), bitorder="little", axis=1)[:, : self.num_detectors]
            .astype(np.bool_)
        )
        # With no accepts or no rejects every score is 0.
        n_acc = int(ca.sum())
        n_rej = n_kept - n_acc
        if n_acc == 0 or n_rej == 0:
            score = np.zeros(self.num_detectors, dtype=np.float64)
        else:
            r_reject = det_full_bool[~ca].mean(axis=0)
            r_accept = det_full_bool[ca].mean(axis=0)
            score = r_reject - r_accept

        ranking = np.argsort(-score).tolist()
        self.canonical_essentials_order = [int(d) for d in ranking]
        self.canonical_essentials_meta = {
            "gap_threshold": float(gap_threshold),
            "shots": int(shots),
            "kept_shots": n_kept,
            "complete_accept_count": n_acc,
            "complete_reject_count": n_rej,
            "score_values": score.tolist(),
        }
        return list(self.canonical_essentials_order)

    def prepare_causal_essentials(
        self,
        *,
        sampler: Optional[Any] = None,
        shots: int = 1_000_000,
        subsample_size: Optional[int] = None,
        rng_seed: int = 0,
    ) -> dict[str, Any]:
        """Stage 1 of 3. Sample, postselect, optionally subsample, and decode
        the baseline complete gaps. Returns the context for score_causal_range
        and finalize_causal_essentials. Scoring detector ranges in parallel
        workers gives bit-identical results (see gen_essentials.py).
        """
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if subsample_size is not None and subsample_size <= 0:
            raise ValueError("subsample_size must be positive (or None).")
        if sampler is None:
            sampler = cultiv.DesaturationSampler()
        task = sinter.Task(circuit=self.circuit, detector_error_model=self.dem)
        dec = sampler.compiled_sampler_for_task(task)
        dets, _actual_obs = dec.gap_circuit_sampler.sample(
            shots, separate_observables=True, bit_packed=True,
        )
        keep = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep]
        n_kept = int(dets.shape[0])
        if n_kept == 0:
            return {"dec": dec, "sub_dets": dets, "sub_gaps_orig": np.zeros(0),
                    "sub_unpacked": np.zeros((0, self.num_detectors), dtype=np.bool_),
                    "shots": int(shots), "kept_shots": 0, "subsample_size": 0}
        if subsample_size is None or subsample_size >= n_kept:
            sub_dets = dets
            sub_n = n_kept
        else:
            rng = np.random.default_rng(rng_seed)
            sub_idx = rng.choice(n_kept, size=int(subsample_size), replace=False)
            sub_dets = dets[sub_idx].copy()
            sub_n = int(subsample_size)
        # Baseline complete-decoder gaps and unpacked syndromes.
        _, sub_gaps_orig = dec._decode_batch_overwrite_last_byte(sub_dets.copy())
        sub_gaps_orig = sub_gaps_orig.astype(np.float64)
        sub_unpacked = (
            np.unpackbits(sub_dets, axis=1, bitorder="little")[:, : self.num_detectors]
            .astype(np.bool_)
        )
        return {"dec": dec, "sub_dets": sub_dets, "sub_gaps_orig": sub_gaps_orig,
                "sub_unpacked": sub_unpacked, "shots": int(shots),
                "kept_shots": n_kept, "subsample_size": sub_n}

    @staticmethod
    def score_causal_range(
        ctx: dict[str, Any], d_lo: int, d_hi: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stage 2 of 3. For each detector d in [d_lo, d_hi), decode the shots
        that fired d once more with d's bit cleared. Returns per detector
        (sum |gap - gap_cleared|, sum (gap - gap_cleared), count).
        """
        dec = ctx["dec"]
        sub_dets = ctx["sub_dets"]
        sub_gaps_orig = ctx["sub_gaps_orig"]
        sub_unpacked = ctx["sub_unpacked"]
        n = d_hi - d_lo
        sum_abs = np.zeros(n, dtype=np.float64)
        sum_signed = np.zeros(n, dtype=np.float64)
        count = np.zeros(n, dtype=np.int64)
        for i, d in enumerate(range(d_lo, d_hi)):
            shots_with_d = np.where(sub_unpacked[:, d])[0]
            if shots_with_d.size == 0:
                continue
            flip_dets = sub_dets[shots_with_d].copy()
            byte_idx = d // 8
            bit_mask = np.uint8(1 << (d % 8))
            flip_dets[:, byte_idx] &= np.uint8(~bit_mask)
            _, gaps_alt = dec._decode_batch_overwrite_last_byte(flip_dets.copy())
            delta = sub_gaps_orig[shots_with_d] - gaps_alt.astype(np.float64)
            sum_abs[i] = float(np.sum(np.abs(delta)))
            sum_signed[i] = float(np.sum(delta))
            count[i] = int(shots_with_d.size)
        return sum_abs, sum_signed, count

    def finalize_causal_essentials(
        self,
        ctx: dict[str, Any],
        sum_abs: np.ndarray,
        sum_signed: np.ndarray,
        count: np.ndarray,
    ) -> list[int]:
        """Stage 3 of 3. score(d) = sum_abs[d] / count[d]. Stores
        causal_essentials_order / _meta and returns the order."""
        if ctx["kept_shots"] == 0:
            self.causal_essentials_order = list(range(self.num_detectors))
            self.causal_essentials_meta = {
                "shots": int(ctx["shots"]),
                "kept_shots": 0,
                "subsample_size": 0,
                "score_values": [0.0] * self.num_detectors,
                "score_signed": [0.0] * self.num_detectors,
                "n_samples_per_det": [0] * self.num_detectors,
                "warning": "no shots survived postselect; ordering is degenerate",
            }
            return list(self.causal_essentials_order)
        sum_abs = np.asarray(sum_abs, dtype=np.float64)
        sum_signed = np.asarray(sum_signed, dtype=np.float64)
        count = np.asarray(count, dtype=np.int64)
        causal_abs = np.divide(sum_abs, count, out=np.zeros_like(sum_abs), where=count > 0)
        causal_signed = np.divide(sum_signed, count, out=np.zeros_like(sum_signed), where=count > 0)
        ranking = np.argsort(-causal_abs).tolist()
        self.causal_essentials_order = [int(d) for d in ranking]
        self.causal_essentials_meta = {
            "shots": int(ctx["shots"]),
            "kept_shots": int(ctx["kept_shots"]),
            "subsample_size": int(ctx["subsample_size"]),
            "score_values": causal_abs.tolist(),
            "score_signed": causal_signed.tolist(),
            "n_samples_per_det": count.tolist(),
        }
        return list(self.causal_essentials_order)

    def compute_causal_essentials_order(
        self,
        *,
        sampler: Optional[Any] = None,
        shots: int = 1_000_000,
        subsample_size: Optional[int] = None,
        rng_seed: int = 0,
    ) -> list[int]:
        """Rank detectors by causal sensitivity:
            score(d) = mean over kept shots where d fired of |gap - gap(d cleared)|
        with both gaps from the complete decoder. Costs about num_detectors
        batched decodes. subsample_size draws that many kept shots at random
        before scoring. gen_essentials.py runs the same scoring in parallel.
        Stores causal_essentials_order / _meta (score_values, score_signed,
        n_samples_per_det) and returns the order.
        """
        ctx = self.prepare_causal_essentials(
            sampler=sampler, shots=shots, subsample_size=subsample_size, rng_seed=rng_seed,
        )
        if ctx["kept_shots"] == 0:
            return self.finalize_causal_essentials(ctx, np.zeros(self.num_detectors),
                                                   np.zeros(self.num_detectors), np.zeros(self.num_detectors, np.int64))
        return self.finalize_causal_essentials(
            ctx, *self.score_causal_range(ctx, 0, self.num_detectors),
        )

    # ---------------- logical-ambiguity essentials (Q) ----------------

    def prepare_logical_ambiguity_essentials(
        self,
        *,
        sampler: Optional[Any] = None,
        shots: int = 200_000,
    ) -> dict[str, Any]:
        """Stage 1 of 3. Sample, postselect and build the complete gap decoder.
        Returns the context for score_logical_ambiguity_range and
        finalize_logical_ambiguity_essentials.
        """
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if sampler is None:
            sampler = cultiv.DesaturationSampler()
        task = sinter.Task(circuit=self.circuit, detector_error_model=self.dem)
        dec = sampler.compiled_sampler_for_task(task)
        dets, _actual_obs = dec.gap_circuit_sampler.sample(
            shots, separate_observables=True, bit_packed=True,
        )
        dets = dets[~np.any(dets & dec._discard_mask, axis=1)]
        # Edges come from the sampler's own pymatching Matching (gap_decoder),
        # the one that computes the gap.
        return {"dec": dec, "matcher": dec.gap_decoder, "dets": dets, "shots": int(shots),
                "kept_shots": int(dets.shape[0]), "num_gap_dets": int(dec.num_dets)}

    @staticmethod
    def score_logical_ambiguity_range(
        ctx: dict[str, Any], s_lo: int, s_hi: int, num_detectors: int,
    ) -> tuple[np.ndarray, np.ndarray, int, float]:
        """Stage 2 of 3. Accumulate over kept shots [s_lo, s_hi):
            count_all[d] = number of shots with d touched by Z(s)
            wexp[d]      = sum over those shots of exp(-dw_s)
            n            = shots scored
            wsum         = sum over scored shots of exp(-dw_s)
        Per shot, decoding with the obs detector forced to 1 gives the
        minimum-weight explanation (w_on, E_on); forced to 0, (w_off, E_off).
            dw   = |w_on - w_off| in the matcher's log-odds units; exp(-dw)
                   is the likelihood ratio of the losing hypothesis
            Z(s) = E_on xor E_off, as sorted (a, b) edge pairs
        d is touched by Z if it is an endpoint of an edge of Z and lies in
        [0, num_detectors): virtual pair nodes, the obs node and the boundary
        (-1) are excluded. A detector counts once per shot.
        """
        matcher = ctx["matcher"]
        dets = ctx["dets"]
        ngap = ctx["num_gap_dets"]
        obs_i = ngap - 1
        count_all = np.zeros(num_detectors, dtype=np.int64)
        wexp = np.zeros(num_detectors, dtype=np.float64)
        n = 0
        wsum = 0.0
        for s in range(s_lo, s_hi):
            fired = np.flatnonzero(np.unpackbits(dets[s], bitorder="little")[:num_detectors])
            syn = np.zeros((1, ngap), dtype=np.uint8)
            syn[0, fired] = 1
            syn[0, obs_i] = 1
            _, w_on = matcher.decode_batch(syn, return_weights=True)
            e_on = matcher.decode_to_edges_array(syn[0])
            syn[0, obs_i] = 0
            _, w_off = matcher.decode_batch(syn, return_weights=True)
            e_off = matcher.decode_to_edges_array(syn[0])
            dw = float(abs(float(w_on[0]) - float(w_off[0])))
            z = set(map(tuple, np.sort(e_on, axis=1))) ^ set(map(tuple, np.sort(e_off, axis=1)))
            touched = {int(x) for a, b in z for x in (a, b) if 0 <= x < num_detectors}
            w = float(np.exp(-dw))
            for d in touched:
                count_all[d] += 1
                wexp[d] += w
            n += 1
            wsum += w
        return count_all, wexp, n, wsum

    def finalize_logical_ambiguity_essentials(
        self,
        ctx: dict[str, Any],
        count_all: np.ndarray,
        wexp: np.ndarray,
        n: int,
        wsum: float,
        *,
        q_type: str = "all",
    ) -> list[int]:
        """Stage 3 of 3. Q_all(d) = count_all[d] / n, Q_exp(d) = wexp[d] / wsum.
        The stored order follows q_type ('all' or 'exp'); meta keeps both."""
        if q_type not in ("all", "exp"):
            raise ValueError("q_type must be 'all' or 'exp'.")
        q_all = np.asarray(count_all, dtype=np.float64) / max(int(n), 1)
        q_exp = np.asarray(wexp, dtype=np.float64) / max(float(wsum), 1e-300)
        chosen = q_all if q_type == "all" else q_exp
        ranking = np.argsort(-chosen, kind="stable").tolist()
        self.logical_ambiguity_essentials_order = [int(d) for d in ranking]
        self.logical_ambiguity_essentials_meta = {
            "shots": int(ctx["shots"]),
            "kept_shots": int(ctx["kept_shots"]),
            "n_scored": int(n),
            "q_type": q_type,
            "score_values": chosen.tolist(),
            "score_all": q_all.tolist(),
            "score_exp": q_exp.tolist(),
        }
        if n == 0:
            self.logical_ambiguity_essentials_meta["warning"] = "no shots survived postselect; ordering is degenerate"
        return list(self.logical_ambiguity_essentials_order)

    def compute_logical_ambiguity_essentials_order(
        self,
        *,
        sampler: Optional[Any] = None,
        shots: int = 200_000,
        q_type: str = "all",
    ) -> list[int]:
        """Rank detectors by logical ambiguity Q. For each kept shot, the
        complete decoder's minimum-weight explanations under the two logical
        hypotheses, E_on and E_off, differ on Z = E_on xor E_off, a
        syndrome-free, logically nontrivial set of matching edges.
            Q_all(d) = P(d touched by Z)
            Q_exp(d) = sum_s exp(-dw_s) 1[d touched by Z(s)] / sum_s exp(-dw_s)
        with dw = |w_on - w_off|. See score_logical_ambiguity_range for the
        exact definitions. gen_essentials.py runs the same scoring in parallel.
        """
        ctx = self.prepare_logical_ambiguity_essentials(sampler=sampler, shots=shots)
        return self.finalize_logical_ambiguity_essentials(
            ctx, *self.score_logical_ambiguity_range(ctx, 0, ctx["kept_shots"], self.num_detectors),
            q_type=q_type,
        )

    def logical_ambiguity_base_mask(
        self,
        K: int,
        *,
        q_type: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Base mask M0 = TopK(Q): the K highest-Q detectors, ties to the lowest
        index. q_type None uses the stored order; 'all' or 'exp' re-ranks from
        the stored scores. Returns (mask_bool, mask_packed, n_added).
        """
        if self.logical_ambiguity_essentials_meta is None:
            raise ValueError(
                "Logical-ambiguity essentials are not computed. "
                "Call compute_logical_ambiguity_essentials_order(...) or load them first."
            )
        if q_type is None:
            ranking = self._ranking_for_type("logical_ambiguity")
        else:
            if q_type not in ("all", "exp"):
                raise ValueError("q_type must be 'all', 'exp' or None.")
            scores = np.asarray(self.logical_ambiguity_essentials_meta[f"score_{q_type}"], dtype=np.float64)
            ranking = np.argsort(-scores, kind="stable").tolist()
        return self._add_top_K_from_ranking(None, ranking, K)

    def logical_ambiguity_coverage(self, mask_bool: np.ndarray, *, q_type: str = "all") -> float:
        """sum_{d in M} Q_d / sum_d Q_d for the requested score array."""
        q = np.asarray(self.logical_ambiguity_essentials_meta[f"score_{q_type}"], dtype=np.float64)
        return float(q[np.asarray(mask_bool, dtype=np.bool_)].sum() / max(q.sum(), 1e-300))

    def compute_blending_essentials_order(
        self,
        *,
        alpha: float,
    ) -> list[int]:
        """Rank detectors by
            blended[d] = (1 - alpha) * canon_n[d] + alpha * causal_n[d]
        with canon_n and causal_n the canonical and causal scores, min-max
        normalized to [0, 1]. alpha=0 gives the canonical order, alpha=1 the
        causal one. Both must be computed first; nothing is sampled or decoded.
        Stores blending_essentials_order / _meta and returns the order.
        """
        if not (0.0 <= float(alpha) <= 1.0):
            raise ValueError("alpha must be in [0, 1].")
        if self.canonical_essentials_meta is None or self.canonical_essentials_order is None:
            raise ValueError(
                "Canonical essentials order is not computed. "
                "Call compute_canonical_essentials_order(gap_threshold=...) first."
            )
        if self.causal_essentials_meta is None or self.causal_essentials_order is None:
            raise ValueError(
                "Causal essentials order is not computed. "
                "Call compute_causal_essentials_order(...) first."
            )

        canon = np.asarray(self.canonical_essentials_meta["score_values"], dtype=np.float64)
        causal = np.asarray(self.causal_essentials_meta["score_values"], dtype=np.float64)

        def _norm01(x: np.ndarray) -> np.ndarray:
            lo, hi = float(x.min()), float(x.max())
            if hi - lo < 1e-30:
                return np.zeros_like(x)
            return (x - lo) / (hi - lo)

        canon_n = _norm01(canon)
        causal_n = _norm01(causal)
        blended = (1.0 - float(alpha)) * canon_n + float(alpha) * causal_n

        ranking = np.argsort(-blended).tolist()
        self.blending_essentials_order = [int(d) for d in ranking]
        self.blending_essentials_meta = {
            "alpha": float(alpha),
            "score_values": blended.tolist(),
            "canonical_normalized": canon_n.tolist(),
            "causal_normalized": causal_n.tolist(),
        }
        return list(self.blending_essentials_order)

    def compute_meangap_essentials_order(
        self,
        *,
        sampler: Optional[Any] = None,
        shots: int = 200_000,
    ) -> list[int]:
        """Rank detectors by
            score(d) = max_d' mean_gap(d') - mean_gap(d)
        with mean_gap(d) the mean complete-decoder gap over kept shots where d
        fired, and the max taken over detectors that fired at least once.
        Detectors that never fire score 0. A high score means d fires mostly
        on low-gap shots: a continuous counterpart of the canonical score.
        Stores meangap_essentials_order / _meta (score_values,
        mean_gap_when_active, count) and returns the order.
        """
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if sampler is None:
            sampler = cultiv.DesaturationSampler()

        task = sinter.Task(circuit=self.circuit, detector_error_model=self.dem)
        dec = sampler.compiled_sampler_for_task(task)
        dets, _actual_obs = dec.gap_circuit_sampler.sample(
            shots, separate_observables=True, bit_packed=True,
        )
        keep = ~np.any(dets & dec._discard_mask, axis=1)
        dets = dets[keep]
        n_kept = int(dets.shape[0])

        if n_kept == 0:
            self.meangap_essentials_order = list(range(self.num_detectors))
            self.meangap_essentials_meta = {
                "shots": int(shots),
                "kept_shots": 0,
                "score_values": [0.0] * self.num_detectors,
                "mean_gap_when_active": [0.0] * self.num_detectors,
                "count": [0] * self.num_detectors,
                "warning": "no shots survived postselect; ordering is degenerate",
            }
            return list(self.meangap_essentials_order)

        _, complete_gaps = dec._decode_batch_overwrite_last_byte(dets.copy())
        complete_gaps = complete_gaps.astype(np.float64)

        det_bits = (
            np.unpackbits(dets, axis=1, bitorder="little")[:, : self.num_detectors]
            .astype(np.float64)
        )
        count = det_bits.sum(axis=0)
        sum_gap = det_bits.T @ complete_gaps

        mean_gap = np.divide(
            sum_gap, count, out=np.zeros_like(sum_gap), where=count > 0,
        )

        # Detectors that never fired keep score 0, so they never outrank an
        # active detector.
        valid = count > 0
        if valid.any():
            max_mean = float(mean_gap[valid].max())
        else:
            max_mean = 0.0
        score = np.zeros_like(mean_gap)
        score[valid] = max_mean - mean_gap[valid]

        ranking = np.argsort(-score).tolist()
        self.meangap_essentials_order = [int(d) for d in ranking]
        self.meangap_essentials_meta = {
            "shots": int(shots),
            "kept_shots": n_kept,
            "score_values": score.tolist(),
            "mean_gap_when_active": mean_gap.tolist(),
            "count": count.astype(np.int64).tolist(),
        }
        return list(self.meangap_essentials_order)

    # ---------------- mask operations: augment, prune, closure ----------------

    def _ranking_for_type(self, type: str) -> list[int]:
        if type == "canonical":
            if self.canonical_essentials_order is None:
                raise ValueError(
                    "Canonical essentials order is not computed. "
                    "Call compute_canonical_essentials_order(gap_threshold=...) first."
                )
            return self.canonical_essentials_order
        if type == "causal":
            if self.causal_essentials_order is None:
                raise ValueError(
                    "Causal essentials order is not computed. "
                    "Call compute_causal_essentials_order(...) first."
                )
            return self.causal_essentials_order
        if type == "blending":
            if self.blending_essentials_order is None:
                raise ValueError(
                    "Blending essentials order is not computed. "
                    "Call compute_blending_essentials_order(alpha=...) first."
                )
            return self.blending_essentials_order
        if type == "meangap":
            if self.meangap_essentials_order is None:
                raise ValueError(
                    "Meangap essentials order is not computed. "
                    "Call compute_meangap_essentials_order(...) first."
                )
            return self.meangap_essentials_order
        if type == "logical_ambiguity":
            if self.logical_ambiguity_essentials_order is None:
                raise ValueError(
                    "Logical-ambiguity essentials order is not computed. "
                    "Call compute_logical_ambiguity_essentials_order(...) first."
                )
            return self.logical_ambiguity_essentials_order
        raise ValueError(
            f"type must be one of {self._ESSENTIALS_TYPES}; got {type!r}."
        )

    def _scores_for_type(self, type: str) -> np.ndarray:
        if type == "canonical":
            if self.canonical_essentials_meta is None:
                raise ValueError(
                    "Canonical essentials order is not computed. "
                    "Call compute_canonical_essentials_order(gap_threshold=...) first."
                )
            return np.asarray(self.canonical_essentials_meta["score_values"], dtype=np.float64)
        if type == "causal":
            if self.causal_essentials_meta is None:
                raise ValueError(
                    "Causal essentials order is not computed. "
                    "Call compute_causal_essentials_order(...) first."
                )
            return np.asarray(self.causal_essentials_meta["score_values"], dtype=np.float64)
        if type == "blending":
            if self.blending_essentials_meta is None:
                raise ValueError(
                    "Blending essentials order is not computed. "
                    "Call compute_blending_essentials_order(alpha=...) first."
                )
            return np.asarray(self.blending_essentials_meta["score_values"], dtype=np.float64)
        if type == "meangap":
            if self.meangap_essentials_meta is None:
                raise ValueError(
                    "Meangap essentials order is not computed. "
                    "Call compute_meangap_essentials_order(...) first."
                )
            return np.asarray(self.meangap_essentials_meta["score_values"], dtype=np.float64)
        if type == "logical_ambiguity":
            if self.logical_ambiguity_essentials_meta is None:
                raise ValueError(
                    "Logical-ambiguity essentials order is not computed. "
                    "Call compute_logical_ambiguity_essentials_order(...) first."
                )
            return np.asarray(self.logical_ambiguity_essentials_meta["score_values"], dtype=np.float64)
        raise ValueError(
            f"type must be one of {self._ESSENTIALS_TYPES}; got {type!r}."
        )

    def _add_top_K_from_ranking(
        self,
        mask_bool: Optional[np.ndarray],
        ranking: list[int],
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if K < 0:
            raise ValueError("K must be non-negative.")
        if mask_bool is None:
            out = np.zeros(self.num_detectors, dtype=np.bool_)
        else:
            if mask_bool.shape != (self.num_detectors,):
                raise ValueError(
                    f"mask_bool must have shape ({self.num_detectors},); got {mask_bool.shape}."
                )
            out = np.array(mask_bool, dtype=np.bool_, copy=True)
        added = 0
        for d in ranking:
            if added >= K:
                break
            if not out[d]:
                out[d] = True
                added += 1
        packed = np.packbits(out, bitorder="little")
        return out, packed, int(added)

    def _prune_lowest_K_by_score(
        self,
        mask_bool: np.ndarray,
        score_values: np.ndarray,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if mask_bool is None:
            raise ValueError("mask_bool is required for prune (cannot be None).")
        if K < 0:
            raise ValueError("K must be non-negative.")
        if mask_bool.shape != (self.num_detectors,):
            raise ValueError(
                f"mask_bool must have shape ({self.num_detectors},); got {mask_bool.shape}."
            )
        out = np.array(mask_bool, dtype=np.bool_, copy=True)
        in_idx = np.where(out)[0]
        if K == 0 or in_idx.size == 0:
            packed = np.packbits(out, bitorder="little")
            return out, packed, 0
        K_eff = min(int(K), int(in_idx.size))
        in_scores = score_values[in_idx]
        drop_local = np.argsort(in_scores, kind="stable")[:K_eff]
        out[in_idx[drop_local]] = False
        packed = np.packbits(out, bitorder="little")
        return out, packed, int(K_eff)

    def augment_mask_with_essentials(
        self,
        mask_bool: Optional[np.ndarray],
        *,
        K: int,
        type: str = "canonical",
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Add the K best-ranked `type` detectors that are not in mask_bool (an
        empty mask if None). `type` is one of _ESSENTIALS_TYPES and must be
        computed or loaded first.
        Returns (mask_bool, mask_packed, n_added), with n_added <= K.
        """
        return self._add_top_K_from_ranking(
            mask_bool, self._ranking_for_type(type), K,
        )

    def prune_mask_with_essentials(
        self,
        mask_bool: np.ndarray,
        *,
        K: int,
        type: str = "canonical",
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Remove the K detectors of mask_bool with the lowest `type` scores.
        `type` is one of _ESSENTIALS_TYPES and must be computed or loaded first.
        Returns (mask_bool, mask_packed, n_removed), with n_removed <= K.
        """
        return self._prune_lowest_K_by_score(
            mask_bool, self._scores_for_type(type), K,
        )

    def _dem_structure(self):
        """Component structure of self.dem, built on first use."""
        if getattr(self, "_dem_structure_cache", None) is None:
            from dem_structure import DemStructure
            self._dem_structure_cache = DemStructure(self.dem)
        return self._dem_structure_cache

    def closure_mask(
        self,
        mask_bool: np.ndarray,
        *,
        tau: float = 0.0,
        max_iter: Optional[int] = None,
        return_info: bool = False,
    ):
        """Add every hidden detector d with S(M + d) - S(M) < -tau, in batches,
        until a fixed point. Add-only. S is the crossing mass
        (mask_crossing_mass). This fills the holes and ragged edges left by
        scattered additions. Semantics and the info dict: DemStructure.closure.

        Returns (mask_bool, mask_packed, n_added), plus info when
        return_info=True.
        """
        if mask_bool is None or mask_bool.shape != (self.num_detectors,):
            raise ValueError(
                f"mask_bool must have shape ({self.num_detectors},); "
                f"got {None if mask_bool is None else mask_bool.shape}."
            )
        out, info = self._dem_structure().closure(
            np.asarray(mask_bool, dtype=np.bool_), tau=float(tau), max_iter=max_iter, return_info=True,
        )
        packed = np.packbits(out, bitorder="little")
        n_added = int(info["detectors_added"])
        if return_info:
            return out, packed, n_added, info
        return out, packed, n_added

    def mask_crossing_mass(self, mask_bool: np.ndarray) -> float:
        """Crossing mass S(M) = sum of p_e over DEM components e with detectors
        both inside and outside M. A first-order proxy for the straddling
        error mass (see dem_structure.py)."""
        return self._dem_structure().S(np.asarray(mask_bool, dtype=np.bool_))

    def _meta_for_type(self, type: str) -> Optional[dict[str, Any]]:
        if type == "canonical":
            return self.canonical_essentials_meta
        if type == "causal":
            return self.causal_essentials_meta
        if type == "blending":
            return self.blending_essentials_meta
        if type == "meangap":
            return self.meangap_essentials_meta
        if type == "logical_ambiguity":
            return self.logical_ambiguity_essentials_meta
        raise ValueError(
            f"type must be one of {self._ESSENTIALS_TYPES}; got {type!r}."
        )

    def save_essentials_to_json(
        self,
        filepath,
        *,
        type: str,
    ) -> pathlib.Path:
        """Write the order and meta of `type` to a JSON file, creating parent
        directories. Returns the path."""
        if type not in self._ESSENTIALS_TYPES:
            raise ValueError(
                f"type must be one of {self._ESSENTIALS_TYPES}; got {type!r}."
            )
        order_attr = f"{type}_essentials_order"
        order = getattr(self, order_attr, None)
        meta = self._meta_for_type(type)
        if order is None or meta is None:
            raise ValueError(
                f"{type} essentials order is not computed. "
                f"Call compute_{type}_essentials_order(...) first."
            )
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": type,
            "num_detectors": int(self.num_detectors),
            "order": [int(d) for d in order],
            "meta": meta,
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        return path

    def load_essentials_from_json(
        self,
        filepath,
        *,
        type: str,
    ) -> list[int]:
        """Load the order and meta of `type` from a file written by
        save_essentials_to_json. Raises ValueError if the file's type or
        num_detectors does not match. Returns the order."""
        if type not in self._ESSENTIALS_TYPES:
            raise ValueError(
                f"type must be one of {self._ESSENTIALS_TYPES}; got {type!r}."
            )
        path = pathlib.Path(filepath)
        with path.open("r") as f:
            payload = json.load(f)

        file_type = payload.get("type")
        if file_type != type:
            raise ValueError(
                f"file at {path} contains type={file_type!r}; expected type={type!r}."
            )
        file_n = payload.get("num_detectors")
        if file_n != self.num_detectors:
            raise ValueError(
                f"num_detectors mismatch: file has {file_n}, current circuit has "
                f"{self.num_detectors}. Refusing to load — circuit likely differs."
            )

        order_raw = payload.get("order")
        if not isinstance(order_raw, list):
            raise ValueError(f"file at {path} is missing or has invalid 'order'.")
        order = [int(d) for d in order_raw]
        if len(order) != self.num_detectors:
            raise ValueError(
                f"loaded order has length {len(order)}; expected {self.num_detectors}."
            )

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            raise ValueError(f"file at {path} is missing or has invalid 'meta'.")

        if type == "canonical":
            self.canonical_essentials_order = order
            self.canonical_essentials_meta = meta
        elif type == "causal":
            self.causal_essentials_order = order
            self.causal_essentials_meta = meta
        elif type == "blending":
            self.blending_essentials_order = order
            self.blending_essentials_meta = meta
        elif type == "meangap":
            self.meangap_essentials_order = order
            self.meangap_essentials_meta = meta
        else:  # logical_ambiguity
            self.logical_ambiguity_essentials_order = order
            self.logical_ambiguity_essentials_meta = meta

        return list(order)
