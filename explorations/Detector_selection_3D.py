import json
import pathlib
import sys
import tempfile
from typing import Any, Optional

import numpy as np
import sinter
import stim

src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))

import cultiv
import gen


class DetectorSelection3D:
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

        self.selected_mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        self.selected_mask_packed = np.packbits(self.selected_mask_bool, bitorder="little")
        self.selection_metadata: dict[str, Any] = {}

        # Four independent base-mask-independent detector orderings:
        # - canonical: r_reject(d) - r_accept(d), correlational ("sentinel" pickup)
        # - causal:    mean |gap_with_d - gap_without_d| over shots where d fired,
        #              causal/sensitivity ("bridge" pickup)
        # - blending:  alpha-weighted blend of normalized canonical and causal scores
        # - meangap:   max(mean_gap_when_active) - mean_gap_when_active[d], a
        #              correlational/continuous score in the canonical family
        # Populated by compute_<type>_essentials_order(...) and consumed by
        # augment_mask_with_<type>(...) / prune_mask_with_<type>(...) / the type-
        # dispatching augment_mask_with_essentials / prune_mask_with_essentials.
        self.canonical_essentials_order: Optional[list[int]] = None
        self.canonical_essentials_meta: Optional[dict[str, Any]] = None
        self.causal_essentials_order: Optional[list[int]] = None
        self.causal_essentials_meta: Optional[dict[str, Any]] = None
        self.blending_essentials_order: Optional[list[int]] = None
        self.blending_essentials_meta: Optional[dict[str, Any]] = None
        self.meangap_essentials_order: Optional[list[int]] = None
        self.meangap_essentials_meta: Optional[dict[str, Any]] = None

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

    @staticmethod
    def _map_slice_index_to_z_start(slice_index: int, z_levels: np.ndarray) -> int:
        # z-levels and cycle-slices are both monotonic but not in the same unit.
        # Use ordinal mapping (index-based) instead of timeline-map-based absolute ticks.
        if len(z_levels) == 0:
            return 0
        return int(z_levels[min(max(slice_index, 0), len(z_levels) - 1)])

    def _extract_spacetime(self):
        ids = []
        x = []
        y = []
        z = []
        for d in range(self.num_detectors):
            c = self.det_coords.get(d, [])
            if len(c) < 3:
                continue
            ids.append(d)
            x.append(float(c[0]))
            y.append(float(c[1]))
            z.append(int(round(float(c[2]))))
        if not ids:
            raise ValueError("No detector coordinates with at least 3 dimensions found.")
        ids_arr = np.asarray(ids, dtype=np.int64)
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        z_arr = np.asarray(z, dtype=np.int64)
        return ids_arr, x_arr, y_arr, z_arr


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

    # Coord-t-driven enumeration. Returns every detector index whose coord[2] equals
    # the given level, regardless of whether it surfaces through any tile.flag entry.
    # This is the path used by mask builders so they cover transitional detectors
    # (e.g. surface-code stabilizers at the cultivation->escape boundary slice).
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

    # Resolve a list of cycle-slice ticks to the union of distinct coord[2] levels
    # those ticks "represent" (via tile-flag membership). Used to drive coord-t
    # iteration in mask builders.
    def _coord_ts_from_ticks(self, ticks_to_use: list[int], codes: dict) -> list[int]:
        seen: set[int] = set()
        for tk in ticks_to_use:
            for d in self._detector_ids_for_tick(codes[tk]):
                c = self.det_coords.get(int(d), [])
                if len(c) >= 3:
                    seen.add(int(round(float(c[2]))))
        return sorted(seen)

    def build_color_region_mask(
        self,
        *,
        left_len: int,
        top_len: int,
        t: int,
        mode: str = "rectangle",
        stage_name: str = "escape",
    ) -> tuple[np.ndarray, np.ndarray]:
        if left_len <= 0 or top_len <= 0 or t <= 0:
            raise ValueError("left_len, top_len, and t must be positive.")
        if mode not in {"rectangle", "triangle"}:
            raise ValueError("mode must be 'rectangle' or 'triangle'.")

        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        if not ticks:
            raise ValueError("No cycle slices found.")

        sel_ticks = ticks[hybrid_idx:hybrid_idx + int(t)]
        if not sel_ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        # Drive enumeration by coord_t levels rather than by ticks. This guarantees
        # we cover transitional detectors (e.g. surface-code stabilizers at the
        # cultivation->escape boundary) that don't surface through any tile.flag,
        # and keeps the spatial-cut computation per-coord_t so each slice's geometry
        # stays coherent (no mixing cultivation and escape patches in one cut).
        sel_coord_ts = self._coord_ts_from_ticks(sel_ticks, codes)
        if not sel_coord_ts:
            raise ValueError("No detector coord[2] levels resolved from the selected ticks.")

        mask_bool = np.zeros(self.num_detectors, dtype=np.bool_)
        z_selected_all: list[int] = []
        eps = 1e-9

        for coord_t in sel_coord_ts:
            tick_ids = self._detector_ids_for_coord_t(coord_t)
            if len(tick_ids) == 0:
                continue

            x_tick = np.asarray([float(self.det_coords[d][0]) for d in tick_ids], dtype=np.float64)
            y_tick = np.asarray([float(self.det_coords[d][1]) for d in tick_ids], dtype=np.float64)
            z_tick = np.full(len(tick_ids), int(coord_t), dtype=np.int64)
            z_selected_all.extend(z_tick.tolist())

            # Patch-aligned coordinates.
            u_tick = x_tick + y_tick
            v_tick = y_tick - x_tick
            u_levels = np.unique(np.sort(u_tick))
            v_levels = np.unique(np.sort(v_tick))

            # Left-top corner selection orientation.
            # Left edge: v is maximal. Top edge: u is minimal.
            u_edge = float(u_levels[0])
            v_edge = float(v_levels[-1])

            # One-step inside lines used for length counting.
            u_inner = float(u_levels[1]) if len(u_levels) > 1 else u_edge
            v_inner = float(v_levels[-2]) if len(v_levels) > 1 else v_edge

            left_inner_idx = np.flatnonzero(np.abs(v_tick - v_inner) <= eps)
            top_inner_idx = np.flatnonzero(np.abs(u_tick - u_inner) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.flatnonzero(np.abs(v_tick - v_edge) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.flatnonzero(np.abs(u_tick - u_edge) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.asarray([int(np.argmax(v_tick))], dtype=np.int64)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.asarray([int(np.argmin(u_tick))], dtype=np.int64)

            # top_len: count on one-step-inside top row, from left to right.
            top_order = top_inner_idx[np.argsort(-v_tick[top_inner_idx], kind="mergesort")]
            top_k = min(int(top_len), len(top_order))
            v_cut = float(v_tick[top_order[top_k - 1]])

            # left_len: count downward on the column selected by top_len (i + top_len).
            right_col_idx = np.flatnonzero(np.abs(v_tick - v_cut) <= eps)
            if len(right_col_idx) == 0:
                nearest = int(np.argmin(np.abs(v_tick - v_cut)))
                right_col_idx = np.asarray([nearest], dtype=np.int64)
            right_col_order = right_col_idx[np.argsort(u_tick[right_col_idx], kind="mergesort")]
            left_k = min(int(left_len), len(right_col_order))
            u_cut = float(u_tick[right_col_order[left_k - 1]])

            # Saturation: when the user asks for a region at least as large as the patch
            # in both directions (left_len >= |u_levels|, top_len >= |v_levels|), include
            # every detector at this coord_t — gives well-defined "cover everything"
            # semantics that doesn't rely on the original cut algorithm working on
            # irregular geometries (e.g. cultivation->escape boundary slice).
            saturate = int(left_len) >= len(u_levels) and int(top_len) >= len(v_levels)
            if saturate:
                in_rect = np.ones(len(tick_ids), dtype=np.bool_)
            else:
                in_rect = (u_tick <= u_cut + eps) & (v_tick >= v_cut - eps)
            keep = in_rect

            if mode == "triangle" and not saturate:
                du_den = max(u_cut - u_edge, eps)
                dv_den = max(v_edge - v_cut, eps)
                du = (u_tick - u_edge) / du_den
                dv = (v_edge - v_tick) / dv_den
                keep = keep & ((du + dv) <= 1.0 + 1e-9)

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

        self.selected_mask_bool = mask_bool
        self.selected_mask_packed = mask_packed
        self.selection_metadata = {
            "mode": mode,
            "stage_name": stage_name,
            "left_len": int(left_len),
            "top_len": int(top_len),
            "t": int(t),
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
        return mask_bool, mask_packed

    def build_partial_region_apart_color_mask(
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
        # t_start / t_end are 1-indexed half-open, counted from the first surface-code (hybrid)
        # slice — mirroring the left_len_*/top_len_* convention. e.g. t_start=1, t_end=2 selects
        # the first hybrid slice; t_start=1, t_end=N+1 reproduces the old single-t behavior with t=N.
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

        # Iterate per coord_t (not per tick) so transitional detectors at cycle
        # boundaries are reachable and the spatial cuts stay coherent per slice.
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

            # Saturation: when [start, end) for both columns and rows extends past the
            # available levels at this coord_t, include every detector at this coord_t.
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

            # Column index from left-to-right order defined on the top-inner row.
            col_rank = np.argmin(np.abs(v_tick[:, None] - top_v_values[None, :]), axis=1) + 1

            # Row index from top-to-bottom order.
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

    def visualize_partial_region_apart_color_svg(
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
        title_prefix = (
            f"Apart region ({metadata.get('mode', 'rectangle')}, half={metadata.get('triangle_half', 'left')}): "
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

    def visualize_partial_region_apart_color_plot(
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
            self.visualize_partial_region_apart_color_svg(
                tmp_path,
                mask_bool=mask_bool,
                metadata=metadata,
            )
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    def build_partial_region_multi_mask(
        self,
        *,
        left_len: int,
        top_len: int,
        t: int,
        mode: str = "rectangle",
        stage_name: str = "escape",
    ) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
        if left_len <= 0 or top_len <= 0 or t <= 0:
            raise ValueError("left_len, top_len, and t must be positive.")
        if mode not in {"rectangle", "triangle"}:
            raise ValueError("mode must be 'rectangle' or 'triangle'.")

        hybrid_idx, slice_ticks, slice_counts = self._find_first_hybrid_slice_index()
        ticks, codes = self._cycle_slices()
        if not ticks:
            raise ValueError("No cycle slices found.")

        sel_ticks = ticks[hybrid_idx:hybrid_idx + int(t)]
        if not sel_ticks:
            sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        # Region order:
        # 0: color-code region (same as build_color_region_mask)
        # 1: right-half triangle (only when mode='triangle'; else all zero)
        # 2: top-right rectangle (top..left_len, top_len..rightmost)
        # 3: bottom rectangle left half
        # 4: bottom rectangle right half
        masks_bool = [np.zeros(self.num_detectors, dtype=np.bool_) for _ in range(5)]

        z_selected_all: list[int] = []
        eps = 1e-9

        for tick in sel_ticks:
            tick_ids = self._detector_ids_for_tick(codes[tick])
            if len(tick_ids) == 0:
                continue

            x_tick = np.asarray([float(self.det_coords[d][0]) for d in tick_ids], dtype=np.float64)
            y_tick = np.asarray([float(self.det_coords[d][1]) for d in tick_ids], dtype=np.float64)
            z_tick = np.asarray([int(round(float(self.det_coords[d][2]))) for d in tick_ids], dtype=np.int64)
            z_selected_all.extend(z_tick.tolist())

            u_tick = x_tick + y_tick
            v_tick = y_tick - x_tick
            u_levels = np.unique(np.sort(u_tick))
            v_levels = np.unique(np.sort(v_tick))

            u_edge = float(u_levels[0])
            v_edge = float(v_levels[-1])
            u_inner = float(u_levels[1]) if len(u_levels) > 1 else u_edge
            v_inner = float(v_levels[-2]) if len(v_levels) > 1 else v_edge

            left_inner_idx = np.flatnonzero(np.abs(v_tick - v_inner) <= eps)
            top_inner_idx = np.flatnonzero(np.abs(u_tick - u_inner) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.flatnonzero(np.abs(v_tick - v_edge) <= eps)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.flatnonzero(np.abs(u_tick - u_edge) <= eps)
            if len(left_inner_idx) == 0:
                left_inner_idx = np.asarray([int(np.argmax(v_tick))], dtype=np.int64)
            if len(top_inner_idx) == 0:
                top_inner_idx = np.asarray([int(np.argmin(u_tick))], dtype=np.int64)

            top_order = top_inner_idx[np.argsort(-v_tick[top_inner_idx], kind="mergesort")]
            top_k = min(int(top_len), len(top_order))
            v_cut = float(v_tick[top_order[top_k - 1]])

            right_col_idx = np.flatnonzero(np.abs(v_tick - v_cut) <= eps)
            if len(right_col_idx) == 0:
                nearest = int(np.argmin(np.abs(v_tick - v_cut)))
                right_col_idx = np.asarray([nearest], dtype=np.int64)
            right_col_order = right_col_idx[np.argsort(u_tick[right_col_idx], kind="mergesort")]
            left_k = min(int(left_len), len(right_col_order))
            u_cut = float(u_tick[right_col_order[left_k - 1]])

            # Base color/triangle rectangle from build_color_region_mask semantics.
            top_left_rect = (u_tick <= u_cut + eps) & (v_tick >= v_cut - eps)
            # Disjoint top-right rectangle: same top rows, columns strictly right of cut.
            top_right_rect = (u_tick <= u_cut + eps) & (v_tick < v_cut - eps)

            # region 2: top-right rectangle (disjoint from region 0/1)
            masks_bool[2][tick_ids[top_right_rect]] = True

            # region 0 and region 1 from triangle split (or rectangle fallback), inside top-left rect.
            if mode == "triangle":
                du_den = max(u_cut - u_edge, eps)
                dv_den = max(v_edge - v_cut, eps)
                du = (u_tick - u_edge) / du_den
                dv = (v_edge - v_tick) / dv_den
                tri_left = top_left_rect & ((du + dv) <= 1.0 + 1e-9)
                tri_right = top_left_rect & (~tri_left)
                masks_bool[0][tick_ids[tri_left]] = True
                masks_bool[1][tick_ids[tri_right]] = True
            else:
                masks_bool[0][tick_ids[top_left_rect]] = True
                # region 1 stays all zero for rectangle mode.

            # Remaining rectangle: rows below left_len cut, all columns.
            remaining = u_tick > u_cut + eps
            n_cols = len(top_order)
            if n_cols <= 1:
                left_half = remaining
                right_half = np.zeros_like(remaining)
            else:
                # Map each detector to left-to-right column rank based on top-inner row ordering.
                top_v_values = v_tick[top_order]
                col_rank = np.argmin(np.abs(v_tick[:, None] - top_v_values[None, :]), axis=1) + 1
                split_col = (n_cols + 1) // 2
                left_half = remaining & (col_rank <= split_col)
                right_half = remaining & (col_rank > split_col)

            masks_bool[3][tick_ids[left_half]] = True
            masks_bool[4][tick_ids[right_half]] = True

        masks_packed = [np.packbits(mask, bitorder="little") for mask in masks_bool]

        if z_selected_all:
            z_levels_selected = sorted(set(int(e) for e in z_selected_all))
            z0 = int(z_levels_selected[0])
            z1 = int(z_levels_selected[-1])
        else:
            z_levels_selected = []
            z0 = 0
            z1 = 0

        metadata = {
            "mode": mode,
            "stage_name": stage_name,
            "left_len": int(left_len),
            "top_len": int(top_len),
            "t": int(t),
            "hybrid_slice_index": int(hybrid_idx),
            "hybrid_slice_tick": int(slice_ticks[hybrid_idx]) if slice_ticks else 0,
            "slice_ticks": [int(e) for e in slice_ticks],
            "slice_qubit_counts": [int(e) for e in slice_counts],
            "selected_slice_ticks": [int(e) for e in sel_ticks],
            "z_start": int(z0),
            "z_end": int(z1),
            "z_levels_selected": [int(e) for e in z_levels_selected],
            "region_sizes": [int(np.count_nonzero(m)) for m in masks_bool],
            "region_names": [
                "region0_color_or_left_triangle",
                "region1_right_triangle",
                "region2_top_right_rectangle",
                "region3_bottom_left_half",
                "region4_bottom_right_half",
            ],
        }
        return masks_bool, masks_packed, metadata

    def visualize_partial_region_multi_svg(
        self,
        path: pathlib.Path,
        *,
        masks_bool: list[np.ndarray],
        metadata: dict[str, Any],
        canvas_height: int = 900,
    ) -> None:
        if masks_bool is None or len(masks_bool) != 5:
            raise ValueError("masks_bool must be a list of 5 boolean masks.")

        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        sel_ticks = [int(e) for e in metadata.get("selected_slice_ticks", [])]
        if not sel_ticks:
            hybrid_idx = int(metadata.get("hybrid_slice_index", 0))
            t_layers = int(metadata.get("t", 1))
            sel_ticks = ticks[hybrid_idx:hybrid_idx + t_layers]
            if not sel_ticks and ticks:
                sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        panels = [codes[k].with_transformed_coords(lambda e: e * (1 + 1j)) for k in sel_ticks]

        colors = [
            (0.95, 0.10, 0.10),
            (0.98, 0.55, 0.10),
            (1.00, 0.41, 0.71),
            (0.12, 0.70, 0.30),
            (0.62, 0.24, 0.85),
        ]

        def tile_color(tile: gen.Tile):
            dr, = tile.flags
            d = int(dr)
            for i, mask in enumerate(masks_bool):
                if d < len(mask) and mask[d]:
                    return colors[i]
            return 0.82, 0.82, 0.82

        titles = [f"tick={k}" for k in sel_ticks]
        title_prefix = (
            f"Multi-region ({metadata.get('mode', 'rectangle')}): "
            f"left_len={metadata.get('left_len')}, "
            f"top_len={metadata.get('top_len')}, "
            f"t={metadata.get('t')}, "
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

    def visualize_partial_region_multi_plot(
        self,
        *,
        masks_bool: list[np.ndarray],
        metadata: dict[str, Any],
    ):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.visualize_partial_region_multi_svg(
                tmp_path,
                masks_bool=masks_bool,
                metadata=metadata,
            )
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    def visualize_selected_region_svg(
        self,
        path: pathlib.Path,
        *,
        canvas_height: int = 900,
    ) -> None:
        if not np.any(self.selected_mask_bool):
            raise ValueError("No selected region mask found. Call build_color_region_mask(...) first.")

        z0 = int(self.selection_metadata.get("z_start", 0))
        z1 = int(self.selection_metadata.get("z_end", z0))

        codes = gen.circuit_to_cycle_code_slices(self.circuit)
        ticks = sorted(codes.keys())
        sel_ticks = [int(e) for e in self.selection_metadata.get("selected_slice_ticks", [])]
        if not sel_ticks:
            hybrid_idx = int(self.selection_metadata.get("hybrid_slice_index", 0))
            t_layers = int(self.selection_metadata.get("t", 1))
            sel_ticks = ticks[hybrid_idx:hybrid_idx + t_layers]
            if not sel_ticks and ticks:
                sel_ticks = [ticks[min(max(hybrid_idx, 0), len(ticks) - 1)]]

        panels = [codes[k].with_transformed_coords(lambda e: e * (1 + 1j)) for k in sel_ticks]

        def tile_color(tile: gen.Tile):
            dr, = tile.flags
            d = int(dr)
            if d >= len(self.selected_mask_bool):
                return 0.6, 0.6, 0.6
            if self.selected_mask_bool[d]:
                return 0.95, 0.1, 0.1
            return 0.82, 0.82, 0.82

        titles = [f"tick={k}" for k in sel_ticks]
        title_prefix = (
            f"Selected region ({self.selection_metadata['mode']}): "
            f"left_len={self.selection_metadata['left_len']}, "
            f"top_len={self.selection_metadata['top_len']}, "
            f"t={self.selection_metadata['t']}, "
            f"z=[{self.selection_metadata['z_start']}..{self.selection_metadata['z_end']}]"
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

    def visualize_selected_region_plot(self):
        try:
            from IPython.display import SVG
        except ImportError as ex:
            raise ImportError("IPython is required for notebook display.") from ex

        with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
            tmp_path = pathlib.Path(tmp.name)
        try:
            self.visualize_selected_region_svg(tmp_path)
            return SVG(data=tmp_path.read_text())
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Essentials orderings (base-mask-independent): canonical, causal, blending.
    # Augment / prune helpers consume whichever ordering is requested by `type`.
    # ------------------------------------------------------------------

    def compute_canonical_essentials_order(
        self,
        *,
        gap_threshold: float,
        sampler: Optional[Any] = None,
        shots: int = 200_000,
    ) -> list[int]:
        """Compute and store a base-mask-independent ordering of all detectors by
        score(d) = P(d fired | complete decoder rejects) - P(d fired | complete decoder accepts),
        with reject vs accept partitioned at `gap_threshold`. The ordering is intrinsic
        to (circuit, sampler, gap_threshold) and is reusable across any base mask.

        Stores the ordering on self.canonical_essentials_order and returns it.
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
        # Guard against degenerate splits.
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

    def compute_causal_essentials_order(
        self,
        *,
        sampler: Optional[Any] = None,
        shots: int = 1_000_000,
        subsample_size: Optional[int] = None,
        rng_seed: int = 0,
    ) -> list[int]:
        """Compute and store a base-mask-independent ordering of all detectors by
        causal sensitivity:
            score(d) = mean over (kept) shots where d fired of |gap_orig - gap_with_d_unfired|
        where the baseline gap and the d-flipped gap come from the complete (full-DEM)
        decoder. High score = d firing carries strong gap-disambiguation signal.

        For each detector d the implementation batches all shots that fired d into
        one decode call (after clearing d's bit), so the total work is roughly
        num_detectors batched decodes.

        If `subsample_size` is None, scoring runs on all kept shots (slower but
        lower-noise). If it is an int, that many kept shots are sampled at random
        before scoring.

        Stores the ordering on self.causal_essentials_order and meta on
        self.causal_essentials_meta (with `score_values`, `score_signed`,
        `n_samples_per_det`).
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
            self.causal_essentials_order = list(range(self.num_detectors))
            self.causal_essentials_meta = {
                "shots": int(shots),
                "kept_shots": 0,
                "subsample_size": 0,
                "score_values": [0.0] * self.num_detectors,
                "score_signed": [0.0] * self.num_detectors,
                "n_samples_per_det": [0] * self.num_detectors,
                "warning": "no shots survived postselect; ordering is degenerate",
            }
            return list(self.causal_essentials_order)

        if subsample_size is None or subsample_size >= n_kept:
            sub_dets = dets
            sub_n = n_kept
        else:
            rng = np.random.default_rng(rng_seed)
            sub_idx = rng.choice(n_kept, size=int(subsample_size), replace=False)
            sub_dets = dets[sub_idx].copy()
            sub_n = int(subsample_size)

        # Baseline (complete-decoder) gaps and unpacked syndromes for fast lookup.
        _, sub_gaps_orig = dec._decode_batch_overwrite_last_byte(sub_dets.copy())
        sub_gaps_orig = sub_gaps_orig.astype(np.float64)
        sub_unpacked = (
            np.unpackbits(sub_dets, axis=1, bitorder="little")[:, : self.num_detectors]
            .astype(np.bool_)
        )

        sum_abs = np.zeros(self.num_detectors, dtype=np.float64)
        sum_signed = np.zeros(self.num_detectors, dtype=np.float64)
        count = np.zeros(self.num_detectors, dtype=np.int64)

        for d in range(self.num_detectors):
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

        ranking = np.argsort(-causal_abs).tolist()
        self.causal_essentials_order = [int(d) for d in ranking]
        self.causal_essentials_meta = {
            "shots": int(shots),
            "kept_shots": n_kept,
            "subsample_size": sub_n,
            "score_values": causal_abs.tolist(),
            "score_signed": causal_signed.tolist(),
            "n_samples_per_det": count.tolist(),
        }
        return list(self.causal_essentials_order)

    def compute_blending_essentials_order(
        self,
        *,
        alpha: float,
    ) -> list[int]:
        """Combine the canonical and causal score arrays into a single ranking via
            blended[d] = (1 - alpha) * canon_n[d] + alpha * causal_n[d]
        where canon_n and causal_n are min-max normalized to [0, 1]. alpha=0
        recovers canonical order, alpha=1 recovers causal order.

        Requires compute_canonical_essentials_order(...) and
        compute_causal_essentials_order(...) to have been called first. No sampling
        or decoding happens here.

        Stores the ordering on self.blending_essentials_order and meta on
        self.blending_essentials_meta.
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
        """Compute and store a base-mask-independent ordering of all detectors by
            score(d) = max_active(mean_gap_when_active) - mean_gap_when_active[d]
        where `mean_gap_when_active[d]` is the average complete-decoder gap over
        kept shots in which detector d fired (zero for detectors that never fire).
        High score means d preferentially fires on hard (low-gap) shots — the
        same correlational family as canonical, but expressed continuously rather
        than via a binary partition at gap_threshold. Maps "darker in the
        overall_detector_gap_sens plot" to "higher score".

        Stores the ordering on self.meangap_essentials_order and meta on
        self.meangap_essentials_meta (with `score_values`, `mean_gap_when_active`,
        `count`).
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

        # Score: high = informative (= low mean_gap_when_active among active dets,
        # i.e. darker tile in the plot). Detectors with count == 0 are pinned at
        # score 0 so they never rank above any active detector.
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

    # ---- internal helpers shared by augment_* and prune_* ----

    _ESSENTIALS_TYPES = ("canonical", "causal", "blending", "meangap")

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

    # ---- public augment functions: one per type, plus a dispatcher ----

    def augment_mask_with_canonical(
        self,
        mask_bool: Optional[np.ndarray],
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Add the top-K canonical-list detectors not already in `mask_bool`.

        If `mask_bool` is None the augmentation starts from an empty mask, so the
        result is exactly the top-K canonical detectors.

        Returns (new_mask_bool, new_mask_packed, n_added). n_added <= K.

        Requires compute_canonical_essentials_order(...) to have been called first.
        """
        return self._add_top_K_from_ranking(
            mask_bool, self._ranking_for_type("canonical"), K,
        )

    def augment_mask_with_causal(
        self,
        mask_bool: Optional[np.ndarray],
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Add the top-K causal-list detectors not already in `mask_bool`.

        If `mask_bool` is None the augmentation starts from an empty mask.
        Returns (new_mask_bool, new_mask_packed, n_added).

        Requires compute_causal_essentials_order(...) to have been called first.
        """
        return self._add_top_K_from_ranking(
            mask_bool, self._ranking_for_type("causal"), K,
        )

    def augment_mask_with_blending(
        self,
        mask_bool: Optional[np.ndarray],
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Add the top-K blending-list detectors not already in `mask_bool`.

        If `mask_bool` is None the augmentation starts from an empty mask.
        Returns (new_mask_bool, new_mask_packed, n_added).

        Requires compute_blending_essentials_order(alpha=...) to have been called first.
        """
        return self._add_top_K_from_ranking(
            mask_bool, self._ranking_for_type("blending"), K,
        )

    def augment_mask_with_meangap(
        self,
        mask_bool: Optional[np.ndarray],
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Add the top-K meangap-list detectors not already in `mask_bool` —
        i.e. the detectors that fire on the lowest-mean-gap (darkest in the
        overall_detector_gap_sens plot) shots.

        If `mask_bool` is None the augmentation starts from an empty mask.
        Returns (new_mask_bool, new_mask_packed, n_added).

        Requires compute_meangap_essentials_order(...) to have been called first.
        """
        return self._add_top_K_from_ranking(
            mask_bool, self._ranking_for_type("meangap"), K,
        )

    def augment_mask_with_essentials(
        self,
        mask_bool: Optional[np.ndarray],
        *,
        K: int,
        type: str = "canonical",
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Dispatcher across the four augment_mask_with_<type> variants.

        type ∈ {"canonical", "causal", "blending", "meangap"}; default
        "canonical" for backward-compat with prior calls. If `mask_bool` is None
        the augmentation starts from an empty mask (= pure top-K from the chosen
        ranking).
        """
        return self._add_top_K_from_ranking(
            mask_bool, self._ranking_for_type(type), K,
        )

    # ---- public prune functions: one per type, plus a dispatcher ----

    def prune_mask_with_canonical(
        self,
        mask_bool: np.ndarray,
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Remove the K detectors currently in `mask_bool` with the smallest
        canonical scores. Returns (new_mask_bool, new_mask_packed, n_removed).
        """
        return self._prune_lowest_K_by_score(
            mask_bool, self._scores_for_type("canonical"), K,
        )

    def prune_mask_with_causal(
        self,
        mask_bool: np.ndarray,
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Remove the K detectors currently in `mask_bool` with the smallest
        causal scores. Returns (new_mask_bool, new_mask_packed, n_removed).
        """
        return self._prune_lowest_K_by_score(
            mask_bool, self._scores_for_type("causal"), K,
        )

    def prune_mask_with_blending(
        self,
        mask_bool: np.ndarray,
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Remove the K detectors currently in `mask_bool` with the smallest
        blending scores. Returns (new_mask_bool, new_mask_packed, n_removed).
        """
        return self._prune_lowest_K_by_score(
            mask_bool, self._scores_for_type("blending"), K,
        )

    def prune_mask_with_meangap(
        self,
        mask_bool: np.ndarray,
        *,
        K: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Remove the K detectors currently in `mask_bool` with the smallest
        meangap scores — i.e. the brightest tiles in the
        overall_detector_gap_sens plot among those currently included.
        Returns (new_mask_bool, new_mask_packed, n_removed).
        """
        return self._prune_lowest_K_by_score(
            mask_bool, self._scores_for_type("meangap"), K,
        )

    def prune_mask_with_essentials(
        self,
        mask_bool: np.ndarray,
        *,
        K: int,
        type: str = "canonical",
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Dispatcher across the four prune_mask_with_<type> variants.

        type ∈ {"canonical", "causal", "blending", "meangap"}; default
        "canonical". Unlike augment, `mask_bool` cannot be None — there is
        nothing to prune from an empty mask.
        """
        if mask_bool is None:
            raise ValueError("mask_bool is required for prune_mask_with_essentials.")
        return self._prune_lowest_K_by_score(
            mask_bool, self._scores_for_type(type), K,
        )

    # ---- persistence: save / load an essentials list to JSON ----

    def _meta_for_type(self, type: str) -> Optional[dict[str, Any]]:
        if type == "canonical":
            return self.canonical_essentials_meta
        if type == "causal":
            return self.causal_essentials_meta
        if type == "blending":
            return self.blending_essentials_meta
        if type == "meangap":
            return self.meangap_essentials_meta
        raise ValueError(
            f"type must be one of {self._ESSENTIALS_TYPES}; got {type!r}."
        )

    def essential_list_to_json(
        self,
        filepath,
        *,
        type: str,
    ) -> pathlib.Path:
        """Serialize the (order, meta) for the requested essentials `type` to a
        JSON file. Use to cache an expensive `compute_<type>_essentials_order`
        result so it can be reloaded across kernel restarts.

        Requires the corresponding compute_<type>_essentials_order(...) to have
        been called first. Returns the resolved Path written to.
        """
        order = self._ranking_for_type(type)   # raises if not computed yet
        meta = self._meta_for_type(type)
        path = pathlib.Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "DetectorSelection3D.essentials/v1",
            "type": type,
            "num_detectors": int(self.num_detectors),
            "order": [int(d) for d in order],
            "meta": meta,
        }
        with path.open("w") as f:
            json.dump(payload, f, indent=2)
        return path

    def from_json_to_essential_list(
        self,
        filepath,
        *,
        type: str,
    ) -> list[int]:
        """Load (order, meta) for the requested essentials `type` from a JSON
        file produced by essential_list_to_json(...). Populates
        self.<type>_essentials_order and self.<type>_essentials_meta on this
        DetectorSelection3D instance.

        Validates that the file's `type` matches the requested one and that the
        stored `num_detectors` matches the current circuit's. Returns the loaded
        order.
        """
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
        else:  # meangap
            self.meangap_essentials_order = order
            self.meangap_essentials_meta = meta
        return list(order)
