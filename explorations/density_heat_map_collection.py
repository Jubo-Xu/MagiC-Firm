import collections
import csv
import dataclasses
import io
import json
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, Optional, Literal

import matplotlib.pyplot as plt
import numpy as np
import stim


src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))
import cultiv
import gen


@dataclasses.dataclass(frozen=True)
class Hyperedge:
    detectors: tuple[int, ...]
    obs_mask: int
    min_t: int
    max_t: int

    def __str__(self) -> str:
        ds = "|".join(str(d) for d in self.detectors)
        return f"{ds}@{self.min_t}:{self.max_t}#obs={self.obs_mask}"

    @staticmethod
    def from_targets(
        targets: list[stim.DemTarget],
        *,
        detector_to_time: dict[int, int],
    ) -> "Hyperedge | None":
        obs_mask = 0
        dets: list[int] = []
        for t in targets:
            if t.is_logical_observable_id():
                obs_mask ^= 1 << t.val
            else:
                assert t.is_relative_detector_id()
                dets.append(t.val)
        if not dets:
            return None
        dets = sorted(set(dets))
        times = [detector_to_time.get(d, 0) for d in dets]
        return Hyperedge(
            detectors=tuple(dets),
            obs_mask=obs_mask,
            min_t=min(times),
            max_t=max(times),
        )


def targets_to_hyperedges(
    targets: list[stim.DemTarget],
    *,
    detector_to_time: dict[int, int],
) -> list[Hyperedge]:
    out: list[Hyperedge] = []
    start = 0
    while start < len(targets):
        end = start + 1
        while end < len(targets) and not targets[end].is_separator():
            end += 1
        he = Hyperedge.from_targets(targets[start:end], detector_to_time=detector_to_time)
        if he is not None:
            out.append(he)
        start = end + 1
    return out


def make_error_to_hyperedges_list(
    dem: stim.DetectorErrorModel,
    *,
    detector_to_time: dict[int, int],
) -> list[list[Hyperedge]]:
    out: list[list[Hyperedge]] = []
    for inst in dem.flattened():
        if inst.type == "error":
            out.append(targets_to_hyperedges(inst.targets_copy(), detector_to_time=detector_to_time))
    return out


class DensityHeatMapCollection:
    CSV_COLUMNS = [
        "apply_postselection",
        "count_mode",
        "decompose_errors",
        "shots",
        "accepted_shots",
        "postselected_shots",
        "postselection_rate",
        "seconds",
        "stage_timeline_map_raw",
        "stage_timeline_map",
        "visualization_stage_timeline_map",
        "stage_postselection_rate",
        "detector_counts",
        "hyperedge_counts",
        "detector_counts_raw",
        "hyperedge_counts_raw",
        "detector_counts_accepted",
        "hyperedge_counts_accepted",
        "hyperedge_metadata",
    ]

    def __init__(
        self,
        *,
        circuit: Optional[stim.Circuit] = None,
        circuit_generator: Optional[Any] = None,
        stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
        apply_postselection: bool = True,
        count_mode: Literal["accepted_only", "all_shots"] = "accepted_only",
        decompose_errors: bool = False,
    ):
        if circuit_generator is not None:
            if getattr(circuit_generator, "ideal_circuit", None) is None:
                raise ValueError("circuit_generator.ideal_circuit is None. Call generate() first.")
            circuit = getattr(circuit_generator, "noisy_circuit", None) or circuit_generator.ideal_circuit
            if stage_timeline_map is None:
                stage_timeline_map = dict(getattr(circuit_generator.params, "StageTimelineMap", {}) or {})
            self._generator_params = getattr(circuit_generator, "params", None)
        else:
            self._generator_params = None
        if circuit is None:
            raise ValueError("Provide either circuit or circuit_generator.")
        if count_mode not in ("accepted_only", "all_shots"):
            raise ValueError(f"Unsupported count_mode={count_mode!r}. Use 'accepted_only' or 'all_shots'.")

        self.circuit = circuit
        self.apply_postselection = bool(apply_postselection)
        self.count_mode: Literal["accepted_only", "all_shots"] = count_mode
        self.decompose_errors = bool(decompose_errors)
        self.stage_timeline_map_raw: dict[str, tuple[int, int]] = {
            k: (int(v[0]), int(v[1])) for k, v in dict(stage_timeline_map or {}).items()
        }
        self.visualization_stage_timeline_map: dict[str, tuple[int, int]] = {
            k: (int(v[0]), int(v[1])) for k, v in dict(visualization_stage_timeline_map or {}).items()
        }

        # Decompose optionally to get more edge-like terms.
        self.dem = self.circuit.detector_error_model(
            decompose_errors=self.decompose_errors,
            ignore_decomposition_failures=self.decompose_errors,
        )
        self.sampler = self.dem.compile_sampler()

        self.detector_coords = self.circuit.get_detector_coordinates()
        self.detector_to_time: dict[int, int] = {
            d: int(round(cs[2])) if len(cs) > 2 else 0
            for d, cs in self.detector_coords.items()
        }
        self.stage_timeline_map: dict[str, tuple[int, int]] = self._normalize_stage_timeline_map_to_detector_time(
            self.stage_timeline_map_raw
        )
        self.num_dets = self.dem.num_detectors
        self.error_hyperedges = make_error_to_hyperedges_list(
            self.dem,
            detector_to_time=self.detector_to_time,
        )

        self.hyperedge_metadata: dict[str, dict[str, Any]] = {}
        for hs in self.error_hyperedges:
            for h in hs:
                key = str(h)
                if key not in self.hyperedge_metadata:
                    self.hyperedge_metadata[key] = {
                        "detectors": list(h.detectors),
                        "min_t": h.min_t,
                        "max_t": h.max_t,
                        "obs_mask": h.obs_mask,
                    }

        self.postselect_mask = self._build_postselect_mask() if self.apply_postselection else None
        self.stage_postselect_masks = self._build_stage_postselect_masks()

        self.shots = 0
        self.accepted_shots = 0
        self.postselected_shots = 0
        self.seconds = 0.0
        self.detector_counts_raw: collections.Counter[str] = collections.Counter()
        self.hyperedge_counts_raw: collections.Counter[str] = collections.Counter()
        self.detector_counts_accepted: collections.Counter[str] = collections.Counter()
        self.hyperedge_counts_accepted: collections.Counter[str] = collections.Counter()
        self.detector_counts: collections.Counter[str] = collections.Counter()
        self.hyperedge_counts: collections.Counter[str] = collections.Counter()
        self.stage_postselected_shots: collections.Counter[str] = collections.Counter()
        self._refresh_public_counters()

    def _normalize_stage_timeline_map_to_detector_time(
        self,
        stage_map: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]]:
        if not stage_map:
            return {}
        det_times = list(self.detector_to_time.values())
        if not det_times:
            return dict(stage_map)
        det_min = int(min(det_times))
        det_max = int(max(det_times))
        s_min = min(v[0] for v in stage_map.values())
        s_max = max(v[1] for v in stage_map.values())
        if s_max <= det_max and s_min >= det_min:
            return {k: (int(v[0]), int(v[1])) for k, v in stage_map.items()}

        # Prefer semantic mapping for end2end circuits to avoid distorted linear normalization.
        p = getattr(self, "_generator_params", None)
        if (
            (p is not None and getattr(p, "circuit_type", None) == "end2end-inplace-distillation")
            or {"injection", "cultivation", "escape"}.issubset(set(stage_map.keys()))
        ):
            d1 = getattr(p, "d1", None)
            basis = getattr(p, "basis", None)
            inject_style = getattr(p, "injection_protocol", None) or "unitary"
            r2 = int(getattr(p, "r_post_escape", 0) or 0)
            if d1 is not None and basis in ("X", "Y", "Z"):
                try:
                    inj_c = cultiv.make_inject_and_cultivate_circuit(
                        dcolor=int(d1),
                        inject_style=inject_style,
                        basis=basis if basis in ("X", "Y") else "Y",
                    )
                    inj_c_rounds = int(gen.count_measurement_layers(inj_c))
                except Exception:
                    inj_c_rounds = max(1, (det_max - det_min + 1) // 3)

                post_start = max(det_min, det_max - r2 + 1) if r2 > 0 else det_max + 1
                stage_marker_ts = sorted({
                    int(round(cs[2]))
                    for cs in self.detector_coords.values()
                    if len(cs) > 4 and cs[3] == -1 and cs[4] == -9
                })
                if stage_marker_ts:
                    # Stage markers correspond to the start of the next phase in this pipeline.
                    inj_end = min(post_start - 1, max(stage_marker_ts) - 1)
                else:
                    inj_end = min(post_start - 1, det_min + inj_c_rounds - 1)
                mid_start = inj_end + 1
                mid_end = post_start - 1

                out: dict[str, tuple[int, int]] = {}

                def split_segment(names: list[str], a: int, b: int) -> None:
                    if not names or a > b:
                        return
                    total = b - a + 1
                    weights = []
                    for n in names:
                        ra, rb = stage_map.get(n, (0, 0))
                        weights.append(max(1, rb - ra + 1))
                    s = sum(weights)
                    spans = [int(total * w / s) for w in weights]
                    rem = total - sum(spans)
                    for i in range(rem):
                        spans[i % len(spans)] += 1
                    cur = a
                    for n, ln in zip(names, spans):
                        if ln <= 0:
                            continue
                        out[n] = (cur, cur + ln - 1)
                        cur += ln

                split_segment([n for n in ("injection", "cultivation") if n in stage_map], det_min, inj_end)
                split_segment([n for n in ("code-grow", "escape") if n in stage_map], mid_start, mid_end)
                if "code-grow" in out and "escape" in out:
                    a0, a1 = out.pop("code-grow")
                    b0, b1 = out["escape"]
                    out["escape"] = (min(a0, b0), max(a1, b1))
                elif "code-grow" in out and "escape" not in out:
                    out["escape"] = out.pop("code-grow")
                if "post-escape" in stage_map and post_start <= det_max:
                    out["post-escape"] = (post_start, det_max)

                if out:
                    return out
            else:
                # Fallback semantic split when generator params aren't available.
                post_len_raw = max(1, stage_map.get("post-escape", (s_max, s_max))[1] - stage_map.get("post-escape", (s_max, s_max))[0] + 1) if "post-escape" in stage_map else 1
                total_raw = max(1, s_max - s_min + 1)
                det_span = det_max - det_min + 1
                post_len = max(1, int(round(det_span * post_len_raw / total_raw))) if "post-escape" in stage_map else 0
                post_start = det_max - post_len + 1 if post_len > 0 else det_max + 1
                stage_marker_ts = sorted({
                    int(round(cs[2]))
                    for cs in self.detector_coords.values()
                    if len(cs) > 4 and cs[3] == -1 and cs[4] == -9
                })
                inj_end = min(post_start - 1, max(stage_marker_ts) - 1) if stage_marker_ts else det_min + max(1, (det_span // 4)) - 1
                mid_start = inj_end + 1
                mid_end = post_start - 1

                out: dict[str, tuple[int, int]] = {}

                def split_segment_fallback(names: list[str], a: int, b: int) -> None:
                    if not names or a > b:
                        return
                    total = b - a + 1
                    weights = [max(1, stage_map[n][1] - stage_map[n][0] + 1) for n in names]
                    s = sum(weights)
                    spans = [int(total * w / s) for w in weights]
                    rem = total - sum(spans)
                    for i in range(rem):
                        spans[i % len(spans)] += 1
                    cur = a
                    for n, ln in zip(names, spans):
                        out[n] = (cur, cur + ln - 1)
                        cur += ln

                split_segment_fallback([n for n in ("injection", "cultivation") if n in stage_map], det_min, inj_end)
                if "escape" in stage_map and mid_start <= mid_end:
                    out["escape"] = (mid_start, mid_end)
                if "post-escape" in stage_map and post_start <= det_max:
                    out["post-escape"] = (post_start, det_max)
                if out:
                    return out

        if s_max == s_min:
            return {k: (det_min, det_max) for k in stage_map}

        items = sorted(stage_map.items(), key=lambda kv: kv[1][0])
        out: dict[str, tuple[int, int]] = {}
        prev_end = det_min - 1
        for i, (stage, (a, b)) in enumerate(items):
            na = round((a - s_min) / (s_max - s_min) * (det_max - det_min) + det_min)
            nb = round((b - s_min) / (s_max - s_min) * (det_max - det_min) + det_min)
            na = max(det_min, min(det_max, int(na)))
            nb = max(det_min, min(det_max, int(nb)))
            if i == 0:
                na = det_min
            else:
                na = max(na, prev_end + 1)
            nb = max(nb, na)
            if i == len(items) - 1:
                nb = det_max
            out[stage] = (na, nb)
            prev_end = nb
        return out

    def _refresh_public_counters(self) -> None:
        if self.count_mode == "accepted_only":
            self.detector_counts = self.detector_counts_accepted.copy()
            self.hyperedge_counts = self.hyperedge_counts_accepted.copy()
        else:
            self.detector_counts = self.detector_counts_raw.copy()
            self.hyperedge_counts = self.hyperedge_counts_raw.copy()

    def _build_postselect_mask(self) -> np.ndarray | None:
        mask = np.zeros(shape=self.num_dets // 8 + 1, dtype=np.uint8)
        found = False
        for d, cs in self.detector_coords.items():
            # Match src/cultiv/_stats_util.py.
            if len(cs) < 4 or (len(cs) > 0 and cs[-1] == -9):
                mask[d >> 3] |= 1 << (d & 7)
                found = True
        return mask if found else None

    def _build_stage_postselect_masks(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        if self.postselect_mask is None:
            return out
        for stage, (t0, t1) in self.stage_timeline_map.items():
            m = np.zeros(shape=self.num_dets // 8 + 1, dtype=np.uint8)
            for d in range(self.num_dets):
                if not (self.postselect_mask[d >> 3] & (1 << (d & 7))):
                    continue
                td = self.detector_to_time.get(d, 0)
                if t0 <= td <= t1:
                    m[d >> 3] |= 1 << (d & 7)
            out[stage] = m
        return out

    def collect(self, *, shots: int, batch_size: int = 4096) -> None:
        if shots <= 0:
            return
        t0 = time.monotonic()
        remaining = shots
        while remaining > 0:
            cur = min(batch_size, remaining)
            dets, _, error_data = self.sampler.sample(
                shots=cur,
                bit_packed=True,
                return_errors=True,
            )
            self.shots += cur
            remaining -= cur
            for k in range(cur):
                udets = np.unpackbits(dets[k], bitorder="little", count=self.num_dets)
                active_errors = np.flatnonzero(np.unpackbits(error_data[k], bitorder="little"))

                # Always record raw counts (all sampled shots).
                for d in np.flatnonzero(udets):
                    self.detector_counts_raw[str(int(d))] += 1
                for err_idx in active_errors:
                    if err_idx >= len(self.error_hyperedges):
                        continue
                    for he in self.error_hyperedges[err_idx]:
                        self.hyperedge_counts_raw[str(he)] += 1

                discarded = self.postselect_mask is not None and np.any(dets[k] & self.postselect_mask)
                if discarded:
                    self.postselected_shots += 1
                    for stage, sm in self.stage_postselect_masks.items():
                        if np.any(dets[k] & sm):
                            self.stage_postselected_shots[stage] += 1
                    continue

                self.accepted_shots += 1
                for d in np.flatnonzero(udets):
                    self.detector_counts_accepted[str(int(d))] += 1

                for err_idx in active_errors:
                    if err_idx >= len(self.error_hyperedges):
                        continue
                    for he in self.error_hyperedges[err_idx]:
                        self.hyperedge_counts_accepted[str(he)] += 1
        self.seconds += time.monotonic() - t0
        self._refresh_public_counters()

    def postselection_rate(self) -> float:
        return self.postselected_shots / self.shots if self.shots else 0.0

    def stage_postselection_rate(self) -> dict[str, float]:
        if self.shots == 0:
            return {k: 0.0 for k in self.stage_timeline_map}
        return {
            stage: self.stage_postselected_shots.get(stage, 0) / self.shots
            for stage in self.stage_timeline_map
        }

    def _ensure_geometry(self) -> None:
        if not self.detector_coords:
            raise ValueError("Detector geometry is unavailable. Initialize with a circuit/circuit_generator, not only from CSV.")

    def _stage_map_for_visualization(
        self,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> dict[str, tuple[int, int]]:
        if visualization_stage_timeline_map:
            return {
                k: (int(v[0]), int(v[1]))
                for k, v in visualization_stage_timeline_map.items()
            }
        if self.visualization_stage_timeline_map:
            return self.visualization_stage_timeline_map
        return self.stage_timeline_map

    @staticmethod
    def _project_coord(coord: Iterable[float]) -> tuple[float, float]:
        c = list(coord)
        x = c[0] if len(c) > 0 else 0.0
        y = c[1] if len(c) > 1 else 0.0
        z = c[2] if len(c) > 2 else 0.0
        # Timeline-oriented oblique projection, tuned for a wider/shorter layout.
        return (x - 0.30 * y) * 24, z * 22 + y * 3.5

    @staticmethod
    def _figure_to_svg_text(fig: Any) -> str:
        buff = io.StringIO()
        fig.savefig(buff, format="svg", bbox_inches="tight")
        return buff.getvalue()

    def _detector_hits_int(self) -> dict[int, int]:
        return {int(k): int(v) for k, v in self.detector_counts.items()}

    def _active_hyperedge_items(self) -> list[tuple[str, int, dict[str, Any]]]:
        out: list[tuple[str, int, dict[str, Any]]] = []
        for k, v in self.hyperedge_counts.items():
            if v <= 0:
                continue
            meta = self.hyperedge_metadata.get(k)
            if meta is None:
                continue
            out.append((k, int(v), meta))
        return out

    def _all_hyperedge_items(self) -> list[tuple[str, int, dict[str, Any]]]:
        out: list[tuple[str, int, dict[str, Any]]] = []
        for k, meta in self.hyperedge_metadata.items():
            out.append((k, int(self.hyperedge_counts.get(k, 0)), meta))
        return out

    @staticmethod
    def _rgb_grad(p: float) -> str:
        p = max(0.0, min(1.0, p))
        r, g, b, _ = plt.get_cmap("cividis")(1 - p, bytes=True)
        return f"#{r:02x}{g:02x}{b:02x}"

    @staticmethod
    def _svg_escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _draw_svg_histogram(
        self,
        *,
        out_lines: list[str],
        values: list[int],
        base_x: float,
        base_y: float,
        total_width: float,
        total_height: float,
        title: str,
        x_label: str,
        y_label: str,
        color: str,
    ) -> None:
        if not values:
            out_lines.append(
                f'<text x="{base_x + total_width/2}" y="{base_y - total_height/2}" '
                f'fill="#555" font-size="6" text-anchor="middle" dominant-baseline="middle">No data</text>'
            )
            return
        max_hit = max(values)
        if max_hit > 100:
            num_per_bucket = (max_hit + 49) // 50
            num_buckets = 50
        else:
            num_per_bucket = 1
            num_buckets = max(max_hit + 1, 2)
        max_hit = num_buckets * num_per_bucket
        bucket_width = total_width / num_buckets
        bucket_hits = [0] * num_buckets
        for h in values:
            idx = min(num_buckets - 1, int(h // num_per_bucket))
            bucket_hits[idx] += 1
        max_bucket = max(bucket_hits, default=1)
        log_top = max(10, max_bucket)
        log_top = 10 ** int(np.ceil(np.log10(log_top)))
        for k in range(num_buckets):
            val = bucket_hits[k]
            if val <= 0:
                h_px = 0
            else:
                h_px = total_height * (np.log(val) / np.log(log_top))
            x = base_x + k * bucket_width + bucket_width / 2
            out_lines.append(
                f'<path d="M {x} {base_y - h_px} L {x} {base_y}" '
                f'fill="none" stroke="{color}" stroke-width="{bucket_width}" />'
            )
        out_lines.append(
            f'<path d="M {base_x} {base_y} L {base_x + total_width} {base_y} '
            f'L {base_x + total_width} {base_y - total_height} L {base_x} {base_y - total_height} Z" '
            f'fill="none" stroke="black" stroke-width="0.5" />'
        )
        out_lines.append(f'<text x="{base_x}" y="{base_y + 2}" fill="black" font-size="6" text-anchor="middle">0</text>')
        out_lines.append(f'<text x="{base_x + total_width}" y="{base_y + 2}" fill="black" font-size="6" text-anchor="middle">{max_hit}</text>')
        out_lines.append(f'<text x="{base_x + total_width/2}" y="{base_y + 7}" fill="black" font-size="6" text-anchor="middle">{self._svg_escape(x_label)}</text>')
        out_lines.append(f'<text x="{base_x + total_width/2}" y="{base_y - total_height - 4}" fill="black" font-size="6" text-anchor="middle">{self._svg_escape(title)}</text>')
        out_lines.append(f'<text x="{base_x - 2}" y="{base_y - total_height/2}" fill="black" font-size="6" text-anchor="end">{self._svg_escape(y_label)}</text>')

    def _draw_svg_hyperedge_histogram_stacked(
        self,
        *,
        out_lines: list[str],
        base_x: float,
        base_y: float,
        total_width: float,
        total_height: float,
    ) -> None:
        active = self._active_hyperedge_items()
        if not active:
            out_lines.append(
                f'<text x="{base_x + total_width/2}" y="{base_y - total_height/2}" '
                f'fill="#555" font-size="6" text-anchor="middle" dominant-baseline="middle">No data</text>'
            )
            return
        max_hit = max(v for _, v, _ in active)
        if max_hit > 100:
            num_per_bucket = (max_hit + 39) // 40
            num_buckets = 40
        else:
            num_per_bucket = 1
            num_buckets = max(max_hit + 1, 2)
        bucket_width = total_width / num_buckets
        spatial = [0] * num_buckets
        cross = [0] * num_buckets
        for _, v, meta in active:
            idx = min(num_buckets - 1, int(v // num_per_bucket))
            if int(meta["min_t"]) < int(meta["max_t"]):
                cross[idx] += 1
            else:
                spatial[idx] += 1
        totals = [a + b for a, b in zip(spatial, cross)]
        max_bucket = max(totals, default=1)
        log_top = max(10, max_bucket)
        log_top = 10 ** int(np.ceil(np.log10(log_top)))
        for k in range(num_buckets):
            x = base_x + k * bucket_width + bucket_width / 2
            s = spatial[k]
            c = cross[k]
            t = s + c
            if t <= 0:
                continue
            h_total = total_height * (np.log(t) / np.log(log_top))
            frac_s = s / t if t else 0
            h_s = h_total * frac_s
            out_lines.append(
                f'<path d="M {x} {base_y - h_s} L {x} {base_y}" '
                f'fill="none" stroke="#2a9d8f" stroke-width="{bucket_width}" />'
            )
            out_lines.append(
                f'<path d="M {x} {base_y - h_total} L {x} {base_y - h_s}" '
                f'fill="none" stroke="#d1495b" stroke-width="{bucket_width}" />'
            )
        out_lines.append(
            f'<path d="M {base_x} {base_y} L {base_x + total_width} {base_y} '
            f'L {base_x + total_width} {base_y - total_height} L {base_x} {base_y - total_height} Z" '
            f'fill="none" stroke="black" stroke-width="0.5" />'
        )
        out_lines.append(f'<text x="{base_x + total_width/2}" y="{base_y - total_height - 4}" fill="black" font-size="6" text-anchor="middle">Hyperedge Hit Histogram (stacked)</text>')
        out_lines.append(f'<text x="{base_x + total_width/2}" y="{base_y + 7}" fill="black" font-size="6" text-anchor="middle">hit-count bucket</text>')
        out_lines.append(f'<text x="{base_x - 2}" y="{base_y - total_height/2}" fill="black" font-size="6" text-anchor="end"># hyperedges</text>')
        out_lines.append(f'<circle cx="{base_x + 6}" cy="{base_y - total_height - 12}" r="1.8" fill="#2a9d8f" />')
        out_lines.append(f'<text x="{base_x + 10}" y="{base_y - total_height - 12}" fill="#2a9d8f" font-size="5" dominant-baseline="middle">pure spatial</text>')
        out_lines.append(f'<circle cx="{base_x + 58}" cy="{base_y - total_height - 12}" r="1.8" fill="#d1495b" />')
        out_lines.append(f'<text x="{base_x + 62}" y="{base_y - total_height - 12}" fill="#d1495b" font-size="5" dominant-baseline="middle">time-spatial-cross</text>')

    def _build_heatmap_svg(
        self,
        *,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> str:
        self._ensure_geometry()
        det_hits = self._detector_hits_int()
        all_hyperedges = self._all_hyperedge_items()
        active_hyperedges = [e for e in all_hyperedges if e[1] > 0]
        det_points = {d: self._project_coord(c) for d, c in self.detector_coords.items()}
        if not det_points:
            return '<svg xmlns="http://www.w3.org/2000/svg"><text x="10" y="20">No detector geometry.</text></svg>'

        graph_min_x = min(p[0] for p in det_points.values())
        graph_max_x = max(p[0] for p in det_points.values())
        graph_min_y = min(p[1] for p in det_points.values())
        graph_max_y = max(p[1] for p in det_points.values())
        center_x = (graph_min_x + graph_max_x) / 2
        center_y = (graph_min_y + graph_max_y) / 2

        min_x = graph_min_x - 40
        min_y = graph_min_y - 40
        max_x = graph_max_x + 40
        max_y = graph_max_y + 40

        svg: list[str] = [f'<svg viewBox="{min_x} {min_y} {max_x - min_x} {max_y - min_y}" xmlns="http://www.w3.org/2000/svg">']

        stage_items = sorted(
            self._stage_map_for_visualization(visualization_stage_timeline_map).items(),
            key=lambda kv: kv[1][0],
        )
        stage_colors = ["#ffe066", "#8ecae6", "#ffadad", "#b8f2a8", "#cdb4db", "#f4a261"]
        for i, (stage, (t0, t1)) in enumerate(stage_items):
            y0 = self._project_coord([0, 0, t0 - 0.5])[1]
            y1 = self._project_coord([0, 0, t1 + 0.5])[1]
            top = min(y0, y1)
            h = max(1, abs(y1 - y0))
            svg.append(
                f'<rect x="{graph_min_x - 20}" y="{top}" width="{graph_max_x - graph_min_x + 40}" height="{h}" '
                f'fill="{stage_colors[i % len(stage_colors)]}" fill-opacity="0.38" stroke="none" />'
            )

        z_values = sorted({int(round(cs[2])) if len(cs) > 2 else 0 for cs in self.detector_coords.values()})
        guide_z = z_values if len(z_values) <= 12 else sorted(set(int(round(v)) for v in np.linspace(z_values[0], z_values[-1], 12)))
        for z in guide_z:
            y = self._project_coord([0, 0, z])[1]
            svg.append(
                f'<path d="M {graph_min_x - 20} {y} L {graph_max_x + 20} {y}" '
                f'fill="none" stroke="#7a7a7a" stroke-width="0.35" stroke-dasharray="1.8,1.8" stroke-opacity="0.6" />'
            )
            svg.append(
                f'<text x="{graph_min_x - 22}" y="{y}" fill="#3f3f3f" font-size="5" text-anchor="end" dominant-baseline="middle">t={z}</text>'
            )

        max_hyper = max((v for _, v, _ in all_hyperedges), default=0)
        for _, h, meta in all_hyperedges:
            dets = [int(d) for d in meta["detectors"]]
            pts = [det_points[d] for d in dets if d in det_points]
            if not pts:
                continue
            is_cross = int(meta["min_t"]) < int(meta["max_t"])
            t = np.log1p(h) / np.log1p(max_hyper) if (max_hyper and h > 0) else 0.0
            if h > 0:
                lw = 0.45 + 2.7 * t
                alpha = 0.35 + 0.55 * t
            else:
                lw = 0.25
                alpha = 0.08
            color = "#d1495b" if is_cross else "#2a9d8f"
            if len(pts) == 1:
                x, y = pts[0]
                svg.append(
                    f'<rect x="{x - 1.0}" y="{y - 1.0}" width="{2.0 + 2*t}" height="{2.0 + 2*t}" '
                    f'fill="{color}" fill-opacity="{alpha}" stroke="none" />'
                )
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            for px, py in pts:
                svg.append(
                    f'<path d="M {cx} {cy} L {px} {py}" fill="none" stroke="{color}" '
                    f'stroke-width="{lw}" stroke-opacity="{alpha}" />'
                )

        max_det = max(det_hits.values(), default=0)
        for d, p in det_points.items():
            h = det_hits.get(d, 0)
            t = np.log1p(h) / np.log1p(max_det) if max_det else 0.0
            c = 0.2 + 0.8 * t
            r = 0.6 + 2.8 * t
            svg.append(
                f'<circle cx="{p[0]}" cy="{p[1]}" r="{r}" fill="{self._rgb_grad(c)}" fill-opacity="0.92" stroke="none" />'
            )

        title_x = (min_x + max_x) / 2
        title_y = min_y + 4
        stats_y = title_y + 11
        subtitle_y = stats_y + 8
        svg.append(f'<text x="{title_x}" y="{title_y}" fill="black" font-size="10" font-weight="bold" text-anchor="middle" dominant-baseline="hanging">Timeline-Enhanced Non-trivial Detector/Hyperedge Heat Map</text>')
        svg.append(f'<text x="{title_x}" y="{stats_y}" fill="black" font-size="7" text-anchor="middle" dominant-baseline="hanging">sampled shots={self.shots}, accepted={self.accepted_shots}, postselected={self.postselected_shots}, postselection_rate={self.postselection_rate():.4f}</text>')
        svg.append(f'<text x="{title_x}" y="{subtitle_y}" fill="#444" font-size="6" text-anchor="middle" dominant-baseline="hanging">all DEM hyperedges shown: inactive=faint, active=highlighted by width/opacity</text>')

        # Stage mapping legend block (kept separate from plot body to avoid overlap).
        legend_x = graph_max_x - 96
        legend_y = graph_min_y + 22
        svg.append(
            f'<rect x="{legend_x - 3}" y="{legend_y - 3}" width="100" height="{max(10, 8 * len(stage_items) + 8)}" '
            f'fill="white" fill-opacity="0.90" stroke="#777" stroke-width="0.5" />'
        )
        for i, (stage, (t0, t1)) in enumerate(stage_items):
            y = legend_y + i * 8
            color = stage_colors[i % len(stage_colors)]
            svg.append(f'<rect x="{legend_x}" y="{y - 2.2}" width="4.4" height="4.4" fill="{color}" stroke="#666" stroke-width="0.35" />')
            svg.append(f'<text x="{legend_x + 6}" y="{y}" fill="#222" font-size="6" dominant-baseline="middle">{self._svg_escape(stage)}: t={t0}..{t1}</text>')
        svg.append(f'<circle cx="{graph_min_x + 4}" cy="{graph_min_y - 8}" r="1.8" fill="#2a9d8f" />')
        svg.append(f'<text x="{graph_min_x + 8}" y="{graph_min_y - 8}" fill="#2a9d8f" font-size="6.5" font-weight="bold" dominant-baseline="middle">pure spatial hyperedge</text>')
        svg.append(f'<circle cx="{graph_min_x + 78}" cy="{graph_min_y - 8}" r="1.8" fill="#d1495b" />')
        svg.append(f'<text x="{graph_min_x + 82}" y="{graph_min_y - 8}" fill="#d1495b" font-size="6.5" font-weight="bold" dominant-baseline="middle">time-spatial-cross hyperedge</text>')
        svg.append("</svg>")
        return "\n".join(svg)

    def _apply_timeline_stage_overlay(
        self,
        ax: Any,
        xs: list[float],
        *,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> None:
        if not xs:
            return
        xmin = min(xs) - 12
        xmax = max(xs) + 12

        viz_stage_map = self._stage_map_for_visualization(visualization_stage_timeline_map)
        if viz_stage_map:
            stage_items = sorted(viz_stage_map.items(), key=lambda kv: kv[1][0])
            stage_colors = ["#ffe066", "#8ecae6", "#ffadad", "#b8f2a8", "#cdb4db", "#f4a261"]
            for i, (stage, (t0, t1)) in enumerate(stage_items):
                y0 = self._project_coord([0, 0, t0 - 0.5])[1]
                y1 = self._project_coord([0, 0, t1 + 0.5])[1]
                top = min(y0, y1)
                bottom = max(y0, y1)
                ax.axhspan(top, bottom, color=stage_colors[i % len(stage_colors)], alpha=0.25, zorder=0)
                ax.text(
                    xmin + 1.5,
                    top + 2.0,
                    f"{stage}: t={t0}..{t1}",
                    fontsize=8,
                    color="#4a4a4a",
                    va="bottom",
                    ha="left",
                )

        z_values = sorted({int(round(cs[2])) if len(cs) > 2 else 0 for cs in self.detector_coords.values()})
        if len(z_values) > 12:
            z_values = sorted(set(int(round(v)) for v in np.linspace(z_values[0], z_values[-1], 12)))
        for z in z_values:
            y = self._project_coord([0, 0, z])[1]
            ax.hlines(y, xmin=xmin, xmax=xmax, linestyles="dashed", linewidth=0.5, color="#7a7a7a", alpha=0.5, zorder=0)
            ax.text(xmin, y, f"t={z}", fontsize=7, color="#4a4a4a", va="center", ha="right")

    def _build_heatmap_figure(
        self,
        *,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> Any:
        self._ensure_geometry()
        fig, ax = plt.subplots(figsize=(12, 8))

        det_hits = self._detector_hits_int()
        det_points: dict[int, tuple[float, float]] = {
            d: self._project_coord(cs)
            for d, cs in self.detector_coords.items()
        }
        xs = [p[0] for p in det_points.values()]
        ys = [p[1] for p in det_points.values()]
        self._apply_timeline_stage_overlay(
            ax,
            xs,
            visualization_stage_timeline_map=visualization_stage_timeline_map,
        )

        max_det_hit = max(det_hits.values(), default=0)
        scat_x: list[float] = []
        scat_y: list[float] = []
        scat_c: list[float] = []
        scat_s: list[float] = []
        for d, h in det_hits.items():
            p = det_points.get(d)
            if p is None:
                continue
            t = np.log1p(h) / np.log1p(max_det_hit) if max_det_hit else 0.0
            scat_x.append(p[0])
            scat_y.append(p[1])
            scat_c.append(h)
            scat_s.append(12 + 80 * t)
        if scat_x:
            sc = ax.scatter(scat_x, scat_y, c=scat_c, s=scat_s, cmap="Blues", alpha=0.9, edgecolors="none", zorder=3)
            cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02)
            cbar.set_label("Non-trivial detector hit count", fontsize=9)

        active_hyperedges = self._active_hyperedge_items()
        max_hyper_hit = max((v for _, v, _ in active_hyperedges), default=0)
        spatial_label_done = False
        cross_label_done = False
        for _, h, meta in active_hyperedges:
            dets = [int(d) for d in meta["detectors"]]
            pts = [det_points[d] for d in dets if d in det_points]
            if not pts:
                continue
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            is_cross = int(meta["min_t"]) < int(meta["max_t"])
            t = np.log1p(h) / np.log1p(max_hyper_hit) if max_hyper_hit else 0.0
            lw = 0.5 + 2.5 * t
            alpha = 0.3 + 0.6 * t
            color = "#d1495b" if is_cross else "#2a9d8f"
            label = None
            if is_cross and not cross_label_done:
                label = "time-spatial-cross hyperedge"
                cross_label_done = True
            if (not is_cross) and not spatial_label_done:
                label = "pure spatial hyperedge"
                spatial_label_done = True
            if len(pts) == 1:
                ax.scatter([pts[0][0]], [pts[0][1]], s=15 + 45 * t, marker="s", color=color, alpha=alpha, zorder=4, label=label)
            else:
                for px, py in pts:
                    ax.plot([cx, px], [cy, py], color=color, linewidth=lw, alpha=alpha, zorder=2, label=label)
                    label = None

        ax.set_title("Hyperedge Density Heat Map with Timeline/Stage Overlay", fontsize=12)
        ax.set_xlabel("Projected x", fontsize=10)
        ax.set_ylabel("Projected timeline axis", fontsize=10)
        ax.grid(False)
        if xs and ys:
            ax.set_xlim(min(xs) - 20, max(xs) + 20)
            ax.set_ylim(min(ys) - 20, max(ys) + 20)
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        return fig

    def _build_detector_histogram_figure(self) -> Any:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        values = [int(v) for v in self.detector_counts.values() if int(v) > 0]
        if not values:
            ax.text(0.5, 0.5, "No non-trivial detector events", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.tight_layout()
            return fig
        max_v = max(values)
        if max_v <= 80:
            bins = np.arange(0, max_v + 2) - 0.5
        else:
            bins = 40
        ax.hist(values, bins=bins, color="#3f88c5", alpha=0.9, edgecolor="white")
        ax.set_yscale("log")
        ax.set_xlabel("Detector hit count")
        ax.set_ylabel("Number of detectors")
        ax.set_title("Histogram of Non-trivial Detector Hits")
        fig.tight_layout()
        return fig

    def _build_hyperedge_histogram_figure(self) -> Any:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        active_hyperedges = self._active_hyperedge_items()
        if not active_hyperedges:
            ax.text(0.5, 0.5, "No active hyperedge events", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.tight_layout()
            return fig

        max_hit = max(v for _, v, _ in active_hyperedges)
        if max_hit <= 80:
            num_per_bucket = 1
            num_buckets = max(max_hit + 1, 2)
        else:
            num_buckets = 40
            num_per_bucket = (max_hit + num_buckets - 1) // num_buckets

        spatial_bins = np.zeros(num_buckets, dtype=int)
        cross_bins = np.zeros(num_buckets, dtype=int)
        for _, h, meta in active_hyperedges:
            idx = min(num_buckets - 1, int(h // num_per_bucket))
            is_cross = int(meta["min_t"]) < int(meta["max_t"])
            if is_cross:
                cross_bins[idx] += 1
            else:
                spatial_bins[idx] += 1

        x = np.arange(num_buckets)
        ax.bar(x, spatial_bins, color="#2a9d8f", alpha=0.9, label="pure spatial hyperedge")
        ax.bar(x, cross_bins, bottom=spatial_bins, color="#d1495b", alpha=0.9, label="time-spatial-cross hyperedge")
        ax.set_yscale("log")
        ax.set_xlabel(f"Hyperedge hit-count bucket (width={num_per_bucket})")
        ax.set_ylabel("Number of hyperedges")
        ax.set_title("Histogram of Hyperedge Errors (Spatial vs Time-Spatial-Cross)")
        if num_buckets > 1:
            ax.set_xticks([0, num_buckets // 2, num_buckets - 1])
            ax.set_xticklabels([
                "0",
                str((num_buckets // 2) * num_per_bucket),
                str((num_buckets - 1) * num_per_bucket),
            ])
        ax.legend(fontsize=9)
        fig.tight_layout()
        return fig

    def heatmap_to_svg(
        self,
        svg_path: str | pathlib.Path | None = None,
        *,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> str:
        svg_text = self._build_heatmap_svg(
            visualization_stage_timeline_map=visualization_stage_timeline_map
        )
        if svg_path is not None:
            pathlib.Path(svg_path).write_text(svg_text)
        return svg_text

    def heatmap_for_notebook(
        self,
        *,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> Any:
        svg_text = self._build_heatmap_svg(
            visualization_stage_timeline_map=visualization_stage_timeline_map
        )
        try:
            from IPython.display import SVG  # type: ignore
            return SVG(data=svg_text)
        except Exception:
            return svg_text

    def detector_histogram_to_svg(self, svg_path: str | pathlib.Path | None = None) -> str:
        fig = self._build_detector_histogram_figure()
        svg_text = self._figure_to_svg_text(fig)
        plt.close(fig)
        if svg_path is not None:
            pathlib.Path(svg_path).write_text(svg_text)
        return svg_text

    def detector_histogram_for_notebook(self) -> Any:
        return self._build_detector_histogram_figure()

    def hyperedge_histogram_to_svg(self, svg_path: str | pathlib.Path | None = None) -> str:
        fig = self._build_hyperedge_histogram_figure()
        svg_text = self._figure_to_svg_text(fig)
        plt.close(fig)
        if svg_path is not None:
            pathlib.Path(svg_path).write_text(svg_text)
        return svg_text

    def hyperedge_histogram_for_notebook(self) -> Any:
        return self._build_hyperedge_histogram_figure()

    def to_CSV(self, csv_path: str | pathlib.Path, *, append: bool = True) -> None:
        csv_path = pathlib.Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append and csv_path.exists() else "w"
        with open(csv_path, mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            if mode == "w":
                w.writeheader()
            w.writerow({
                "apply_postselection": int(self.apply_postselection),
                "count_mode": self.count_mode,
                "decompose_errors": int(self.decompose_errors),
                "shots": self.shots,
                "accepted_shots": self.accepted_shots,
                "postselected_shots": self.postselected_shots,
                "postselection_rate": self.postselection_rate(),
                "seconds": self.seconds,
                "stage_timeline_map_raw": json.dumps(self.stage_timeline_map_raw, sort_keys=True),
                "stage_timeline_map": json.dumps(self.stage_timeline_map, sort_keys=True),
                "visualization_stage_timeline_map": json.dumps(self.visualization_stage_timeline_map, sort_keys=True),
                "stage_postselection_rate": json.dumps(self.stage_postselection_rate(), sort_keys=True),
                "detector_counts": json.dumps(dict(self.detector_counts), sort_keys=True),
                "hyperedge_counts": json.dumps(dict(self.hyperedge_counts), sort_keys=True),
                "detector_counts_raw": json.dumps(dict(self.detector_counts_raw), sort_keys=True),
                "hyperedge_counts_raw": json.dumps(dict(self.hyperedge_counts_raw), sort_keys=True),
                "detector_counts_accepted": json.dumps(dict(self.detector_counts_accepted), sort_keys=True),
                "hyperedge_counts_accepted": json.dumps(dict(self.hyperedge_counts_accepted), sort_keys=True),
                "hyperedge_metadata": json.dumps(self.hyperedge_metadata, sort_keys=True),
            })

    @staticmethod
    def from_CSV(
        csv_path: str | pathlib.Path,
        *,
        row_index: int = -1,
    ) -> "DensityHeatMapCollection":
        csv_path = pathlib.Path(csv_path)
        csv.field_size_limit(sys.maxsize)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No rows found in {csv_path}")
        row = rows[row_index]

        obj = DensityHeatMapCollection.__new__(DensityHeatMapCollection)
        obj.circuit = None
        obj.dem = None
        obj.sampler = None
        obj.detector_coords = {}
        obj.detector_to_time = {}
        obj.num_dets = 0
        obj.error_hyperedges = []
        obj.postselect_mask = None
        obj.stage_postselect_masks = {}
        obj.apply_postselection = bool(int(row.get("apply_postselection", "1") or 1))
        obj.count_mode = row.get("count_mode", "accepted_only")
        obj.decompose_errors = bool(int(row.get("decompose_errors", "0") or 0))

        obj.shots = int(row["shots"])
        obj.accepted_shots = int(row["accepted_shots"])
        obj.postselected_shots = int(row["postselected_shots"])
        obj.seconds = float(row["seconds"])
        if "stage_timeline_map_raw" in row and row["stage_timeline_map_raw"]:
            obj.stage_timeline_map_raw = {
                k: tuple(v) for k, v in json.loads(row["stage_timeline_map_raw"]).items()
            }
        else:
            obj.stage_timeline_map_raw = {
                k: tuple(v) for k, v in json.loads(row["stage_timeline_map"]).items()
            }
        obj.stage_timeline_map = {
            k: tuple(v) for k, v in json.loads(row["stage_timeline_map"]).items()
        }
        obj.visualization_stage_timeline_map = {
            k: tuple(v) for k, v in json.loads(row.get("visualization_stage_timeline_map", "{}")).items()
        }

        obj.detector_counts_raw = collections.Counter(json.loads(row.get("detector_counts_raw", "{}")))
        obj.hyperedge_counts_raw = collections.Counter(json.loads(row.get("hyperedge_counts_raw", "{}")))
        obj.detector_counts_accepted = collections.Counter(json.loads(row.get("detector_counts_accepted", "{}")))
        obj.hyperedge_counts_accepted = collections.Counter(json.loads(row.get("hyperedge_counts_accepted", "{}")))
        obj.detector_counts = collections.Counter(json.loads(row.get("detector_counts", "{}")))
        obj.hyperedge_counts = collections.Counter(json.loads(row.get("hyperedge_counts", "{}")))
        obj.hyperedge_metadata = json.loads(row["hyperedge_metadata"])
        stage_rates = json.loads(row["stage_postselection_rate"])
        obj.stage_postselected_shots = collections.Counter({
            k: int(round(float(v) * obj.shots))
            for k, v in stage_rates.items()
        })
        if not obj.detector_counts and obj.count_mode == "accepted_only":
            obj.detector_counts = obj.detector_counts_accepted.copy()
        if not obj.detector_counts and obj.count_mode == "all_shots":
            obj.detector_counts = obj.detector_counts_raw.copy()
        if not obj.hyperedge_counts and obj.count_mode == "accepted_only":
            obj.hyperedge_counts = obj.hyperedge_counts_accepted.copy()
        if not obj.hyperedge_counts and obj.count_mode == "all_shots":
            obj.hyperedge_counts = obj.hyperedge_counts_raw.copy()
        return obj
