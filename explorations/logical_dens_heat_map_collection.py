import collections
import csv
import dataclasses
import io
import json
import pathlib
import sys
import time
from typing import Any, Dict, Iterable, Optional, Literal, AbstractSet, Set

import matplotlib.pyplot as plt
import numpy as np
import pymatching
import stim


src_path = pathlib.Path(__file__).parent.parent / "magic_state_cultivation" / "upstream" / "src"
assert src_path.exists()
sys.path.append(str(src_path))
import cultiv
import gen


@dataclasses.dataclass(frozen=True)
class Edge:
    a: int
    b: Optional[int]
    obs_mask: int
    min_t: int
    max_t: int

    def __post_init__(self) -> None:
        if self.b is not None:
            assert self.a < self.b

    def __lt__(self, other: "Edge") -> bool:
        return (self.a, -1 if self.b is None else self.b) < (other.a, -1 if other.b is None else other.b)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.a == other.a and self.b == other.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))

    def __str__(self) -> str:
        if self.b is None:
            return f"{self.a}:"
        return f"{self.a}:{self.b}"

    @staticmethod
    def from_targets(
        targets: list[stim.DemTarget],
        *,
        detector_to_time: dict[int, int],
    ) -> "Edge | None":
        obs_mask = 0
        pair: list[int] = []
        for t in targets:
            if t.is_logical_observable_id():
                obs_mask ^= 1 << t.val
            else:
                assert t.is_relative_detector_id()
                pair.append(t.val)
        if not (1 <= len(pair) <= 2):
            return None
        if len(pair) == 1:
            a = pair[0]
            b = None
            min_t = detector_to_time.get(a, 0)
            max_t = min_t
        else:
            pair = sorted(pair)
            a, b = pair
            ta = detector_to_time.get(a, 0)
            tb = detector_to_time.get(b, 0)
            min_t = min(ta, tb)
            max_t = max(ta, tb)
        return Edge(a=a, b=b, obs_mask=obs_mask, min_t=min_t, max_t=max_t)


def _targets_to_edge_no_separator(
    targets: list[stim.DemTarget],
    *,
    detector_to_time: dict[int, int],
) -> Edge | None:
    return Edge.from_targets(targets, detector_to_time=detector_to_time)


def targets_to_edges(
    targets: list[stim.DemTarget],
    *,
    detector_to_time: dict[int, int],
) -> list[Edge]:
    edges: list[Edge] = []
    start = 0
    while start < len(targets):
        end = start + 1
        while end < len(targets) and not targets[end].is_separator():
            end += 1
        edge = _targets_to_edge_no_separator(
            targets[start:end],
            detector_to_time=detector_to_time,
        )
        if edge is not None:
            edges.append(edge)
        start = end + 1
    return edges


def make_error_to_edges_list(
    dem: stim.DetectorErrorModel,
    *,
    detector_to_time: dict[int, int],
) -> list[list[Edge]]:
    out: list[list[Edge]] = []
    for inst in dem.flattened():
        if inst.type == "error":
            out.append(targets_to_edges(inst.targets_copy(), detector_to_time=detector_to_time))
    return out


def edge_set_without_vacuous_components(
    relevant_edges: AbstractSet[Edge],
    *,
    node_to_edges: Dict[int, Iterable[Edge]],
) -> Set[Edge]:
    involved: list[Edge] = []
    seen: set[Edge] = set()
    for root_edge in relevant_edges:
        if not root_edge.obs_mask or root_edge in seen:
            continue
        mask = 0
        stack = [root_edge]
        component: list[Edge] = []
        while stack:
            e = stack.pop()
            if e in seen or e not in relevant_edges:
                continue
            seen.add(e)
            component.append(e)
            mask ^= e.obs_mask
            stack.extend(node_to_edges.get(e.a, ()))
            if e.b is not None:
                stack.extend(node_to_edges.get(e.b, ()))
        if mask:
            involved.extend(component)
    return set(involved)


class LogicalDensityHeatMapCollection:
    CSV_COLUMNS = [
        "apply_postselection",
        "count_mode",
        "decompose_errors",
        "shots",
        "accepted_shots",
        "postselected_shots",
        "logical_error_shots",
        "logical_error_shots_raw",
        "logical_error_shots_accepted",
        "postselection_rate",
        "logical_error_rate",
        "seconds",
        "stage_timeline_map_raw",
        "stage_timeline_map",
        "visualization_stage_timeline_map",
        "stage_postselection_rate",
        "detector_counts",
        "edge_counts",
        "detector_counts_raw",
        "edge_counts_raw",
        "detector_counts_accepted",
        "edge_counts_accepted",
        "edge_metadata",
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
        decompose_errors: bool = True,
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

        self.dem = self.circuit.detector_error_model(
            decompose_errors=self.decompose_errors,
            ignore_decomposition_failures=self.decompose_errors,
        )
        self.matching = pymatching.Matching.from_detector_error_model(self.dem)
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
        self.num_obs = self.dem.num_observables

        self.error_edge_list: list[list[Edge]] = make_error_to_edges_list(
            self.dem,
            detector_to_time=self.detector_to_time,
        )
        self.e2i: dict[tuple[int, int], int] = {}
        self.i2es: dict[int, list[Edge]] = {}
        self.node_to_edges: Dict[int, Set[Edge]] = {}
        for k, es in enumerate(self.error_edge_list):
            if len(es) == 1:
                (e,) = es
                b = -1 if e.b is None else e.b
                self.e2i[(e.a, b)] = k
                self.e2i[(b, e.a)] = k
            self.i2es[k] = es
        for es in self.error_edge_list:
            for e in es:
                self.node_to_edges.setdefault(e.a, set()).add(e)
                if e.b is not None:
                    self.node_to_edges.setdefault(e.b, set()).add(e)

        self.edge_metadata: dict[str, dict[str, Any]] = {}
        for es in self.error_edge_list:
            for e in es:
                key = str(e)
                if key not in self.edge_metadata:
                    self.edge_metadata[key] = {
                        "a": int(e.a),
                        "b": None if e.b is None else int(e.b),
                        "min_t": int(e.min_t),
                        "max_t": int(e.max_t),
                        "obs_mask": int(e.obs_mask),
                    }

        self.postselect_mask = self._build_postselect_mask() if self.apply_postselection else None
        self.stage_postselect_masks = self._build_stage_postselect_masks()

        self.shots = 0
        self.accepted_shots = 0
        self.postselected_shots = 0
        self.logical_error_shots_raw = 0
        self.logical_error_shots_accepted = 0
        self.logical_error_shots = 0
        self.seconds = 0.0
        self.detector_counts_raw: collections.Counter[str] = collections.Counter()
        self.edge_counts_raw: collections.Counter[str] = collections.Counter()
        self.detector_counts_accepted: collections.Counter[str] = collections.Counter()
        self.edge_counts_accepted: collections.Counter[str] = collections.Counter()
        self.detector_counts: collections.Counter[str] = collections.Counter()
        self.edge_counts: collections.Counter[str] = collections.Counter()
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
                    inj_end = min(post_start - 1, max(stage_marker_ts) - 1)
                else:
                    inj_end = min(post_start - 1, det_min + inj_c_rounds - 1)
                inj_end = max(det_min, inj_end)
                cult_start = min(inj_end, det_min + max(0, inj_c_rounds - 1))
                cult_end = inj_end
                esc_start = inj_end + 1
                esc_end = post_start - 1

                out: dict[str, tuple[int, int]] = {}
                out["injection"] = (det_min, max(det_min, cult_start - 1))
                out["cultivation"] = (cult_start, cult_end)
                out["escape"] = (esc_start, max(esc_start, esc_end))
                if post_start <= det_max:
                    out["post-escape"] = (post_start, det_max)
                return {
                    k: (max(det_min, a), min(det_max, b))
                    for k, (a, b) in out.items()
                    if a <= b
                }

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
            self.edge_counts = self.edge_counts_accepted.copy()
            self.logical_error_shots = self.logical_error_shots_accepted
        else:
            self.detector_counts = self.detector_counts_raw.copy()
            self.edge_counts = self.edge_counts_raw.copy()
            self.logical_error_shots = self.logical_error_shots_raw

    def _build_postselect_mask(self) -> np.ndarray | None:
        mask = np.zeros(shape=self.num_dets // 8 + 1, dtype=np.uint8)
        found = False
        for d, cs in self.detector_coords.items():
            # Postselection identification rule matching _stats_util.py.
            if len(cs) < 4 or ((len(cs) > 0 and cs[-1] == -9)):
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

    def collect(self, *, shots: int, batch_size: int = 2048) -> None:
        if shots <= 0:
            return
        t0 = time.monotonic()

        def process_shot(k: int, dets: np.ndarray, obs: np.ndarray, error_data: np.ndarray) -> None:
            discarded = self.postselect_mask is not None and np.any(dets[k] & self.postselect_mask)
            if discarded:
                self.postselected_shots += 1
                for stage, sm in self.stage_postselect_masks.items():
                    if np.any(dets[k] & sm):
                        self.stage_postselected_shots[stage] += 1
            else:
                self.accepted_shots += 1

            udets = np.unpackbits(dets[k], bitorder="little", count=self.num_dets)
            prediction = self.matching.decode_batch(
                dets[k:k + 1],
                bit_packed_predictions=True,
                bit_packed_shots=True,
            )
            if np.array_equal(prediction[0], obs[k]):
                return

            predicted_edges = self.matching.decode_to_edges_array(udets)
            err_bits = np.unpackbits(error_data[k], bitorder="little")
            for a, b in predicted_edges:
                k2 = self.e2i.get((int(a), int(b)))
                if k2 is None:
                    continue
                if k2 < len(err_bits):
                    err_bits[k2] ^= 1

            all_differences: set[Edge] = set()
            for err_idx in np.flatnonzero(err_bits):
                for edge in self.i2es.get(int(err_idx), []):
                    if edge in all_differences:
                        all_differences.remove(edge)
                    else:
                        all_differences.add(edge)
            relevant = edge_set_without_vacuous_components(
                all_differences,
                node_to_edges=self.node_to_edges,
            )

            self.logical_error_shots_raw += 1
            for d in np.flatnonzero(udets):
                self.detector_counts_raw[str(int(d))] += 1
            for e in relevant:
                self.edge_counts_raw[str(e)] += 1

            if discarded:
                return

            self.logical_error_shots_accepted += 1
            for d in np.flatnonzero(udets):
                self.detector_counts_accepted[str(int(d))] += 1
            for e in relevant:
                self.edge_counts_accepted[str(e)] += 1

        if not self.apply_postselection:
            dets, obs, error_data = self.sampler.sample(
                shots=shots,
                bit_packed=True,
                return_errors=True,
            )
            self.shots += shots
            for k in range(shots):
                process_shot(k, dets, obs, error_data)
        else:
            # With postselection enabled, interpret `shots` as accepted-shot target.
            target_accepted = self.accepted_shots + shots
            while self.accepted_shots < target_accepted:
                cur = min(batch_size, max(1, target_accepted - self.accepted_shots))
                dets, obs, error_data = self.sampler.sample(
                    shots=cur,
                    bit_packed=True,
                    return_errors=True,
                )
                self.shots += cur
                for k in range(cur):
                    process_shot(k, dets, obs, error_data)
                    if self.accepted_shots >= target_accepted:
                        break

        self.seconds += time.monotonic() - t0
        self._refresh_public_counters()

    def postselection_rate(self) -> float:
        return self.postselected_shots / self.shots if self.shots else 0.0

    def logical_error_rate(self) -> float:
        denom = self.accepted_shots if self.count_mode == "accepted_only" else self.shots
        return self.logical_error_shots / denom if denom else 0.0

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
        return (x - 0.30 * y) * 24, z * 22 + y * 3.5

    @staticmethod
    def _figure_to_svg_text(fig: Any) -> str:
        buff = io.StringIO()
        fig.savefig(buff, format="svg", bbox_inches="tight")
        return buff.getvalue()

    def _detector_hits_int(self) -> dict[int, int]:
        return {int(k): int(v) for k, v in self.detector_counts.items()}

    def _active_edge_items(self) -> list[tuple[str, int, dict[str, Any]]]:
        out: list[tuple[str, int, dict[str, Any]]] = []
        for k, v in self.edge_counts.items():
            if v <= 0:
                continue
            meta = self.edge_metadata.get(k)
            if meta is None:
                continue
            out.append((k, int(v), meta))
        return out

    def _all_edge_items(self) -> list[tuple[str, int, dict[str, Any]]]:
        out: list[tuple[str, int, dict[str, Any]]] = []
        for k, meta in self.edge_metadata.items():
            out.append((k, int(self.edge_counts.get(k, 0)), meta))
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

    def _draw_svg_edge_histogram_stacked(
        self,
        *,
        out_lines: list[str],
        base_x: float,
        base_y: float,
        total_width: float,
        total_height: float,
    ) -> None:
        active = self._active_edge_items()
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
        out_lines.append(f'<text x="{base_x + total_width/2}" y="{base_y - total_height - 4}" fill="black" font-size="6" text-anchor="middle">Edge Hit Histogram (stacked)</text>')
        out_lines.append(f'<text x="{base_x + total_width/2}" y="{base_y + 7}" fill="black" font-size="6" text-anchor="middle">hit-count bucket</text>')
        out_lines.append(f'<text x="{base_x - 2}" y="{base_y - total_height/2}" fill="black" font-size="6" text-anchor="end"># edges</text>')
        out_lines.append(f'<circle cx="{base_x + 6}" cy="{base_y - total_height - 12}" r="1.8" fill="#2a9d8f" />')
        out_lines.append(f'<text x="{base_x + 10}" y="{base_y - total_height - 12}" fill="#2a9d8f" font-size="5" dominant-baseline="middle">pure spatial edge</text>')
        out_lines.append(f'<circle cx="{base_x + 52}" cy="{base_y - total_height - 12}" r="1.8" fill="#d1495b" />')
        out_lines.append(f'<text x="{base_x + 56}" y="{base_y - total_height - 12}" fill="#d1495b" font-size="5" dominant-baseline="middle">time-spatial-cross edge</text>')

    def _build_heatmap_svg(
        self,
        *,
        visualization_stage_timeline_map: Optional[dict[str, tuple[int, int]]] = None,
    ) -> str:
        self._ensure_geometry()
        det_hits = self._detector_hits_int()
        all_edges = self._all_edge_items()
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

        max_edge_hit = max((v for _, v, _ in all_edges), default=0)
        for _, h, meta in all_edges:
            a = int(meta["a"])
            b = meta["b"]
            if a not in det_points:
                continue
            ax, ay = det_points[a]
            if b is None:
                dx = ax - center_x
                dy = ay - center_y
                norm = (dx * dx + dy * dy) ** 0.5
                if norm > 0:
                    dx /= norm
                    dy /= norm
                bx = ax + dx * 20
                by = ay + dy * 20
            else:
                b = int(b)
                if b not in det_points:
                    continue
                bx, by = det_points[b]
            is_cross = int(meta["min_t"]) < int(meta["max_t"])
            t = np.log1p(h) / np.log1p(max_edge_hit) if (max_edge_hit and h > 0) else 0.0
            if h > 0:
                lw = 0.45 + 2.7 * t
                alpha = 0.35 + 0.55 * t
            else:
                lw = 0.25
                alpha = 0.08
            color = "#d1495b" if is_cross else "#2a9d8f"
            svg.append(
                f'<path d="M {ax} {ay} L {bx} {by}" fill="none" stroke="{color}" '
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
        svg.append(f'<text x="{title_x}" y="{title_y}" fill="black" font-size="10" font-weight="bold" text-anchor="middle" dominant-baseline="hanging">Timeline-Enhanced Logical-Error Detector/Edge Heat Map</text>')
        svg.append(f'<text x="{title_x}" y="{stats_y}" fill="black" font-size="7" text-anchor="middle" dominant-baseline="hanging">sampled shots={self.shots}, accepted={self.accepted_shots}, postselected={self.postselected_shots}, logical_errors={self.logical_error_shots}, logical_error_rate={self.logical_error_rate():.6f}</text>')
        svg.append(f'<text x="{title_x}" y="{subtitle_y}" fill="#444" font-size="6" text-anchor="middle" dominant-baseline="hanging">all DEM edges shown: inactive=faint, active=highlighted by width/opacity</text>')

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
        svg.append(f'<text x="{graph_min_x + 8}" y="{graph_min_y - 8}" fill="#2a9d8f" font-size="6.5" font-weight="bold" dominant-baseline="middle">pure spatial edge</text>')
        svg.append(f'<circle cx="{graph_min_x + 58}" cy="{graph_min_y - 8}" r="1.8" fill="#d1495b" />')
        svg.append(f'<text x="{graph_min_x + 62}" y="{graph_min_y - 8}" fill="#d1495b" font-size="6.5" font-weight="bold" dominant-baseline="middle">time-spatial-cross edge</text>')
        svg.append("</svg>")
        return "\n".join(svg)

    def _apply_timeline_stage_overlay(self, ax: Any, xs: list[float]) -> None:
        if not xs:
            return
        xmin = min(xs) - 12
        xmax = max(xs) + 12

        viz_stage_map = self._stage_map_for_visualization()
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

    def _build_heatmap_figure(self) -> Any:
        self._ensure_geometry()
        fig, ax = plt.subplots(figsize=(12, 8))

        det_hits = self._detector_hits_int()
        det_points: dict[int, tuple[float, float]] = {
            d: self._project_coord(cs)
            for d, cs in self.detector_coords.items()
        }
        xs = [p[0] for p in det_points.values()]
        ys = [p[1] for p in det_points.values()]
        self._apply_timeline_stage_overlay(ax, xs)

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
            cbar.set_label("Logical-error detector hit count", fontsize=9)

        active_edges = self._active_edge_items()
        max_edge_hit = max((v for _, v, _ in active_edges), default=0)
        spatial_label_done = False
        cross_label_done = False
        for _, h, meta in active_edges:
            a = int(meta["a"])
            b = meta["b"]
            pa = det_points.get(a)
            if pa is None:
                continue
            is_cross = int(meta["min_t"]) < int(meta["max_t"])
            t = np.log1p(h) / np.log1p(max_edge_hit) if max_edge_hit else 0.0
            lw = 0.5 + 2.5 * t
            alpha = 0.3 + 0.6 * t
            color = "#d1495b" if is_cross else "#2a9d8f"
            label = None
            if is_cross and not cross_label_done:
                label = "time-spatial-cross edge"
                cross_label_done = True
            if (not is_cross) and not spatial_label_done:
                label = "pure spatial edge"
                spatial_label_done = True
            if b is None or int(b) not in det_points:
                ax.scatter([pa[0]], [pa[1]], s=15 + 45 * t, marker="s", color=color, alpha=alpha, zorder=4, label=label)
                continue
            pb = det_points[int(b)]
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, linewidth=lw, alpha=alpha, zorder=2, label=label)

        ax.set_title("Logical-Error Edge Density Heat Map with Timeline/Stage Overlay", fontsize=12)
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
            ax.text(0.5, 0.5, "No logical-error detector events", ha="center", va="center", transform=ax.transAxes)
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
        ax.set_title("Histogram of Logical-Error Detector Hits")
        fig.tight_layout()
        return fig

    def _build_edge_histogram_figure(self) -> Any:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        active_edges = self._active_edge_items()
        if not active_edges:
            ax.text(0.5, 0.5, "No active logical-error edges", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            fig.tight_layout()
            return fig

        max_hit = max(v for _, v, _ in active_edges)
        if max_hit <= 80:
            num_per_bucket = 1
            num_buckets = max(max_hit + 1, 2)
        else:
            num_buckets = 40
            num_per_bucket = (max_hit + num_buckets - 1) // num_buckets

        spatial_bins = np.zeros(num_buckets, dtype=int)
        cross_bins = np.zeros(num_buckets, dtype=int)
        for _, h, meta in active_edges:
            idx = min(num_buckets - 1, int(h // num_per_bucket))
            is_cross = int(meta["min_t"]) < int(meta["max_t"])
            if is_cross:
                cross_bins[idx] += 1
            else:
                spatial_bins[idx] += 1

        x = np.arange(num_buckets)
        ax.bar(x, spatial_bins, color="#2a9d8f", alpha=0.9, label="pure spatial edge")
        ax.bar(x, cross_bins, bottom=spatial_bins, color="#d1495b", alpha=0.9, label="time-spatial-cross edge")
        ax.set_yscale("log")
        ax.set_xlabel(f"Edge hit-count bucket (width={num_per_bucket})")
        ax.set_ylabel("Number of edges")
        ax.set_title("Histogram of Logical-Error Edges (Spatial vs Time-Spatial-Cross)")
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

    def edge_histogram_to_svg(self, svg_path: str | pathlib.Path | None = None) -> str:
        fig = self._build_edge_histogram_figure()
        svg_text = self._figure_to_svg_text(fig)
        plt.close(fig)
        if svg_path is not None:
            pathlib.Path(svg_path).write_text(svg_text)
        return svg_text

    def edge_histogram_for_notebook(self) -> Any:
        return self._build_edge_histogram_figure()

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
                "logical_error_shots": self.logical_error_shots,
                "logical_error_shots_raw": self.logical_error_shots_raw,
                "logical_error_shots_accepted": self.logical_error_shots_accepted,
                "postselection_rate": self.postselection_rate(),
                "logical_error_rate": self.logical_error_rate(),
                "seconds": self.seconds,
                "stage_timeline_map_raw": json.dumps(self.stage_timeline_map_raw, sort_keys=True),
                "stage_timeline_map": json.dumps(self.stage_timeline_map, sort_keys=True),
                "visualization_stage_timeline_map": json.dumps(self.visualization_stage_timeline_map, sort_keys=True),
                "stage_postselection_rate": json.dumps(self.stage_postselection_rate(), sort_keys=True),
                "detector_counts": json.dumps(dict(self.detector_counts), sort_keys=True),
                "edge_counts": json.dumps(dict(self.edge_counts), sort_keys=True),
                "detector_counts_raw": json.dumps(dict(self.detector_counts_raw), sort_keys=True),
                "edge_counts_raw": json.dumps(dict(self.edge_counts_raw), sort_keys=True),
                "detector_counts_accepted": json.dumps(dict(self.detector_counts_accepted), sort_keys=True),
                "edge_counts_accepted": json.dumps(dict(self.edge_counts_accepted), sort_keys=True),
                "edge_metadata": json.dumps(self.edge_metadata, sort_keys=True),
            })

    @staticmethod
    def from_CSV(
        csv_path: str | pathlib.Path,
        *,
        row_index: int = -1,
    ) -> "LogicalDensityHeatMapCollection":
        csv_path = pathlib.Path(csv_path)
        csv.field_size_limit(sys.maxsize)
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"No rows found in {csv_path}")
        row = rows[row_index]

        obj = LogicalDensityHeatMapCollection.__new__(LogicalDensityHeatMapCollection)
        obj.circuit = None
        obj.dem = None
        obj.matching = None
        obj.sampler = None
        obj.detector_coords = {}
        obj.detector_to_time = {}
        obj.num_dets = 0
        obj.error_edge_list = []
        obj.e2i = {}
        obj.i2es = {}
        obj.node_to_edges = {}
        obj.postselect_mask = None
        obj.stage_postselect_masks = {}
        obj.apply_postselection = bool(int(row.get("apply_postselection", "1") or 1))
        obj.count_mode = row.get("count_mode", "accepted_only")
        obj.decompose_errors = bool(int(row.get("decompose_errors", "1") or 1))

        obj.shots = int(row["shots"])
        obj.accepted_shots = int(row["accepted_shots"])
        obj.postselected_shots = int(row["postselected_shots"])
        obj.logical_error_shots = int(row.get("logical_error_shots", "0") or 0)
        obj.logical_error_shots_raw = int(row.get("logical_error_shots_raw", "0") or 0)
        obj.logical_error_shots_accepted = int(row.get("logical_error_shots_accepted", "0") or 0)
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
        obj.edge_counts_raw = collections.Counter(json.loads(row.get("edge_counts_raw", "{}")))
        obj.detector_counts_accepted = collections.Counter(json.loads(row.get("detector_counts_accepted", "{}")))
        obj.edge_counts_accepted = collections.Counter(json.loads(row.get("edge_counts_accepted", "{}")))
        obj.detector_counts = collections.Counter(json.loads(row.get("detector_counts", "{}")))
        obj.edge_counts = collections.Counter(json.loads(row.get("edge_counts", "{}")))
        obj.edge_metadata = json.loads(row["edge_metadata"])
        stage_rates = json.loads(row["stage_postselection_rate"])
        obj.stage_postselected_shots = collections.Counter({
            k: int(round(float(v) * obj.shots))
            for k, v in stage_rates.items()
        })

        if not obj.detector_counts and obj.count_mode == "accepted_only":
            obj.detector_counts = obj.detector_counts_accepted.copy()
        if not obj.detector_counts and obj.count_mode == "all_shots":
            obj.detector_counts = obj.detector_counts_raw.copy()
        if not obj.edge_counts and obj.count_mode == "accepted_only":
            obj.edge_counts = obj.edge_counts_accepted.copy()
        if not obj.edge_counts and obj.count_mode == "all_shots":
            obj.edge_counts = obj.edge_counts_raw.copy()

        if obj.logical_error_shots == 0:
            if obj.count_mode == "accepted_only":
                obj.logical_error_shots = obj.logical_error_shots_accepted
            else:
                obj.logical_error_shots = obj.logical_error_shots_raw

        return obj
