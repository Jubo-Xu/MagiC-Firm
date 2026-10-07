#!/usr/bin/env python3
"""Overall gap sensitivity of the complete decoder for one circuit folder.

    python experiments/generate/run_gap_sensitivity.py experiments/data/circuits/<config> \\
        --shots 1000000 --seed 0 --workers 16 --cmap black_red

Samples `--shots` attempts, keeps the accepted (postselect-passing) ones,
complete-decodes them and accumulates, per detector d,
    count[d]                  accepted shots in which d fired
    mean_gap_when_active[d]   mean complete gap over those shots
    error_rate_when_active[d] complete-decoder error rate over those shots
(the statistic of explorations/gap_sensitivity_collect.collect_overall_gap_sens,
accumulated in chunks so memory is O(chunk x num_dets)).

Outputs (experiments/result/gap_sensitivity/<config>/):
    overall_gap_sens_s<seed>_n<shots>.json         per-detector arrays, accepted-gap
                                                   summary (min/max/mean/median/std,
                                                   histogram), config
    overall_gap_sens_s<seed>_n<shots>_<cmap>[_median|_linear].svg
                                                   heatmap of mean_gap_when_active over
                                                   the cycle-code slices: per-detector
                                                   min -> darkest, max -> brightest, and
                                                   (--center mean, default) the mean gap
                                                   over accepted shots -> middle of the
                                                   colour range (two-slope map; --center
                                                   median / none = linear); grey = never
                                                   fired / postselected; the colour scale
                                                   is ticked at equal colour positions and
                                                   marks the centre value
    overall_err_sens_s<seed>_n<shots>_<cmap>.svg   same for error_rate_when_active, linear
                                                   in [0, max] (colour inverted: dark =
                                                   high error), LER over accepted shots
                                                   marked
`--cmap` accepts the built-in `black_red` (default) or any matplotlib colormap
name (viridis, magma, inferno, Blues, ..., `_r` for reversed).
"""
import argparse
import math
import json
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702

import gen

from circuit_folder import load_circuit_folder, compile_samplers

_G: dict = {}          # fork-shared state for worker processes


# ----------------------------------------------------------------------------
# collection
# ----------------------------------------------------------------------------
def collect_overall_gap_sens(dec, *, shots, seed, chunk=20_000):
    """Per-detector gap/error sensitivity of the complete decoder, plus the
    per-shot gap summary over accepted (postselect-passing) shots.

    Same statistic as explorations/gap_sensitivity_collect.collect_overall_gap_sens,
    accumulated in chunks so memory is O(chunk x num_dets), not O(shots x num_dets).
      count[d]                  accepted shots in which d fired
      mean_gap_when_active[d]   mean complete gap over those shots
      error_rate_when_active[d] complete-decoder error rate over those shots
    Returns raw sums too, so worker results can be merged by addition.
    """
    nd = dec.task.circuit.num_detectors
    sampler = dec.gap_circuit.compile_detector_sampler(seed=seed)
    count = np.zeros(nd, np.float64)
    sum_gap = np.zeros(nd, np.float64)
    sum_err = np.zeros(nd, np.float64)
    gaps_all = []                      # one float32 per accepted shot (for min/max/mean/hist)
    errors = 0
    done = 0
    while done < shots:
        n = min(chunk, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True, bit_packed=True)
        keep = ~np.any(dets & dec._discard_mask, axis=1)
        dets, obs = dets[keep], obs[keep]
        done += n
        if dets.shape[0] == 0:
            continue
        pred, gaps = dec._decode_batch_overwrite_last_byte(bit_packed_dets=dets.copy())
        err = (pred ^ (obs[:, 0] & 1).astype(np.bool_)).astype(np.float32)
        errors += int(err.sum())
        bits = np.unpackbits(dets, axis=1, bitorder="little")[:, :nd].astype(np.float32)
        count += bits.sum(axis=0)
        sum_gap += bits.T @ gaps.astype(np.float32)
        sum_err += bits.T @ err
        gaps_all.append(gaps.astype(np.float32))
    gaps_all = np.concatenate(gaps_all) if gaps_all else np.zeros(0, np.float32)
    return {"shots": int(shots), "kept": int(gaps_all.size), "errors": errors,
            "count": count, "sum_gap": sum_gap, "sum_err": sum_err,
            "gaps": gaps_all}


# ----------------------------------------------------------------------------
# colour
# ----------------------------------------------------------------------------
def get_cmap(name):
    """name -> callable u in [0,1] -> (r, g, b) floats. `black_red` is the
    built-in default (the exploration's look: black = low, red = high); any
    matplotlib colormap name works too (dark-at-low for the sequential ones;
    append `_r` to reverse)."""
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    if name == "black_red":
        cm = LinearSegmentedColormap.from_list("black_red", [(0, 0, 0), (1, 0, 0)])
    elif name == "black_red_r":
        cm = LinearSegmentedColormap.from_list("black_red_r", [(1, 0, 0), (0, 0, 0)])
    else:
        cm = matplotlib.colormaps[name]          # KeyError -> unknown name, let it raise
    return lambda u: tuple(float(x) for x in cm(float(u))[:3])


def make_norm(vmin, vmax, vcenter=None):
    """-> (fwd, inv, centered). fwd: value -> u in [0,1] (clipped); inv: u -> value.
    Linear when vcenter is None or not strictly inside (vmin, vmax); otherwise
    two-slope: [vmin, vcenter] -> [0, 0.5], [vcenter, vmax] -> [0.5, 1], so
    vcenter (e.g. the mean gap over shots) sits at the middle of the colour range."""
    vmin, vmax = float(vmin), float(vmax)
    if vcenter is None or not (vmin < vcenter < vmax):
        span = max(vmax - vmin, 1e-12)
        return (lambda v: min(max((v - vmin) / span, 0.0), 1.0),
                lambda u: vmin + u * span, False)
    lo, hi = vcenter - vmin, vmax - vcenter

    def fwd(v):
        v = min(max(v, vmin), vmax)
        return 0.5 * (v - vmin) / lo if v <= vcenter else 0.5 + 0.5 * (v - vcenter) / hi

    def inv(u):
        return vmin + 2 * u * lo if u <= 0.5 else vcenter + 2 * (u - 0.5) * hi

    return fwd, inv, True


def make_norm_deviation(L, scale="linear"):
    """Symmetric norm for a signed deviation d in [-L, L]: 0 -> u = 0.5, +-L -> 1 / 0
    (clipped), so equal |d| gets equal colour intensity on both sides. scale='sqrt'
    uses a signed square root (more resolution near zero, tails compressed)."""
    L = max(float(L), 1e-12)
    g = (lambda x: x) if scale == "linear" else (lambda x: math.copysign(math.sqrt(abs(x)), x))
    gi = (lambda y: y) if scale == "linear" else (lambda y: math.copysign(y * y, y))
    G = g(1.0)
    fwd = lambda v: 0.5 + 0.5 * g(min(max(v, -L), L) / L) / G
    inv = lambda u: L * gi((u - 0.5) * 2 * G)
    return fwd, inv, True


def make_tile_color_func(values, counts, *, norm, cmap, invert=False):
    """gen.write_svg tile_color_func: colour of detector d is cmap(norm(v[d])) with
    norm = the forward map of make_norm; grey for detectors with count 0 (never
    fired in an accepted shot, e.g. postselected). invert=True maps high values
    to the dark end (used for the error-rate map where high = bad)."""
    grey = (0.5, 0.5, 0.5)

    def tile_color(tile):
        d, = tile.flags
        d = int(d)
        if d >= len(values) or counts[d] <= 0:
            return grey
        u = norm(float(values[d]))
        return cmap(1.0 - u if invert else u)

    return tile_color


# ----------------------------------------------------------------------------
# heatmap SVG (tiles via gen.write_svg + a colour scale in a right-hand margin)
# ----------------------------------------------------------------------------
def write_heatmap_svg(path, circuit, values, counts, *, vmin, vmax, cmap, title, vcenter=None,
                      marks=None, invert=False, unit="", fmt="{:.1f}", canvas_height=1000,
                      norm=None, mark_text=None, subtitle=None, edge_tags=(" (min)", " (max)")):
    """Tile colour = cmap(norm(values[d])) with norm = make_norm(vmin, vmax, vcenter):
    linear in [vmin, vmax], or two-slope with vcenter at the middle of the colour
    range; grey = count 0. Then the viewBox is widened by a right margin holding a
    vertical colour bar sampled from `cmap`, ticks at equal COLOUR positions
    (u = 0, .25, .5, .75, 1) labelled with their values, and a labelled marker for
    every entry of `marks` ({label: value}); a mark outside [vmin, vmax] is pinned
    to the edge but labelled with its true value. invert=True flips the bar like
    the tiles."""
    import re
    path = pathlib.Path(path)
    fwd, inv, centered = norm if norm is not None else make_norm(vmin, vmax, vcenter)
    codes = gen.circuit_to_cycle_code_slices(circuit)
    ticks = sorted(codes)
    panels = [codes[t].with_transformed_coords(lambda e: e * (1 + 1j)) for t in ticks]
    panels[0].write_svg(path, canvas_height=canvas_height, other=panels[1:],
                        title=[f"tick={t}" for t in ticks],
                        tile_color_func=make_tile_color_func(values, counts, norm=fwd,
                                                             cmap=cmap, invert=invert),
                        show_coords=False, show_obs=False)
    text = path.read_text()
    m = re.search(r'viewBox="0 0 (\d+) (\d+)"', text)
    W, H = int(m.group(1)), int(m.group(2))

    # --- colour bar geometry (right margin) ---
    margin, bar_w = 330, 22
    x0, y0, bar_h = W + 40, 60, H - 120
    col = lambda u: "rgb({},{},{})".format(*(int(round(255 * c)) for c in cmap(1 - u if invert else u)))
    y_of = lambda v: y0 + bar_h * (1.0 - fwd(v))                                   # vmin bottom, vmax top
    centre_label = next((lbl for lbl, v in (marks or {}).items() if v == vcenter), "centre") if centered else None
    if subtitle is None:
        subtitle = ("dark = low" if not invert else "dark = high") + (f" [{unit}]" if unit else "") \
            + (f" | centre = {centre_label}" if centered else " | linear")
    n = 64                                   # stacked solid slabs: renders in every rasterizer (no gradient support needed)
    bar = "".join(f'<rect x="{x0}" y="{y0 + bar_h * (1 - (i + 1) / n):.2f}" width="{bar_w}" '
                  f'height="{bar_h / n + 0.5:.2f}" fill="{col((i + 0.5) / n)}"/>' for i in range(n))
    g = [f'<g id="colour-scale" font-family="sans-serif" fill="#111">',
         f'<text x="{x0}" y="{y0 - 34}" font-size="13" font-weight="700">{title}</text>',
         f'<text x="{x0}" y="{y0 - 18}" font-size="10">{subtitle}</text>',
         f'{bar}<rect x="{x0}" y="{y0}" width="{bar_w}" height="{bar_h}" fill="none" stroke="#444" stroke-width="0.6"/>']
    for k in range(5):                       # ticks at equal colour positions u = 0, .25, .5, .75, 1
        if centered and k == 2:              # the centre is labelled by its marker
            continue
        u = k / 4
        v = inv(u)
        y = y0 + bar_h * (1.0 - u)
        tag = edge_tags[0] if k == 0 else edge_tags[1] if k == 4 else ""
        g.append(f'<line x1="{x0 + bar_w}" y1="{y}" x2="{x0 + bar_w + 5}" y2="{y}" stroke="#111" stroke-width="0.8"/>')
        g.append(f'<text x="{x0 + bar_w + 8}" y="{y + 3.5}" font-size="10">{fmt.format(v)}{tag}</text>')
    for label, v in (marks or {}).items():   # labelled markers (e.g. mean gap over shots)
        y = y_of(v)
        g.append(f'<polygon points="{x0 - 10},{y - 5} {x0 - 10},{y + 5} {x0 - 2},{y}" fill="#111"/>')
        g.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + bar_w}" y2="{y}" stroke="white" stroke-width="1.6"/>')
        g.append(f'<line x1="{x0}" y1="{y}" x2="{x0 + bar_w}" y2="{y}" stroke="#111" stroke-width="0.8" stroke-dasharray="2,2"/>')
        txt = (mark_text or {}).get(label, f"{label} = {fmt.format(v)}")
        g.append(f'<text x="{x0 + bar_w + 8}" y="{y - 6}" font-size="10" font-weight="700">{txt}</text>')
    g.append(f'<rect x="{x0}" y="{y0 + bar_h + 16}" width="14" height="10" fill="rgb(128,128,128)"/>')
    g.append(f'<text x="{x0 + 20}" y="{y0 + bar_h + 25}" font-size="10">no data / postselected</text>')
    g.append('</g>')
    text = text.replace(m.group(0), f'viewBox="0 0 {W + margin} {H}"', 1)
    text = text.replace("</svg>", "\n".join(g) + "\n</svg>", 1)
    path.write_text(text)


# ----------------------------------------------------------------------------
# driver
# ----------------------------------------------------------------------------
def _worker(job):
    k, n = job
    cf, seed, chunk = (_G[x] for x in ("cf", "seed", "chunk"))
    dec, _ = compile_samplers(cf)
    return collect_overall_gap_sens(dec, shots=n, seed=seed + k, chunk=chunk)


def collect(cf, *, shots, seed, workers, chunk):
    """Worker k samples shots//workers attempts (remainder to the last) with
    seed + k; sums merge by addition, per-shot gaps concatenate. workers=1 runs
    inline. The total is reproducible for fixed (seed, workers)."""
    per = [shots // workers] * workers
    per[-1] += shots - sum(per)
    jobs = [(k, n) for k, n in enumerate(per) if n > 0]
    _G.update(cf=cf, seed=seed, chunk=chunk)
    if workers == 1:
        parts = [_worker(jobs[0])]
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("fork")) as ex:
            parts = list(ex.map(_worker, jobs))
    return {"shots": int(shots),
            "kept": sum(p["kept"] for p in parts), "errors": sum(p["errors"] for p in parts),
            "count": sum(p["count"] for p in parts), "sum_gap": sum(p["sum_gap"] for p in parts),
            "sum_err": sum(p["sum_err"] for p in parts),
            "gaps": np.concatenate([p["gaps"] for p in parts])}


def summarize(res):
    """-> (mean_gap_when_active, error_rate_when_active, summary dict).
    Summary = accepted-shot gap statistics (min/max/mean/median/std, integer-dB
    histogram, LER) + per-detector value ranges (the heatmap colour ranges)."""
    c = res["count"]; act = c > 0
    mg = np.divide(res["sum_gap"], c, out=np.zeros_like(c), where=act)
    er = np.divide(res["sum_err"], c, out=np.zeros_like(c), where=act)
    g = res["gaps"]
    kept = int(res["kept"])
    edges = np.arange(0, int(np.ceil(g.max())) + 2) if kept else np.arange(0, 2)
    hist, _ = np.histogram(g, bins=edges)
    stat = lambda a: {"min": float(a.min()), "max": float(a.max()), "mean": float(a.mean()),
                      "median": float(np.median(a)), "std": float(a.std())} if a.size else None
    summary = {
        "shots": {"sampled": int(res["shots"]), "accepted": kept,
                  "accept_rate": kept / max(res["shots"], 1),
                  "errors": int(res["errors"]), "ler": res["errors"] / max(kept, 1),
                  "gap": stat(g),
                  "gap_histogram": {"bin_edges_db": edges.tolist(), "counts": hist.tolist()}},
        "detectors": {"num_detectors": int(c.size), "active": int(act.sum()),
                      "mean_gap_when_active": stat(mg[act]),
                      "error_rate_when_active": stat(er[act])},
    }
    return mg, er, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=pathlib.Path, help="circuit data folder (read only)")
    ap.add_argument("--shots", type=int, default=1_000_000, help="attempts to sample (accepted = postselect-passing)")
    ap.add_argument("--seed", type=int, default=0, help="stim sampler seed (worker k uses seed+k)")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=20_000, help="shots per decode batch (memory bound)")
    ap.add_argument("--cmap", default=None,
                    help="colormap: any matplotlib name (default RdBu for the deviation map: red = below the mean "
                         "gap, blue = above; black_red for the absolute maps)")
    ap.add_argument("--center", choices=["mean", "median", "none"], default="mean",
                    help="reference of the gap map: deviation from the mean/median gap over accepted shots "
                         "(diverging, symmetric), or none = absolute values, linear")
    ap.add_argument("--absolute", action="store_true",
                    help="legacy gap map: absolute values with the centre at the middle of the range (two-slope)")
    ap.add_argument("--clip-quantile", type=float, default=0.98,
                    help="deviation map: symmetric colour range +-L with L = this quantile of |deviation| over "
                         "active detectors (1.0 = max); values beyond are clipped (bar ends marked)")
    ap.add_argument("--scale", choices=["linear", "sqrt"], default="linear",
                    help="deviation map: linear, or signed sqrt (more resolution near the mean)")
    ap.add_argument("--replot", type=pathlib.Path, default=None, metavar="JSON",
                    help="re-draw the maps from a stored overall_gap_sens_*.json (no sampling)")
    ap.add_argument("--out", type=pathlib.Path, default=None,
                    help="output folder (default experiments/result/gap_sensitivity/<folder name>)")
    args = ap.parse_args()
    if args.workers < 1 or args.shots < 1 or args.chunk < 1:
        ap.error("--workers, --shots and --chunk must be >= 1")
    deviation = not args.absolute and args.center != "none"
    cmap_name = args.cmap or ("RdBu" if deviation else "black_red")
    cmap = get_cmap(cmap_name)                       # fail early on a bad name

    t0 = time.time()
    cf = load_circuit_folder(args.folder, None)
    out = args.out or (paths.RESULT / "gap_sensitivity" / cf.folder.name)
    out.mkdir(parents=True, exist_ok=True)
    if args.replot:
        doc = json.loads(args.replot.read_text()); pd = doc["per_detector"]
        res = {k: np.asarray(pd[k], dtype=float) for k in ("count", "sum_gap", "sum_err")}
        mg, er, summary = np.asarray(pd["mean_gap_when_active"], float), np.asarray(pd["error_rate_when_active"], float), doc["summary"]
        args.seed, args.shots = doc["config"]["seed"], doc["config"]["shots"]
    else:
        res = collect(cf, shots=args.shots, seed=args.seed, workers=args.workers, chunk=args.chunk)
        mg, er, summary = summarize(res)
    s, d = summary["shots"], summary["detectors"]
    print(f"{cf.folder.name}: {s['accepted']}/{s['sampled']} accepted ({s['accept_rate']:.3f}), "
          f"LER {s['ler']:.2e}, gap min/mean/median/max {s['gap']['min']:.1f}/{s['gap']['mean']:.1f}/"
          f"{s['gap']['median']:.1f}/{s['gap']['max']:.1f} dB; {d['active']}/{d['num_detectors']} detectors active, "
          f"mean_gap_when_active {d['mean_gap_when_active']['min']:.1f}..{d['mean_gap_when_active']['max']:.1f} dB "
          f"({time.time() - t0:.0f}s)")

    base = f"overall_gap_sens_s{args.seed}_n{args.shots}"
    if not args.replot: (out / f"{base}.json").write_text(json.dumps({
        "folder": cf.folder.name, "circuit_file": cf.stem + ".stim",
        "config": {k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(args).items()},
        "summary": summary,
        "per_detector": {"count": res["count"].astype(int).tolist(),
                         "mean_gap_when_active": mg.tolist(), "error_rate_when_active": er.tolist(),
                         "sum_gap": res["sum_gap"].tolist(), "sum_err": res["sum_err"].tolist()},
    }, indent=1))
    mgr = d["mean_gap_when_active"]
    centre = None if args.center == "none" else s["gap"][args.center]
    if deviation:
        # diverging map of the DEVIATION from the reference gap: symmetric range +-L (robust quantile),
        # neutral colour at the reference, equal |deviation| -> equal intensity on both sides
        active = res["count"] > 0
        dev = np.where(active, mg - centre, 0.0)
        L = float(np.quantile(np.abs(dev[active]), args.clip_quantile)) if args.clip_quantile < 1 else float(np.abs(dev[active]).max())
        n_clip = int((np.abs(dev[active]) > L).sum())
        gap_svg = f"{base}_{cmap_name}_dev{'' if args.center == 'mean' else '_' + args.center}{'' if args.scale == 'linear' else '_sqrt'}.svg"
        write_heatmap_svg(out / gap_svg, cf.circuit, dev, res["count"], vmin=-L, vmax=L, vcenter=0.0, cmap=cmap,
                          norm=make_norm_deviation(L, args.scale), title="mean gap when active - reference",
                          marks={"ref": 0.0}, mark_text={"ref": f"{args.center} gap (shots) = {centre:.1f} dB"},
                          subtitle=f"deviation [dB], {args.scale}; range +-{L:.1f} = q{args.clip_quantile:g} of |dev| ({n_clip} clipped)",
                          fmt="{:+.1f}", edge_tags=(" (clip &lt;=)", " (clip &gt;=)"))
    else:
        marks = {f"{args.center} gap (shots)": centre} if centre is not None else {"mean gap (shots)": s["gap"]["mean"]}
        gap_svg = f"{base}_{cmap_name}" + ("" if args.center == "mean" else f"_{'linear' if centre is None else args.center}") + ".svg"
        write_heatmap_svg(out / gap_svg, cf.circuit, mg, res["count"],
                          vmin=mgr["min"], vmax=mgr["max"], vcenter=centre, cmap=cmap,
                          title="mean gap when active", marks=marks, unit="dB")
    err_cmap = get_cmap(args.cmap or "black_red")
    err_svg = f"overall_err_sens_s{args.seed}_n{args.shots}_{args.cmap or 'black_red'}.svg"
    write_heatmap_svg(out / err_svg, cf.circuit, er, res["count"],
                      vmin=0.0, vmax=d["error_rate_when_active"]["max"], cmap=err_cmap, invert=True,
                      title="error rate when active", marks={"LER (shots)": s["ler"]}, fmt="{:.3g}")
    print(f"saved {out}/{base}.json, {gap_svg}, {err_svg}")


if __name__ == "__main__":
    main()
