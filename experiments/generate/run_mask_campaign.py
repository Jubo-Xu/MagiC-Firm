#!/usr/bin/env python3
"""Overnight campaign: for every circuit folder with essentials, pick a
reasonable partial mask, generate its folder, and characterize the complete
and partial decoders on micro-blossom — so the runtime estimator (Mode A)
can be run per circuit afterwards.

Per circuit (resumable — every step is skipped when its output exists):
  1. sweep   run_eval_partial_mask.py over a K_Q grid with the final recipe
             Q_exp base -> closure   (ablation 2026-08-27: canonical augment
             and causal prune cancel each other; Q_exp + closure is best)
             (100k shots, tc 40, iso false-high <= 0.1%); results saved under
             experiments/result/mask_campaign/<circuit>/sweep/K<K>/
  2. select  maximise modelled saving  T3(th*) x (L_complete - L_partial)  with
             L = cycles(v) / f_circuit, cycles(v) = 319 + 0.644 (v - 395)
             (linear latency model validated on the flagship at 43 MHz);
             among candidates within --tie-us of the best take the SMALLEST
             mask. Circuits with no admissible th* get no partial mask.
  3. generate gen_partial_mask_dem.py --contract for the chosen K_Q
  4. characterize run_mb_characterization.py for complete and chosen partial
             (1000 shots, seed 0, 8 workers) at the circuit's clock:
             d2 -> {11: 77 MHz, 13: 62 MHz, 15..21: 43 MHz}
             (micro-blossom paper Fmax for surface codes of that distance;
             d >= 17 has no published value -> 43 MHz, conservative)
  5. record  experiments/result/mask_campaign/summary.json + summary.md with
             the estimator command (--th th*) per circuit.

    python experiments/generate/run_mask_campaign.py [--only PATTERN ...] [--exclude PATTERN ...] [--skip-p 0.003 0.005]
        [--eval-workers 48] [--mb-workers 8] [--sweep-shots 100000] [--no-mb]
Order: d1=3 p=1e-3 first, then 5e-4/7e-4/9e-4, then 1.5e-3, then d1=5 (1e-3, 7e-4, 5e-4).
Only ONE micro-blossom characterization may run on the machine at a time.
Requires only the logical_ambiguity essentials list per circuit.
"""
import argparse
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from naming import parse_name, freq_tag, FREQ_BY_D2                        # noqa: E402
from campaign import essentials_flags, WC                                  # noqa: E402
REPO = paths.REPO
PY = sys.executable
CIRCUITS = paths.CIRCUITS
OUT = paths.RESULT / "mask_campaign"
MB_RESULT = paths.RESULT / "mb_characterization"

K_GRID = {3: [100, 150, 200, 250, 300, 350], 5: [250, 350, 450, 550]}
TC, ISO_FH = 40.0, 0.001
CYC0, V0, CYC_PER_V = 319.0, 395.0, (763.0 - 319.0) / (1084.0 - 395.0)   # 43 MHz model, cycles


def log(msg):
    print(f"[{datetime.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def cycles(v):
    return CYC0 + CYC_PER_V * (v - V0)


def run(cmd, log_path):
    with open(log_path, "a") as f:
        f.write(f"\n$ {' '.join(map(str, cmd))}\n")
        f.flush()
        r = subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=REPO)
    return r.returncode


def ordered_circuits(args):
    names = []
    for d in sorted(CIRCUITS.iterdir()):
        if not d.is_dir() or not d.name.startswith("end2end_"):
            continue
        if not all(list(d.glob(f"*_{k}_*.json")) for k in ESSENTIALS):
            continue
        d1, d2, p, inj = parse_name(d.name)
        if p in args.skip_p or (args.only and not any(pat in d.name for pat in args.only)):
            continue
        if args.exclude and any(pat in d.name for pat in args.exclude):
            continue
        names.append((d1, d2, p, d.name))
    prio = {0.001: 0, 0.0005: 1, 0.0007: 1, 0.0009: 1, 0.0015: 2, 0.003: 3, 0.005: 3}
    prio5 = {0.001: 0, 0.0007: 1, 0.0005: 2}
    names.sort(key=lambda t: (t[0], prio5.get(t[2], 9) if t[0] == 5 else prio.get(t[2], 9), t[1], t[2]))
    return [n for _, _, _, n in names]


def sweep(name, folder, d1, out_dir, args, log_path):
    """Run the evaluator for every K in the grid; return list of candidate dicts."""
    cands = []
    for K in K_GRID[d1]:
        save = out_dir / "sweep" / f"K{K}"
        stats = save / "stats.json"
        if not stats.exists():
            rc = run([PY, paths.EXPERIMENTS / "generate" / "run_eval_partial_mask.py", folder,
                      "--base-size", K, "--q-type", "exp", "--closure",
                      "--weight-cutoff", WC, *essentials_flags(folder),
                      "--shots", args.sweep_shots, "--seed", 0, "--workers", args.eval_workers,
                      "--tc", TC, "--tl", 0, "--th", 60, "--iso-fh", ISO_FH, "--save", save], log_path)
            if rc != 0 or not stats.exists():
                log(f"  K{K}: evaluator FAILED (rc {rc})")
                continue
        s = json.loads(stats.read_text())
        r = s["results"][-1]
        iso = r.get("iso")
        cands.append({"K": K, "selected": r["structure"]["selected"], "vertices": r["structure"]["contracted_vertices"],
                      "S": r["structure"]["S_crossing_mass"], "p_eq": r["gap"]["p_equal"],
                      "th": None if iso is None else iso["th"], "T3": None if iso is None else iso["tier_share"]["T3_accept"],
                      "false_high": None if iso is None else iso["false_high"],
                      "accept": r["gating"]["complete_only_accept_rate"]})
    return cands


def select(cands, v_complete, f, tie_us):
    lc = cycles(v_complete) / f * 1e6
    for c in cands:
        c["L_partial_us"] = cycles(c["vertices"]) / f * 1e6
        c["L_complete_us"] = lc
        c["saving_us"] = None if (c["th"] is None or not c["T3"]) else c["T3"] * (lc - c["L_partial_us"])
    ok = [c for c in cands if c["saving_us"] is not None and c["saving_us"] > 0]
    if not ok:
        return None
    best = max(c["saving_us"] for c in ok)
    near = [c for c in ok if c["saving_us"] >= best - tie_us]
    return min(near, key=lambda c: c["vertices"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help="substring filters on circuit names")
    ap.add_argument("--exclude", nargs="*", default=None, help="substring filters to skip")
    ap.add_argument("--skip-p", nargs="*", type=float, default=[])
    ap.add_argument("--sweep-shots", type=int, default=100_000)
    ap.add_argument("--eval-workers", type=int, default=48)
    ap.add_argument("--mb-workers", type=int, default=8)
    ap.add_argument("--mb-shots", type=int, default=1000)
    ap.add_argument("--tie-us", type=float, default=0.3)
    ap.add_argument("--no-mb", action="store_true", help="sweep + select + generate only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    circuits = ordered_circuits(args)
    log(f"campaign over {len(circuits)} circuits: {circuits}")
    for name in circuits:
        d1, d2, p, inj = parse_name(name)
        folder = CIRCUITS / name
        out_dir = OUT / name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "campaign.log"
        f = FREQ_BY_D2[d2]
        rec = summary.get(name, {"circuit": name, "d1": d1, "d2": d2, "p": p, "frequency_hz": f})
        t0 = time.time()
        log(f"=== {name}  (f = {freq_tag(f)})")

        # complete-graph vertex count = gap DEM detectors
        ps = json.loads((folder / "postselected_detectors.json").read_text())
        v_complete = int(ps["n_gap_dets"])

        # 1-2. sweep + select
        cands = sweep(name, folder, d1, out_dir, args, log_path)
        chosen = select(cands, v_complete, f, args.tie_us)
        rec["sweep"] = cands
        rec["chosen"] = chosen
        for c in cands:
            log(f"  K{c['K']:<4} |M| {c['selected']:4d} v {c['vertices']:4d} S {c['S']:.2f} eq {c['p_eq']:.3f} "
                + (f"th* {c['th']} T3 {c['T3']:.3f} FH {c['false_high']:.4f} saving {c['saving_us']:.2f}us"
                   if c.get("saving_us") is not None else "no admissible th / nothing certifiable at tc 40"))
        if chosen is None:
            log("  -> no partial mask qualifies (nothing certifiable at tc 40); complete only")
            rec["partial_tag"] = None
        else:
            tag = f"partial_qexp{chosen['K']}_cl_wc{WC:g}"
            rec["partial_tag"] = tag
            log(f"  -> chosen K_Q {chosen['K']} ({chosen['vertices']} v, th* {chosen['th']}, T3 {chosen['T3']:.3f}, "
                f"saving {chosen['saving_us']:.2f}us at {freq_tag(f)}) -> {tag}")
            # 3. generate
            if not (folder / tag / "contracted.dem").exists():
                rc = run([PY, paths.ALGORITHMS / "gen_partial_mask_dem.py", "--out", folder,
                          "--base-size", chosen["K"], "--q-type", "exp", "--closure",
                          "--contract", "--weight-cutoff", WC,
                          *essentials_flags(folder)], log_path)
                log(f"  generate: rc {rc}")
                if rc != 0:
                    rec["error"] = "generate failed"
        # 4. characterize
        if not args.no_mb:
            for variant in (["complete"] + ([rec["partial_tag"]] if rec.get("partial_tag") else [])):
                key = f"{name}__{variant}"
                res = MB_RESULT / f"{key}__s0_n{args.mb_shots}_{freq_tag(f)}_realsched.json"
                if res.exists():
                    log(f"  mb {variant}: exists, skip")
                else:
                    cmd = [PY, paths.EXPERIMENTS / "generate" / "run_mb_characterization.py", folder]
                    if variant != "complete":
                        cmd += ["--partial", variant]
                    cmd += ["--shots", args.mb_shots, "--seed", 0, "--frequency", f, "--workers", args.mb_workers]
                    t1 = time.time()
                    rc = run(cmd, log_path)
                    log(f"  mb {variant}: rc {rc} ({(time.time() - t1) / 60:.1f} min)")
                if res.exists():
                    j = json.loads(res.read_text())["latency"]
                    rec.setdefault("mb", {})[variant] = {"parallel_mean_us": j["parallel"]["mean_us"],
                                                         "parallel_p99_us": j["parallel"]["p99_us"],
                                                         "parallel_mean_cycles": j["parallel"]["mean_cycles"],
                                                         "serial_mean_us": j["serial"]["mean_us"]}
        if rec.get("partial_tag"):
            rec["estimator_cmd"] = (f"python experiments/generate/run_runtime_estimation.py experiments/data/circuits/{name} "
                                    f"--partial {rec['partial_tag']} --mode pcp --tc {TC:g} --tl 0 --th {chosen['th']} "
                                    f"--epochs 1000000 --latency mb --mb-freq {f:g} --trace --workers 64")
        rec["complete_cmd"] = (f"python experiments/generate/run_runtime_estimation.py experiments/data/circuits/{name} "
                               f"--mode single --tc {TC:g} --epochs 1000000 --latency mb --mb-freq {f:g} --trace --workers 64")
        rec["minutes"] = round((time.time() - t0) / 60, 1)
        summary[name] = rec
        summary_path.write_text(json.dumps(summary, indent=1))
        write_markdown(summary)
        log(f"=== done {name} in {rec['minutes']} min")
    log("CAMPAIGN DONE")


def write_markdown(summary):
    lines = ["| circuit | f | chosen K_Q | \\|M\\| / vertices | th* | T3 | false-high | model saving µs | MB complete mean/p99 µs | MB partial mean/p99 µs |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in summary.items():
        c = r.get("chosen"); mb = r.get("mb", {})
        fmt = lambda v: f"{v['parallel_mean_us']:.2f} / {v['parallel_p99_us']:.2f}" if v else "—"
        lines.append(f"| {name.replace('end2end_', '').replace('_inj=unitary_b=Y', '')} | {freq_tag(r['frequency_hz'])} | "
                     + (f"{c['K']} | {c['selected']} / {c['vertices']} | {c['th']} | {c['T3']:.3f} | {c['false_high']:.4f} | {c['saving_us']:.2f}"
                        if c else "— | — | — | — | — | —")
                     + f" | {fmt(mb.get('complete'))} | {fmt(mb.get(r.get('partial_tag')))} |")
    lines += ["", "Estimator commands:", ""] + [f"- `{r['estimator_cmd']}`" for r in summary.values() if r.get("estimator_cmd")]
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
