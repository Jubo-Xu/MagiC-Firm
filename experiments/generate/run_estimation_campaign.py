#!/usr/bin/env python3
"""Systematic runtime-estimation collection over all circuits that have a
chosen partial mask and micro-blossom characterizations (from
experiments/result/mask_campaign/summary.json).

Per circuit (resumable: existing result JSONs are reused, never re-run):
  baseline   run_runtime_estimation --mode single --tc TC            (complete only)
  two-stage  run_runtime_estimation --mode pcp --tc TC --tl 0 --th th  for th = TC, TC+2, ...
             until the fast-accept tier is empty (the gap cap scales with
             ln((1-p)/p), so the sweep end is data-driven, not fixed)
             (fast-accept only; same trace shots as the baseline, so the LER
             comparison is paired — only tier-3 accepts can differ)
  iso-LER    th* = smallest th whose accepted-shot error count is
             <= baseline errors + 2*sqrt(baseline errors)   (2 sigma, Poisson:
             LER not significantly different from the baseline at 95%)
All runs use --latency mb (Mode A constants from the characterization JSONs at
the circuit's clock), --trace, --epochs EPOCHS. Results stay under
experiments/result/runtime_estimation/<circuit>/ and the selection is written
to experiments/result/runtime_estimation/summary.json (+ summary.md), which
plot_runtime_figures.py consumes.

    python experiments/generate/run_estimation_campaign.py [--workers 32] [--epochs 1000000] [--only PATTERN ...]
"""
import argparse
import datetime
import json
import math
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from naming import freq_tag, parse_name                                    # noqa: E402
from campaign import (log, run_estimator, valid, expected_path, summarize, profile_path,  # noqa: E402
                      write_md, PY, CIRCUITS, RESULT, MB_RESULT, MASK_SUMMARY, TC, SIGMA, MB_SHOTS, MB_SEED)
REPO = paths.REPO
def targets(args):
    """[(circuit, partial_tag, frequency_hz)] from the mask campaign with both MB JSONs present."""
    out = []
    entries = []
    if MASK_SUMMARY.exists():
        for name, r in json.loads(MASK_SUMMARY.read_text()).items():
            if r.get("partial_tag"):
                entries.append((name, r["partial_tag"], float(r["frequency_hz"])))
    for name, tag, f in entries:
        if args.only and not any(p in name for p in args.only):
            continue
        ok = all((MB_RESULT / f"{name}__{v}__s0_n1000_{freq_tag(f)}_realsched.json").exists()
                 for v in ("complete", tag))
        if args.cs and not profile_path(name, args).exists():
            log(f"skip {name}: control-latency profile missing ({profile_path(name, args).name})")
            continue
        if ok and (CIRCUITS / name / tag / "contracted.dem").exists():
            out.append((name, tag, f))
        else:
            log(f"skip {name}: characterization or mask folder missing")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1_000_000)
    ap.add_argument("--th-step", type=int, default=2)
    ap.add_argument("--th-max", type=int, default=120, help="safety bound; the sweep stops when the fast-accept tier empties")
    ap.add_argument("--cs", default=None, metavar="Q,F",
                    help="control-system latency profile <folder>/control_system/latency_q<Q>_f<F>.json "
                         "(algorithms/control_latency.py --save) passed to every run; results carry "
                         "_csq<Q>f<F> and the summary is summary_csq<Q>f<F>.{json,md}")
    ap.add_argument("--reuse-th-from", type=pathlib.Path, default=None, metavar="SUMMARY.json",
                    help="skip the th sweep: take th* per circuit from this summary and run only the "
                         "baseline and pcp@th* (operating-point re-runs, e.g. under a latency profile)")
    args = ap.parse_args()
    args.cs_tag = ""
    if args.cs:
        q, fo = (int(x) for x in args.cs.split(","))
        args.cs_q, args.cs_f, args.cs_tag = q, fo, f"_csq{q}f{fo}"
    ref = json.loads(args.reuse_th_from.read_text()) if args.reuse_th_from else None
    RESULT.mkdir(parents=True, exist_ok=True)
    summary_path = RESULT / f"summary{args.cs_tag}.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for name, tag, f in targets(args):
        if name in summary and summary[name].get("partial_tag") != tag:
            log(f"drop stale summary entry for {name}: mask {summary[name].get('partial_tag')} != campaign {tag}")
            del summary[name]
        if ref is not None and ref.get(name, {}).get("th_star") is None:
            log(f"skip {name}: no th* in {args.reuse_th_from.name}")
            continue
        d1, d2, p, _ = parse_name(name)
        log_path = RESULT / name / "estimation.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log(f"=== {name}  mask {tag}  f {freq_tag(f)}")
        base_path = expected_path(name, "complete", f"single_tc{TC:g}", args)
        if not valid(base_path, name, "complete", f, args, ["--mode", "single", "--tc", TC]):
            base_path = run_estimator(name, f, args, ["--mode", "single", "--tc", TC], log_path)
        base = summarize(base_path)
        tol = base["errors"] + SIGMA * math.sqrt(max(base["errors"], 1))
        log(f"  baseline: {base['prep_us']:.2f}us, LER {base['ler']:.1e} ({base['errors']} errs) -> tolerance <= {tol:.1f} errs")
        sweep = {}
        ths = [int(ref[name]["th_star"])] if ref is not None else range(int(TC), args.th_max + 1, args.th_step)
        for th in ths:
            path = expected_path(name, tag, f"pcp_tc{TC:g}_tl0_th{th}", args)
            if not valid(path, name, tag, f, args, ["--mode", "pcp", "--tc", TC, "--tl", 0, "--th", th]):
                path = run_estimator(name, f, args, ["--partial", tag, "--mode", "pcp", "--tc", TC, "--tl", 0, "--th", th], log_path)
            s = summarize(path); sweep[th] = s
            log(f"  th {th}: {s['prep_us']:.2f}us, LER {s['ler']:.1e} ({s['errors']} errs), T3 {s['tier3_share']:.3f}"
                + ("  <- meets tolerance" if s["errors"] <= tol else ""))
            if s["tier3_share"] == 0.0:      # past the gap cap: no shot can fast-accept any more
                break
        if ref is not None:
            th_star = ths[0]                 # operating point reused from the reference summary
        else:
            ok = [th for th, s in sweep.items() if s["errors"] <= tol]
            th_star = min(ok) if ok else None
        rec = {"circuit": name, "d1": d1, "d2": d2, "p": p, "frequency_hz": f, "partial_tag": tag, "tc": TC,
               "baseline": base, "tolerance_sigma": SIGMA, "tolerance_errors": tol, "th_star": th_star,
               "partial": sweep.get(th_star), "sweep": {str(k): v for k, v in sweep.items()},
               "control_latency": (paths.rel(profile_path(name, args)) if args.cs else None),
               "th_from": (str(args.reuse_th_from) if ref is not None else None)}
        if th_star is not None:
            pr = sweep[th_star]
            rec["reduction"] = 1 - pr["prep_us"] / base["prep_us"]
            log(f"  -> th* {th_star}: {pr['prep_us']:.2f}us vs {base['prep_us']:.2f}us ({rec['reduction']:.1%} lower), "
                f"LER {pr['ler']:.1e} vs {base['ler']:.1e}")
        else:
            log("  -> no th meets the LER tolerance")
        summary[name] = rec
        summary_path.write_text(json.dumps(summary, indent=1))
        write_md(summary, args.cs_tag)
    log("ESTIMATION CAMPAIGN DONE")


if __name__ == "__main__":
    main()
