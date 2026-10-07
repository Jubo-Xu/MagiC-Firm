#!/usr/bin/env python3
"""LER-matched (tc-matched) baselines: for every circuit in a runtime-estimation
summary with an operating point (baseline single tc35 + pcp@th*), run the
complete-only decoder at tc in --tc-grid (same epochs, same trace shots), then
find the complete-only threshold whose accepted-shot error count equals the
two-stage run's and compare preparation times at strictly equal LER.

Gaps are integers, so the matched point is realised as a RANDOMIZED mixture of
the two bracketing measured thresholds: q solves q*E(t2) + (1-q)*E(t1) = E_pcp
and the same q mixes the times (and their gate/feedback/decode components) —
one consistent operating point, no separate approximation for the time.

Writes a `tc_matched` block per circuit into the summary (grid summaries,
bracket, q, tc_b, mixed baseline, iso-LER reduction) and <summary>_iso_ler.md.

    python experiments/generate/run_tc_match.py [--summary summary.json] [--cs 14,29] [--epochs 10000000]
        [--tc-grid 29 31 33] [--workers 32] [--only PATTERN ...]
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
import campaign as rec                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=None, help="default summary<cs_tag>.json")
    ap.add_argument("--cs", default=None, metavar="Q,F", help="control-latency profile tag (see run_estimation_campaign)")
    ap.add_argument("--epochs", type=int, default=10_000_000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--tc-grid", type=int, nargs="+", default=[29, 31, 33])
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    args.cs_tag = ""
    if args.cs:
        q, fo = (int(x) for x in args.cs.split(","))
        args.cs_q, args.cs_f, args.cs_tag = q, fo, f"_csq{q}f{fo}"
    summary_path = rec.RESULT / (args.summary or f"summary{args.cs_tag}.json")
    summary = json.loads(summary_path.read_text())

    for name, r in sorted(summary.items(), key=lambda kv: (kv[1]["d1"], kv[1]["d2"], kv[1]["p"])):
        if r.get("th_star") is None or (args.only and not any(p in name for p in args.only)):
            continue
        if r["partial"]["epochs"] != args.epochs:
            rec.log(f"skip {name}: operating point has {r['partial']['epochs']} epochs, not {args.epochs}")
            continue
        f = float(r["frequency_hz"])
        log_path = rec.RESULT / name / "tcmatch.log"
        grid = {}
        for tc in sorted(set(args.tc_grid) | {int(r["tc"])}):
            path = rec.expected_path(name, "complete", f"single_tc{tc}", args)
            if not rec.valid(path, name, "complete", f, args, ["--mode", "single", "--tc", tc]):
                path = rec.run_estimator(name, f, args, ["--mode", "single", "--tc", tc], log_path)
            grid[tc] = rec.summarize(path)
        tcs = sorted(grid); E = [grid[t]["errors"] for t in tcs]; Ep = r["partial"]["errors"]
        note = ""
        if Ep > E[0]:
            t1 = t2 = tcs[0]; q = 0.0; note = f"pcp LER above tc={tcs[0]} baseline (clamped)"
        elif Ep < E[-1]:
            t1 = t2 = tcs[-1]; q = 0.0; note = f"pcp LER below tc={tcs[-1]} baseline (clamped)"
        else:
            for a, b in zip(tcs, tcs[1:]):
                if grid[a]["errors"] >= Ep >= grid[b]["errors"]:
                    t1, t2 = a, b
                    q = ((grid[a]["errors"] - Ep) / (grid[a]["errors"] - grid[b]["errors"])
                         if grid[a]["errors"] != grid[b]["errors"] else 0.0)
                    break
        mix = lambda k: (1 - q) * grid[t1][k] + q * grid[t2][k]
        base = {k: mix(k) for k in ("prep_us", "gate_us", "feedback_us", "gap_decode_us", "ler", "errors", "accept")}
        base["tc"] = t1 + q * (t2 - t1)
        r["tc_matched"] = dict(grid={str(t): grid[t] for t in tcs}, bracket=[t1, t2], q=q, tc_b=base["tc"],
                               baseline=base, reduction=1 - r["partial"]["prep_us"] / base["prep_us"],
                               method="randomized-threshold mixture of the two bracketing measured tc runs; "
                                      "q from error counts, same q for times", note=note)
        rec.log(f"{name}: tc_b {base['tc']:.1f}  complete {base['prep_us']:.2f}us vs partial {r['partial']['prep_us']:.2f}us "
                f"-> iso-LER reduction {r['tc_matched']['reduction']:.1%} (th-matched {r['reduction']:.1%}) {note}")
        summary_path.write_text(json.dumps(summary, indent=1))

    lines = ["| d1 | d2 | p | tc_b | complete µs @tc_b | partial µs | iso-LER reduction | th-matched reduction | errors (complete@tc_b / partial) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, r in sorted(summary.items(), key=lambda kv: (kv[1]["d1"], kv[1]["d2"], kv[1]["p"])):
        t = r.get("tc_matched")
        if t:
            lines.append(f"| {r['d1']} | {r['d2']} | {r['p']:g} | {t['tc_b']:.1f} | {t['baseline']['prep_us']:.2f} | "
                         f"{r['partial']['prep_us']:.2f} | {t['reduction']:.1%} | {r['reduction']:.1%} | "
                         f"{t['baseline']['errors']:.0f} / {r['partial']['errors']} {t['note']} |")
    (rec.RESULT / f"{summary_path.stem}_iso_ler.md").write_text("\n".join(lines) + "\n")
    rec.log("TC-MATCH DONE")


if __name__ == "__main__":
    main()
