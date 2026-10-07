#!/usr/bin/env python3
"""Re-select the two-stage operating point t_h* and the LER-matched baseline from the
per-shot gap tables (collect_gap_table.py) instead of the estimator's own accepted-shot
error counts — the tables have 20M+ shots, so the LER side of the comparison is exact
to a few percent and free of the winner's-curse bias of picking the smallest passing
t_h on a 1M-epoch sweep.

Per circuit in the summary:
    E_base            complete-decoder errors at t_c (table)
    t_h*              smallest t_h in [t_c, cap] with LER_pcp(t_c, t_l=0, t_h) <= (1 + rel_tol) * LER_base,
                      selected on the FIRST half of the table; every reported LER and the matching use the
                      SECOND half (held-out), so the choice of t_h* cannot bias the evaluation.
                      t_h = cap always qualifies (it is the complete rule).
                      A strict tolerance is not meaningful: the fast accept carries a false-high floor of a
                      few percent at any t_h < cap, so rel_tol is the explicit LER budget of the operating
                      point. The iso-LER reduction is insensitive to it (the matched baseline moves with it).
    partial time      estimator run at (t_c, 0, t_h*), reused if it exists, else run (--epochs, control model)
    tc-matched base   bracket t1 <= t_c_b <= t2 on the --tc-grid with LER_c(t1) >= LER_pcp >= LER_c(t2) on the
                      evaluation half. Operational definition: each PREPARATION uses t1 w.p. 1-q, t2 w.p. q
                      (rule fixed for its whole retry sequence), so LER and mean time are BOTH linear
                      mixtures with the same q: q = (L1 - LER_pcp)/(L1 - L2); T = (1-q) T1 + q T2 from the
                      estimator runs at t1, t2 (reused)
    fingerprint       the table's mask_sha must equal the current mask/DEM fingerprint (else skipped)
The summary entries are rewritten in place (previous file copied to <summary>.pre_tables.json),
with `th_from = "tables"`, the table statistics, and `tc_matched` recomputed. Then run
plot_runtime_figures.py --iso-ler as before.

    python experiments/generate/select_th_from_tables.py --cs 14,29 [--rel-tol 0.25] [--tc-grid 29 31 33 35] [--only ...]
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
REPO = paths.REPO
import campaign as rec                                                     # noqa: E402
from gap_table import rule_stats, mask_fingerprint, GAP_TABLE_DIR as GAP_OUT  # noqa: E402



def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cs", default=None, metavar="Q,F")
    ap.add_argument("--summary", default=None)
    ap.add_argument("--rel-tol", type=float, default=0.25,
                    help="LER budget of the fast accept: t_h* = smallest t_h with LER_pcp <= (1 + rel_tol) * LER_complete@tc "
                         "(default 0.25, the looseness the old 2-sigma-at-1M rule amounted to); the comparison itself is "
                         "always made at equal LER via the tc-matched baseline")
    ap.add_argument("--tc-grid", type=int, nargs="+", default=[29, 31, 33, 35])
    ap.add_argument("--epochs", type=int, default=1_000_000, help="epochs for a partial run that has to be (re)done")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true", help="select and report only; no estimator runs, no summary write")
    ap.add_argument("--report-only", action="store_true", help="only (re)write the md tables from the stored summary")
    args = ap.parse_args()
    args.cs_tag = ""
    if args.cs:
        q, fo = (int(x) for x in args.cs.split(",")); args.cs_q, args.cs_f, args.cs_tag = q, fo, f"_csq{q}f{fo}"
    summary_path = rec.RESULT / (args.summary or f"summary{args.cs_tag}.json")
    summary = json.loads(summary_path.read_text())
    if not args.dry_run and not args.report_only:
        shutil.copy(summary_path, summary_path.with_suffix(".pre_tables.json"))

    for name, r in sorted(summary.items(), key=lambda kv: (kv[1]["d1"], kv[1]["d2"], kv[1]["p"])):
        if args.report_only or (args.only and not any(p in name for p in args.only)):
            continue
        tag, tc, f = r["partial_tag"], float(r["tc"]), float(r["frequency_hz"])
        table = GAP_OUT / name / f"{tag}_s0.npz"
        if not table.exists():
            rec.log(f"skip {name}: no gap table"); continue
        z = np.load(table); gc, gp, err = z["gc"], z["gp"], z["err"]
        meta = json.loads(table.with_suffix(".meta.json").read_text())
        if meta.get("mask_sha") != mask_fingerprint(rec.CIRCUITS / name, tag):
            rec.log(f"skip {name}: gap table fingerprint missing or stale (verify_gap_table.py)"); continue
        cap = int(gp.max())
        # held-out split: SELECT t_h* on the first half, EVALUATE (LER, matching) on the second half
        h = len(gc) // 2
        A = (gc[:h], gp[:h], err[:h]); B = (gc[h:], gp[h:], err[h:])
        ler_bA = rule_stats(*A, tc=tc)[2]
        th_star = None
        for th in range(int(tc), cap + 1):
            if rule_stats(*A, tc=tc, tl=0, th=th)[2] <= (1 + args.rel_tol) * ler_bA:
                th_star = th; break
        n_b, e_b, ler_b = rule_stats(*B, tc=tc)
        n_p, e_p, ler_p = rule_stats(*B, tc=tc, tl=0, th=th_star)
        old = r.get("th_star")
        # tc-matched baseline on the evaluation half. Operational definition: each magic-state
        # PREPARATION uses threshold t1 with probability 1-q and t2 with probability q for its whole
        # retry sequence, so both the LER and the mean preparation time are linear mixtures:
        # LER_mix = (1-q) L1 + q L2,  T_mix = (1-q) T1 + q T2  (same q, same definition).
        grid_s = {t: rule_stats(*B, tc=t) for t in sorted(set(args.tc_grid) | {int(tc)})}
        tcs = sorted(grid_s); note = ""
        if ler_p > grid_s[tcs[0]][2]:
            t1 = t2 = tcs[0]; qq = 0.0; note = f"pcp LER above tc={tcs[0]} (clamped)"
        elif ler_p < grid_s[tcs[-1]][2]:
            t1 = t2 = tcs[-1]; qq = 0.0; note = f"pcp LER below tc={tcs[-1]} (clamped)"
        else:
            for a, b in zip(tcs, tcs[1:]):
                L1, L2 = grid_s[a][2], grid_s[b][2]
                if L1 >= ler_p >= L2:
                    t1, t2 = a, b; qq = (L1 - ler_p) / (L1 - L2) if L1 != L2 else 0.0; break
        rec.log(f"{name[8:36]}: table {len(gc)/1e6:.0f}M (select/eval halves)  LER_base {ler_b:.2e} ({e_b} errs)  th* {old} -> {th_star}  "
                f"LER_pcp {ler_p:.2e} ({e_p} errs, {ler_p/ler_b:.3f}x)  tc_b {t1 + qq*(t2-t1):.2f} [{t1},{t2}] {note}")
        if args.dry_run:
            continue
        # estimator runs (reused when present)
        log_path = rec.RESULT / name / "estimation.log"
        ppath = rec.expected_path(name, tag, f"pcp_tc{tc:g}_tl0_th{th_star}", args)
        if not rec.valid(ppath, name, tag, f, args, ["--mode", "pcp", "--tc", tc, "--tl", 0, "--th", th_star]):
            ppath = rec.run_estimator(name, f, args, ["--partial", tag, "--mode", "pcp", "--tc", tc, "--tl", 0, "--th", th_star], log_path)
        part = rec.summarize(ppath)
        grid = {}
        for t in (t1, t2):
            alt = argparse.Namespace(**vars(args)); alt.epochs = 10_000_000
            gpath = rec.expected_path(name, "complete", f"single_tc{t}", alt)     # 10M refinement first
            if not rec.valid(gpath, name, "complete", f, alt, ["--mode", "single", "--tc", t]):
                gpath = rec.expected_path(name, "complete", f"single_tc{t}", args)  # then the campaign epochs
                if not rec.valid(gpath, name, "complete", f, args, ["--mode", "single", "--tc", t]):
                    gpath = rec.run_estimator(name, f, args, ["--mode", "single", "--tc", t], log_path)
            grid[t] = rec.summarize(gpath)
        mix = lambda k: (1 - qq) * grid[t1][k] + qq * grid[t2][k]
        base = {k: mix(k) for k in ("prep_us", "gate_us", "feedback_us", "gap_decode_us", "accept")}
        base.update(tc=t1 + qq * (t2 - t1), ler=ler_p, errors=e_p, ler_source="table")
        r.update(th_star=th_star, partial=part, th_from="tables", th_previous=old,
                 table={"file": paths.rel(table), "shots": int(len(gc)), "split": "select on first half, evaluate on second",
                        "mask_sha": meta["mask_sha"], "rel_tol": args.rel_tol,
                        "eval_half": {"shots": int(len(gc) - h), "n_base": int(n_b), "E_base": int(e_b), "ler_base": ler_b,
                                      "n_pcp": int(n_p), "E_pcp": int(e_p), "ler_pcp": ler_p,
                                      "grid": {str(t): {"n": int(v[0]), "E": int(v[1]), "ler": v[2]} for t, v in grid_s.items()}},
                        "cap": cap},
                 reduction=1 - part["prep_us"] / r["baseline"]["prep_us"],
                 tc_matched=dict(grid={str(t): grid[t] for t in grid}, bracket=[t1, t2], q=qq, tc_b=base["tc"],
                                 baseline=base, reduction=1 - part["prep_us"] / base["prep_us"],
                                 method="per-preparation randomized threshold (t1 w.p. 1-q, t2 w.p. q): LER and time "
                                        "are both linear mixtures; q from held-out TABLE LERs, times from the "
                                        "estimator runs at t1, t2", note=note))
        summary_path.write_text(json.dumps(summary, indent=1))
        rec.log(f"  -> partial {part['prep_us']:.2f}us vs matched complete {base['prep_us']:.2f}us: "
                f"iso-LER reduction {r['tc_matched']['reduction']:.1%}")
    if not args.dry_run:
        rec.write_md(summary, args.cs_tag)
        lines = ["| d1 | d2 | p | th* (old) | table shots | LER complete@tc / pcp@th* | tc_b | complete µs @tc_b | partial µs | iso-LER reduction |",
                 "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in sorted(summary.items(), key=lambda kv: (kv[1]["d1"], kv[1]["d2"], kv[1]["p"])):
            t = r.get("tc_matched"); tb = r.get("table")
            if t and tb:
                ev = tb["eval_half"]
                lines.append(f"| {r['d1']} | {r['d2']} | {r['p']:g} | {r['th_star']} ({r.get('th_previous')}) | {tb['shots']/1e6:.0f}M | "
                             f"{ev['ler_base']:.2e} / {ev['ler_pcp']:.2e} | {t['tc_b']:.1f} | {t['baseline']['prep_us']:.2f} | "
                             f"{r['partial']['prep_us']:.2f} | {t['reduction']:.1%} {t['note']} |")
        (rec.RESULT / f"{summary_path.stem}_iso_ler.md").write_text("\n".join(lines) + "\n")
    rec.log("SELECT-TH-FROM-TABLES DONE")


if __name__ == "__main__":
    main()
