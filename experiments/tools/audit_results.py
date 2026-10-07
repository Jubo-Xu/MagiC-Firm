#!/usr/bin/env python3
"""One-time migration for estimator result JSONs written before fingerprints existed:
a result is stamped with the CURRENT mask/DEM fingerprint only if it was written after
every input it depends on (complete DEM, mask, contracted DEM, MB characterization);
otherwise it is reported and left unstamped (every consumer then re-runs it).

    python experiments/tools/audit_results.py [--apply]
"""
import argparse, json, os, pathlib, sys
sys.path.insert(0, next(str(p / "lib") for p in pathlib.Path(__file__).resolve().parents
                        if (p / "lib" / "paths.py").is_file()))
import paths; paths.setup_imports()                                         # noqa: E402,E702
from gap_table import mask_fingerprint, result_fingerprint
RESULT = paths.RESULT / "runtime_estimation"; CIRCUITS = paths.CIRCUITS
MB = paths.RESULT / "mb_characterization"

ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--apply", action="store_true"); args = ap.parse_args()
ok = stale = done = 0
for f in sorted(RESULT.glob("end2end_*/*.json")):
    j = json.loads(f.read_text()); cfg = j.get("config", {})
    if "inputs" in cfg:
        done += 1; continue
    name = f.parent.name; partial = cfg.get("partial"); folder = CIRCUITS / name
    deps = [folder / "desaturated_with_obs.dem"]
    if partial: deps += [folder / partial / "mask_bool.npy", folder / partial / "contracted.dem"]
    key = f"{name}__{partial or 'complete'}"
    lat = cfg.get("latency", {}); ftag = f"{lat.get('mb_freq_hz', 0) / 1e6:g}MHz"
    deps += list(MB.glob(f"{key}__s{lat.get('mb_seed', 0)}_n{lat.get('mb_shots', 1000)}_{ftag}_realsched.json"))
    if partial: deps += list(MB.glob(f"{name}__complete__s{lat.get('mb_seed', 0)}_n{lat.get('mb_shots', 1000)}_{ftag}_realsched.json"))
    cl = cfg.get("estimator", {}).get("control_latency")
    if cl: deps.append(pathlib.Path(cl))
    newest = max(os.path.getmtime(d) for d in deps)
    if os.path.getmtime(f) >= newest:
        ok += 1
        if args.apply:
            cfg["inputs"] = result_fingerprint(folder, partial, mb_shots=lat.get("mb_shots", 1000), mb_seed=lat.get("mb_seed", 0),
                                               mb_freq_hz=lat.get("mb_freq_hz", 0), control_latency=cl)
            cfg["mask_sha"] = cfg["inputs"]["mask_sha"]
            cfg["fingerprint_source"] = "audit_results.py: mtime audit (written after DEM, mask, MB characterization, control profile)"
            j["config"] = cfg; f.write_text(json.dumps(j, indent=1))
    else:
        stale += 1; print("STALE", f.relative_to(RESULT))
print(f"{done} already stamped, {ok} pass the mtime audit{' (stamped)' if args.apply else ''}, {stale} stale")
