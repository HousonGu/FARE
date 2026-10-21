"""Run the supplementary experiments on a server; no manuscript edits."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from data_provider.retail import (HERE, ROOT, RESULT_ROOT, CONFIG, KEYS, RetailData, schedules, digest,
                    json_write, checked_merge, metric, check_prediction, RAW_ROOT)

STAGES = ["references", "chronos", "lightgbm", "tft", "gates", "score"]


def fingerprint(schedule, backbone="all"):
    # Bind resume to all executable project dependencies, not just the CLI.
    from exp.dependencies import dependency_files
    paths = dependency_files() + [ROOT/"configs/retail.json", ROOT/"configs/favorita_origins.json"]
    sources = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(paths))}
    return dict(config=CONFIG, schedule=schedule, sources=sources, backbone=backbone)


def audit(data, schedule, folder):
    files = (["sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv"] if data.dataset == "m5"
             else ["train.csv", "stores.csv", "holidays_events.csv"])
    raw = RAW_ROOT/data.dataset
    if data.dataset == "m5":
        if len(data.ids)!=30490 or len(data.days)!=1941 or data.days[0]!="2011-01-29" or data.days[-1]!="2016-05-22":
            raise ValueError("M5 must contain all 30,490 series and 1,941 labelled dates")
    provenance = {name: digest(raw/name) for name in files}
    versions = {}
    for name in ("numpy", "pandas", "scipy", "pyarrow", "lightgbm", "torch", "chronos-forecasting", "pytorch-forecasting", "lightning", "scikit-learn", "statsmodels"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    path = folder/"audit.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old["raw_sha256"] != provenance:
            raise ValueError("Raw data changed; choose a new run directory")
        if old["packages"] != versions or old["python"] != platform.python_version():
            raise ValueError("Runtime packages changed; restore environment or choose a new run directory")
        # Imported V3 audit lacks the final four origins; extend counts below.
    counts = {}
    for split in ("fit", "dev", "test"):
        for origin in schedule[split]:
            data.indices(origin)  # all 56 history + 7 target dates must exist
            counts[origin] = len(data.eligible(origin))
            if counts[origin] == 0:
                raise ValueError(f"No eligible series {origin}")
    json_write(path, dict(raw_sha256=provenance, series_by_origin=counts,
        raw_days=len(data.days), raw_series=len(data.ids), raw_start=data.days[0], raw_end=data.days[-1],
        python=platform.python_version(), platform=platform.platform(),
        packages=versions))


from exp.retail_scoring import score, families


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", choices=["all", "chronos", "lightgbm", "tft"], default="all")
    p.add_argument("--run", choices=["all"]+list(schedules()), default="m5_full")
    p.add_argument("--stage", choices=["all", "audit"]+STAGES, default="all")
    p.add_argument("--out", type=Path, default=RESULT_ROOT/"supplementary")
    p.add_argument("--device", default="cuda")
    p.add_argument("--chronos-model", type=Path)
    p.add_argument("--list", action="store_true")
    args = p.parse_args()
    if args.list:
        print(json.dumps(schedules(), ensure_ascii=False, indent=2)); return
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out/"RUNNING.lock"
    # Exclusive lock prevents two tmux invocations from corrupting shared state.
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run directory is locked: {lock}; verify no runner is alive before removing a stale lock")
    with os.fdopen(fd, "w") as f:
        f.write(str(os.getpid()))
    try:
        chosen = schedules()
        if args.run != "all":
            chosen = {args.run: chosen[args.run]}
        for name, schedule in chosen.items():
            folder = args.out/name
            folder.mkdir(parents=True, exist_ok=True)
            manifest = folder/"manifest.json"
            expected = fingerprint(schedule, args.backbone)
            if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")) != expected:
                raise ValueError("Code/config/schedule differs from previous run; use a new --out directory")
            if not manifest.exists():
                json_write(manifest, expected)
            data = RetailData(schedule["dataset"])
            audit(data, schedule, folder)
            stages = STAGES if args.stage == "all" else ([] if args.stage == "audit" else [args.stage])
            if args.backbone != "all":
                stages = [stage for stage in stages if stage not in ["chronos", "lightgbm", "tft"] or stage == args.backbone]
            for stage in stages:
                complete = folder/(stage+".complete.json")
                if complete.exists():
                    saved = json.loads(complete.read_text())
                    for rel, sha in saved["artifacts"].items():
                        if digest(folder/rel) != sha:
                            raise ValueError(f"Completed artifact changed: {rel}")
                    print("skip complete", name, stage, flush=True)
                    continue
                started = time.time()
                json_write(folder/"status.json", dict(run=name, stage=stage, status="running"))
                print("START", name, stage, flush=True)
                if stage == "references":
                    from exp.retail_references import build
                    build(data, schedule, folder/"references")
                    artifact_roots = [folder/"references"]
                elif stage == "chronos":
                    from backbones.retail import chronos
                    chronos(data, schedule, folder/"chronos", args.device, args.chronos_model)
                    artifact_roots = [folder/"chronos"]
                elif stage == "lightgbm":
                    from backbones.retail import lightgbm
                    lightgbm(data, schedule, folder/"lightgbm")
                    artifact_roots = [folder/"lightgbm"]
                elif stage == "tft":
                    from backbones.retail_tft import run
                    run(data, schedule, folder/"tft", args.device)
                    artifact_roots = [folder/"tft"]
                elif stage == "gates":
                    from exp.retail_gates import run
                    for backbone in families(args.backbone):
                        run(data, schedule, folder, backbone)
                    artifact_roots = [folder/("gates_"+b) for b in families(args.backbone)]
                else:
                    score(data, schedule, folder, args.backbone)
                    artifact_roots = [folder/"metrics.json", folder/"实验效果.md"]
                paths = [p for r in artifact_roots for p in (r.rglob("*") if r.is_dir() else [r]) if p.is_file()]
                json_write(complete, dict(elapsed_seconds=time.time()-started,
                    artifacts={str(p.relative_to(folder)): digest(p) for p in paths}))
                json_write(folder/"status.json", dict(run=name, stage=stage, status="complete"))
            del data
            import gc
            gc.collect()
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
