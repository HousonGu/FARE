"""Unified FARE experiment entry point; optional model packages load only on execution."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
BACKBONES = ("chronos", "lightgbm", "tft", "patchtst", "sparsetsf")
COVERAGE = {
    "favorita": list(BACKBONES),
    "m5": list(BACKBONES),
    "rossmann": list(BACKBONES),
    "rohlik": list(BACKBONES),
}
STAGES = ("audit", "references", "backbone", "fusion", "score", "all")


@dataclass(frozen=True)
class Plan:
    module: str
    command: list[str]
    environment: dict[str, str]
    notes: tuple[str, ...] = ()

    def record(self):
        return {"module": self.module, "command": self.command,
                "environment": self.environment, "working_directory": str(ROOT),
                "notes": list(self.notes)}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", choices=tuple(COVERAGE))
    result.add_argument("--backbone", choices=BACKBONES, default="chronos")
    result.add_argument("--stage", choices=STAGES, default="all")
    result.add_argument("--raw-root", type=Path, default=ROOT / "data/raw",
                        help="Parent of the dataset's raw CSV directory")
    result.add_argument("--raw", type=Path,
                        help="Direct raw CSV directory override for Rossmann/Rohlik raw-data runners")
    result.add_argument("--data-root", type=Path, default=ROOT / "data/prepared",
                        help="Parent of prepared panels for PatchTST/SparseTSF")
    result.add_argument("--out", type=Path, help="Output root; defaults to results/DATASET/BACKBONE")
    result.add_argument("--device", default="cuda")
    result.add_argument("--list", action="store_true", help="Show the available dataset/backbone combinations")
    result.add_argument("--dry-run", action="store_true", help="Print the command without loading data or models")
    return result


def build_plan(args):
    if args.dataset not in COVERAGE:
        raise ValueError("--dataset is required unless --list is used")
    if args.backbone not in COVERAGE[args.dataset]:
        raise ValueError("Unsupported dataset/backbone combination; see docs/REPRODUCE.md.")
    raw_root = args.raw_root.expanduser().resolve()
    if args.raw is not None and (args.dataset in ("favorita", "m5") or args.backbone in ("patchtst", "sparsetsf")):
        raise ValueError("--raw is for the Rossmann/Rohlik raw-data runners. Use --raw-root for Favorita/M5 "
                         "or --data-root for prepared neural panels.")
    raw_directory = (args.raw or raw_root / args.dataset).expanduser().resolve()
    out = (args.out or ROOT / "results" / args.dataset / args.backbone).expanduser().resolve()
    environment = {"FARE_RAW_ROOT": str(raw_root), "FARE_RESULTS_ROOT": str(out)}
    common = ["--out", str(out), "--device", args.device]
    notes = []
    if args.backbone in ("patchtst", "sparsetsf"):
        if args.stage == "references":
            raise ValueError("Build references with --backbone chronos --stage references, then export a panel "
                             "with python -m exp.prepare_panel. See docs/REPRODUCE.md.")
        module = "exp.neural"
        stage = {"backbone": "train", "score": "fusion"}.get(args.stage, args.stage)
        native = ["--dataset", args.dataset, "--model", args.backbone,
                  "--data-root", str(args.data_root.expanduser().resolve()), "--stage", stage]
        notes.append("Requires a prepared panel and separately fetched upstream model source.")
    elif args.dataset in ("favorita", "m5"):
        module = "exp.retail"
        stage = {"backbone": args.backbone, "fusion": "gates"}.get(args.stage, args.stage)
        native = ["--run", "favorita_original" if args.dataset == "favorita" else "m5_full",
                  "--backbone", args.backbone, "--stage", stage]
        notes.append("The native output adds a favorita_original/ or m5_full/ directory below --out.")
    elif args.backbone == "chronos":
        module = "exp." + args.dataset
        stage = {"backbone": "chronos", "fusion": "gates"}.get(args.stage, args.stage)
        native = ["--raw", str(raw_directory), "--stage", stage]
    else:
        module = "exp." + args.dataset + "_supervised"
        stage = {"backbone": args.backbone, "fusion": "score"}.get(args.stage, args.stage)
        native = ["--raw", str(raw_directory), "--backbone", args.backbone, "--stage", stage]
        notes.append("This runner fits the development gates and scores test outputs together in its score stage.")
    return Plan(module, [sys.executable, "-m", module, *native, *common], environment, tuple(notes))


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.list:
        print(json.dumps({"datasets": COVERAGE, "stages": list(STAGES)}, indent=2))
        return 0
    try:
        plan = build_plan(args)
    except ValueError as error:
        command_parser.error(str(error))
    if args.dry_run:
        print(json.dumps(plan.record(), indent=2))
        return 0
    environment = dict(os.environ, **plan.environment)
    return subprocess.run(plan.command, cwd=ROOT, env=environment).returncode


if __name__ == "__main__":
    raise SystemExit(main())
