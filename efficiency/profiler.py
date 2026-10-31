"""Profiler: time and memory of named functions inside an existing pipeline run.

The pipeline code is not edited. Functions listed in a spec file are wrapped in
place before the original entry script runs, so the measured numbers come from
the same code path that produced the paper results.

Usage
-----
Check that every target resolves (cheap, no pipeline run):
    python efficiency/profiler.py --spec efficiency/specs/fare.json --check

Profile one pipeline run:
    python efficiency/profiler.py --spec efficiency/specs/fare.json --label m5_last_r1 \
        --out-dir runs -- efficiency/entry.py --method fare --dataset m5 ...

Bank-size probe (keeps a fixed, time-ordered subset of the candidate bank):
    python efficiency/profiler.py --spec efficiency/specs/fare.json --label m5_last_f0.25_r1 \
        --bank-fraction 0.25 --out-dir runs -- efficiency/entry.py ...

Limits
------
* Only the current process is instrumented. Worker processes started with
  multiprocessing "spawn" are not wrapped; run the stage single-process.
* Peak RSS from getrusage is the authoritative process peak. Per-function RSS
  comes from a 10 ms sampler and can miss very short spikes.
* Probe runs change predictions. Point the entry script at a new output
  directory; probe outputs must not be mixed with formal accuracy results.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import inspect
import json
import math
import os
import platform
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import psutil
except ImportError:  # psutil is optional; /proc is used on Linux instead.
    psutil = None


# ---------------------------------------------------------------- memory
_PROCESS = psutil.Process() if psutil is not None else None
_CUDA_DEVICE = os.environ.get("EFFICIENCY_DEVICE", "cuda")


def current_rss(children: bool = False) -> int:
    """Resident set size in bytes; children are added only when asked (slow)."""
    if _PROCESS is not None:
        total = _PROCESS.memory_info().rss
        if not children:
            return total
        for child in _PROCESS.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total
    try:
        with open("/proc/self/status", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def peak_rss_getrusage() -> int | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def torch_module():
    return sys.modules.get("torch")


def cuda_ready() -> bool:
    torch = torch_module()
    return bool(str(_CUDA_DEVICE).startswith("cuda") and torch is not None
                and torch.cuda.is_available() and torch.cuda.is_initialized())


def cuda_sync() -> None:
    if cuda_ready():
        torch_module().cuda.synchronize(_CUDA_DEVICE)


# ---------------------------------------------------------------- recorder
class Recorder:
    def __init__(self, interval: float = 0.01):
        self.stats: dict[str, dict] = {}
        self.stack: list[list] = []  # [name, start, child_seconds]
        self.lock = threading.Lock()
        self.interval = interval
        self.sampled_peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.bank_sizes: list[tuple[int, int]] = []
        self.gpu_allocated_peak = None
        self.gpu_reserved_peak = None

    def entry(self, name: str) -> dict:
        return self.stats.setdefault(name, dict(
            calls=0, inclusive=0.0, exclusive=0.0, durations=[], rss_peak=0,
            rss_entry_max=0, gpu_allocated_peak=None, gpu_reserved_peak=None,
            call_details=[]))

    def observe_gpu(self):
        if not cuda_ready():
            return None, None
        allocated = int(torch_module().cuda.max_memory_allocated(_CUDA_DEVICE))
        reserved = int(torch_module().cuda.max_memory_reserved(_CUDA_DEVICE))
        self.gpu_allocated_peak = max(self.gpu_allocated_peak or 0, allocated)
        self.gpu_reserved_peak = max(self.gpu_reserved_peak or 0, reserved)
        return allocated, reserved

    def _sample(self) -> None:
        while not self.stop.wait(self.interval):
            rss = current_rss(children=True)
            with self.lock:
                self.sampled_peak = max(self.sampled_peak, rss)
                for frame in self.stack:
                    frame[3] = max(frame[3], rss)
                for name in {frame[0] for frame in self.stack}:
                    stat = self.stats[name]
                    stat["rss_peak"] = max(stat["rss_peak"], rss)

    def start(self) -> None:
        self.sampled_peak = current_rss(children=True)
        self.thread.start()

    def finish(self) -> None:
        self.sampled_peak = max(self.sampled_peak, current_rss(children=True))
        self.stop.set()
        self.thread.join()
        self.observe_gpu()

    def wrap(self, name: str, function, before=None, keep_call_details=False):
        recorder = self

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if before is not None:
                args, kwargs = before(args, kwargs)
            outermost = not recorder.stack
            torch = torch_module()
            if outermost and cuda_ready():
                recorder.observe_gpu()
                torch.cuda.reset_peak_memory_stats(_CUDA_DEVICE)
            cuda_sync()
            rss = current_rss()
            with recorder.lock:
                stat = recorder.entry(name)
                stat["rss_entry_max"] = max(stat["rss_entry_max"], rss)
                stat["rss_peak"] = max(stat["rss_peak"], rss)
                recorder.stack.append([name, time.perf_counter(), 0.0, rss, rss])
            try:
                return function(*args, **kwargs)
            finally:
                cuda_sync()
                end = time.perf_counter()
                rss = current_rss()
                allocated, reserved = recorder.observe_gpu() if outermost else (None, None)
                with recorder.lock:
                    frame = recorder.stack.pop()
                    elapsed = end - frame[1]
                    stat = recorder.stats[name]
                    stat["calls"] += 1
                    stat["inclusive"] += elapsed
                    stat["exclusive"] += elapsed - frame[2]
                    stat["durations"].append(elapsed)
                    stat["rss_peak"] = max(stat["rss_peak"], rss)
                    if keep_call_details:
                        stat["call_details"].append(dict(
                            inclusive=elapsed, exclusive=max(0.0, elapsed-frame[2]),
                            rss_peak=max(frame[3], rss), rss_entry=frame[4],
                            gpu_allocated_peak=allocated, gpu_reserved_peak=reserved))
                    if recorder.stack:
                        recorder.stack[-1][2] += elapsed
                if outermost and cuda_ready():
                    stat["gpu_allocated_peak"] = max(stat["gpu_allocated_peak"] or 0, allocated)
                    stat["gpu_reserved_peak"] = max(stat["gpu_reserved_peak"] or 0, reserved)
        wrapper.__profiled_original__ = function
        return wrapper


# ---------------------------------------------------------------- targets
def resolve(target: str):
    """'package.module:Class.method' -> (owner, attribute, function)."""
    module_name, _, qualname = target.partition(":")
    if not qualname:
        raise ValueError(f"Target must look like module:function, got {target!r}")
    owner = importlib.import_module(module_name)
    parts = qualname.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    function = inspect.getattr_static(owner, parts[-1])
    if isinstance(function, (staticmethod, classmethod)):
        raise TypeError(f"{target}: static/class methods are not supported; wrap the caller")
    if not callable(function):
        raise TypeError(f"{target} is not callable")
    return owner, parts[-1], function


def module_callables(module_name: str) -> list[str]:
    module = importlib.import_module(module_name)
    lines = []
    for name, value in sorted(vars(module).items()):
        if inspect.isfunction(value) and value.__module__ == module.__name__:
            lines.append(f"  {module_name}:{name}{inspect.signature(value)}")
        elif inspect.isclass(value) and value.__module__ == module.__name__:
            for method, member in sorted(vars(value).items()):
                if inspect.isfunction(member):
                    lines.append(f"  {module_name}:{name}.{method}{inspect.signature(member)}")
    return lines


def make_bank_hook(recorder: Recorder, params: list[str], signature: inspect.Signature,
                   fraction: float, seed: int, axis=0, min_keep=1):

    def hook(args, kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        values = {name: bound.arguments[name] for name in params if name in bound.arguments}
        if len(values) != len(params):
            missing = [name for name in params if name not in values]
            raise RuntimeError(f"Bank parameter(s) not passed in this call: {missing}")
        lengths = {name: np.shape(value)[axis] for name, value in values.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"Bank parameters have different lengths: {lengths}")
        n = next(iter(lengths.values()))
        keep = n if fraction >= 1 else min(n, max(min_keep, math.ceil(fraction * n)))
        recorder.bank_sizes.append((n, keep))
        if keep == n:
            return args, kwargs
        # Identical ordered banks receive identical subsets across query batches.
        # Taking a permutation prefix also nests 1/8 inside 1/4 inside 1/2.
        rng = np.random.default_rng([seed, n])
        index = np.sort(rng.permutation(n)[:keep])
        for name, value in values.items():
            if hasattr(value, "iloc"):
                bound.arguments[name] = value.iloc[index] if axis == 0 else value.iloc[:, index]
            else:
                bound.arguments[name] = np.take(np.asarray(value), index, axis=axis)
        return bound.args, bound.kwargs

    return hook


def install(spec: dict, recorder: Recorder, fraction: float, seed: int) -> list[str]:
    originals, wrappers = {}, {}
    bank = spec.get("bank")
    for item in spec["targets"]:
        owner, attribute, function = resolve(item["fn"])
        before = None
        if bank and bank["fn"] == item["fn"]:
            signature = inspect.signature(function)
            unknown = [name for name in bank["params"] if name not in signature.parameters]
            if unknown:
                raise SystemExit(f"Bank parameter(s) {unknown} not in {item['fn']}{signature}")
            before = make_bank_hook(recorder, bank["params"], signature, fraction, seed,
                                    axis=bank.get("axis", 0), min_keep=bank.get("min_keep", 1))
        wrapper = recorder.wrap(item["name"], function, before,
                                keep_call_details=bool(item.get("split_by_call_order")))
        setattr(owner, attribute, wrapper)
        originals[id(function)] = function
        wrappers[id(function)] = wrapper
    if bank and bank["fn"] not in {item["fn"] for item in spec["targets"]}:
        raise SystemExit("spec.bank.fn must also be listed in spec.targets")
    if fraction < 1 and not bank:
        raise SystemExit("--bank-fraction needs a 'bank' entry in the spec")
    # Rebind names already imported elsewhere with `from module import function`.
    rebound = []
    for module_name, module in list(sys.modules.items()):
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        for key, value in list(namespace.items()):
            if id(value) in originals and value is originals[id(value)]:
                namespace[key] = wrappers[id(value)]
                rebound.append(f"{module_name}.{key}")
    return rebound


def check_bank(spec):
    bank = spec.get("bank")
    if not bank:
        return
    if bank["fn"] not in {item["fn"] for item in spec["targets"]}:
        raise ValueError("spec.bank.fn must also be listed in spec.targets")
    _, _, function = resolve(bank["fn"])
    signature = inspect.signature(function)
    unknown = [name for name in bank["params"] if name not in signature.parameters]
    if unknown:
        raise ValueError(f"Bank parameter(s) {unknown} not in {bank['fn']}{signature}")
    if not bank["params"]:
        raise ValueError("Bank params must not be empty")
    if not isinstance(bank.get("axis", 0), int):
        raise ValueError("Bank axis must be an integer")
    minimum = bank.get("min_keep", 1)
    if not isinstance(minimum, int) or minimum < 1:
        raise ValueError("Bank min_keep must be a positive integer")


# ---------------------------------------------------------------- environment
def environment(root: Path) -> dict:
    cpu = platform.processor()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    info = dict(
        python=platform.python_version(), platform=platform.platform(), cpu=cpu,
        logical_cpus=os.cpu_count(),
        usable_cpus=len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        threads={key: os.environ.get(key) for key in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "CUDA_VISIBLE_DEVICES")},
        numpy=np.__version__,
        total_ram=psutil.virtual_memory().total if psutil is not None else None,
    )
    for package in ("pandas", "torch", "scipy", "sklearn", "lightgbm"):
        module = sys.modules.get(package)
        if module is not None:
            info[package] = getattr(module, "__version__", None)
    torch = torch_module()
    if torch is not None:
        info["torch_threads"] = torch.get_num_threads()
    if torch is not None and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(_CUDA_DEVICE if str(_CUDA_DEVICE).startswith("cuda") else 0)
        info["cuda"] = torch.version.cuda
    info["measurement_device"] = _CUDA_DEVICE
    return info


def split_calls(stats: dict, spec: dict) -> dict:
    """Attribute ordered calls to phases, e.g. design -> design@dev, design@test.

    Only applied when the call count equals the number of labels; otherwise the
    target stays combined and the mismatch is reported.
    """
    notes = {}
    for item in spec["targets"]:
        labels = item.get("split_by_call_order")
        stat = stats.get(item["name"])
        if not labels or stat is None:
            continue
        if stat["calls"] != len(labels):
            notes[item["name"]] = f"not split: {stat['calls']} calls, {len(labels)} labels"
            continue
        details = stat.get("call_details", [])
        for index, (label, seconds) in enumerate(zip(labels, stat["durations"])):
            detail = details[index] if len(details) == len(labels) else {}
            stats[f"{item['name']}@{label}"] = dict(
                calls=1, inclusive=seconds, exclusive=detail.get("exclusive"), durations=[seconds],
                rss_peak=detail.get("rss_peak"), rss_entry_max=detail.get("rss_entry"),
                gpu_allocated_peak=detail.get("gpu_allocated_peak"),
                gpu_reserved_peak=detail.get("gpu_reserved_peak"), split_from=item["name"])
        notes[item["name"]] = "split by verified call order; per-call memory only, never copied from parent"
    return notes


def summarize(stat: dict) -> dict:
    durations = np.asarray(stat.pop("durations"), dtype=float)
    stat.pop("call_details", None)
    stat["inclusive"] = round(stat["inclusive"], 6)
    if stat["exclusive"] is not None:
        stat["exclusive"] = round(stat["exclusive"], 6)
    if len(durations):
        stat["per_call_p50"] = float(np.quantile(durations, 0.5))
        stat["per_call_p95"] = float(np.quantile(durations, 0.95))
        stat["per_call_max"] = float(durations.max())
    return stat


# ---------------------------------------------------------------- main
def main() -> None:
    global _CUDA_DEVICE
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--root", type=Path,
                        help="code root; overrides spec 'root' (local and server layouts differ)")
    parser.add_argument("--check", action="store_true", help="resolve targets and list signatures")
    parser.add_argument("--label", help="run name, e.g. m5_last_r1")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--bank-fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--cuda-device", help="CUDA device to synchronize/measure, e.g. cuda:1; defaults to device metadata or EFFICIENCY_DEVICE")
    parser.add_argument("--meta", action="append", default=[],
                        help="key=value stored with the run, e.g. dataset=m5 position=last")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- entry_script.py [args] or -- -m package.module [args]")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    root = (args.root or args.spec.parent / spec.get("root", ".")).resolve()
    for entry in spec.get("sys_path", []):
        sys.path.insert(0, str((root / entry).resolve()))
    if spec.get("chdir_root"):
        os.chdir(root)

    if args.check:
        failed = False
        for item in spec["targets"]:
            try:
                _, _, function = resolve(item["fn"])
                print(f"OK  {item['name']:<24} {item['fn']}{inspect.signature(function)}")
            except Exception as error:  # report every target, then fail
                failed = True
                print(f"ERR {item['name']:<24} {item['fn']}: {type(error).__name__}: {error}")
                module_name = item["fn"].partition(":")[0]
                try:
                    print("    callables in module:")
                    print("\n".join(module_callables(module_name)) or "    (none)")
                except Exception as inner:
                    print(f"    cannot import {module_name}: {inner}")
        try:
            check_bank(spec)
        except Exception as error:
            failed = True
            print(f"ERR bank: {type(error).__name__}: {error}")
        raise SystemExit(1 if failed else 0)

    if not args.label:
        parser.error("--label is required for a profiling run")
    if not 0 < args.bank_fraction <= 1:
        parser.error("--bank-fraction must be in (0, 1]")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("give the entry command after --")
    destination = (args.out_dir / args.label).resolve()
    if destination.exists():
        parser.error(f"{destination} exists; choose a new --label")

    if any("=" not in item or not item.split("=", 1)[0] for item in args.meta):
        parser.error("--meta values must be nonempty-key=value")
    meta = dict(item.split("=", 1) for item in args.meta)
    command_device = None
    for index, value in enumerate(command):
        if value == "--device" and index+1 < len(command):
            command_device = command[index+1]
        elif value.startswith("--device="):
            command_device = value.split("=", 1)[1]
    _CUDA_DEVICE = args.cuda_device or meta.get("device") or os.environ.get("EFFICIENCY_DEVICE") or command_device or "cuda"
    check_bank(spec)
    recorder = Recorder()
    rss_start = current_rss(children=True)
    rebound = install(spec, recorder, args.bank_fraction, args.seed)
    recorder.start()
    started = time.perf_counter()
    status, error = "complete", None
    exit_code = 0
    saved_argv = sys.argv
    try:
        if command[0] == "-m":
            sys.argv = [command[1], *command[2:]]
            runpy.run_module(command[1], run_name="__main__", alter_sys=True)
        else:
            sys.argv = command
            runpy.run_path(command[0], run_name="__main__")
    except SystemExit as stop:
        if stop.code not in (None, 0):
            status, error = "failed", f"SystemExit({stop.code})"
            exit_code = stop.code if isinstance(stop.code, int) else 1
    except BaseException as caught:  # keep partial measurements, then re-raise
        status, error = "failed", f"{type(caught).__name__}: {caught}"
        raise
    finally:
        sys.argv = saved_argv
        cuda_sync()
        wall = time.perf_counter() - started
        recorder.finish()
        gpu = None if recorder.gpu_allocated_peak is None else dict(
            note="maximum observed across profiler windows and before profiler resets; pipeline-internal resets may limit coverage",
            allocated_peak=recorder.gpu_allocated_peak,
            reserved_peak=recorder.gpu_reserved_peak)
        sizes = np.asarray(recorder.bank_sizes, dtype=float).reshape(-1, 2)
        if "query_count" not in meta and os.environ.get("EFFICIENCY_QUERY_COUNT"):
            meta["query_count"] = os.environ["EFFICIENCY_QUERY_COUNT"]
        split_notes = split_calls(recorder.stats, spec)
        result = dict(
            label=args.label, status=status, error=error, meta=meta,
            spec=str(args.spec.resolve()), method=spec.get("method"),
            spec_snapshot=spec,
            command=command, bank_fraction=args.bank_fraction, seed=args.seed,
            wall_seconds=round(wall, 6), rss_at_start=rss_start,
            rss_peak_sampled=recorder.sampled_peak, rss_peak_getrusage=peak_rss_getrusage(),
            gpu_last_window=gpu, rebound_names=rebound, split_notes=split_notes,
            gpu_peak_observed=gpu,
            uncalled_targets=[item["name"] for item in spec["targets"] if item["name"] not in recorder.stats],
            code_root=str(root),
            bank=None if not len(sizes) else dict(
                calls=int(len(sizes)), full_median=float(np.median(sizes[:, 0])),
                full_max=int(sizes[:, 0].max()), used_median=float(np.median(sizes[:, 1])),
                used_max=int(sizes[:, 1].max()), weighting="one observation per bank-hook call"),
            targets={name: summarize(dict(stat)) for name, stat in recorder.stats.items()},
            environment=environment(root),
        )
        destination.mkdir(parents=True)
        (destination / "profile.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[profile] {status}: {destination / 'profile.json'}", flush=True)
    if exit_code:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
