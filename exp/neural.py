"""PatchTST/SparseTSF runner: backbone training and FARE fusion on prepared panels."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
import argparse
import importlib.metadata
import json
import platform
import sys
import time
import torch
from threadpoolctl import threadpool_limits
from data_provider.panel import Panel
from exp.io import HERE, write_json, verify
from backbones.neural import configure, run_backbone
from exp.neural_fusion import run_fusion


def environment(device):
    return dict(python=platform.python_version(), device=device, torch=torch.__version__,
        packages={n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scipy', 'pyarrow', 'threadpoolctl']},
        gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None,
        cuda_runtime=torch.version.cuda if device.startswith('cuda') else None)


def identity(panel, config, device):
    return dict(config=config, data=str(panel.folder), schedule=panel.schedule, environment=environment(device))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=HERE.parent/'data/prepared')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dataset', choices=['all', 'favorita', 'm5', 'rossmann', 'rohlik'], default='all')
    parser.add_argument('--model', choices=['all', 'patchtst', 'sparsetsf'], default='all')
    parser.add_argument('--stage', choices=['audit', 'train', 'fusion', 'all'], default='all')
    return parser.parse_args()


def main():
    args = arguments()
    config = json.loads((HERE.parent/'configs/neural.json').read_text(encoding='utf-8'))
    configure(config, args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    datasets = config['datasets'] if args.dataset == 'all' else [args.dataset]
    kinds = config['backbones'] if args.model == 'all' else [args.model]
    for dataset in datasets:
        panel = Panel(args.data_root/dataset)
        destination = args.out/dataset
        destination.mkdir(parents=True, exist_ok=True)
        manifest = destination/'manifest.json'
        stamp = identity(panel, config, args.device)
        if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != stamp:
            raise ValueError('Code/data/config/environment changed; use a new output directory')
        lock = destination/'RUNNING.lock'
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
        try:
            write_json(manifest, stamp)
            write_json(destination/'status.json', dict(status='running', stage=args.stage))
            with threadpool_limits(limits=config['cpu_threads']):
                if not (destination/'audit.json').exists():
                    write_json(destination/'audit.json', panel.audit())
                if args.stage in ['all', 'train']:
                    for kind in kinds:
                        run_backbone(panel, kind, destination/'models'/kind, args.device, config)
                if args.stage in ['all', 'fusion']:
                    for kind in kinds:
                        for seed in config['seeds']:
                            run_fusion(panel, destination/'models'/kind/str(seed), destination/'fusion'/kind/str(seed), kind, seed, config)
            write_json(destination/'status.json', dict(status='stage_complete', stage=args.stage))
        except BaseException as error:
            write_json(destination/'status.json', dict(status='failed_or_interrupted', stage=args.stage,
                       error=type(error).__name__+': '+str(error)))
            raise
        finally:
            lock.unlink(missing_ok=True)
        print('finished requested stage', dataset, args.stage, flush=True)


if __name__ == '__main__':
    main()
