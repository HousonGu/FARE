"""Run one measured stage of one method at one test origin.

FARE, TimeFuse and RAF stages at the first/middle/last origin are in runtime.py;
the one-time (development) and per-origin stages of the cost table are in
runtime_extra.py.
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {
    'fare': ('reference', 'fusion', 'dev_reference'), 'timefuse': ('reference', 'fusion'),
    'raf': ('raf', 'raf_plan', 'raf_test'), 'backbone': ('backbone', 'backbone_dev'),
    'residual': ('fit', 'test'), 'pir': ('fit', 'test'), 'pfrp': ('fit', 'test'), 'saraf': ('fit', 'test')}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--method', choices=list(ALLOWED), required=True)
    parser.add_argument('--dataset', choices=['favorita', 'm5', 'rossmann', 'rohlik'], required=True)
    parser.add_argument('--phase', choices=sorted({p for v in ALLOWED.values() for p in v}), default='fusion')
    parser.add_argument('--position', choices=['first', 'middle', 'last'], default='first')
    parser.add_argument('--check', action='store_true', help='print the dispatched functions and exit')
    parser.add_argument('--inputs', type=Path, default=ROOT / 'data/baselines',
                        help='parent of the shared input folders from baselines/prepare_inputs.py')
    parser.add_argument('--model-path', type=Path, default=ROOT / 'models/chronos-2')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default=os.environ.get('EFFICIENCY_DEVICE', 'cuda'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--smoke-queries', type=int, help='limit the query series (not a full measurement)')
    parser.add_argument('--fit-dir', type=Path, help='fit/raf_plan output of the same repetition (test phases)')
    parser.add_argument('--threads', type=int, default=int(os.environ.get('EFFICIENCY_THREADS', 8)))
    args = parser.parse_args()
    if args.phase not in ALLOWED[args.method]:
        parser.error(f'{args.method} requires --phase from {ALLOWED[args.method]}')
    for name, value in [('EFFICIENCY_METHOD', args.method), ('EFFICIENCY_DATASET', args.dataset)]:
        if name in os.environ and os.environ[name].lower() != value:
            parser.error(name + ' differs from the requested run')
        os.environ[name] = value
    os.environ['EFFICIENCY_INPUTS'] = str(args.inputs.resolve())
    extra = args.method in ('backbone', 'residual', 'pir', 'pfrp', 'saraf') or \
        args.phase in ('dev_reference', 'raf_plan', 'raf_test')
    if extra:
        import runtime_extra as stages
    else:
        import runtime as stages
    if args.check:
        print(json.dumps(stages.source_functions(), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output is required for a measurement run')
    if args.smoke_queries is not None and args.smoke_queries < 1:
        parser.error('--smoke-queries must be positive')
    stages.run(args)


if __name__ == '__main__':
    main()
