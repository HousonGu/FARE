"""PIR on frozen Chronos-2 forecasts for a prepared retail panel: three seeds, test scoring."""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import Panel
from io_utils import HERE, write_json, verify

ROOT = HERE.parents[1]

def identity(panel,c,device):
    return dict(data=str(panel.folder),config=c,schedule=panel.schedule,
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,
            torch=torch.__version__,cuda=torch.version.cuda,device=device,
            gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='One dataset folder produced by baselines/prepare_inputs.py')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'config.json');p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=['audit','all'],default='all')
    a=p.parse_args()
    from engine import Budget, make_bank, cache_week, train, predict, score
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    lock=out/'RUNNING.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    start=time.monotonic();budget=Budget(float('inf'))
    try:
        c=json.loads(a.config.read_text(encoding='utf-8'))
        if c['context']!=56 or c['horizon']!=7 or len(c['seeds'])<2:raise ValueError('Invalid protocol')
        torch.set_num_threads(c['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        panel=Panel(a.data)
        manifest=identity(panel,c,a.device)
        if (out/'manifest.json').exists():
            if json.loads((out/'manifest.json').read_text(encoding='utf-8'))!=manifest:
                raise ValueError('Data/code/config/environment changed; use new output')
        else:write_json(out/'manifest.json',manifest)
        dev=panel.schedule['dev'];cut=len(dev)-c['tune_weeks']
        if cut<1:raise ValueError('Insufficient revision train weeks')
        write_json(out/'audit.json',dict(dataset=panel.meta['dataset'],series=len(panel.ids),
            selection_train=dev[:cut],selection_validation=dev[cut:],refit=dev,
            bank=panel.schedule['fit'],test=panel.schedule['test'],
            test_rows=sum(len(panel.rows(o))*7 for o in panel.schedule['test']),
            method='PIR-Chronos-retail-univariate',future_inputs='five deterministic calendar features',
            test_labels_used_for_training=False))
        if a.stage=='audit':status='audit_complete'
        elif verify(out,'complete.json'):status='complete_test'
        else:
            bank=make_bank(panel,out,budget)
            for o in dev+panel.schedule['test']:cache_week(panel,o,out,bank,c,a.device,budget)
            for seed in c['seeds']:
                selection=train(panel,out,c,a.device,budget,seed,'select',dev[:cut],c['epochs'],dev[cut:])
                train(panel,out,c,a.device,budget,seed,'refit',dev,selection['best_epoch'],[])
                predict(panel,out,c,a.device,budget,seed)
            score(panel,out,c,budget);status='complete_test'
        write_json(out/'status.json',dict(status=status,elapsed_seconds=time.monotonic()-start));return 0
    except Exception as e:
        write_json(out/'status.json',dict(status='failed',reason=repr(e)));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':sys.exit(main())
