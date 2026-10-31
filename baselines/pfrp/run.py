"""PFRP and its DLinear backbone on a prepared retail panel: three seeds, test scoring."""
import argparse,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from author import author_api,AUTHOR
from data import Panel
from io_utils import HERE,write_json
from engine import Budget,run_seed,score

ROOT=HERE.parents[1]

def run_identity(panel,config,device):
    return dict(data=str(panel.folder),config=config,
        environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,cuda=torch.version.cuda,device=device,gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None),
        schedule=panel.schedule,memory_origins=panel.schedule['train'][:len(panel.schedule['train'])//2],
        fusion_origins=panel.schedule['train'][len(panel.schedule['train'])//2:],
        adaptations=['Global shared univariate retail model, weekly windows 56/7','Chronological disjoint memory/fusion within fit','Author PCL, retrieval and DLinear fusion architecture retained','Author cosine alternate K-medoids evaluated in blocks; deterministic stable initialization ties','No future covariates for PFRP/DLinear','Tune selects epoch only; encoder last epoch; three fixed seeds; no test selection'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='One dataset folder produced by baselines/prepare_inputs.py')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=['audit','all'],default='all')
    a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True);lock=out/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time();budget=Budget(float('inf'))
    try:
        c=json.loads(a.config.read_text(encoding='utf-8'));torch.set_num_threads(c['cpu_threads'])
        if c['context']!=56 or c['horizon']!=7 or c['batch_size']<2:raise ValueError('Invalid protocol dimensions')
        from threadpoolctl import threadpool_limits
        limiter=threadpool_limits(limits=c['cpu_threads'])
        panel=Panel(a.data)
        identity=run_identity(panel,c,a.device);manifest=out/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8'))!=identity:raise ValueError('Data/code/config/environment changed: use a new output directory')
        else:write_json(manifest,identity)
        write_json(out/'audit.json',dict(dataset=panel.meta['dataset'],series=len(panel.ids),memory_origins=identity['memory_origins'],fusion_origins=identity['fusion_origins'],coverage={s:{o:len(panel.rows(o))*7 for o in panel.schedule[s]} for s in ['train','tune','dev','test']}))
        if a.stage=='audit':status='audit_complete'
        else:
            for seed in c['seeds']:run_seed(panel,c,a.device,budget,out,seed)
            score(panel,c,out);status='complete_test'
        write_json(out/'status.json',dict(status=status,elapsed_seconds=time.time()-started))
        return 0
    except Exception as e:
        write_json(out/'status.json',dict(status='failed',reason=repr(e),elapsed_seconds=time.time()-started));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':sys.exit(main())
