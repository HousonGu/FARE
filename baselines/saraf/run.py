"""SARAF and its linear control on a prepared retail panel: three seeds, test scoring."""
import argparse,json,os,platform,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from data import Panel
from retrieval import Bank
from engine import Budget,run_seed,score
from io_utils import HERE,write_json

ROOT=HERE.parents[1]

def run_identity(panel,c,device):
    return dict(method=c['method'],config=c,data=str(panel.folder),
        schedule=panel.schedule,environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,torch=torch.__version__,cuda=torch.version.cuda,device=device,gpu=torch.cuda.get_device_name(torch.device(device)) if device.startswith('cuda') else None),
        adaptations=['Univariate same-series retrieval; shared model across eligible series','Source train bank with strict per-query completed-target mask; later half-train fits forecasting model','56/7 daily windows, 1 period, true five-field calendar marks','Scaler and stationarity fitted only to first-half-train contexts','Batched author-equivalent stochastic MMR and Gaussian aggregation','Zero residual for empty banks; min(K,available) for short banks','Tune WAPE selects epoch; no tune/dev/test targets enter retrieval bank','Independent author linear_x control'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='One dataset folder produced by baselines/prepare_inputs.py')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,default=HERE/'config.json')
    p.add_argument('--device',default='cuda')
    p.add_argument('--stage',choices=['audit','all'],default='all')
    args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=True);lock=out/'RUNNING.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.time();budget=Budget(float('inf'))
    try:
        c=json.loads(args.config.read_text(encoding='utf-8'))
        if (c['context'],c['horizon'],c['n_period'])!=(56,7,1):raise ValueError('Only the 56/7 one-period adapter is supported')
        if c['topm']<1 or c['candidate_pool']<c['topm'] or c['retrieval_batch_size']<1 or c['batch_size']<1:raise ValueError('Invalid retrieval configuration')
        if not c['seeds'] or len(set(c['seeds']))!=len(c['seeds']) or c['epochs']<1 or not 0<=c['time_aware_weight']<=1:raise ValueError('Invalid training configuration')
        torch.set_num_threads(c['cpu_threads'])
        from threadpoolctl import threadpool_limits
        limiter=threadpool_limits(limits=c['cpu_threads'])
        panel=Panel(args.data)
        identity=run_identity(panel,c,args.device);path=out/'manifest.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8'))!=identity:raise ValueError('Source/data/config/environment changed; use a new output directory')
        else:write_json(path,identity)
        write_json(out/'audit.json',dict(dataset=panel.schedule['dataset'],series=len(panel.ids),coverage={s:{o:len(panel.rows(o))*7 for o in panel.schedule[s]} for s in ['train','tune','dev','test']}))
        if args.stage=='audit':status='audit_complete'
        else:
            bank=Bank(panel,c,out,budget)
            for seed in c['seeds']:run_seed(bank,c,args.device,budget,out,seed)
            score(bank,c,out);status='complete_test'
        write_json(out/'status.json',dict(status=status,elapsed_seconds=time.time()-started));return 0
    except Exception as e:
        write_json(out/'status.json',dict(status='failed',reason=repr(e),elapsed_seconds=time.time()-started));raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':sys.exit(main())
