"""Rossmann Chronos-2 runner: references, backbone forecasts, gates, and scoring."""
from pathlib import Path
import argparse,json,os,sys,time,platform,importlib.metadata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data_provider.rossmann import RossmannData,KEYS
from exp.rossmann_references import references
from exp.rossmann_gates import run_gates
HERE=Path(__file__).resolve().parent
METHODS=['backbone','shape_half','factor_half','global_static','history_hard','soft_history','soft_no_history','flat_softmax']

def json_write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8'); os.replace(tmp,path)

def parquet_write(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix('.tmp')
    frame.to_parquet(tmp,index=False); os.replace(tmp,path)

def merge(left,right):
    result=left.merge(right,on=KEYS,how='outer',validate='one_to_one',indicator=True)
    if not result._merge.eq('both').all(): raise ValueError('Prediction keys differ')
    return result.drop(columns='_merge')

def verified_week(path,expected,columns):
    frame=pd.read_parquet(path); merge(expected,frame)
    if frame.duplicated(KEYS).any() or not np.isfinite(frame[columns].to_numpy(float)).all(): raise ValueError('Invalid weekly cache')
    return frame

def save_week(path,frame):
    parquet_write(path,frame)

from backbones.rossmann_chronos import chronos_week


def metric(frame,col):
    y=frame.actual.to_numpy(float); p=frame[col].to_numpy(float)
    return dict(rows=len(y),actual_sum=float(y.sum()),wape=float(abs(y-p).sum()/y.sum()) if y.sum()>0 else None,mae=float(abs(y-p).mean()) if len(y) else None)

def environment():
    versions={}
    for name in ['numpy','pandas','scipy','pyarrow','torch','chronos-forecasting']:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    return dict(python=platform.python_version(),platform=platform.platform(),packages=versions)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True); parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--stage',choices=['audit','references','chronos','gates','score','all'],required=True)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--config',type=Path,default=HERE.parent/'configs/rossmann.json')
    args=parser.parse_args(); data=RossmannData(args.raw); config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    expected=json.loads((HERE.parent/'configs/rossmann.json').read_text(encoding='utf-8-sig'))
    if config!=expected: raise ValueError('Protocol change requires a new version of the code and output directory')
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    identity=dict(config=config,schedule=data.schedule,environment=environment(),device=args.device)
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity: raise ValueError('Config/schedule/environment/device changed: use a new output directory')
    with (out/'RUNNING.lock').open('x') as f: f.write(str(os.getpid()))
    started=time.perf_counter()
    try:
        if (out/'complete.json').exists():
            for rel in json.loads((out/'complete.json').read_text(encoding='utf-8'))['artifacts']:
                if not (out/rel).is_file(): raise ValueError('Completed artifact missing: '+rel)
            return
        json_write(manifest,identity); json_write(out/'audit.json',data.audit)
        if args.stage=='audit': return
        origins=data.schedule['dev']+data.schedule['test']
        model=None
        stages=['references','chronos','gates','score'] if args.stage=='all' else [args.stage]
        with threadpool_limits(limits=config['cpu_threads']):
            for stage in stages:
                if stage in ['references','chronos']:
                    for origin in origins:
                        path=out/stage/(origin+'.parquet'); keys=data.window(origin)[-1]
                        cols=['shape_retrieval','factor_retrieval'] if stage=='references' else ['backbone_forecast']
                        if path.exists(): verified_week(path,keys,cols); continue
                        if stage=='references': frame=references(data,origin,config['reference_batch'])
                        else:
                            if model is None:
                                if importlib.metadata.version('chronos-forecasting')!=config['chronos_package']: raise ValueError('Requires chronos-forecasting=='+config['chronos_package'])
                                import torch
                                from chronos import Chronos2Pipeline
                                if args.device.startswith('cuda'):
                                    # Force a real kernel before loading the model, catching incompatible GPU builds.
                                    (torch.ones(8,device=args.device)*2).sum().item()
                                model=Chronos2Pipeline.from_pretrained(config['chronos_model'],device_map=args.device)
                            frame=chronos_week(data,origin,model,config)
                        save_week(path,frame); verified_week(path,keys,cols)
                        print(stage,origin,len(frame),flush=True)
                elif stage=='gates':
                    pieces=[]
                    for origin in origins:
                        keys=data.window(origin)[-1]
                        ref=verified_week(out/'references'/(origin+'.parquet'),keys,['shape_retrieval','factor_retrieval'])
                        b=verified_week(out/'chronos'/(origin+'.parquet'),keys,['backbone_forecast'])
                        frame=merge(ref,b)
                        if origin in data.schedule['dev']: frame=merge(frame,data.truth(origin)[KEYS+['actual']])
                        else: frame['actual']=np.nan
                        pieces.append(frame)
                    pred,fit=run_gates(pd.concat(pieces,ignore_index=True),data.schedule['dev'])
                    save_week(out/'predictions.parquet',pred); json_write(out/'fit.json',fit)
                elif stage=='score':
                    truths=pd.concat([data.truth(o) for o in data.schedule['test']],ignore_index=True)
                    pred=verified_week(out/'predictions.parquet',truths[KEYS],METHODS)
                    frame=merge(truths,pred); valid=frame.actual.notna(); scored=frame[valid]
                    coverage=[dict(origin=o,planned_rows=len(p),scored_rows=int(p.actual.notna().sum()),missing_target_rows=int(p.actual.isna().sum())) for o,p in frame.groupby('origin')]
                    result=dict(status='complete',test_weeks=26,coverage=coverage,overall={m:metric(scored,m) for m in METHODS},business_days={m:metric(scored[scored.open.eq(1)],m) for m in METHODS},weekly={o:{m:metric(p[p.actual.notna()],m) for m in METHODS} for o,p in frame.groupby('origin')})
                    if set(frame.origin)!=set(data.schedule['test']): raise ValueError('Incomplete test origins')
                    parquet_write(out/'scored_predictions.parquet',scored); json_write(out/'metrics.json',result)
                    artifacts=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['RUNNING.lock','complete.json','status.json'])
                    json_write(out/'complete.json',dict(status='complete',artifacts=artifacts))
        json_write(out/'status.json',dict(status='complete' if (out/'complete.json').exists() else 'incomplete',last_stage=args.stage,elapsed_seconds=time.perf_counter()-started))
    finally: (out/'RUNNING.lock').unlink(missing_ok=True)

if __name__=='__main__': main()
