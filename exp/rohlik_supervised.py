"""Rohlik: independently fitted LightGBM/TFT backbones and seven fusion variants."""
from pathlib import Path
import argparse,json,os,platform,sys,time,importlib.metadata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from data_provider.rohlik_supervised import Data
from data_provider.supervised_features import CONFIG,verify
from data_provider.rohlik import KEYS
from exp.rohlik_references import references
from exp.rohlik_gates import run_gates
from exp.rohlik import json_write,save_week,verified_week,merge,metric,METHODS
H=Path(__file__).resolve().parent


def identity(data,device):
    versions={}
    for n in ['numpy','pandas','scipy','torch','lightgbm','pytorch-forecasting','lightning','pyarrow']:
        try: versions[n]=importlib.metadata.version(n)
        except importlib.metadata.PackageNotFoundError: versions[n]=None
    return dict(config=CONFIG,schedule=data.schedule,device=device,
                environment=dict(python=platform.python_version(),packages=versions))


def validate_models(out):
    for p in (out/'models').rglob('*'):
        if p.is_file() and p.suffix in ['.json','.pt','.parquet','.txt']:
            verify(p)


def reference_stage(data,out,source):
    origins=data.schedule['dev']+data.schedule['test']
    if source:
        manifest=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if manifest['schedule']!=data.source.schedule:
            raise ValueError('Reference source schedule differs')
    for o in origins:
        p=out/'references'/f'{o}.parquet'; keys=data.source.window(o)[-1]
        if p.exists(): verified_week(p,keys,['shape_retrieval','factor_retrieval']); continue
        tick=time.perf_counter()
        if source:
            f=verified_week(source/'references'/f'{o}.parquet',keys,['shape_retrieval','factor_retrieval'])
        else: f=references(data.source,o,CONFIG['reference_query_batch'])
        save_week(p,f)
        print('references',o,'seconds',time.perf_counter()-tick,flush=True)


def fit_backbone(data,out,kind,device):
    import data_provider.supervised_features as common
    validate_models(out)
    if kind=='lightgbm':
        from backbones.rohlik import lightgbm
        for seed in CONFIG['backbone_seeds']:
            common.CONFIG['seed']=seed
            folder=out/'models'/'lightgbm'/str(seed)
            # One predeclared hyperparameter search, reused across refit seeds.
            selection=out/'models'/'lightgbm'/str(CONFIG['backbone_seeds'][0])/'selection.json'
            if seed!=CONFIG['backbone_seeds'][0] and not (folder/'selection.json').exists():
                verify(selection)
                common.json_write(folder/'selection.json',json.loads(selection.read_text()))
            lightgbm(data,data.schedule,folder)
        common.CONFIG['seed']=CONFIG['backbone_seeds'][0]
    else:
        from backbones.rohlik_tft import run
        run(data,data.schedule,out/'models'/'tft',device)


def fuse_and_score(data,out,backbone='all'):
    validate_models(out)
    origins=data.schedule['dev']+data.schedule['test']
    for kind in (['lightgbm','tft'] if backbone == 'all' else [backbone]):
        for seed in CONFIG['backbone_seeds']:
            dest=out/'comparisons'/kind/str(seed); pieces=[]; inputs=[]
            for o in origins:
                refpath=out/'references'/f'{o}.parquet'
                bp=out/'models'/'lightgbm'/str(seed)/f'{o}.parquet' if kind=='lightgbm' else out/'models'/'tft'/f'{o}_seed_{seed}.parquet'
                verify(bp)
                for p in [refpath,bp]: inputs.append(str(p.relative_to(out)))
                ref=verified_week(refpath,data.source.window(o)[-1],['shape_retrieval','factor_retrieval'])
                b=pd.read_parquet(bp).rename(columns={'forecast':'backbone_forecast'})
                f=merge(ref,b)
                if o in data.schedule['dev']: f=merge(f,data.source.truth(o))
                else: f['actual']=np.nan
                pieces.append(f)
            marker=dest/'complete.json'
            if marker.exists():
                old=json.loads(marker.read_text())
                if old['inputs']!=inputs: raise ValueError('Gate inputs changed')
                for rel in old['outputs']:
                    if not (dest/rel).is_file(): raise ValueError('Gate/score artifact missing')
                continue
            pred,fit=run_gates(pd.concat(pieces,ignore_index=True),data.schedule['dev'])
            save_week(dest/'predictions.parquet',pred); json_write(dest/'fit.json',fit)
            truth=pd.concat([data.source.truth(o) for o in data.schedule['test']],ignore_index=True)
            scored=merge(truth,pred)
            if len(scored)!=283185 or scored.origin.nunique()!=26: raise ValueError('Frozen test keys changed')
            if not np.isfinite(scored.actual).all(): raise ValueError('Test targets missing; do not silently change the scored set')
            result=dict(backbone=kind,seed=seed,status='complete',overall={m:metric(scored,m) for m in METHODS},
                        weekly={o:{m:metric(f,m) for m in METHODS} for o,f in scored.groupby('origin')})
            save_week(dest/'scored_predictions.parquet',scored); json_write(dest/'metrics.json',result)
            json_write(marker,dict(inputs=inputs,outputs=sorted(p.name for p in dest.iterdir() if p.is_file() and p.name!='complete.json')))
    json_write(out/'complete.json',dict(status='complete',artifacts=sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name not in ['RUNNING.lock','complete.json','status.json'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--reference-source',type=Path)
    p.add_argument('--backbone',choices=['all','lightgbm','tft'],default='all')
    p.add_argument('--stage',choices=['audit','references','lightgbm','tft','score','all'],default='all')
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True)
    data=Data(a.raw); stamp=identity(data,a.device); stamp['backbone']=a.backbone; manifest=a.out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text())!=stamp: raise ValueError('Config/schedule/environment changed: use a new output directory')
    lock=a.out/'RUNNING.lock'
    with lock.open('x') as f: f.write(str(os.getpid()))
    try:
        json_write(manifest,stamp); json_write(a.out/'audit.json',dict(**data.source.audit,backbone_schedule=data.schedule))
        if (a.out/'complete.json').exists():
            for rel in json.loads((a.out/'complete.json').read_text())['artifacts']:
                if not (a.out/rel).is_file(): raise ValueError('Completed output missing: '+rel)
            return
        with threadpool_limits(limits=CONFIG['cpu_threads']):
            stages=['references','lightgbm','tft','score'] if a.stage=='all' else [a.stage]
            if a.backbone != 'all': stages=[s for s in stages if s not in ['lightgbm','tft'] or s == a.backbone]
            for stage in stages:
                print('START',stage,flush=True)
                if stage=='references': reference_stage(data,a.out,a.reference_source)
                elif stage in ['lightgbm','tft']: fit_backbone(data,a.out,stage,a.device)
                elif stage=='score': fuse_and_score(data,a.out,a.backbone)
        json_write(a.out/'status.json',dict(stage=a.stage,status='complete' if (a.out/'complete.json').exists() else 'stage_complete_full_pending'))
    finally: lock.unlink(missing_ok=True)


if __name__=='__main__': main()
