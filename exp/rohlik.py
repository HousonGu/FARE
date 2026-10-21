"""Resumable Rohlik pipeline. No test scoring in profile mode."""
from pathlib import Path
import argparse,json,os,sys,time,platform,importlib.metadata
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from exp.io import source_hashes
from data_provider.rohlik import RohlikData,KEYS,digest
from exp.rohlik_references import references
from exp.rohlik_gates import run_gates
HERE=Path(__file__).resolve().parent
METHODS=['backbone','shape_half','factor_half','global_static','soft_history','soft_no_history','flat_softmax']

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
    marker=path.with_suffix('.sha256.json')
    if not marker.exists() or json.loads(marker.read_text())['sha256']!=digest(path): raise ValueError('Weekly checksum missing or changed')
    return frame

def save_week(path,frame):
    parquet_write(path,frame); json_write(path.with_suffix('.sha256.json'),dict(sha256=digest(path),rows=len(frame)))

def gate_inputs(out,origins):
    return {f'{stage}/{o}.parquet':digest(out/stage/(o+'.parquet'))
            for stage in ['references','chronos'] for o in origins}

from backbones.rohlik_chronos import chronos_week


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
    parser.add_argument('--stage',choices=['audit','profile','references','chronos','gates','score','all'],required=True)
    parser.add_argument('--device',default='cuda'); parser.add_argument('--config',type=Path,default=HERE.parent/'configs/rohlik.json')
    args=parser.parse_args(); data=RohlikData(args.raw); config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    expected=json.loads((HERE.parent/'configs/rohlik.json').read_text(encoding='utf-8-sig'))
    if config!=expected: raise ValueError('Protocol change requires a new version of the code and output directory')
    out=args.out; out.mkdir(parents=True,exist_ok=True)
    identity=dict(config=config,raw=data.audit['raw_sha256'],schedule=data.schedule,code=source_hashes(),environment=environment(),device=args.device)
    manifest=out/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8'))!=identity: raise ValueError('Input/code/environment/device changed: use a new output directory')
    with (out/'RUNNING.lock').open('x') as f: f.write(str(os.getpid()))
    started=time.perf_counter()
    try:
        if (out/'complete.json').exists():
            for rel,sha in json.loads((out/'complete.json').read_text(encoding='utf-8'))['artifacts'].items():
                if digest(out/rel)!=sha: raise ValueError('Completed artifact changed: '+rel)
            if args.stage!='profile': return
        json_write(manifest,identity); json_write(out/'audit.json',data.audit)
        if args.stage=='audit': return
        if args.stage=='profile':
            origins=[data.schedule['dev'][i] for i in [0,12,25]]; base=out/'profile'
        else: origins=data.schedule['dev']+data.schedule['test']; base=out
        model=None; timings=[]
        stages=['references','chronos','gates','score'] if args.stage=='all' else (['references','chronos'] if args.stage=='profile' else [args.stage])
        with threadpool_limits(limits=config['cpu_threads']):
            for stage in stages:
                if stage in ['references','chronos']:
                    for origin in origins:
                        path=base/stage/(origin+'.parquet'); keys=data.window(origin)[-1]
                        cols=['shape_retrieval','factor_retrieval'] if stage=='references' else ['backbone_forecast']
                        if path.exists() and path.with_suffix('.sha256.json').exists(): verified_week(path,keys,cols); continue
                        tick=time.perf_counter()
                        if stage=='references': frame=references(data,origin,config['reference_batch'])
                        else:
                            if model is None:
                                if importlib.metadata.version('chronos-forecasting')!=config['chronos_package']: raise ValueError('Requires chronos-forecasting=='+config['chronos_package'])
                                import torch
                                from chronos import Chronos2Pipeline
                                if args.device.startswith('cuda'):
                                    # Force a real kernel before loading the model, catching incompatible GPU builds.
                                    (torch.ones(8,device=args.device)*2).sum().item()
                                    torch.cuda.reset_peak_memory_stats()
                                model=Chronos2Pipeline.from_pretrained(config['chronos_model'],revision=config['chronos_revision'],device_map=args.device)
                            frame=chronos_week(data,origin,model,config)
                        save_week(path,frame); verified_week(path,keys,cols)
                        record=dict(stage=stage,origin=origin,seconds=time.perf_counter()-tick,rows=len(frame),bytes=path.stat().st_size)
                        if stage=='references': record.update(max_bank_size=int(frame.bank_size.max()),fallback_series=int(frame.groupby('series_id').fallback.first().sum()))
                        try:
                            import resource
                            record['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        except ImportError: pass
                        if stage=='chronos' and args.device.startswith('cuda'): record['cuda_peak_bytes']=torch.cuda.max_memory_allocated()
                        timings.append(record); json_write(base/f'timing_{stage}_{origin}.json',record); print(record,flush=True)
                elif stage=='gates':
                    inputs=gate_inputs(out,origins)
                    marker=out/'gates.complete.json'
                    if marker.exists():
                        record=json.loads(marker.read_text(encoding='utf-8'))
                        if record['inputs']!=inputs:raise ValueError('Gate inputs changed')
                        for name,sha in record['outputs'].items():
                            if digest(out/name)!=sha:raise ValueError('Gate artifact changed: '+name)
                        continue
                    gate_started=time.perf_counter()
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
                    json_write(marker,dict(inputs=inputs,outputs={n:digest(out/n) for n in ['predictions.parquet','predictions.sha256.json','fit.json']},seconds=time.perf_counter()-gate_started))
                elif stage=='score':
                    record=json.loads((out/'gates.complete.json').read_text(encoding='utf-8'))
                    if record['inputs']!=gate_inputs(out,origins):raise ValueError('Score inputs changed after gate fit')
                    for name,sha in record['outputs'].items():
                        if digest(out/name)!=sha:raise ValueError('Gate artifact changed: '+name)
                    truths=pd.concat([data.truth(o) for o in data.schedule['test']],ignore_index=True)
                    pred=verified_week(out/'predictions.parquet',truths[KEYS],METHODS)
                    frame=merge(truths,pred); valid=frame.actual.notna(); scored=frame[valid]
                    coverage=[dict(origin=o,planned_rows=len(p),scored_rows=int(p.actual.notna().sum()),missing_target_rows=int(p.actual.isna().sum())) for o,p in frame.groupby('origin')]
                    result=dict(status='complete',test_weeks=26,coverage=coverage,overall={m:metric(scored,m) for m in METHODS},weekly={o:{m:metric(p[p.actual.notna()],m) for m in METHODS} for o,p in frame.groupby('origin')})
                    if set(frame.origin)!=set(data.schedule['test']): raise ValueError('Incomplete test origins')
                    parquet_write(out/'scored_predictions.parquet',scored); json_write(out/'metrics.json',result)
                    artifacts={str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and 'profile' not in p.parts and p.name not in ['RUNNING.lock','complete.json','status.json','budget_status.json','server.log','BUDGET.lock']}
                    json_write(out/'complete.json',dict(status='complete',artifacts=artifacts))
        if args.stage=='profile':
            records=[json.loads(p.read_text()) for p in base.glob('timing_*.json')]
            json_write(base/'summary.json',dict(status='development_profile_only',records=records,admission='Extrapolate each stage conservatively to 52 origins; account for larger final banks, add 50% margin and reserve 30/60 minutes. No test scoring.'))
        json_write(out/'status.json',dict(status='complete' if (out/'complete.json').exists() else 'incomplete',last_stage=args.stage,elapsed_seconds=time.perf_counter()-started))
    finally: (out/'RUNNING.lock').unlink(missing_ok=True)

if __name__=='__main__': main()
