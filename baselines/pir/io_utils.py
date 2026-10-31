import json,os
from pathlib import Path
import numpy as np
import pandas as pd

KEYS=['origin','series_id','target_date']
HERE=Path(__file__).resolve().parent
def write_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def parquet(p,frame):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.tmp.parquet');frame.to_parquet(t,index=False);os.replace(t,p)
def seal(folder,name,files,extra=None):
    write_json(Path(folder)/name,dict(artifacts=sorted(Path(p).relative_to(folder).as_posix() for p in files),**(extra or {})))
def verify(folder,name):
    p=Path(folder)/name
    if not p.exists():return None
    r=json.loads(p.read_text(encoding='utf-8'))
    for n in r['artifacts']:
        if not (Path(folder)/n).is_file():raise ValueError('Missing artifact: '+n)
    return r

def checked_merge(a,b):
    if a.duplicated(KEYS).any() or b.duplicated(KEYS).any():raise ValueError('Duplicate keys')
    if '__left_order__' in a or '__left_order__' in b:raise ValueError('Reserved column')
    # Pandas outer joins sort keys even with sort=False. Preserve panel order
    # before reshaping into [series, horizon], especially M5's nonlexical IDs.
    c=a.assign(__left_order__=np.arange(len(a))).merge(b,on=KEYS,how='outer',validate='one_to_one',indicator=True)
    if not c['_merge'].eq('both').all():raise ValueError('Evaluation keys differ')
    return c.sort_values('__left_order__').drop(columns=['_merge','__left_order__']).reset_index(drop=True)
def metric(y,p):
    y=np.asarray(y,np.float64);p=np.asarray(p,np.float64)
    if y.shape!=p.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(p).all() or (y<0).any():raise ValueError('Invalid score values')
    ae=np.abs(y-p)
    return dict(rows=len(y),actual_sum=float(y.sum()),absolute_error_sum=float(ae.sum()),wape=float(ae.sum()/y.sum()) if y.sum()>0 else None,mae=float(ae.mean()))
