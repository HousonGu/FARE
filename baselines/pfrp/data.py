import json
from pathlib import Path
import numpy as np
import pandas as pd
from io_utils import KEYS,verify

class Panel:
    def __init__(self,folder,check=True):
        self.folder=Path(folder)
        if check and not verify(self.folder,'complete.json'):raise ValueError('Prepared data incomplete')
        self.meta=json.loads((self.folder/'panel.json').read_text(encoding='utf-8'))
        self.schedule=self.meta['schedule'];self.ids=np.array(self.meta['ids']);self.days=self.meta['days']
        self.first=np.array(self.meta['first'],dtype='datetime64[D]')
        self.values=np.load(self.folder/'values.npy',mmap_mode='r');self.columns={v:i for i,v in enumerate(self.days)}
        validate_schedule(self.schedule)
    def rows(self,origin):
        if 'eligible_rows' in self.meta:
            return np.asarray(self.meta['eligible_rows'][origin],dtype=int)
        return np.flatnonzero(self.first<=np.datetime64(origin)-np.timedelta64(56,'D'))
    def window(self,origin,rows=None,labels=False):
        rows=self.rows(origin) if rows is None else np.asarray(rows)
        if not np.isin(rows,self.rows(origin)).all():raise ValueError('Ineligible series')
        days=pd.date_range(pd.Timestamp(origin)-pd.Timedelta(days=56),periods=63).strftime('%Y-%m-%d').tolist()
        ix=[self.columns[d] for d in days]
        x=np.asarray(self.values[np.ix_(rows,ix[:56])],dtype=np.float32)
        if not np.isfinite(x).all() or (x<0).any():raise ValueError('Invalid context')
        y=np.asarray(self.values[np.ix_(rows,ix[56:])],dtype=np.float32) if labels else None
        if labels and (not np.isfinite(y).all() or (y<0).any()):raise ValueError('Invalid target')
        keys=pd.DataFrame({'origin':origin,'series_id':np.repeat(self.ids[rows],7),'target_date':np.tile(days[56:],len(rows))})
        return x,y,keys
    def truth(self,origin):
        _,_,keys=self.window(origin,labels=False)
        ix=[self.columns[d] for d in pd.date_range(origin,periods=7).strftime('%Y-%m-%d')]
        keys['actual']=np.asarray(self.values[np.ix_(self.rows(origin),ix)],dtype=np.float64).ravel()
        return keys

def validate_schedule(s):
    for key in ['fit','train','tune','dev','test']:
        if not s[key] or s[key]!=sorted(set(s[key])):raise ValueError('Empty or unordered split '+key)
        dates=pd.to_datetime(s[key])
        if any(d.weekday()!=0 for d in dates) or any((b-a).days<7 for a,b in zip(dates,dates[1:])):raise ValueError('Overlapping/non-Monday origins')
    if s['train']+s['tune']!=s['fit']:raise ValueError('Train/tune must partition fit')
    for a,b in [('train','tune'),('fit','dev'),('dev','test')]:
        if pd.Timestamp(s[a][-1])+pd.Timedelta(days=6)>=pd.Timestamp(s[b][0]):raise ValueError('Split overlap')
    if len(s['train'])<4:raise ValueError('Insufficient training origins')

class Windows:
    def __init__(self,panel,origins):
        self.panel=panel;self.origins=origins
        self.rows=[panel.rows(o) for o in origins];self.ends=np.cumsum([len(r) for r in self.rows])
        self.flat_rows=np.concatenate(self.rows)
        self.indices=np.array([[panel.columns[d] for d in pd.date_range(pd.Timestamp(o)-pd.Timedelta(days=56),periods=63).strftime('%Y-%m-%d')] for o in origins])
        self.times=np.array([pd.Timestamp(o).timestamp() for o in origins],np.float32)
    def __len__(self):return int(self.ends[-1])
    def batch(self,indices):
        indices=np.asarray(indices);which=np.searchsorted(self.ends,indices,side='right')
        values=np.asarray(self.panel.values[self.flat_rows[indices,None],self.indices[which]],dtype=np.float32)
        if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid training window')
        return values[:,:56],values[:,56:],self.times[which,None]
