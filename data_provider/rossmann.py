"""Rossmann panel: observed zeros are retained; missing days remain NaN."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

KEYS = ['origin','series_id','target_date']
FIELDS = ['Sales','Promo','SchoolHoliday','Open']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

class RossmannData:
    def __init__(self, raw):
        self.raw=Path(raw)
        frame=pd.read_csv(self.raw/'train.csv',usecols=['Store','Date']+FIELDS)
        meta=pd.read_csv(self.raw/'store.csv',usecols=['Store','StoreType','Assortment'])
        frame['Date']=pd.to_datetime(frame.Date,errors='raise')
        if frame.duplicated(['Store','Date']).any() or meta.Store.duplicated().any(): raise ValueError('Duplicate keys')
        if frame[['Store','Date','Promo','SchoolHoliday','Open']].isna().any().any(): raise ValueError('Missing key or factors')
        if not np.isfinite(frame.Sales.dropna()).all() or (frame.Sales.dropna()<0).any(): raise ValueError('Invalid Sales')
        for c in FIELDS[1:]:
            if not frame[c].isin([0,1]).all(): raise ValueError('Nonbinary '+c)
        frame=frame.merge(meta,on='Store',how='left',validate='many_to_one')
        if frame[['StoreType','Assortment']].isna().any().any(): raise ValueError('Missing metadata')
        self.ids=np.array(sorted(frame.Store.unique()))
        self.dates=pd.date_range(frame.Date.min(),frame.Date.max())
        self.arr={c:frame.pivot(index='Store',columns='Date',values=c).reindex(index=self.ids,columns=self.dates).to_numpy(float) for c in FIELDS}
        metadata=meta.set_index('Store').loc[self.ids]
        self.groups=[np.flatnonzero((metadata.StoreType.to_numpy()==a)&(metadata.Assortment.to_numpy()==b)) for a,b in sorted(set(zip(metadata.StoreType,metadata.Assortment)))]
        origins=pd.date_range(self.dates[0]+pd.Timedelta(days=56),self.dates[-1]-pd.Timedelta(days=6),freq='W-MON').strftime('%Y-%m-%d').tolist()
        if len(origins)<61: raise ValueError('Need fit + 26 development + 26 test weeks')
        self.schedule=dict(fit=origins[:-52],dev=origins[-52:-26],test=origins[-26:])
        before=frame[frame.Date<pd.Timestamp(self.schedule['test'][0])]
        self.audit=dict(series=len(self.ids),rows=len(frame),missing_sales=int(frame.Sales.isna().sum()),
            missing_panel_days=int(len(self.ids)*len(self.dates)-len(frame)),closed_rows=int(frame.Open.eq(0).sum()),
            closed_nonzero_sales=int((frame.Open.eq(0)&frame.Sales.ne(0)&frame.Sales.notna()).sum()),
            pretest_factor_support=before.groupby(['Promo','SchoolHoliday']).size().rename('rows').reset_index().to_dict('records'),
            raw_sha256={n:digest(self.raw/n) for n in ['train.csv','store.csv']},schedule=self.schedule,
            fit_tune=self.schedule['fit'][-8:])

    def columns(self,origin):
        pos=int((pd.Timestamp(origin)-self.dates[0]).days)
        if pos<56 or pos+7>len(self.dates): raise ValueError('Window out of range')
        return np.arange(pos-56,pos+7)

    def eligible(self,origin,reference=False):
        c=self.columns(origin)
        valid=np.isfinite(self.arr['Sales'][:,c[:56]]).all(1)
        for name in ['Promo','SchoolHoliday','Open']:
            valid &= np.isfinite(self.arr[name][:,c]).all(1)
        if reference: valid &= np.isfinite(self.arr['Sales'][:,c[56:]]).all(1)
        return np.flatnonzero(valid)

    def window(self,origin,rows=None):
        rows=self.eligible(origin) if rows is None else np.asarray(rows)
        c=self.columns(origin)
        h=self.arr['Sales'][np.ix_(rows,c[:56])]
        cov={name:self.arr[name][np.ix_(rows,c)] for name in ['Promo','SchoolHoliday','Open']}
        keys=pd.DataFrame(dict(origin=origin,series_id=np.repeat(self.ids[rows],7),target_date=np.tile(self.dates[c[56:]].strftime('%Y-%m-%d'),len(rows))))
        return h,cov,keys

    def truth(self,origin):
        rows=self.eligible(origin); c=self.columns(origin)
        _,cov,keys=self.window(origin,rows)
        keys['actual']=self.arr['Sales'][np.ix_(rows,c[56:])].ravel()
        keys['open']=cov['Open'][:,56:].ravel()
        return keys
