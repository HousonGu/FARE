"""Rossmann adapter for the existing LightGBM and actual PyTorch Forecasting TFT."""
import numpy as np
import pandas as pd
from data_provider.rossmann import RossmannData
from data_provider.supervised_features import RetailData


class Data:
    tree = RetailData.tree

    def __init__(self, raw):
        self.source = RossmannData(raw)
        self.ids = self.source.ids
        self.schedule = dict(self.source.schedule)
        self.schedule['train'] = self.schedule['fit'][:-8]
        self.schedule['tune'] = self.schedule['fit'][-8:]
        meta = pd.read_csv(self.source.raw/'store.csv').set_index('Store').loc[self.ids]
        self.static_names = ['StoreType', 'Assortment']
        self.static = np.column_stack([pd.Categorical(meta[c]).codes for c in self.static_names]).astype(np.int64)
        self.cardinalities = {c: int(self.static[:, i].max())+1 for i,c in enumerate(self.static_names)}
        for a,b in [('train','tune'),('fit','dev'),('dev','test')]:
            assert pd.Timestamp(self.schedule[a][-1])+pd.Timedelta(days=6)<pd.Timestamp(self.schedule[b][0])

    def eligible(self, origin):
        # Fit labels can be required only in the pre-development fit interval.
        return self.source.eligible(origin, reference=origin in self.schedule['fit'])

    def window(self, origin, rows=None, labels=False):
        if labels and origin not in self.schedule['fit']:
            raise ValueError('Backbone training/selection may only read fit labels')
        rows = self.eligible(origin) if rows is None else np.asarray(rows)
        if not np.isin(rows,self.eligible(origin)).all():
            raise ValueError('Ineligible rows')
        h,c,keys = self.source.window(origin, rows)
        scale = np.maximum(h.mean(1),1).astype(np.float32)
        cols = self.source.columns(origin)
        dow = np.broadcast_to(np.eye(7,dtype=np.float32)[self.source.dates[cols].dayofweek],(len(rows),63,7))
        cov = np.concatenate([dow,c['Promo'][...,None],c['SchoolHoliday'][...,None]],axis=2).astype(np.float32)
        y = self.source.arr['Sales'][np.ix_(rows,cols[56:])] if labels else None
        if y is not None and not np.isfinite(y).all():
            raise ValueError('Nonfinite training labels')
        return rows,h,scale,cov,y,keys
