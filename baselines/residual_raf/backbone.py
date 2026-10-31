"""Frozen Chronos-2 forecasts and the unmodified Chronos-2 inference pipeline."""
from pathlib import Path
import json
import importlib.metadata
import numpy as np
import pandas as pd
from io_utils import HERE, verify, verify_file, merge

FACTOR_NAMES={'favorita':['promo_bin','holiday'],'m5':['snap','event'],
 'rossmann':['promo','school_holiday'],'rohlik':['discount','holiday']}

class FrozenBackbone:
    def __init__(self,panel,input_root=None):
        self.panel=panel
        self.root=Path(input_root)/panel.dataset if input_root else panel.folder
        record=verify(self.root)
        if record is None:raise ValueError('Frozen B inputs missing')
        self._origin=None
        self._array=None

    def __call__(self,origin,rows=None):
        rows=self.panel._rows(origin,rows)
        if self._origin!=origin:
            path=self.root/'backbone'/f'{origin}.parquet'
            verify_file(path)
            frame=merge(self.panel.keys(origin),pd.read_parquet(path))
            arr=frame.forecast.to_numpy(float).reshape(-1,7)
            if not np.isfinite(arr).all() or (arr<0).any():raise ValueError('Invalid B')
            self._array=np.full((len(self.panel.ids),7),np.nan)
            self._array[self.panel.rows(origin)]=arr
            self._origin=origin
        return self._array[rows].copy()

def covariates_for(panel,origin,rows):
    indices=panel.indices(origin)
    cov=np.asarray(panel.covariates[np.ix_(rows,indices,np.arange(len(panel.covariate_names)))],np.float32)
    dates=pd.date_range(pd.Timestamp(origin)-pd.Timedelta(days=56),periods=63)
    weekday=np.eye(7,dtype=np.float32)[dates.dayofweek]
    return np.concatenate([np.broadcast_to(weekday,(len(rows),63,7)),cov],axis=-1)

def load_pipeline(model_path,device='cuda'):
    import torch
    from chronos import Chronos2Pipeline
    if importlib.metadata.version('chronos-forecasting')!='2.3.2':raise RuntimeError('Need chronos-forecasting 2.3.2')
    for name in ['config.json','model.safetensors']:
        if not (Path(model_path)/name).is_file():raise ValueError('Chronos-2 weights missing: '+str(Path(model_path)/name))
    model=Chronos2Pipeline.from_pretrained(str(model_path),device_map=device,local_files_only=True,torch_dtype=torch.float32)
    model.model.eval()
    for p in model.model.parameters():p.requires_grad_(False)
    return model

def predict_context(pipeline,panel,origin,rows,context,past_covariates=None,future_covariates=None,batch_size=64):
    import torch
    context=np.asarray(context,np.float32)
    n,length=context.shape
    if n!=len(rows) or length<56 or not np.isfinite(context).all():raise ValueError('Invalid context')
    cov=covariates_for(panel,origin,rows)
    if past_covariates is None:past_covariates=cov[:,:56]
    if future_covariates is None:future_covariates=cov[:,56:]
    names=[f'weekday_{i}' for i in range(7)]+FACTOR_NAMES[panel.dataset]
    if past_covariates.shape!=(n,length,len(names)) or future_covariates.shape!=(n,7,len(names)):
        raise ValueError('Covariates must cover the full augmented input')
    dates=pd.date_range(pd.Timestamp(origin)-pd.Timedelta(days=length),periods=length+7)
    # IDs are temporary task identifiers. cross_learning=False prevents sharing.
    ids=np.arange(n).astype(str)
    past=pd.DataFrame({'item_id':np.repeat(ids,length),'timestamp':np.tile(dates[:length],n),'target':context.ravel()})
    future=pd.DataFrame({'item_id':np.repeat(ids,7),'timestamp':np.tile(dates[length:],n)})
    for j,name in enumerate(names):
        past[name]=past_covariates[:,:,j].ravel()
        future[name]=future_covariates[:,:,j].ravel()
    with torch.inference_mode():
        result=pipeline.predict_df(past,future_df=future,prediction_length=7,context_length=length,
            quantile_levels=[.5],batch_size=batch_size,cross_learning=False,validate_inputs=True,
            freq='D',id_column='item_id',timestamp_column='timestamp',target='target')
    result['item_id']=result.item_id.astype(str)
    expected=pd.DataFrame({'item_id':np.repeat(ids,7),'timestamp':np.tile(dates[length:],n)})
    if len(expected)!=len(result):raise ValueError('Chronos output row count changed')
    result=expected.merge(result[['item_id','timestamp','0.5']],on=['item_id','timestamp'],how='left',validate='one_to_one',indicator=True,sort=False)
    if not result['_merge'].eq('both').all():raise ValueError('Chronos output keys changed')
    output=result['0.5'].to_numpy(float).reshape(n,7)
    if not np.isfinite(output).all():raise ValueError('Nonfinite Chronos output')
    return output
