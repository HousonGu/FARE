"""Pinned Chronos-2 dataset adapter; no test labels enter prediction."""
import numpy as np
import pandas as pd
from data_provider.rohlik import KEYS
from backbones.chronos_keys import merge

def chronos_week(data,origin,model,config):
    import torch
    h,cov,keys=data.window(origin); rows=data.eligible(origin); dates=pd.date_range(pd.Timestamp(origin)-pd.Timedelta(days=56),periods=63)
    past=pd.DataFrame(dict(item_id=np.repeat(data.ids[rows],56),timestamp=np.tile(dates[:56],len(rows)),target=h.astype(np.float32).ravel()))
    future=pd.DataFrame(dict(item_id=np.repeat(data.ids[rows],7),timestamp=np.tile(dates[56:],len(rows))))
    for day in range(7):
        past[f'weekday_{day}']=np.tile((dates[:56].dayofweek==day).astype(float),len(rows))
        future[f'weekday_{day}']=np.tile((dates[56:].dayofweek==day).astype(float),len(rows))
    for field,name in [('Discount','discount'),('Holiday','holiday')]:
        past[name]=cov[field][:,:56].ravel(); future[name]=cov[field][:,56:].ravel()
    with torch.inference_mode():
        pred=model.predict_df(past,future_df=future,prediction_length=7,context_length=56,quantile_levels=[.5],batch_size=config['chronos_batch'],cross_learning=False,validate_inputs=True,freq='D',id_column='item_id',timestamp_column='timestamp',target='target')
    pred=pred.rename(columns={'item_id':'series_id','timestamp':'target_date','0.5':'backbone_forecast'})
    pred['series_id']=pred.series_id.astype(int); pred['target_date']=pd.to_datetime(pred.target_date).dt.strftime('%Y-%m-%d'); pred['origin']=origin
    pred['backbone_forecast']=np.maximum(pred.backbone_forecast,0)
    return merge(keys,pred[KEYS+['backbone_forecast']])
