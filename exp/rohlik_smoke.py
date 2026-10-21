"""Real-data, real-model preflight, never a formal test score."""
import time
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from data_provider.supervised_features import CONFIG,json_write,metric
from backbones.rohlik_tft import Windows,make_model,train,forward,loader,seed_all


class SmallData:
    def __init__(self,data): self.data=data
    def __getattr__(self,name): return getattr(self.data,name)
    def eligible(self,origin): return self.data.eligible(origin)[:32]
    def window(self,origin,rows=None,labels=False):
        return self.data.window(origin,self.eligible(origin) if rows is None else rows,labels)
    from data_provider.supervised_features import RetailData
    tree=RetailData.tree


def smoke(data,out,device):
    out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(CONFIG['cpu_threads'])
    if device.startswith('cuda'):
        (torch.ones(8,device=device)*2).sum().item(); torch.cuda.reset_peak_memory_stats(device)
    tiny=SmallData(data); o=data.schedule['train'][-1]; v=data.schedule['tune'][0]; d=data.schedule['dev'][0]
    tick=time.perf_counter(); x,y,w,_=tiny.tree(o,True); vx,vy,vw,_=tiny.tree(v,True)
    model=lgb.LGBMRegressor(n_estimators=10,num_leaves=7,min_child_samples=5,n_jobs=2,verbosity=-1,random_state=20261005)
    model.fit(x,y,sample_weight=w,categorical_feature=data.static_names+['horizon'])
    pred=np.maximum(model.predict(vx),0)*vw
    report=dict(status='smoke_complete_not_formal_result',lightgbm=dict(seconds=time.perf_counter()-tick,tune_metric=metric(vy*vw,pred)))
    seed_all(CONFIG['seed']); tick=time.perf_counter()
    network=make_model(data,dict(hidden_size=32,dropout=.1,learning_rate=.001))
    ts=Windows(tiny,[o],True); valid=Windows(tiny,[v],True)
    record=train(network,ts,device,2,valid,checkpoint=out/'tft_resume.pt')
    windows=Windows(tiny,[d],False); batch=next(iter(loader(windows)))
    network.eval()
    with torch.inference_mode():
        p=forward(network,batch[0],batch[1],device)
        changed={k:z.clone() for k,z in batch[0].items()}
        changed['decoder_cont'][:,:,0]=123456
        q=forward(network,changed,batch[1],device)
    torch.testing.assert_close(p,q,rtol=0,atol=0)
    assert p.shape==(32,7) and torch.isfinite(p).all()
    report['tft']=dict(seconds=time.perf_counter()-tick,training=record,prediction_shape=list(p.shape),decoder_sales_invariance=True)
    # Test target perturbation cannot change supervised feature/eligibility paths.
    origin=data.schedule['test'][0]; cols=data.source.columns(origin)[56:]
    before=data.tree(origin,False); saved=data.source.arr['Sales'][:,cols].copy()
    try:
        data.source.arr['Sales'][:,cols]=np.nan
        after=data.tree(origin,False)
        pd.testing.assert_frame_equal(before[0],after[0]); pd.testing.assert_frame_equal(before[3],after[3])
    finally: data.source.arr['Sales'][:,cols]=saved
    report['test_target_feature_invariance']=True
    report['frozen_test_rows']=sum(len(data.eligible(o))*7 for o in data.schedule['test'])
    assert report['frozen_test_rows']==283185
    report['device']=device
    if device.startswith('cuda'): report['cuda_peak_bytes']=torch.cuda.max_memory_allocated(device)
    json_write(out/'report.json',report); print(report,flush=True)
