"""Resumable exact retrieval, validation selection, full-dev refit and scoring."""
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from author import RetailPIR, objective
from retrieval import normalize_windows, search
from io_utils import write_json, seal, verify, parquet, checked_merge, metric

class BudgetExpired(Exception): pass

class Budget:
    def __init__(self, seconds): self.end=time.monotonic()+seconds
    def tick(self):
        if time.monotonic() >= self.end: raise BudgetExpired('Resume same command and output directory')

def save_torch(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); torch.save(value,tmp); os.replace(tmp,path)

def load_torch(path, device):
    path=Path(path)
    if not path.is_file():
        raise ValueError('Missing checkpoint '+str(path))
    return torch.load(path,map_location=device,weights_only=False)

def make_bank(panel,out,budget):
    folder=out/'bank'; folder.mkdir(exist_ok=True)
    if verify(folder,'complete.json'):
        return np.load(folder/'keys.npy',mmap_mode='r'),np.load(folder/'values.npy',mmap_mode='r')
    origins=panel.schedule['fit']
    if pd.Timestamp(origins[-1])+pd.Timedelta(days=6)>=pd.Timestamp(panel.schedule['dev'][0]):
        raise ValueError('Bank target crosses first revision query')
    n=sum(len(panel.rows(o)) for o in origins)
    keys=np.lib.format.open_memmap(folder/'keys.tmp.npy',mode='w+',dtype='float32',shape=(n,56))
    vals=np.lib.format.open_memmap(folder/'values.tmp.npy',mode='w+',dtype='float32',shape=(n,7))
    offset=0
    for o in origins:
        budget.tick(); x,y,_=panel.window(o,labels=True)
        scale=np.maximum(x.mean(1,keepdims=True),1)
        a,b=normalize_windows(x/scale,y/scale)
        keys[offset:offset+len(x)]=a;vals[offset:offset+len(x)]=b;offset+=len(x)
    keys.flush();vals.flush();del keys,vals
    os.replace(folder/'keys.tmp.npy',folder/'keys.npy');os.replace(folder/'values.tmp.npy',folder/'values.npy')
    write_json(folder/'bank.json',dict(windows=n,origins=origins,order='origin then original panel series order',
        latest_target=str(pd.Timestamp(origins[-1])+pd.Timedelta(days=6)),scope='all eligible series; frozen fit only'))
    seal(folder,'complete.json',[folder/n for n in ['keys.npy','values.npy','bank.json']])
    return np.load(folder/'keys.npy',mmap_mode='r'),np.load(folder/'values.npy',mmap_mode='r')

def marks(origin):
    d=pd.date_range(origin,periods=7)
    # Five known calendar channels; no sales, availability or unknown future input.
    return np.stack([d.month/12-.5,d.day/31-.5,d.dayofweek/6-.5,d.dayofyear/366-.5,np.zeros(7)],axis=1).astype('float32')

def week(panel, origin, labels=False):
    x,y,keys=panel.window(origin,labels=labels)
    b=checked_merge(keys,pd.read_parquet(panel.folder/'backbone'/f'{origin}.parquet'))
    scale=np.maximum(x.mean(1,keepdims=True),1)
    return x/scale, None if y is None else y/scale, b.forecast.to_numpy(dtype='float32').reshape(-1,7)/scale,scale,keys

def cache_week(panel,origin,out,bank,c,device,budget):
    folder=out/'retrieval'/origin;folder.mkdir(parents=True,exist_ok=True)
    if verify(folder,'complete.json'): return
    x,_,_,_,_=week(panel,origin)
    x,_=normalize_windows(x,np.zeros((len(x),7),dtype='float32'))
    results=[]; similarities=[]; started=time.monotonic()
    for start in range(0,len(x),c['retrieval_batch']):
        budget.tick()
        g,s,_=search(x[start:start+c['retrieval_batch']],*bank,c['retrieval_k'],c['retrieval_block'],device,budget.tick)
        results.append(g);similarities.append(s)
    for name,array in [('global',np.concatenate(results)),('similarities',np.concatenate(similarities))]:
        np.save(folder/(name+'.tmp.npy'),array);os.replace(folder/(name+'.tmp.npy'),folder/(name+'.npy'))
    write_json(folder/'audit.json',dict(origin=origin,queries=len(x),bank_windows=len(bank[0]),seconds=time.monotonic()-started))
    seal(folder,'complete.json',[folder/(n+'.npy') for n in ['global','similarities']]+[folder/'audit.json'])

def tensors(arrays,indices,device,origin):
    x,y,b,g,s=arrays
    tensor=lambda a:torch.as_tensor(a[indices],device=device)
    args=(tensor(x)[...,None],tensor(b)[...,None],tensor(g)[...,None],tensor(s)[:,None,:],
          torch.tensor(marks(origin),device=device).expand(len(indices),-1,-1))
    return args,None if y is None else tensor(y)[...,None]

def arrays_for(panel,o,out,labels):
    x,y,b,scale,keys=week(panel,o,labels)
    folder=out/'retrieval'/o
    if not verify(folder,'complete.json'):raise ValueError('Unfinished retrieval '+o)
    return (x,y,b,np.load(folder/'global.npy'),np.load(folder/'similarities.npy')),scale,keys

def evaluate(model,panel,origins,out,c,device,budget):
    model.eval();ae=total=0.
    with torch.no_grad():
        for o in origins:
            a,scale,_=arrays_for(panel,o,out,True)
            for start in range(0,len(a[0]),c['batch_size']):
                budget.tick();ix=np.arange(start,min(start+c['batch_size'],len(a[0])))
                args,y=tensors(a,ix,device,o)
                pred=model.revise(*args)[0].cpu().numpy()[:,:,0]*scale[ix]
                true=a[1][ix]*scale[ix]
                ae+=np.abs(np.maximum(pred,0)-true).sum(dtype=np.float64);total+=true.sum(dtype=np.float64)
    if total<=0:raise ValueError('Zero validation sum')
    return float(ae/total)

def train(panel,out,c,device,budget,seed,phase,origins,epochs,tune):
    folder=out/f'seed_{seed}'/phase;folder.mkdir(parents=True,exist_ok=True)
    if verify(folder,'complete.json'):return json.loads((folder/'training.json').read_text())
    torch.manual_seed(seed)
    if device.startswith('cuda'):torch.cuda.manual_seed_all(seed)
    model=RetailPIR(c).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=c['learning_rate'])
    state=dict(epoch=0,week=0,batch=0,best=None,best_epoch=None,history=[])
    ckpt=folder/'resume.pt'
    if ckpt.exists():
        loaded=load_torch(ckpt,device);model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer']);state=loaded['state']
        torch.set_rng_state(loaded['rng'].cpu())
        if device.startswith('cuda'):torch.cuda.set_rng_state_all([s.cpu() for s in loaded['cuda_rng']])
    def checkpoint():
        save_torch(ckpt,dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else []))
    try:
        while state['epoch']<epochs:
            epoch=state['epoch'];model.train()
            # Deterministic epoch permutation; state cursor exactly identifies next batch.
            order=np.random.default_rng(seed+epoch).permutation(len(origins))
            for wi in range(state['week'],len(origins)):
                o=origins[order[wi]];a,_,_=arrays_for(panel,o,out,True)
                indices=np.random.default_rng(seed+epoch*10000+int(order[wi])).permutation(len(a[0]))
                for start in range(state['batch'],len(indices),c['batch_size']):
                    budget.tick();ix=indices[start:start+c['batch_size']];args,y=tensors(a,ix,device,o)
                    optimizer.zero_grad();loss=objective(model.revise(*args),y,c['quality_weight'])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite PIR loss')
                    loss.backward();optimizer.step();state['batch']=start+len(ix)
                state['week']=wi+1;state['batch']=0;checkpoint()
            value=evaluate(model,panel,tune,out,c,device,budget) if tune else None
            state['history'].append(dict(epoch=epoch+1,tune_wape=value))
            if tune and (state['best'] is None or value<state['best']):
                state['best']=value;state['best_epoch']=epoch+1
            state['epoch']=epoch+1;state['week']=0;state['batch']=0;checkpoint()
            print(seed,phase,state['epoch'],value,flush=True)
        save_torch(folder/'model.pt',model.state_dict());write_json(folder/'training.json',state)
        seal(folder,'complete.json',[folder/'model.pt',folder/'training.json'])
        return state
    except BudgetExpired:
        checkpoint();raise

def predict(panel,out,c,device,budget,seed):
    folder=out/f'seed_{seed}';model=RetailPIR(c).to(device)
    model.load_state_dict(load_torch(folder/'refit/model.pt',device));model.eval()
    # Dev outputs are explicitly in-sample refit diagnostics, not validation results.
    with torch.no_grad():
        for o in panel.schedule['dev']+panel.schedule['test']:
            dest=folder/'predictions';dest.mkdir(exist_ok=True)
            if verify(dest,o+'.complete.json'):continue
            a,scale,keys=arrays_for(panel,o,out,False);chunks=[]
            for start in range(0,len(a[0]),c['batch_size']):
                budget.tick();ix=np.arange(start,min(start+c['batch_size'],len(a[0])))
                args,_=tensors(a,ix,device,o);chunks.append(model.revise(*args)[0].cpu().numpy()[:,:,0]*scale[ix])
            raw=np.concatenate(chunks).ravel()
            if not np.isfinite(raw).all():raise ValueError('Invalid prediction')
            keys['pir_raw']=raw;keys['pir']=np.maximum(raw,0);parquet(dest/(o+'.parquet'),keys)
            seal(dest,o+'.complete.json',[dest/(o+'.parquet')])

def score(panel,out,c,budget):
    records=[]; all_artifacts=[]
    for seed in c['seeds']:
        folder=out/f'seed_{seed}';totals={};weeks=[];artifacts=[]
        for o in panel.schedule['test']:
            budget.tick();dest=folder/'predictions'
            if not verify(dest,o+'.complete.json'):raise ValueError('Missing complete prediction '+o)
            frame=checked_merge(panel.truth(o),pd.read_parquet(dest/(o+'.parquet')))
            m=metric(frame.actual,frame['pir'])
            t=totals.setdefault('pir',dict(rows=0,actual_sum=0.,absolute_error_sum=0.))
            for key in t:t[key]+=m[key]
            weeks.append(dict(origin=o,metrics={'pir':m},negative_raw=int((frame.pir_raw<0).sum())))
        for t in totals.values():
            t['wape']=t['absolute_error_sum']/t['actual_sum'];t['mae']=t['absolute_error_sum']/t['rows']
        record=dict(seed=seed,metrics=totals,weeks=weeks)
        write_json(folder/'metrics.json',record)
        artifacts=list((folder/'predictions').glob('*'))+[folder/'metrics.json']
        for phase in ['select','refit']:
            if not verify(folder/phase,'complete.json'):raise ValueError('Missing training completion')
            artifacts += [folder/phase/n for n in ['model.pt','training.json','complete.json']]
        seal(folder,'complete.json',artifacts)
        records.append(record);all_artifacts += artifacts+[folder/'complete.json']
    write_json(out/'metrics.json',dict(dataset=panel.meta['dataset'],method='PIR-Chronos-retail-univariate',records=records))
    all_artifacts += [out/'metrics.json',out/'manifest.json',out/'audit.json']
    all_artifacts += [p for p in (out/'bank').rglob('*') if p.is_file() and '.tmp' not in p.name]
    all_artifacts += [p for p in (out/'retrieval').rglob('*') if p.is_file() and '.tmp' not in p.name]
    seal(out,'complete.json',all_artifacts)
