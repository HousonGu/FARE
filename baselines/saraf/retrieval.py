"""Batched author-equivalent same-series retrieval for the frozen retail adapter."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from author import author_api
from io_utils import write_json,seal,verify

def date_marks(origin):
    d=pd.date_range(pd.Timestamp(origin)-pd.Timedelta(days=56),periods=56)
    return np.array([d.year,d.month,d.day,d.dayofweek,d.hour],dtype=np.float32).T
def stable_seed(seed,origin,start):
    return int.from_bytes(hashlib.sha256(f'{seed}/{origin}/{start}'.encode()).digest()[:8],'little')%(2**63-1)

def select_and_aggregate(sim,future,tool):
    """Same TopM, stochastic MMR proxy and Gaussian aggregation as author; B batched."""
    b,n=sim.shape;k=min(tool.topm,n);m=min(tool.mmr_candidate_pool,n)
    top=torch.topk(sim,m,dim=1);s=top.values;indices=top.indices
    mask=torch.zeros((b,m),dtype=torch.bool,device=sim.device);chosen=torch.zeros(b,dtype=torch.long,device=sim.device)
    redundancy=torch.full_like(s,-float('inf'));rows=torch.arange(b,device=sim.device)
    for step in range(k):
        if step:
            scores=tool.adaptive_lambda*s-(1-tool.adaptive_lambda)*redundancy
            probs=F.softmax(scores.masked_fill(mask,-float('inf'))/.1,dim=1)
            chosen=torch.multinomial(probs,1).squeeze(1)
        mask[rows,chosen]=True
        redundancy=torch.maximum(redundancy,1-torch.abs(s-s[rows,chosen,None]))
    weights=torch.exp(-((1-s)**2)/(2*tool.adaptive_sigma**2))*mask
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12)
    selected_future=torch.gather(future,1,indices[:,:,None].expand(-1,-1,7))
    return (weights[:,:,None]*selected_future).sum(1),indices,mask

class Bank:
    def __init__(self,panel,c,out,budget):
        self.panel=panel;self.c=c;self.folder=Path(out)/'bank';self.folder.mkdir(parents=True,exist_ok=True)
        self.origins=panel.schedule['train'];self.calibration_count=len(self.origins)//2
        self.minimum_query=self.origins[self.calibration_count]
        _,Tool=author_api();self.tool=Tool(56,7,1,n_period=1,topm=c['topm'],freq='d',time_aware_weight=c['time_aware_weight'],mmr_candidate_pool=c['candidate_pool'])
        if not verify(self.folder,'complete.json'):self.build(budget)
        self.x=np.load(self.folder/'x.npy',mmap_mode='r');self.y=np.load(self.folder/'y.npy',mmap_mode='r');self.valid=np.load(self.folder/'valid.npy')
        params=np.load(self.folder/'scaler.npz');self.mean=params['mean'];self.scale=params['scale']
        self.meta=json.loads((self.folder/'bank.json').read_text(encoding='utf-8'))
        self.tool.dataset_stationarity=self.meta['stationarity'];self.tool.adaptive_sigma=self.tool.compute_adaptive_sigma(self.meta['stationarity']);self.tool.adaptive_lambda=self.tool.compute_adaptive_lambda(self.meta['stationarity'])
        self.marks=torch.tensor(np.stack([date_marks(o) for o in self.origins]))
    def build(self,budget):
        n=len(self.panel.ids);r=len(self.origins)
        x=np.lib.format.open_memmap(self.folder/'x.npy',mode='w+',dtype='float32',shape=(n,r,56));y=np.lib.format.open_memmap(self.folder/'y.npy',mode='w+',dtype='float32',shape=(n,r,7));valid=np.zeros((n,r),bool)
        sums=np.zeros(n,np.float64);squares=sums.copy();counts=sums.copy()
        for j,o in enumerate(self.origins):
            budget.tick();rows=self.panel.rows(o);xx,yy,_=self.panel.window(o,rows,labels=True);x[rows,j]=xx;y[rows,j]=yy;valid[rows,j]=True
            if j<self.calibration_count:
                sums[rows]+=xx.astype(np.float64).sum(1);squares[rows]+=(xx.astype(np.float64)**2).sum(1);counts[rows]+=56
        mean=np.divide(sums,counts,out=np.zeros(n),where=counts>0);variance=np.divide(squares,counts,out=np.ones(n),where=counts>0)-mean**2
        scale=np.sqrt(np.maximum(variance,0));scale[scale<1e-6]=1
        total_score=0.;total_count=0
        for start in range(0,n,256):
            budget.tick();end=min(n,start+256)
            x[start:end]=(x[start:end]-mean[start:end,None,None])/scale[start:end,None,None]
            y[start:end]=(y[start:end]-mean[start:end,None,None])/scale[start:end,None,None]
            samples=np.array(x[start:end,:self.calibration_count][valid[start:end,:self.calibration_count]])
            if len(samples):
                s=self.tool.compute_stationarity_scores(torch.tensor(samples[:,:,None]));total_score+=float(s.double().sum());total_count+=len(s)
        if not total_count:raise ValueError('Empty legal bank')
        x.flush();y.flush();del x,y
        np.save(self.folder/'valid.npy',valid);np.savez(self.folder/'scaler.npz',mean=mean,scale=scale)
        write_json(self.folder/'bank.json',dict(stationarity=total_score/total_count,stationarity_windows=total_count,calibration_origins=self.origins[:self.calibration_count],windows=int(valid.sum()),origins=self.origins,bank_last_target=str((pd.Timestamp(self.origins[-1])+pd.Timedelta(days=6)).date()),minimum_query=self.minimum_query,series_without_bank=int((~valid.any(1)).sum()),series_without_calibration=int((counts==0).sum()),min_available=int(valid.sum(1).min()),max_available=int(valid.sum(1).max()),legality='candidate origin + 6 days < query origin; tune/dev/test excluded'))
        seal(self.folder,'complete.json',[p for p in self.folder.iterdir() if p.is_file() and p.name!='complete.json'])
    def normalize(self,x,rows):return ((x-self.mean[rows,None])/self.scale[rows,None]).astype(np.float32)
    def restore(self,y,rows):return y.astype(np.float64)*self.scale[rows,None]+self.mean[rows,None]
    def close(self):
        for value in [self.x,self.y]:
            if isinstance(value,np.memmap):value._mmap.close()
    def retrieve(self,origin,rows,device,seed,start=0):
        if pd.Timestamp(origin)<pd.Timestamp(self.minimum_query):raise ValueError('Query predates frozen calibration boundary')
        rows=np.asarray(rows);x,_,_=self.panel.window(origin,rows,labels=False)
        q=torch.tensor(self.normalize(x,rows),device=device);result=torch.zeros((len(rows),7),device=device)
        completed=np.asarray(pd.to_datetime(self.origins)+pd.Timedelta(days=6)<pd.Timestamp(origin))
        valid=self.valid[rows]&completed[None,:];counts=valid.sum(1)
        # Per-seed/origin/batch RNG makes interrupted week recomputation stable.
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if device.startswith('cuda') else []):
            torch.manual_seed(stable_seed(seed,origin,start))
            masks,groups=np.unique(valid,axis=0,return_inverse=True)
            for group,mask in enumerate(masks):
                if not mask.any():continue
                pos=np.flatnonzero(groups==group);selected_rows=rows[pos]
                past=torch.tensor(np.array(self.x[selected_rows][:,mask]),device=device);future=torch.tensor(np.array(self.y[selected_rows][:,mask]),device=device)
                centered=past-past.mean(2,keepdim=True);query=q[pos]-q[pos].mean(1,keepdim=True)
                sim=(F.normalize(centered,dim=2)*F.normalize(query,dim=1)[:,None]).sum(2)
                marks=torch.tensor(date_marks(origin)[None],device=device)
                bonus=self.tool.compute_time_aware_bonus(marks,self.marks[mask],torch.device(device))
                sim=(1-self.c['time_aware_weight'])*sim+self.c['time_aware_weight']*bonus
                residual=future-future[:,:,-1:]
                result[pos]=select_and_aggregate(sim,residual,self.tool)[0]
        return result.cpu().numpy(),dict(queries=len(rows),empty_bank=int((counts==0).sum()),short_bank=int(((counts>0)&(counts<self.c['topm'])).sum()))

def retrieve_week(bank,origin,seed,out,device,budget):
    folder=Path(out)/'retrieval';folder.mkdir(parents=True,exist_ok=True);path=folder/(origin+'.npy');marker=origin+'.complete.json'
    record=verify(folder,marker)
    if record:return np.load(path,mmap_mode='r')
    rows=bank.panel.rows(origin);pred=np.empty((len(rows),7),np.float32);audit=dict(queries=0,empty_bank=0,short_bank=0)
    for start in range(0,len(rows),bank.c['retrieval_batch_size']):
        budget.tick();end=min(len(rows),start+bank.c['retrieval_batch_size']);p,a=bank.retrieve(origin,rows[start:end],device,seed,start)
        pred[start:end]=p
        for k in audit:audit[k]+=a[k]
    if not np.isfinite(pred).all():raise ValueError('Nonfinite retrieval')
    tmp=path.with_suffix('.tmp.npy');np.save(tmp,pred);tmp.replace(path)
    seal(folder,marker,[path],dict(origin=origin,seed=seed,**audit));return pred
