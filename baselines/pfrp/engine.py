import json,os,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from author import author_api
from data import Windows
from io_utils import write_json,parquet,seal,verify,metric,checked_merge
from medoids import unit,initialize,assign,update

class BudgetExpired(Exception):pass
class Budget:
    def __init__(self,seconds):self.end=time.monotonic()+seconds
    def tick(self):
        if time.monotonic()>=self.end:raise BudgetExpired('Batch time budget exhausted; resume same command/output')
def save_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)
def load_torch(path,device):return torch.load(path,map_location=device,weights_only=False)
def settings(c):return SimpleNamespace(seq_len=56,pred_len=7,topk_PFRP=c['topk'],d_ff_PFRP=c['hidden'],data_path='retail.csv',task_name='long_term_forecast',moving_avg=25,enc_in=1)
def construct(folder,c,device):
    api,_=author_api();cwd=Path.cwd()
    try:os.chdir(folder);model=api['DLinear_PFRP'](settings(c))
    finally:os.chdir(cwd)
    return model.to(device)
def forecast(model,x,device,batch,tick=lambda:None,decoder_check=False):
    model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            tick();q=torch.as_tensor(x[start:start+batch,:,None],device=device)
            y=model(q,None,None,None)
            if decoder_check:
                z=model(q,None,torch.full((len(q),7,1),1e8,device=device),None)
                torch.testing.assert_close(y,z,rtol=0,atol=0)
            if not torch.isfinite(y).all():raise ValueError('Nonfinite prediction')
            out.append(y.cpu().numpy()[:,:,0])
    return np.concatenate(out)
def tune_score(model,panel,origins,c,device,budget):
    absolute=0.;total=0.
    for o in origins:
        x,_,_=panel.window(o,labels=False);truth=panel.truth(o)['actual'].to_numpy()
        p=np.maximum(forecast(model,x,device,c['batch_size'],budget.tick),0).ravel()
        absolute+=np.abs(truth-p).sum();total+=truth.sum()
    if total<=0:raise ValueError('Zero tune denominator')
    return float(absolute/total)

def train(model,windows,c,device,budget,folder,name,seed,epochs,encoder=False,tune=None):
    """Resume exact next batch, deterministic epoch permutation, optimizer and RNG state."""
    folder=Path(folder);path=folder/(name+'.last.pt');best=folder/(name+'.best.pt')
    if verify(folder,name+'.complete.json'):
        model.load_state_dict(load_torch(best,device)['model']);return
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=c['learning_rate'])
    state=dict(epoch=0,offset=0,best_score=None,best_epoch=None,history=[])
    if path.exists():
        state=load_torch(path,device);model.load_state_dict(state.pop('model'));optimizer.load_state_dict(state.pop('optimizer'))
        torch.set_rng_state(state.pop('rng').cpu())
        cuda=state.pop('cuda_rng',None)
        if device.startswith('cuda') and cuda is not None:torch.cuda.set_rng_state_all([t.cpu() for t in cuda])
    _,pcl=author_api()
    def checkpoint():
        save_torch(path,dict(state,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None))
    try:
        while state['epoch']<epochs:
            order=np.random.default_rng(seed+state['epoch']).permutation(len(windows));model.train()
            for start in range(state['offset'],len(order),c['batch_size']):
                budget.tick();idx=order[start:start+c['batch_size']]
                x,y,marks=windows.batch(idx)
                if encoder:
                    # Author objective needs at least one temporally distinct positive for every row.
                    valid=(np.abs(marks[:,0,None]-marks[:,0][None,:])>=4*3600).any(axis=1)
                    if not valid.all():
                        # Add deterministic samples from a different origin; never drop panel windows.
                        candidates=[0,int(windows.ends[0])]
                        idx=np.r_[idx,candidates];x,y,marks=windows.batch(idx)
                    a=torch.as_tensor(x,device=device);b=torch.as_tensor(y,device=device);m=torch.as_tensor(marks,device=device)
                    loss=pcl['contrastive_loss'](model(pcl['normalize_series'](a)),a,m,pcl['normalize_series'](b),None)
                else:
                    a=torch.as_tensor(x[:,:,None],device=device);b=torch.as_tensor(y[:,:,None],device=device)
                    loss=(model(a,None,None,None)-b).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite '+name+' training objective')
                optimizer.zero_grad();loss.backward();optimizer.step();state['offset']=start+len(order[start:start+c['batch_size']])
            score=float(state['epoch']) if encoder else tune_score(model,*tune,c,device,budget)
            if encoder or state['best_score'] is None or score<state['best_score']:
                state['best_score']=score;state['best_epoch']=state['epoch']+1
                save_torch(best,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=state['epoch']+1,tune_wape=None if encoder else score))
            state['history'].append(dict(epoch=state['epoch']+1,tune_wape=None if encoder else score))
            state['epoch']+=1;state['offset']=0;checkpoint()
            write_json(folder/(name+'.training.json'),{k:state[k] for k in ['epoch','best_score','best_epoch','history']})
            print(name,'epoch',state['epoch'],'tune_wape',None if encoder else score,flush=True)
    except BudgetExpired:
        checkpoint();raise
    seal(folder,name+'.complete.json',[best,folder/(name+'.training.json')],{'status':'trained','epochs':epochs})
    model.load_state_dict(load_torch(best,device)['model'])

def memory(encoder,windows,c,device,budget,folder):
    folder=Path(folder)
    if verify(folder,'memory.complete.json'):return
    checkpoint=folder/'checkpoints_CL/checkpoint_retail.pth';checkpoint.parent.mkdir(parents=True,exist_ok=True)
    save_torch(checkpoint,{k:v.detach().cpu() for k,v in encoder.state_dict().items()})
    feature_path=folder/'all_features.npy'
    if not verify(folder,'features.complete.json'):
        a=np.lib.format.open_memmap(feature_path,mode='w+',dtype='float32',shape=(len(windows),16));encoder.eval()
        with torch.no_grad():
            for start in range(0,len(windows),c['batch_size']):
                budget.tick();x,_,_=windows.batch(np.arange(start,min(start+c['batch_size'],len(windows))))
                sd=x.std(1,keepdims=True);sd[sd==0]=1e-6;x=(x-x.mean(1,keepdims=True))/sd
                a[start:start+len(x)]=torch.nn.functional.normalize(encoder(torch.as_tensor(x,device=device)),dim=1).cpu().numpy()
        a.flush();del a;seal(folder,'features.complete.json',[feature_path])
    features=np.load(feature_path,mmap_mode='r');z=unit(features);state_path=folder/'medoids.progress.json'
    if state_path.exists():
        state=json.loads(state_path.read_text());medoids=np.array(state['medoids']);iteration=state['iteration']
    else:medoids=initialize(z,c['medoids']);iteration=0
    converged=state.get('converged',False) if state_path.exists() else False;objective=None
    while iteration<c['medoid_max_iter'] and not converged:
        budget.tick();labels,objective=assign(z,medoids,c['distance_block'],budget.tick);new=update(z,medoids,labels)
        converged=np.array_equal(new,medoids);medoids=new;iteration+=1
        write_json(state_path,dict(medoids=medoids.tolist(),iteration=iteration,converged=converged,objective_before_update=objective))
        print('medoids iteration',iteration,'objective',objective,'converged',converged,flush=True)
        if converged:break
    if len(medoids)<c['topk']:raise ValueError('Bank smaller than fixed retrieval K')
    x,y,_=windows.batch(medoids)
    def normalize(v):
        sd=v.std(1,keepdims=True);sd[sd==0]=1e-6;return ((v-v.mean(1,keepdims=True))/sd).astype(np.float32)
    bank=folder/'GMB/retail';bank.mkdir(parents=True,exist_ok=True)
    np.save(bank/'past_96.npy',normalize(x));np.save(bank/'future_96_720.npy',normalize(y));np.save(bank/'feature_96.npy',np.asarray(features[medoids]))
    write_json(folder/'memory.json',dict(candidate_windows=len(windows),medoids=len(medoids),iterations=iteration,converged=converged,origins=windows.origins,selected_flat_indices=medoids.tolist(),algorithm='exact cosine alternate K-medoids; stable heuristic initialization; full candidate pool'))
    seal(folder,'memory.complete.json',[checkpoint,folder/'memory.json',*bank.glob('*.npy')])

def predict(model,panel,c,device,budget,folder,name):
    for split in ['dev','test']:
        for origin in panel.schedule[split]:
            path=Path(folder)/'predictions'/name/(origin+'.parquet');mark=path.with_suffix('.complete.json')
            if verify(path.parent,mark.name):continue
            budget.tick();x,_,keys=panel.window(origin,labels=False)
            pred=forecast(model,x,device,c['batch_size'],budget.tick,decoder_check=True).ravel()
            keys[name+'_raw']=pred.astype(np.float64);keys[name]=np.maximum(pred,0).astype(np.float64);parquet(path,keys)
            seal(path.parent,mark.name,[path],{'origin':origin,'split':split,'decoder_label_isolation':True})

def run_seed(panel,c,device,budget,out,seed):
    folder=Path(out)/('seed_'+str(seed));folder.mkdir(parents=True,exist_ok=True)
    if verify(folder,'complete.json'):return
    torch.manual_seed(seed);np.random.seed(seed)
    train_origins=panel.schedule['train'];cut=len(train_origins)//2
    bank_windows=Windows(panel,train_origins[:cut]);fusion=Windows(panel,train_origins[cut:])
    api,pcl=author_api();encoder=pcl['PastSeriesEncoder'](56).to(device)
    train(encoder,bank_windows,c,device,budget,folder,'encoder',seed,c['encoder_epochs'],encoder=True)
    memory(encoder,bank_windows,c,device,budget,folder)
    for name in ['pfrp','dlinear']:
        torch.manual_seed(seed)
        model=construct(folder,c,device) if name=='pfrp' else api['DLinear'](settings(c)).to(device)
        train(model,fusion,c,device,budget,folder,name,seed,c['fusion_epochs'],tune=(panel,panel.schedule['tune']))
        predict(model,panel,c,device,budget,folder,name)
        del model
    seal(folder,'complete.json',[p for p in folder.rglob('*') if p.is_file() and (p.name.endswith('.best.pt') or p.suffix in ['.json','.parquet','.npy','.pth'])],{'status':'all_dev_test_predictions_complete','seed':seed})

def score(panel,c,out):
    out=Path(out);weekly=[];methods=['pfrp','dlinear']
    for seed in c['seeds']:
        folder=out/('seed_'+str(seed))
        if not verify(folder,'complete.json'):raise ValueError('Seed incomplete '+str(seed))
        for origin in panel.schedule['test']:
            merged=panel.truth(origin)
            for name in methods:
                p=pd.read_parquet(folder/'predictions'/name/(origin+'.parquet'));merged=checked_merge(merged,p)
            for name in methods:
                weekly.append(dict(seed=seed,origin=origin,method=name,**metric(merged.actual,merged[name]),negative_raw_rows=int((merged[name+'_raw']<0).sum())))
    overall=[]
    for seed in c['seeds']:
        for name in methods:
            subset=[r for r in weekly if r['seed']==seed and r['method']==name]
            n=sum(r['rows'] for r in subset);total=sum(r['actual_sum'] for r in subset);ae=sum(r['absolute_error_sum'] for r in subset)
            overall.append(dict(seed=seed,method=name,rows=n,actual_sum=total,absolute_error_sum=ae,wape=ae/total,mae=ae/n))
    write_json(out/'metrics.json',dict(status='complete_test',overall=overall,weekly=weekly))
    seal(out,'complete.json',[out/'metrics.json',out/'manifest.json',*[p for p in out.glob('seed_*/complete.json')]],{'status':'complete_test'})
