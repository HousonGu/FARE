import json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from author import construct
from data import Windows
from retrieval import retrieve_week
from io_utils import write_json,parquet,seal,verify,metric,checked_merge

class BudgetExpired(Exception):pass
class Budget:
    def __init__(self,seconds):self.end=time.monotonic()+seconds
    def tick(self):
        if time.monotonic()>=self.end:raise BudgetExpired('Time budget exhausted; repeat same command/output to resume')
def save_torch(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(value,tmp);os.replace(tmp,path)
def load_torch(path,device):return torch.load(path,map_location=device,weights_only=False)

class TrainingWindows(Windows):
    def __init__(self,bank,origins,seed,folder,device,budget):
        super().__init__(bank.panel,origins);self.bank=bank
        self.retrieval=np.concatenate([np.asarray(retrieve_week(bank,o,seed,folder,device,budget)) for o in origins])
    def tensors(self,idx,device):
        x,y,_=self.batch(idx);rows=self.flat_rows[idx]
        return [torch.tensor(v,device=device) for v in [self.bank.normalize(x,rows),self.bank.normalize(y,rows),self.retrieval[idx]]]

def predict_origin(model,bank,origin,seed,folder,device,budget,c):
    rows=bank.panel.rows(origin);x,_,keys=bank.panel.window(origin,labels=False)
    retrieved=retrieve_week(bank,origin,seed,folder,device,budget);outputs=[];model.eval()
    with torch.no_grad():
        for start in range(0,len(rows),c['batch_size']):
            budget.tick();end=min(len(rows),start+c['batch_size'])
            a=torch.tensor(bank.normalize(x[start:end],rows[start:end]),device=device);r=torch.tensor(np.array(retrieved[start:end]),device=device)
            pred=model(a,r)
            if not torch.isfinite(pred).all():raise ValueError('Nonfinite prediction')
            outputs.append(bank.restore(pred.cpu().numpy(),rows[start:end]))
    return np.concatenate(outputs).ravel(),keys

def tune(model,bank,seed,folder,device,budget,c):
    error=0.;total=0.
    for origin in bank.panel.schedule['tune']:
        p,_=predict_origin(model,bank,origin,seed,folder,device,budget,c);y=bank.panel.truth(origin).actual.to_numpy()
        error+=np.abs(y-np.maximum(p,0)).sum();total+=y.sum()
    if total<=0:raise ValueError('Zero tune target sum')
    return float(error/total)

def train(model,windows,bank,seed,folder,device,budget,c,name):
    folder=Path(folder);last=folder/(name+'.last.pt');best=folder/(name+'.best.pt')
    if verify(folder,name+'.complete.json'):
        model.load_state_dict(load_torch(best,device)['model']);return
    opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate']);state=dict(epoch=0,offset=0,best_score=None,history=[])
    if last.exists():
        state=load_torch(last,device);model.load_state_dict(state.pop('model'));opt.load_state_dict(state.pop('optimizer'))
        torch.set_rng_state(state.pop('rng').cpu());gpu_rng=state.pop('cuda_rng',None)
        if device.startswith('cuda') and gpu_rng is not None:torch.cuda.set_rng_state_all([v.cpu() for v in gpu_rng])
    def checkpoint():save_torch(last,dict(state,model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None))
    try:
        while state['epoch']<c['epochs']:
            # Author type1 is called after each epoch: epoch 1 and 2 both use initial lr.
            lr=c['learning_rate']*(.5**max(0,state['epoch']-1))
            for group in opt.param_groups:group['lr']=lr
            order=np.random.default_rng(seed+state['epoch']).permutation(len(windows));model.train()
            for start in range(state['offset'],len(order),c['batch_size']):
                budget.tick();idx=order[start:start+c['batch_size']];x,y,r=windows.tensors(idx,device)
                loss=(model(x,r)-y).square().mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite objective')
                opt.zero_grad();loss.backward();opt.step();state['offset']=start+len(idx)
            value=tune(model,bank,seed,folder,device,budget,c)
            if state['best_score'] is None or value<state['best_score']:
                state['best_score']=value;save_torch(best,dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=state['epoch']+1,tune_wape=value))
            state['history'].append(dict(epoch=state['epoch']+1,tune_wape=value,learning_rate=lr));state['epoch']+=1;state['offset']=0;checkpoint()
            write_json(folder/(name+'.training.json'),state);print(seed,name,'epoch',state['epoch'],'tune_wape',value,flush=True)
    except BudgetExpired:checkpoint();raise
    seal(folder,name+'.complete.json',[best,folder/(name+'.training.json')],{'status':'trained'})
    model.load_state_dict(load_torch(best,device)['model'])

def run_seed(bank,c,device,budget,out,seed):
    folder=Path(out)/('seed_'+str(seed));folder.mkdir(parents=True,exist_ok=True)
    if verify(folder,'complete.json'):return
    origins=bank.panel.schedule['train'][len(bank.panel.schedule['train'])//2:]
    windows=TrainingWindows(bank,origins,seed,folder,device,budget)
    for name in ['saraf','linear_control']:
        torch.manual_seed(seed);model=construct(c,device,control=name=='linear_control')
        train(model,windows,bank,seed,folder,device,budget,c,name)
        for split in ['dev','test']:
            for origin in bank.panel.schedule[split]:
                p=folder/'predictions'/name/(origin+'.parquet')
                if verify(p.parent,origin+'.complete.json'):continue
                pred,keys=predict_origin(model,bank,origin,seed,folder,device,budget,c);keys[name+'_raw']=pred;keys[name]=np.maximum(pred,0)
                parquet(p,keys);seal(p.parent,origin+'.complete.json',[p],{'origin':origin,'split':split})
        del model
    seal(folder,'complete.json',[p for p in folder.rglob('*') if p.is_file() and (p.name.endswith('.best.pt') or p.suffix in ['.json','.parquet','.npy'])],{'status':'full_seed_complete','seed':seed})

def score(bank,c,out):
    out=Path(out);weekly=[];methods=['saraf','linear_control']
    for seed in c['seeds']:
        folder=out/('seed_'+str(seed))
        if not verify(folder,'complete.json'):raise ValueError('Incomplete seed')
        for origin in bank.panel.schedule['test']:
            merged=bank.panel.truth(origin)
            for name in methods:merged=checked_merge(merged,pd.read_parquet(folder/'predictions'/name/(origin+'.parquet')))
            for name in methods:
                weekly.append(dict(seed=seed,origin=origin,method=name,**metric(merged.actual,merged[name]),negative_raw_rows=int((merged[name+'_raw']<0).sum())))
    overall=[]
    for seed in c['seeds']:
        for name in methods:
            r=[v for v in weekly if v['seed']==seed and v['method']==name];n=sum(v['rows'] for v in r);s=sum(v['actual_sum'] for v in r);e=sum(v['absolute_error_sum'] for v in r)
            overall.append(dict(seed=seed,method=name,rows=n,actual_sum=s,absolute_error_sum=e,wape=e/s,mae=e/n))
    write_json(out/'metrics.json',dict(status='complete_test',method=c['method'],overall=overall,weekly=weekly,bank=bank.meta))
    seal(out,'complete.json',[out/'manifest.json',out/'metrics.json',out/'bank/complete.json',*out.glob('seed_*/complete.json')],{'status':'complete_test'})
