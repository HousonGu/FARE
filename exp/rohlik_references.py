"""Natural same-group bank with frozen Top-8 alternating retrieval."""
import numpy as np
import pandas as pd
from fare.retrieval import select

def references(data,origin,batch=8):
    from data_provider.rohlik import KEYS
    legal=[o for split in ['fit','dev','test'] for o in data.schedule[split] if pd.Timestamp(o)+pd.Timedelta(days=6)<pd.Timestamp(origin)]
    eligible=data.eligible(origin); chunks=[]
    reference_rows={past:data.eligible(past,reference=True) for past in legal}
    for members in data.groups:
        queries=np.intersect1d(members,eligible)
        if not len(queries): continue
        hs,ys,ps,ss=[],[],[],[]
        for past in legal:
            rows=np.intersect1d(members,reference_rows[past])
            if not len(rows): continue
            h,cov,_=data.window(past,rows)
            hs.append(h); ys.append(data.arr['Sales'][np.ix_(rows,data.columns(past)[56:])])
            ps.append(cov['Discount'][:,56:]); ss.append(cov['Holiday'][:,56:])
        size=sum(map(len,hs))
        if size>=8:
            h,y,p,s=map(np.concatenate,[hs,ys,ps,ss])
            scale=np.maximum(h.mean(1),1); norm=h/scale[:,None]; target=y/scale[:,None]
        for start in range(0,len(queries),batch):
            q,cov,keys=data.window(origin,queries[start:start+batch]); qs=np.maximum(q.mean(1),1)
            if size<8:
                shape_pred=factor_pred=q[:,-7:]; pdist=sdist=np.zeros(len(q))
            else:
                # Batched direct MSE retains the legacy ranking arithmetic, without truncating the bank.
                shape=((q[:,None,:]/qs[:,None,None]-norm[None,:,:])**2).mean(2)
                discount=(cov['Discount'][:,None,56:]!=p[None,:,:]).mean(2)
                school=(cov['Holiday'][:,None,56:]!=s[None,:,:]).mean(2)
                choice,dist=select(shape,discount,school)
                shape_pred=target[choice['shape']].mean(1)*qs[:,None]
                factor_pred=target[choice['factor']].mean(1)*qs[:,None]
                pdist=np.take_along_axis(dist['factor1_branch'],choice['factor'][:,::2],1).mean(1)
                sdist=np.take_along_axis(dist['holiday_branch'],choice['factor'][:,1::2],1).mean(1)
            keys['shape_retrieval']=shape_pred.ravel(); keys['factor_retrieval']=factor_pred.ravel()
            for name,val in [('history_mean',q.mean(1)),('history_std',q.std(1)),('history_zero_fraction',(q==0).mean(1)),('discount_distance',pdist),('holiday_distance',sdist)]: keys[name]=np.repeat(val,7)
            keys['bank_size']=size; keys['fallback']=size<8
            keys['discount']=cov['Discount'][:,56:].ravel(); keys['holiday']=cov['Holiday'][:,56:].ravel()
            chunks.append(keys)
    if not chunks: raise ValueError('No eligible queries at '+origin)
    return pd.concat(chunks,ignore_index=True).sort_values(KEYS).reset_index(drop=True)
