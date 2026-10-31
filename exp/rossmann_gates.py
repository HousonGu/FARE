"""Explicit Rossmann features, forward errors and shared-candidate controls."""
import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.optimize import minimize
from exp.reference_math import fit_gate,predict_gate,select_global_weights
from exp.static_controls import history_hard_control
from data_provider.rossmann import KEYS

def history(frame,parts,train):
    codes,ids=pd.factorize(frame.series_id,sort=True)
    sums=np.zeros((len(ids),3)); counts=np.zeros(len(ids)); total=np.zeros(3); count=0
    out=np.empty((len(frame),4)); latest=None
    for origin in sorted(frame.origin.unique()):
        rows=np.flatnonzero(frame.origin.eq(origin)); c=codes[rows]
        if latest is not None and latest>=pd.Timestamp(origin): raise ValueError('Overlapping development targets')
        prior=total/count if count else np.zeros(3)
        out[rows,:3]=(sums[c]+14*prior)/(counts[c,None]+14)
        out[rows,3]=np.log1p(counts[c])
        use=rows[train[rows]&np.isfinite(frame.actual.to_numpy()[rows])]
        if len(use):
            error=np.clip(np.abs(frame.actual.to_numpy()[use,None]-parts[use])/np.maximum(frame.history_mean.to_numpy()[use,None],1),0,10)
            for j in range(3): sums[:,j]+=np.bincount(codes[use],weights=error[:,j],minlength=len(ids))
            counts+=np.bincount(codes[use],minlength=len(ids)); total+=error.sum(0); count+=len(use)
            latest=pd.to_datetime(frame.iloc[use].target_date).max()
    return out,dict(series_ids=[int(i) for i in ids],sums=sums.tolist(),counts=counts.tolist(),global_sums=total.tolist(),global_count=count,shrink_rows=14)

def design(frame,train):
    parts=frame[['backbone_forecast','shape_retrieval','factor_retrieval']].to_numpy(float)
    scale=np.maximum(frame.history_mean.to_numpy(),1)
    names=['log_history_mean','history_cv','history_zero_fraction','log_bank_size','log_promo_distance','log_school_holiday_distance','promo','school_holiday','backbone_shape_disagreement','factor_shape_disagreement']
    raw=np.column_stack([np.log1p(frame.history_mean),np.clip(frame.history_std/scale,0,10),frame.history_zero_fraction,np.log1p(frame.bank_size),np.log1p(frame.promo_distance),np.log1p(frame.school_holiday_distance),frame.promo,frame.school_holiday,np.clip(abs(parts[:,0]-parts[:,1])/scale,0,10),np.clip(abs(parts[:,2]-parts[:,1])/scale,0,10)])
    hist,state=history(frame,parts,train)
    raw=np.column_stack([raw,hist]); names+=['hist_backbone_error','hist_shape_error','hist_factor_error','hist_log_support']
    continuous=[i for i,n in enumerate(names) if n not in ['promo','school_holiday']]
    mean=raw[train][:,continuous].mean(0); std=np.maximum(raw[train][:,continuous].std(0),1e-8)
    raw[:,continuous]=(raw[:,continuous]-mean)/std
    if not np.isfinite(raw).all() or not np.isfinite(parts).all(): raise ValueError('Nonfinite design')
    return np.column_stack([np.ones(len(raw)),raw]),parts,dict(names=names,continuous=continuous,mean=mean.tolist(),std=std.tolist(),history=state)

def flat_objective(theta,x,y,parts):
    k=x.shape[1]; w=softmax(np.column_stack([np.zeros(len(x)),x@theta[:k],x@theta[k:]]),axis=1)
    pred=(parts*w).sum(1); residual=pred-y; smooth=np.sqrt(residual**2+1); g=residual/smooth/y.sum()
    grad=np.r_[x.T@(g*w[:,1]*(parts[:,1]-pred)),x.T@(g*w[:,2]*(parts[:,2]-pred))]
    mask=np.ones(2*k,bool); mask[[0,k]]=False
    grad[mask]+=.002*theta[mask]
    return smooth.sum()/y.sum()+.001*np.square(theta[mask]).sum(),grad

def run_gates(frame,dev):
    frame=frame.sort_values(KEYS).reset_index(drop=True).copy()
    train=frame.origin.isin(dev).to_numpy(); frame.loc[~train,'actual']=np.nan
    x,parts,meta=design(frame,train); y=frame.actual.to_numpy(float); fit=train&np.isfinite(y)
    if not fit.any() or y[fit].sum()<=0: raise ValueError('No usable development labels')
    out=frame.loc[~train,KEYS].copy(); fits={}
    out['backbone']=parts[~train,0]; out['shape_half']=parts[~train][:,[0,1]].mean(1); out['factor_half']=parts[~train][:,[0,2]].mean(1)
    weights,wape=select_global_weights(y[fit],parts[fit]); out['global_static']=parts[~train]@(np.asarray(weights)/20)
    fits['global_static']=dict(weights=list(weights),development_wape=wape)
    out['history_hard'],fits['history_hard']=history_hard_control(frame,parts,y,fit,weights,~train)
    for name,xx in [('soft_history',x),('soft_no_history',x[:,:-4])]:
        theta,record=fit_gate(xx[fit],y[fit],*parts[fit].T)
        pred,beta,alpha=predict_gate(xx[~train],theta,*parts[~train].T)
        out[name]=pred; fits[name]=record
        if name=='soft_history': out['alpha']=alpha; out['beta']=beta
    theta=np.zeros(2*x.shape[1]); theta[[0,x.shape[1]]]=-np.log(2)
    result=minimize(flat_objective,theta,args=(x[fit],y[fit],parts[fit]),jac=True,method='L-BFGS-B',options=dict(maxiter=250,ftol=1e-9,gtol=1e-6))
    if not result.success: raise RuntimeError('Softmax failed: '+str(result.message))
    k=x.shape[1]; w=softmax(np.column_stack([np.zeros((~train).sum()),x[~train]@result.x[:k],x[~train]@result.x[k:]]),axis=1)
    out['flat_softmax']=(w*parts[~train]).sum(1)
    fits['flat_softmax']=dict(parameters=result.x.tolist(),iterations=int(result.nit))
    return out,dict(design=meta,fits=fits,full_parameter_count=2*k,development_labeled_rows=int(fit.sum()))
