"""Exact cosine alternate K-medoids without allocating an N x N distance matrix.

For unit vectors, sum_j distance(i,j) = n - dot(x_i,sum_j x_j).
Zero-vector self-distance is corrected to zero, matching cosine distance matrices.
Stable row order resolves ties. Empty clusters retain their previous medoid.
"""
import numpy as np

def unit(features):
    x=np.asarray(features,dtype=np.float64)
    norm=np.linalg.norm(x,axis=1,keepdims=True)
    return np.divide(x,norm,out=np.zeros_like(x),where=norm>0)
def costs(x):
    return len(x)-x@x.sum(axis=0)-(np.linalg.norm(x,axis=1)==0)
def initialize(x,k):
    return np.argsort(costs(x),kind='stable')[:min(k,len(x))]
def assign(x,medoids,block=512,tick=lambda:None):
    labels=np.empty(len(x),np.int32);nearest=np.empty(len(x),np.float64)
    for start in range(0,len(x),block):
        tick();end=min(start+block,len(x));d=np.clip(1-x[start:end]@x[medoids].T,0,2)
        j=np.flatnonzero((medoids>=start)&(medoids<end));d[medoids[j]-start,j]=0
        labels[start:end]=d.argmin(axis=1);nearest[start:end]=d[np.arange(end-start),labels[start:end]]
    return labels,float(nearest.sum())
def update(x,medoids,labels):
    new=medoids.copy()
    order=np.argsort(labels,kind='stable');ends=np.cumsum(np.bincount(labels,minlength=len(medoids)));starts=np.r_[0,ends[:-1]]
    for j,m in enumerate(medoids):
        rows=order[starts[j]:ends[j]]
        if len(rows):
            c=costs(x[rows]);best=int(c.argmin())
            old=np.flatnonzero(rows==m)
            if not len(old) or c[best]<c[old[0]]:new[j]=rows[best]
    if len(np.unique(new))!=len(new):raise ValueError('Duplicate medoids')
    return new
