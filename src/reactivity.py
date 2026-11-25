"""V14: four-arm first-probe causal design and prediction-powered comparator.
All checked-in outcomes synthetic; never interpret as measured human behavior.
Order: incumbent/no prior probe, candidate/no probe, incumbent/prior, candidate/prior.
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.stats import norm
from src.paired_probe import cp, paired_ci, joint_probs

C = np.array([[-1,1,0,0],[0,0,-1,1],[1,-1,-1,1]])

def contrasts(p):
    return np.asarray(p) @ C.T

def box_contrasts(lo,hi):
    lo,hi=np.asarray(lo),np.asarray(hi)
    return lo@np.maximum(C,0).T+hi@np.minimum(C,0).T, hi@np.maximum(C,0).T+lo@np.minimum(C,0).T

def factorial_ci(s,n,alpha=.05):
    l,u=cp(s,n,alpha/4)
    return box_contrasts(l,u)

def masking_decision(l,u,tau=.03):
    # Joint CI: clean contrast below -tau, reactive contrast >= -tau.
    return (u[...,0]<-tau)&(l[...,1]>=-tau)

def sharp_missing_clean(reactive, b0=None,b1=None):
    """Known reactive means; optional absolute per-policy carryover bounds."""
    a,b=reactive
    if not 0<=a<=1 or not 0<=b<=1:raise ValueError('invalid means')
    x=1 if b0 is None else b0;y=1 if b1 is None else b1
    if not 0<=x<=1 or not 0<=y<=1:raise ValueError('invalid bounds')
    return max(0,b-y)-min(1,a+x),min(1,b+y)-max(0,a-x)

def bernkl(p,q):
    if not 0<p<1 or not 0<q<1:raise ValueError('interior means required')
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

def necessary_clean_cost(safe,unsafe,cost=1,alpha=.05,beta=.05):
    """Only clean candidate law changes; all other observations identical."""
    if not 0<alpha<1-beta<1 or cost<=0:raise ValueError('invalid design')
    return cost*bernkl(1-beta,alpha)/bernkl(safe,unsafe)

def wilson(s,n,alpha=.05):
    p=np.asarray(s)/n;z=norm.ppf(1-alpha/2);a=z*z/n;center=(p+a/2)/(1+a);rad=z*np.sqrt(p*(1-p)/n+a/(4*n))/(1+a)
    return center-rad,center+rad

def hybrid_paired(counts,alpha=.05):
    """Standard approximate hybrid Wilson score construction with correlation.
    Degenerate marginals use correlation zero; not exact coverage.
    """
    n=counts.sum(-1);a=counts[...,1]+counts[...,3];b=counts[...,2]+counts[...,3];pa=a/n;pb=b/n;d=pb-pa
    den=np.sqrt(pa*(1-pa)*pb*(1-pb));rho=np.divide(counts[...,3]/n-pa*pb,den,out=np.zeros_like(pa),where=den>0);rho=np.clip(rho,-1,1)
    al,au=wilson(a,n,alpha);bl,bu=wilson(b,n,alpha)
    x=pb-bl;y=au-pa;low=d-np.sqrt(np.maximum(0,x*x+y*y-2*rho*x*y));x=bu-pb;y=pa-al;high=d+np.sqrt(np.maximum(0,x*x+y*y-2*rho*x*y))
    return np.maximum(-1,low),np.minimum(1,high)

def ppi(y,f,fu,lam=1.,alpha=.05):
    """Frozen-predictor PPI-style mean with finite-sample Hoeffding CI.
    Independent labeled/unlabeled iid target populations; f in [0,1], binary y.
    Lambda fixed externally (e.g. independent pilot), not fitted here.
    """
    y,f,fu=map(np.asarray,(y,f,fu))
    if y.shape!=f.shape or not y.shape[-1] or not fu.shape[-1] or not 0<=lam<=1:raise ValueError('invalid arrays')
    est=(y-lam*f).mean(-1)+lam*fu.mean(-1)
    rad=math.sqrt(math.log(2/alpha)/2*((1+lam)**2/y.shape[-1]+lam**2/fu.shape[-1]))
    return np.clip(est-rad,0,1),np.clip(est+rad,0,1)

def ppi_normal(y,f,fu,lam=1.,alpha=.05):
    r=y-lam*f;est=r.mean(-1)+lam*fu.mean(-1);se=np.sqrt(r.var(-1,ddof=1)/r.shape[-1]+lam**2*fu.var(-1,ddof=1)/fu.shape[-1]);h=norm.ppf(1-alpha/2)*se
    return np.clip(est-h,0,1),np.clip(est+h,0,1)

def assignment(path,n=120,seed=20261125):
    if n<4 or n%4:raise ValueError('n must be a positive multiple of 4')
    rng=np.random.default_rng(seed)
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(['participant_id','policy','prior_probe','terminal_correct','status'])
        for block in range(n//4):
            for j,arm in enumerate(rng.permutation(4)):w.writerow([f'P{block*4+j+1:04d}',int(arm)%2,int(arm)//2,'',''])

def analyze(path):
    n=np.zeros(4,int);s=np.zeros(4,int);miss=np.zeros(4,int);seen=set()
    with open(path,newline='') as f:
        for r in csv.DictReader(f):
            pid=r['participant_id'];a=int(r['policy']);b=int(r['prior_probe']);v=r['terminal_correct'];st=r['status']
            if pid in seen or a not in (0,1) or b not in (0,1) or v not in ('','0','1') or st not in ('completed','withdrawn','missing'):raise ValueError('invalid or incomplete pilot row')
            if (st=='completed')!=(v!=''):raise ValueError('status/outcome mismatch')
            seen.add(pid);i=a+2*b;n[i]+=1;s[i]+=v=='1';miss[i]+=v==''
    if np.any(n==0):raise ValueError('all four arms required')
    # Only use records ethically authorized for retention. Noncompletion is not success.
    l,u=factorial_ci(s,n);lo,hi=box_contrasts(s/n,(s+miss)/n)
    return {'n':n.tolist(),'success':s.tolist(),'missing':miss.tolist(),'ITT_contrasts':contrasts(s/n).tolist(),'joint_CI':[l.tolist(),u.tolist()],'missing_outcome_identification_bounds':[lo.tolist(),hi.tolist()]}
