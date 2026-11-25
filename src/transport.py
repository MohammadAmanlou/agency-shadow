"""Causal history-bias transport certificates; synthetic tests only."""
import json
import math
from pathlib import Path
import numpy as np


def certificate(proxy_lo, proxy_hi, edges, anchors=None):
    """Outer bounds on clean means and contrasts using difference constraints.

    edges: (i,j,rho) certify |(proxy_i-clean_i)-(proxy_j-clean_j)| <= rho.
    Exact/sharp if proxy_lo==proxy_hi and the supplied constraints define the class.
    With sampling intervals this is a conservative outer confidence set.
    D[i,j] bounds clean_j-clean_i; last node has fixed value zero.
    """
    lo, hi = np.asarray(proxy_lo,float), np.asarray(proxy_hi,float)
    n=len(lo)
    if lo.ndim!=1 or lo.shape!=hi.shape or not n or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo>hi) or np.any(lo<0) or np.any(hi>1):
        raise ValueError('invalid proxy intervals')
    D=np.full((n+1,n+1),np.inf);np.fill_diagonal(D,0.)
    D[n,:n]=1.;D[:n,n]=0.
    for i,j,rho in edges:
        if not (0<=i<n and 0<=j<n and np.isfinite(rho) and rho>=0):raise ValueError('invalid edge')
        D[i,j]=min(D[i,j],hi[j]-lo[i]+rho)
        D[j,i]=min(D[j,i],hi[i]-lo[j]+rho)
    for i,(a,b) in (anchors or {}).items():
        if not 0<=i<n or not 0<=a<=b<=1:raise ValueError('invalid anchor')
        D[n,i]=min(D[n,i],b);D[i,n]=min(D[i,n],-a)
    for k in range(n+1):D=np.minimum(D,D[:,k,None]+D[None,k,:])
    if np.any(np.diag(D)<-1e-9):raise ValueError('incompatible transport/anchor constraints')
    return D


def ci(means,n,alpha):
    means=np.asarray(means,float)
    if n<=0 or not 0<alpha<1:raise ValueError('invalid confidence parameters')
    r=math.sqrt(math.log(2*means.size/alpha)/(2*n))
    return np.maximum(0,means-r),np.minimum(1,means+r)


def run(out='results',seed=20261125,trials=2000):
    """Same total cost; no human/LLM data; falsify transport in one scenario."""
    rng=np.random.default_rng(seed)
    truth=np.array([.55,.66,.62,.58,.51,.44,.60]);k=len(truth)
    base_bias=np.array([.15,.16,.145,.155,.14,.15,.15])
    budget=28000; expensive=20; cheap=1
    n_clean=budget//(k*expensive);n_proxy=budget//(k*cheap)
    edges=[(0,i,.015) for i in range(1,k)];tau=.03;alpha=.05
    unsafe=truth[1:]-truth[0]<-tau; safe=~unsafe
    result={'seed':seed,'trials':trials,'truth':truth.tolist(),'clean_cost':expensive,'proxy_cost':cheap,'budget_each':budget,'clean_n_per_policy':n_clean,'proxy_n_per_policy':n_proxy,'rho':.015,'tau':tau,'alpha':alpha,'scenarios':{}}
    for name in ('valid_transport','violated_transport'):
        bias=base_bias.copy()
        if name=='violated_transport':bias[5]=.33
        proxy=truth+bias
        counts={m:{'false_release_trials':0,'safe_candidates_released':0,'simultaneous_contrast_coverage_trials':0} for m in ('direct_clean','proxy_naive','transport')}
        for _ in range(trials):
            a=rng.binomial(n_clean,truth)/n_clean;b=rng.binomial(n_proxy,proxy)/n_proxy
            for method,means,n in (('direct_clean',a,n_clean),('proxy_naive',b,n_proxy),('transport',b,n_proxy)):
                l,u=ci(means,n,alpha)
                if method=='transport':
                    D=certificate(l,u,edges); lower=-D[1:k,0];upper=D[0,1:k]
                else:lower=l[1:]-u[0];upper=u[1:]-l[0]
                release=lower>=-tau;delta=truth[1:]-truth[0]
                counts[method]['false_release_trials']+=int(np.any(release&unsafe))
                counts[method]['safe_candidates_released']+=int(np.sum(release&safe))
                counts[method]['simultaneous_contrast_coverage_trials']+=int(np.all((lower<=delta)&(delta<=upper)))
        result['scenarios'][name]={m:{'false_release_rate':v['false_release_trials']/trials,'mean_safe_releases':v['safe_candidates_released']/trials,'simultaneous_coverage':v['simultaneous_contrast_coverage_trials']/trials} for m,v in counts.items()}
    Path(out).mkdir(exist_ok=True,parents=True);(Path(out)/'transport.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':run()
