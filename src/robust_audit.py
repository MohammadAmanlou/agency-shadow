"""Fault-tolerant transport certificates and adaptive first-probe audits.
All experiments are synthetic. k is an external sensitivity assumption, not learned.
"""
from itertools import combinations
import json
import math
from pathlib import Path
import numpy as np
def lifted_certificate(lo, hi, edges, anchors=None):
    """Exact difference-constraint representation, even with proxy intervals.
    Nodes 0..n-1: clean means; n..2n-1: negative biases; 2n: zero.
    Return all-pairs upper bounds on x[j]-x[i].
    """
    lo,hi=np.asarray(lo,float),np.asarray(hi,float)
    if lo.ndim!=1 or not len(lo) or lo.shape!=hi.shape or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo<0) or np.any(hi>1) or np.any(lo>hi):
        raise ValueError('invalid proxy intervals')
    n=len(lo);z=2*n;D=np.full((z+1,z+1),np.inf);np.fill_diagonal(D,0.)
    for i in range(n):
        a,b=(anchors or {}).get(i,(0.,1.))
        if not 0<=a<=b<=1:raise ValueError('invalid anchor')
        D[z,i]=b;D[i,z]=-a
        D[n+i,i]=hi[i];D[i,n+i]=-lo[i]
    for i,j,r in edges:
        if not (0<=i<n and 0<=j<n and np.isfinite(r) and r>=0):raise ValueError('invalid edge')
        D[n+i,n+j]=min(D[n+i,n+j],r);D[n+j,n+i]=min(D[n+j,n+i],r)
    for k in range(z+1):D=np.minimum(D,D[:,k,None]+D[None,k,:])
    if np.any(np.diag(D)<-1e-9):raise ValueError('incompatible constraints')
    return D


def robust_bounds(lo, hi, edges, k, anchors=None):
    """Union over feasible models discarding at most k undirected constraints.
    Sharp for both point and interval proxy means. None = empty class.
    Returns (lower, upper, number of feasible deletion sets), relative to node 0.
    """
    if not isinstance(k,int) or not 0<=k<=len(edges):
        raise ValueError('invalid deletion budget')
    low=[]; high=[]
    # Exactly k suffices: removing additional constraints never removes a feasible point.
    for omit in combinations(range(len(edges)),k):
        keep=[e for i,e in enumerate(edges) if i not in omit]
        try:D=lifted_certificate(lo,hi,keep,anchors)
        except ValueError as e:
            if 'incompatible' not in str(e):raise
            continue
        low.append(-D[1:len(lo),0]);high.append(D[0,1:len(lo)])
    if not low:return None
    return np.min(low,axis=0),np.max(high,axis=0),len(low)


def sentinel_edge(lo, hi, anchors, i, j, rho):
    """Simultaneous bias-difference interval from independent clean/proxy CIs.
    No extrapolation to unmeasured edges. A radius can be certified or falsified,
    otherwise the edge remains unresolved. CI validity is inherited from inputs.
    """
    ai,bi=anchors[i];aj,bj=anchors[j]
    lower=lo[j]-hi[i]-bj+ai;upper=hi[j]-lo[i]-aj+bi
    status='certified' if lower>=-rho and upper<=rho else ('falsified' if lower>rho or upper< -rho else 'unresolved')
    return lower,upper,status


def radius(n,delta):
    return math.sqrt(math.log(2/delta)/(2*n))


def interval(s,n,delta):
    if n==0:return (0.,1.)
    r=radius(n,delta);return max(0.,s/n-r),min(1.,s/n+r)


def audit(truth,proxy,edges,k,budget,rng,*,proxy_n=2000,batch=100,cost=20,alpha=.05,tau=.03,mode='robust'):
    """Finite-look adaptive gate; no optional-stopping loophole.
    All per-node sample sizes are multiples of batch; union over all possible looks.
    Modes: direct adaptive, robust conditional; empty model class => direct fallback.
    """
    j=len(truth);spend=0; n=np.zeros(j,dtype=int);s=np.zeros(j,dtype=int)
    if mode not in ('direct','robust'):raise ValueError('invalid mode')
    if mode=='robust':
        spend=j*proxy_n
        if spend>budget:raise ValueError('proxy budget exceeds total')
        ps=rng.binomial(proxy_n,proxy)
        pi=[interval(x,proxy_n,alpha/(2*j)) for x in ps];lo,hi=np.array(pi).T
    else:lo=np.zeros(j);hi=np.ones(j)
    max_looks=max(1,budget//(batch*cost));d=alpha/(2*j*max_looks)
    fallback=False;looks=0
    while True:
        anchors={i:interval(s[i],n[i],d) for i in range(j)}
        dl=np.array([anchors[i][0]-anchors[0][1] for i in range(1,j)])
        du=np.array([anchors[i][1]-anchors[0][0] for i in range(1,j)])
        if mode=='robust' and not fallback:
            ans=robust_bounds(lo,hi,edges,k,anchors)
            if ans is None:fallback=True;lower,upper=dl,du
            else:lower,upper,_=ans
        else:lower,upper=dl,du
        unresolved=np.flatnonzero((lower < -tau)&(upper>=-tau))+1
        if not len(unresolved) or spend+batch*cost>budget:break
        # Query only unresolved contenders, share reference observations across them.
        # A fixed heuristic; no optimal allocation claim.
        candidates=[0]+unresolved.tolist()
        q=max(candidates,key=lambda i: ((anchors[i][1]-anchors[i][0])*(min(len(unresolved),2) if i==0 else 1),-n[i],-i))
        s[q]+=rng.binomial(batch,truth[q]);n[q]+=batch;spend+=batch*cost;looks+=1
    safe=truth[1:]-truth[0]>=-tau
    passed=lower>=-tau
    return {'false':int(np.any(passed&~safe)),'safe_passes':int(np.sum(passed&safe)),
            'coverage':int(np.all((lower<=truth[1:]-truth[0]+1e-10)&(upper>=truth[1:]-truth[0]-1e-10))),
            'spend':spend,'clean_n':int(n.sum()),'fallback':int(fallback)}


def run(out='results',seed=20261126,trials=300):
    truth=np.array([.55,.72,.67,.60,.46,.40,.68]);j=len(truth)
    # Tight within-cluster constraints; cross-cluster links more permissive.
    edges=[(0,1,.01),(1,2,.01),(0,2,.01),(0,3,.01),(2,3,.01),
           (3,4,.20),(4,5,.01),(5,6,.20),(3,6,.01),(0,6,.01),(0,5,.01)]
    budget=70000;tau=.03;results={}
    for name,bias in [('valid',np.full(j,.10)),('one_bad_edge',np.array([.1,.1,.1,.1,.28,.28,.1])),('many_bad_edges',np.array([.1,.1,.1,.1,.35,.40,.1]))]:
        proxy=truth+bias
        bad=sum(abs(bias[i]-bias[l])>r+1e-10 for i,l,r in edges)
        block={'true_bad_edges':int(bad),'methods':{}}
        for method,k in [('direct',0),('transport_k0',0),('robust_k1',1)]:
            rng=np.random.default_rng(seed);rows=[]
            for _ in range(trials):
                rows.append(audit(truth,proxy,edges,k,budget,rng,tau=tau,mode='direct' if method=='direct' else 'robust'))
            block['methods'][method]={key:float(np.mean([r[key] for r in rows])) for key in rows[0]}
        results[name]=block
    result={'seed':seed,'trials_per_method_scenario':trials,'budget':budget,'clean_cost':20,'proxy_n_per_version':2000,'tau':tau,'alpha':.05,'truth':truth.tolist(),'edges':edges,'scenarios':results}
    Path(out).mkdir(parents=True,exist_ok=True);(Path(out)/'robust_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':run()
