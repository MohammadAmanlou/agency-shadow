"""V12: self-calibrating composite noninferiority gate (synthetic only)."""
from itertools import combinations
from functools import lru_cache
import math
import json
from pathlib import Path
import numpy as np
from src.robust_audit import lifted_certificate, sentinel_edge, interval
from src.cut_audit import upper_bad_count

@lru_cache(maxsize=4096)
def count_upper(M, n, x, delta):
    return upper_bad_count(M, n, x, delta)


def calibration(lo, hi, anchors, edges, sample, delta):
    """Frozen uniformly sampled edge indices, no replacement; no oracle labels."""
    M=len(edges); sample=tuple(sample)
    if len(set(sample))!=len(sample) or any(not 0<=i<M for i in sample):
        raise ValueError('invalid edge sample')
    status=[sentinel_edge(lo,hi,anchors,*e)[2] for e in edges]
    good=[i for i,s in enumerate(status) if s=='certified']
    bad=[i for i,s in enumerate(status) if s=='falsified']
    unknown=[i for i,s in enumerate(status) if s=='unresolved']
    x=sum(status[i]!='certified' for i in sample)
    U=min(M-len(good),count_upper(M,len(sample),x,delta))
    return good,bad,unknown,U


def composite(lo, hi, anchors, edges, good, bad, unknown, U, cap=200):
    """Sharp envelope of feasible graphs; direct outer bound on cap/infeasibility."""
    J=len(lo)
    dl=np.array([anchors[i][0]-anchors[0][1] for i in range(1,J)])
    du=np.array([anchors[i][1]-anchors[0][0] for i in range(1,J)])
    k=U-len(bad)
    if k<0:return dl,du,'inconsistent'
    k=min(k,len(unknown))
    if k==len(unknown):return dl,du,'direct_no_transfer'
    if math.comb(len(unknown),k)>cap:return dl,du,'direct_cap'
    lower=[];upper=[]
    for drop in combinations(unknown,k):
        keep=good+[i for i in unknown if i not in drop]
        try:D=lifted_certificate(lo,hi,[edges[i] for i in keep],anchors)
        except ValueError as e:
            if 'incompatible' not in str(e):raise
            continue
        lower.append(-D[1:J,0]);upper.append(D[0,1:J])
    if not lower:return dl,du,'inconsistent'
    return np.maximum(dl,np.min(lower,axis=0)),np.minimum(du,np.max(upper,axis=0)),'graph'


def run_gate(mu, proxy, edges, budget, rng, mode='calibrated', *, sample_size=6, proxy_n=3000, batch=1000, cost=20, alpha=.05, tau=.03, oracle_k=None):
    """All clean draws cost money, including edge calibration and reference.
    Finite-look simultaneous intervals; graph and cohort population frozen.
    Sampling is adaptive, but each per-node stream is iid. Oracle is diagnostic only.
    """
    mu,proxy=np.asarray(mu,float),np.asarray(proxy,float);J=len(mu);M=len(edges)
    if mode not in ('direct','calibrated','targeted','oracle') or J<2 or not 0<alpha<1 or budget<0 or batch<1 or cost<1:raise ValueError('invalid design')
    if not 0<=sample_size<=M:raise ValueError('invalid sample size')
    if mode=='oracle' and (oracle_k is None or not 0<=oracle_k<=M):raise ValueError('oracle k required')
    n=np.zeros(J,int);s=np.zeros(J,int);spend=0;looks=0;graph_looks=0
    if mode=='direct':lo=np.zeros(J);hi=np.ones(J);sample=()
    else:
        spend=J*proxy_n
        if spend>budget:raise ValueError('proxy cost exceeds budget')
        sample=tuple(rng.choice(M,sample_size,replace=False))
        ps=rng.binomial(proxy_n,proxy);lo,hi=np.array([interval(x,proxy_n,alpha/(3*J)) for x in ps]).T
    max_looks=max(1,budget//(cost*batch));d=alpha/(3*J*max_looks)
    while True:
        anchors={i:interval(s[i],n[i],d) for i in range(J)}
        good=[];bad=[];unknown=list(range(M));U=M
        if mode in ('calibrated','targeted'):good,bad,unknown,U=calibration(lo,hi,anchors,edges,sample,alpha/3)
        elif mode=='oracle':U=oracle_k
        if mode=='direct':
            lower=np.array([anchors[i][0]-anchors[0][1] for i in range(1,J)])
            upper=np.array([anchors[i][1]-anchors[0][0] for i in range(1,J)])
            status='direct'
        else:lower,upper,status=composite(lo,hi,anchors,edges,good,bad,unknown,U)
        graph_looks+=int(status=='graph')
        unresolved=np.flatnonzero((lower < -tau)&(upper>=-tau))+1
        if not len(unresolved) or spend+batch*cost>budget:break
        # Both methods use the same direct allocation. Calibration reuses those
        # samples; no benefit from pretending an edge label is a free observation.
        candidates=[0]+unresolved.tolist()
        i=max(candidates,key=lambda i:((anchors[i][1]-anchors[i][0])*(min(len(unresolved),2) if i==0 else 1),-n[i],-i))
        # Spend the first half of the clean budget on sampled-edge validation.
        if mode=='targeted' and spend < (budget+J*proxy_n)/2:
            degrees=np.zeros(J)
            for e in sample:
                if e in unknown:
                    a,b,_=edges[e];degrees[a]+=1;degrees[b]+=1
            if degrees.max()>0:
                i=max(range(J),key=lambda v:(degrees[v]*(anchors[v][1]-anchors[v][0]),-n[v],-v))
        s[i]+=rng.binomial(batch,mu[i]);n[i]+=batch;spend+=batch*cost;looks+=1
    contrast=mu[1:]-mu[0];passed=lower>=-tau;safe=contrast>=-tau
    true_k=sum(abs((proxy[i]-mu[i])-(proxy[j]-mu[j]))>r+1e-10 for i,j,r in edges)
    return {'false_pass':int(np.any(passed&~safe)),'safe_passes':int(np.sum(passed&safe)),
            'coverage':int(np.all((lower<=contrast+1e-10)&(upper>=contrast-1e-10))),
            'k_coverage':int(U>=true_k),'final_U':U,'spend':spend,'clean_draws':int(n.sum()),
            'true_K':true_k,'used_graph':int(graph_looks>0),'certified_edges':len(good),'falsified_edges':len(bad)}


def run(out='results',seed=20261125,trials=40):
    # Unknown reference varies by trial; continuous bias, including dense corruption.
    # Both sides of threshold are present; hypothetical costs only.
    edges=[(0,1,.16),(0,2,.16),(1,2,.16),(0,3,.16),(1,3,.16),(2,3,.16),
           (3,4,.16),(4,5,.16),(3,5,.16),(0,5,.16)]
    budgets=[80000,240000]; report={'seed':seed,'trials':trials,'edges':edges,'alpha':.05,'tau':.03,'proxy_n':3000,'sample_size':6,'cost_clean':20,'scenarios':{}}
    for scenario in ('smooth','sparse','dense'):
        for budget in budgets:
            block={}
            for mode in ('direct','calibrated','targeted','oracle'):
                rng=np.random.default_rng(seed);world_rng=np.random.default_rng(seed+1);rows=[]
                for _ in range(trials):
                    ref=world_rng.uniform(.40,.50);mu=ref+np.array([0,.27,.19,.09,-.13,-.20]);bias=world_rng.uniform(.035,.065,len(mu))
                    if scenario=='sparse':bias[5]+=.24
                    if scenario=='dense':bias[3:]+=np.array([.27,.34,.42])
                    proxy=mu+bias;K=sum(abs(bias[i]-bias[j])>r+1e-10 for i,j,r in edges)
                    rows.append(run_gate(mu,proxy,edges,budget,rng,mode,oracle_k=K))
                block[mode]={k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
            report['scenarios'][f'{scenario}_{budget}']=block
    p=Path(out);p.mkdir(parents=True,exist_ok=True);(p/'calibrated_gate.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':run()
