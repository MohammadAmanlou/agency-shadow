"""V11: cut-witness audit design, restricted finite hypotheses; synthetic only."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from scipy.stats import hypergeom


def witnesses(n, edges, k, reference=0):
    """All nonempty vertex sets not containing the reference with cut <= k."""
    if not 0 <= reference < n or k < 0 or n < 2 or n > 20:
        raise ValueError('invalid small-graph problem')
    nodes=[i for i in range(n) if i!=reference]; out=[]
    for size in range(1,len(nodes)+1):
        for s in itertools.combinations(nodes,size):
            a=set(s)
            if sum((i in a)!=(j in a) for i,j in edges)<=k:out.append(s)
    return out


def kl(p,q):
    p,q=np.asarray(p,float),np.asarray(q,float)
    if np.any((p<=0)|(p>=1)|(q<=0)|(q>=1)):raise ValueError('interior probabilities required')
    return p*np.log(p/q)+(1-p)*np.log((1-p)/(1-q))


def alternatives(mu, sets, shift):
    """Known baseline versus finite, downward-shift cut alternatives."""
    mu=np.asarray(mu,float);out=[]
    for s in sets:
        v=mu.copy();v[list(s)]-=shift;out.append(v)
    a=np.asarray(out).reshape((-1,len(mu)))
    if np.any((a<=0)|(a>=1)):raise ValueError('shift outside interior')
    return a


def allocation(mu, alt, costs, alpha=.05, beta=.05):
    """KL covering LP: expected-cost lower bound for alpha risk, 1-beta power.
    The solution normalized by cost is a proposed fixed allocation, not a finite
    sample optimum. Every listed alternative must be unsafe.
    """
    mu=np.asarray(mu,float);alt=np.asarray(alt,float);c=np.asarray(costs,float)
    if not len(alt) or c.shape!=mu.shape or np.any(c<=0) or not 0<alpha<1-beta<1:raise ValueError('invalid design')
    K=kl(mu[None,:],alt);b=float(kl(1-beta,alpha))
    sol=linprog(c,A_ub=-K,b_ub=np.full(len(alt),-b),bounds=(0,None),method='highs')
    if not sol.success:raise ValueError('no informative audit')
    dual=-sol.ineqlin.marginals
    return sol.x,float(sol.fun),dual,K


def upper_bad_count(M,n,x,alpha):
    """Exact one-sided finite-population upper count, uniform W/O replacement.
    x may conservatively count uncertain edges as bad on a separate CI event.
    """
    if not 0<=x<=n<=M or not 0<alpha<1:raise ValueError('invalid audit count')
    return max(k for k in range(M+1) if hypergeom.cdf(x,M,k,n)>=alpha)


def run_many(mu,alt,costs,weights,truth,budget,rng,trials,alpha=.05):
    """Sequential LR gate with deterministic fixed-design tracking, batched trials."""
    c=np.asarray(costs,int);w=np.asarray(weights,float);truth=np.asarray(truth,float)
    active=np.flatnonzero(w>1e-10);w=w/w.sum();n=np.zeros(len(mu));lr=np.zeros((trials,len(alt)));spend=0
    passed=np.zeros(trials,bool);used=np.zeros(trials,int)
    pos=np.log(mu[None,:]/alt);neg=np.log((1-mu[None,:])/(1-alt));threshold=math.log(1/alpha)
    while not np.all(passed):
        affordable=[i for i in active if c[i]<=budget-spend]
        if not affordable:break
        i=max(affordable,key=lambda i:(w[i]*(n.sum()+1)-n[i],-i))
        b=min(5,(budget-spend)//c[i]);s=rng.binomial(b,truth[i],size=trials);n[i]+=b;spend+=b*c[i]
        lr+=s[:,None]*pos[:,i]+(b-s[:,None])*neg[:,i]
        fresh=(~passed)&np.all(lr>=threshold,axis=1);used[fresh]=spend;passed|=fresh
    used[~passed]=spend
    return passed,used


def run_gate(mu,alt,costs,weights,truth,budget,rng,alpha=.05):
    """Single-trial convenience wrapper."""
    p,c=run_many(mu,alt,costs,weights,truth,budget,rng,1,alpha)
    return bool(p[0]),int(c[0])


def graph_suite():
    # Three dense components connected by thin cuts. Two possible corrupt edges.
    cliques=[range(0,4),range(4,8),range(8,12)]
    e=[x for g in cliques for x in itertools.combinations(g,2)]+[(0,4),(1,5),(0,8),(2,9)]
    return {'cluster':(12,e,2,[9,8,5,4,1,5,7,9,2,5,7,9]),
            'cycle':(8,[(i,(i+1)%8) for i in range(8)],2,[9,1,2,3,4,5,6,7]),
            'path':(8,[(i,i+1) for i in range(7)],1,[9,2,3,4,5,6,7,8])}


def run(out='results',seed=20261125,trials=250):
    rng=np.random.default_rng(seed);report={'seed':seed,'trials':trials,'alpha':.05,'beta_for_LP':.05,'shift':.15,'tau':.03,'graphs':{}}
    for name,(n,e,k,c) in graph_suite().items():
        mu=np.full(n,.6);sets=witnesses(n,e,k);alt=alternatives(mu,sets,.15);x,lb,dual,K=allocation(mu,alt,c)
        # Same fixed budget for all methods within graph. Includes null and each alternative.
        budget=math.ceil(2*lb);uniform=np.ones(n);uniform[0]=0
        # Strong baseline: uniform over informative nodes only, equal spend per node.
        informative=np.any(K>1e-12,axis=0);equal_cost=informative/np.array(c)
        methods={'cut_lp':x,'uniform_candidates':uniform,'equal_spend_informative':equal_cost}
        results={}
        for m,w in methods.items():
            passed,used=run_many(mu,alt,c,w,mu,budget,rng,trials)
            byalt=[]
            for a in alt:
                rr,_=run_many(mu,alt,c,w,a,budget,rng,trials)
                byalt.append(float(np.mean(rr)))
            results[m]={'baseline_pass':float(np.mean(passed)), 'baseline_mean_cost':float(np.mean(used)), 'baseline_mean_pass_cost':float(np.mean(used[passed])) if np.any(passed) else None,'worst_alt_false_pass':max(byalt),'mean_alt_false_pass':float(np.mean(byalt))}
        report['graphs'][name]={'vertices':n,'edges':len(e),'k':k,'witnesses':[list(s) for s in sets],'costs':c,'LP_lower_cost':lb,'LP_dual_cost':float(dual.sum()*kl(.95,.05)),'LP_expected_counts':x.tolist(),'budget':budget,'methods':results}
    # Outside-class counterexample: the selected cheap nodes do not observe a hidden shift.
    n,e,k,c=graph_suite()['cluster'];mu=np.full(n,.6);alt=alternatives(mu,witnesses(n,e,k),.15);x,lb,*_=allocation(mu,alt,c);outside=mu.copy();outside[1]-=.15
    p,_=run_many(mu,alt,c,x,outside,math.ceil(2*lb),rng,trials)
    report['outside_class']={'shifted_node':1,'violated_edges':sum((i==1)!=(j==1) for i,j in e),'assumed_k':k,'false_pass':float(p.mean())}
    report['edge_count_examples']=[{'M':50,'n':n,'observed_bad':x,'upper_95':upper_bad_count(50,n,x,.05)} for n,x in [(0,0),(10,0),(20,0),(30,0),(50,0),(20,1),(30,2)]]
    Path(out).mkdir(exist_ok=True,parents=True);(Path(out)/'cut_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':run()
