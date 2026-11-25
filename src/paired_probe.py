"""V13: fresh matched-pair first-probe design; synthetic only.
No repeated human probes, known proxy biases or graph assumptions.
"""
import json
import math
from pathlib import Path
import numpy as np
from scipy.stats import beta


def cp(s,n,a):
    """Two-sided Clopper-Pearson, broadcasting arrays; total error a."""
    s,n=np.broadcast_arrays(np.asarray(s),np.asarray(n))
    if np.any(s<0) or np.any(s>n) or not 0<a<1:raise ValueError('invalid counts')
    l=np.where(s==0,0,beta.ppf(a/2,s,n-s+1))
    u=np.where(s==n,1,beta.ppf(1-a/2,s+1,n-s))
    return l,u


def paired_ci(plus,minus,n,alpha=.05):
    """Conservative exact rectangle for q*(2r-1); plus=B only, minus=A only."""
    plus,minus=np.broadcast_arrays(np.asarray(plus),np.asarray(minus))
    if np.any(plus<0) or np.any(minus<0):raise ValueError('negative count')
    k=plus+minus;ql,qu=cp(k,n,alpha/2);rl,ru=cp(plus,k,alpha/2)
    corners=np.stack([ql*(2*rl-1),ql*(2*ru-1),qu*(2*rl-1),qu*(2*ru-1)])
    return corners.min(0),corners.max(0)


def direct_ci(a,b,n,alpha=.05):
    al,au=cp(a,n,alpha/2);bl,bu=cp(b,n,alpha/2)
    return bl-au,bu-al


def joint_probs(delta,q):
    """Order 00, A-only, B-only, 11; marginals (.5-delta/2,.5+delta/2)."""
    if not abs(delta)<=q<=1:raise ValueError('infeasible discordance')
    return np.array([(1-q)/2,(q-delta)/2,(q+delta)/2,(1-q)/2])


def kl_binary(p,q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


def cost_lower(q,gap,cost,alpha=.05,beta_error=.05):
    """Expected null cost for testing delta=0 vs delta=-gap, fixed q."""
    if not 0<gap<q<=1 or cost<=0 or not 0<alpha<1-beta_error<1:raise ValueError('invalid design')
    divergence=-q/2*math.log1p(-(gap/q)**2)
    return cost*kl_binary(1-beta_error,alpha)/divergence


def run(out='results',seed=20261125,trials=1000):
    rng=np.random.default_rng(seed);rows=[];tau=.03;alpha=.05;base_cost=40;pilot_n=50
    for scenario,q,pc in [('positive_low_overhead',.12,40),('positive_high_overhead',.12,80),('weak_positive',.40,50),('negative_coupling',.80,50)]:
        for delta in (.04,0.,-.06):
            p=joint_probs(delta,q);pa=.5-delta/2;pb=.5+delta/2
            for budget in (12000,40000):
                nd=budget//base_cost;npair=budget//pc
                a=rng.binomial(nd,pa,trials);b=rng.binomial(nd,pb,trials)
                pl=rng.multinomial(npair,p,trials)
                l,u=paired_ci(pl[:,2],pl[:,1],npair,alpha)
                # Strong variance-aware comparator: direct confidence from observed discordance.
                # Rectangle CP is our finite-sample implementation; paired Hoeffding is a weaker bound.
                mean=(pl[:,2]-pl[:,1])/npair;h=math.sqrt(2*math.log(2/alpha)/npair)
                methods={'direct_CP':(*direct_ci(a,b,nd,alpha),np.full(trials,nd*base_cost),np.full(trials,2*nd)),
                         'paired_CP':(l,u,np.full(trials,npair*pc),np.full(trials,2*npair)),
                         'paired_Hoeffding':(mean-h,mean+h,np.full(trials,npair*pc),np.full(trials,2*npair))}
                # Pilot chooses the cheaper estimated variance-cost mode; inference uses fresh pairs only.
                pilot=rng.multinomial(pilot_n,p,trials);dq=(pilot[:,1]+pilot[:,2])/pilot_n
                da=(pilot[:,1]+pilot[:,3])/pilot_n;db=(pilot[:,2]+pilot[:,3])/pilot_n
                choose=pc*np.maximum(0,dq-(db-da)**2)<base_cost*(da*(1-da)+db*(1-db))
                rem=budget-pilot_n*pc;nhp=rem//pc;nhd=rem//base_cost
                hp=rng.multinomial(nhp,p,trials);ha=rng.binomial(nhd,pa,trials);hb=rng.binomial(nhd,pb,trials)
                lp,up=paired_ci(hp[:,2],hp[:,1],nhp,alpha);ld,ud=direct_ci(ha,hb,nhd,alpha)
                methods['pilot_select']=(np.where(choose,lp,ld),np.where(choose,up,ud),pilot_n*pc+np.where(choose,nhp*pc,nhd*base_cost),2*(pilot_n+np.where(choose,nhp,nhd)))
                for method,(l,u,c,n) in methods.items():
                    passed=l>=-tau;safe=delta>=-tau
                    rows.append(dict(scenario=scenario,q=q,delta=delta,budget=budget,method=method,pass_rate=float(passed.mean()),false_pass_rate=float(passed.mean()) if not safe else 0.,coverage=float(np.mean((l<=delta)&(delta<=u))),mean_width=float(np.mean(u-l)),mean_cost=float(np.mean(c)),mean_people=float(np.mean(n)),pilot_paired_fraction=float(choose.mean()) if method=='pilot_select' else None))
    result={'seed':seed,'trials_per_cell':trials,'tau':tau,'alpha':alpha,'rows':rows,'note':'Synthetic joint laws; two distinct first-probe participants per pair; costs include pilot.'}
    p=Path(out);p.mkdir(exist_ok=True,parents=True);(p/'paired_probe.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    r=run();print('cells:',len(r['rows']),'replications:',len(r['rows'])*r['trials_per_cell'])
