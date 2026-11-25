"""Bounded-bias first-probe design: synthetic, not human validation."""
import json
import math
from pathlib import Path
import numpy as np


def design(costs, biases, epsilon, alpha):
    """Minimize the sufficient fixed-design cost, given valid bias bounds."""
    c, d = np.asarray(costs, float), np.asarray(biases, float)
    if c.ndim != 1 or c.shape != d.shape or not len(c) or not np.all(np.isfinite(c)) or not np.all(np.isfinite(d)) or np.any(c <= 0) or np.any(d < 0) or not 0 < epsilon < .5 or not 0 < alpha < .5:
        raise ValueError('invalid design parameters')
    feasible = np.flatnonzero(d < epsilon)
    if not len(feasible):
        raise ValueError('no history with sufficiently small certified bias')
    n = [math.ceil(math.log(2/alpha)/(2*(epsilon-d[i])**2)) for i in feasible]
    k = min(range(len(n)), key=lambda k: c[feasible[k]]*n[k])
    i = int(feasible[k])
    return i, n[k], float(c[i]*n[k])


def minimax_lower(costs, biases, epsilon, alpha):
    """Worst-case expected cost lower bound for 0<epsilon<1/4, alpha<1/2."""
    if not 0 < epsilon < .25:
        raise ValueError('epsilon must be strictly below 1/4')
    design(costs, biases, epsilon, alpha)
    eta = max(max(epsilon-d,0)**2/c for c,d in zip(costs,biases))
    return (1-2*alpha)*math.log((1-alpha)/alpha)/(16*eta)


def run(out='results', seed=20261125, trials=2000):
    costs, biases = [20.,4.,1.], [0.,.02,.12]
    epsilon, alpha, truth = .08,.05,.4
    i,n,cost = design(costs,biases,epsilon,alpha)
    _,nc,cc = design(costs[:1],biases[:1],epsilon,alpha)
    rng=np.random.default_rng(seed)
    draws=rng.binomial(n,truth+biases[i],size=trials)/n
    result={'seed':seed,'trials':trials,'costs':costs,'bias_bounds':biases,'epsilon':epsilon,'alpha':alpha,'selected_history':i,'n':n,'cost':cost,'clean_n':nc,'clean_cost':cc,'cost_reduction_fraction':1-cost/cc,'minimax_cost_lower_bound':minimax_lower(costs,biases,epsilon,alpha),'synthetic_coverage':float(np.mean(abs(draws-truth)<=epsilon))}
    Path(out).mkdir(parents=True,exist_ok=True)
    (Path(out)/'cost_design.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':run()
