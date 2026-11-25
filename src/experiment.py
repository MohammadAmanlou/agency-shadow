"""Synthetic measurement-reactivity example; no human or LLM data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def probe_probability(world, gap):
    """gap = consecutive previous rounds without a probe; probe is scored before reset."""
    if world == 'stable':
        return np.full_like(np.asarray(gap, dtype=float), 0.8)
    if world == 'erosion':
        return np.maximum(0.0, 0.8 - 0.05 * np.asarray(gap))
    raise ValueError(world)


def trajectory(world, schedule):
    gap = 0
    observations = []
    for probe in schedule:
        if probe:
            observations.append(float(probe_probability(world, gap)))
            gap = 0
        else:
            observations.append(None)
            gap += 1
    return observations


def sharp_interval(observed, weights, target=None):
    """Sharp pointwise interval under coordinatewise bounded history effects.
    observed: dict of binary histories -> exact expected outcomes; validate compatibility.
    """
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise ValueError('negative weights')
    target = tuple([0] * len(w)) if target is None else tuple(target)
    def distance(a,b):
        return float(np.dot(w, np.abs(np.array(a)-np.array(b))))
    if len(target) != len(w) or any(v not in (0,1) for v in target):
        raise ValueError('invalid target')
    for a, x in observed.items():
        if len(a) != len(w) or any(v not in (0,1) for v in a) or not 0 <= x <= 1:
            raise ValueError('invalid observation')
        for b, y in observed.items():
            if abs(x-y) > distance(a,b)+1e-10:
                raise ValueError('incompatible observations')
    lower = max([0.0]+[m-distance(h,target) for h,m in observed.items()])
    upper = min([1.0]+[m+distance(h,target) for h,m in observed.items()])
    return lower, upper


def first_probe_hazard(horizon, hazard):
    return hazard * (1-hazard)**np.arange(horizon)


def run(n=1200, horizon=12, trials=300, seed=20261125, out=Path('results')):
    if n < horizon or horizon < 2 or trials < 1:
        raise ValueError('need n >= horizon >= 2, trials >= 1')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    truth = np.array([float(probe_probability('erosion',t)) for t in range(horizon)])
    stable = np.full(horizon,0.8)
    balanced_n = np.full(horizon, n//horizon, dtype=int); balanced_n[:n%horizon] += 1
    # Equal audit-budget: repeated-probe arm can afford floor(n/horizon) people.
    repeated_n = n//horizon
    repeated = rng.binomial(repeated_n,0.8,size=(trials,horizon))/repeated_n
    staggered = rng.binomial(balanced_n,truth,size=(trials,horizon))/balanced_n
    stable_staggered = rng.binomial(balanced_n,stable,size=(trials,horizon))/balanced_n
    probs = first_probe_hazard(horizon,.5)
    # Single first probe, geometric waiting schedule; subjects without probe by T are unmeasured.
    counts = rng.multinomial(n, np.r_[probs, max(0,1-probs.sum())], size=trials)[:,:horizon]
    draws = rng.binomial(counts,truth)
    geometric = np.divide(draws, counts, out=np.full(draws.shape,np.nan), where=counts>0)
    summary = {'seed':seed,'participants_per_single_probe_arm':n,'horizon':horizon,'trials':trials,
               'repeated_participants':repeated_n,'audit_budget_staggered':n,'audit_budget_repeated':repeated_n*horizon,
               'true_terminal_stable':float(stable[-1]),'true_terminal_erosion':float(truth[-1]),
               'repeated_terminal_mean':float(repeated[:,-1].mean()),
               'staggered_terminal_mean':float(staggered[:,-1].mean()),
               'staggered_stable_terminal_mean':float(stable_staggered[:,-1].mean()),
               'repeated_terminal_rmse_as_clean_estimator':float(np.sqrt(np.mean((repeated[:,-1]-truth[-1])**2))),
               'staggered_terminal_rmse':float(np.sqrt(np.mean((staggered[:,-1]-truth[-1])**2))),
               'geometric_terminal_expected_clean_n':float(n*probs[-1]),
               'geometric_terminal_empty_rate':float(np.mean(counts[:,-1]==0)),
               'geometric_terminal_theoretical_empty_rate':float((1-probs[-1])**n),
               'example_sharp_interval':sharp_interval({(1,0,0):.65,(0,1,0):.55},[.2,.1,.1])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (out/'trajectory.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['round','true_clean_stable','true_clean_erosion','repeated_observed_mean','staggered_clean_mean','geometric_clean_n_mean','geometric_empty_rate'])
        for t in range(horizon):writer.writerow([t+1,stable[t],truth[t],repeated[:,t].mean(),staggered[:,t].mean(),counts[:,t].mean(),(counts[:,t]==0).mean()])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t=np.arange(1,horizon+1);fig,ax=plt.subplots(figsize=(8.8,4.5),layout='constrained')
    ax.plot(t,stable,'--',label='Stable: true first-probe outcome')
    ax.plot(t,truth,'-',linewidth=2.5,label='Erosion: true first-probe outcome')
    ax.plot(t,repeated.mean(0),'o:',label='Repeated probes (different estimand)')
    ax.plot(t,staggered.mean(0),'s',markersize=4,label='Randomized first-probe cohorts')
    ax.set(xlabel='Interaction round',ylabel='Probability of successful correction',ylim=(0,1),xticks=t,title='Synthetic example: measurement can conceal deterioration')
    ax.legend(loc='lower left',fontsize=8);ax.grid(alpha=.2);fig.savefig(out/'trajectory.png',dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=1200);p.add_argument('--horizon',type=int,default=12);p.add_argument('--trials',type=int,default=300);p.add_argument('--seed',type=int,default=20261125);p.add_argument('--out',type=Path,default=Path('results'));a=p.parse_args();run(**vars(a))
