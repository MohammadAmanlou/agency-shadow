"""Resumable synthetic rule benchmark on a local OpenAI-compatible LLM endpoint."""
import argparse, hashlib, json, time
from pathlib import Path
from src.benchmark import generate, llm_answer, verify


def summarize(rows):
    def metrics(items):
        valid=[r for r in items if r['status']=='ok'];n=len(items)
        return {'n':n,'completed':len(valid),'errors':n-len(valid),'accuracy_all':sum(r['correct'] for r in valid)/n if n else None,'certified_all':sum(r['certified'] for r in valid)/n if n else None}
    pairs={}
    for r in rows:pairs.setdefault(r['pair'],[]).append(r)
    complete=[p for p in pairs.values() if len(p)==2]
    return {'all':metrics(rows),'heldout':metrics([r for r in rows if r['split']=='heldout_rule_family']), 'complete_pairs':len(complete),'both_correct_and_certified_fraction':sum(all(r['correct'] and r['certified'] for r in p) for p in complete)/len(complete) if complete else None}


def run(endpoint,model,out,pairs_per_family=60,seed=20261126,answer_fn=None):
    p=Path(out);p.mkdir(parents=True,exist_ok=True);cases=generate(pairs_per_family,seed)
    meta={'endpoint':endpoint,'model':model,'pairs_per_family':pairs_per_family,'seed':seed,'dataset_sha256':hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest(),'mock':answer_fn is not None}
    manifest=p/'manifest.json';raw=p/'responses.jsonl'
    if manifest.exists() and json.loads(manifest.read_text())!=meta:raise ValueError('run configuration differs; use a new output directory')
    manifest.write_text(json.dumps(meta,indent=2)+'\n');rows={}
    if raw.exists():
        for line in raw.read_text().splitlines():
            if line.strip():
                r=json.loads(line);rows[r['id']]=r
    for c in cases:
        if c['id'] in rows and rows[c['id']]['status']=='ok':continue
        start=time.monotonic();r={k:c[k] for k in ('id','pair','family','split')}
        try:
            a=answer_fn(c) if answer_fn else llm_answer(c,endpoint,model)
            r.update(status='ok',response=a,correct=isinstance(a,dict) and type(a.get('refund')) is bool and a['refund']==c['truth'],certified=verify(c,a))
        except Exception as e:r.update(status='error',error=type(e).__name__,correct=False,certified=False)
        r['seconds']=time.monotonic()-start;rows[c['id']]=r
        with raw.open('a') as f:f.write(json.dumps(r)+'\n')
    result={**meta,**summarize(list(rows.values())),'real_model_inference':answer_fn is None and any(r['status']=='ok' for r in rows.values())}
    (p/'summary.json').write_text(json.dumps(result,indent=2)+'\n');return result

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--endpoint',default='http://127.0.0.1:8000/v1/chat/completions');a.add_argument('--model',required=True);a.add_argument('--out',required=True);a.add_argument('--pairs-per-family',type=int,default=60);print(json.dumps(run(**vars(a.parse_args())),indent=2))
