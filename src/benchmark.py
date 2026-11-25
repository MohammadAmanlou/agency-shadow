"""Rule-grounded paired-world benchmark. No human or LLM observations."""
import argparse
import json
from pathlib import Path
import random
import urllib.request

RULES={
 'duplicate_fee':('Refund iff posted and duplicate and not already_refunded.',('posted','duplicate','already_refunded'),(True,True,False), 'duplicate'),
 'late_fee':('Refund iff hardship_approved and timely_request and not prior_waiver.',('hardship_approved','timely_request','prior_waiver'),(True,True,False),'hardship_approved'),
 'transfer_fee':('Refund iff service_failed and charge_posted and not reversal_completed.',('service_failed','charge_posted','reversal_completed'),(True,True,False),'service_failed'),
 'account_fee':('Refund iff plan_exempt and fee_posted and not exemption_expired.',('plan_exempt','fee_posted','exemption_expired'),(True,True,False),'plan_exempt'),
}


def generate(pairs_per_family=60,seed=20261126):
    rng=random.Random(seed);cases=[]
    for fam,(rule,fields,values,key) in RULES.items():
        for i in range(pairs_per_family):
            pair=f'{fam}-{i:03d}';common={k:v for k,v in zip(fields,values) if k!=key}
            for label in (False,True):
                facts={**common,key:label};records=[{'id':f'e{j+1}','field':f,'value':facts[f]} for j,f in enumerate(fields)]
                records.append({'id':'e4','field':'unrelated_flag','value':bool(i%2)})
                rng.shuffle(records)
                cases.append({'id':f'{pair}-{int(label)}','pair':pair,'family':fam,'rule':rule,
                    'split':'heldout_rule_family' if fam=='account_fee' else 'development',
                    'initial_evidence':[x for x in records if x['field']!=key],
                    'authorized_records':records,'truth':label,'critical_field':key})
    rng.shuffle(cases);return cases


def verify(case,answer):
    """Independently verify Boolean rule and provenance; reject missing/duplicate fields."""
    if not isinstance(answer,dict) or type(answer.get('refund')) is not bool or not isinstance(answer.get('evidence_ids'),list):return False
    ids=answer['evidence_ids']
    if any(type(x) is not str for x in ids) or len(ids)!=len(set(ids)):return False
    byid={r['id']:r for r in case['authorized_records']}
    if any(x not in byid for x in ids):return False
    facts={byid[x]['field']:byid[x]['value'] for x in ids}
    fields=RULES[case['family']][1]
    if any(f not in facts or type(facts[f]) is not bool for f in fields):return False
    rule_value=(facts[fields[0]] and facts[fields[1]] and not facts[fields[2]])
    return answer['refund']==rule_value


def answer_rule(case):
    fields=RULES[case['family']][1];f={x['field']:x['value'] for x in case['authorized_records']}
    decision=all([f[fields[0]],f[fields[1]],not f[fields[2]]])
    return {'refund':decision,'evidence_ids':[r['id'] for r in case['authorized_records'] if r['field'] in fields]}


def llm_answer(case,endpoint,model):
    """Optional local OpenAI-compatible endpoint; never sends truth or pair ID."""
    prompt=('Return ONLY a JSON object {"refund":true or false,"evidence_ids":["e1",...]}. '
        'Apply the rule to authorized records. Cite every necessary record.\nRule: '+case['rule']+'\nRecords: '+json.dumps(case['authorized_records']))
    body=json.dumps({'model':model,'messages':[{'role':'user','content':prompt}],'temperature':0,'max_tokens':256}).encode()
    req=urllib.request.Request(endpoint,data=body,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as res:raw=json.load(res)['choices'][0]['message']['content']
    try:return json.loads(raw)
    except (json.JSONDecodeError,TypeError):return {'invalid':True}


def evaluate(cases,fn):
    rows=[]
    for c in cases:
        a=fn(c);a=a if isinstance(a,dict) else {};rows.append({'id':c['id'],'family':c['family'],'split':c['split'],
            'correct':type(a.get('refund')) is bool and a['refund']==c['truth'],
            'certificate':verify(c,a)})
    def metrics(rs):return {'n':len(rs),'accuracy':sum(r['correct'] for r in rs)/len(rs),'certified':sum(r['certificate'] for r in rs)/len(rs)}
    return {'all':metrics(rows),'heldout':metrics([r for r in rows if r['split']=='heldout_rule_family'])}


def run(out='results',endpoint=None,model=None):
    cases=generate();p=Path(out);p.mkdir(parents=True,exist_ok=True)
    (p/'paired_worlds.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in cases))
    result={'cases':len(cases),'pairs':len(cases)//2,'families':len(RULES),'rule_oracle':evaluate(cases,answer_rule),
            'initial_only_constant':evaluate(cases,lambda c:{'refund':True,'evidence_ids':[r['id'] for r in c['initial_evidence']]}),
            'llm_executed':False}
    if endpoint:
        if not model:raise ValueError('--model required with --endpoint')
        result['llm']=evaluate(cases,lambda c:llm_answer(c,endpoint,model));result['llm_executed']=True;result['model']=model
    (p/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default='results');p.add_argument('--endpoint');p.add_argument('--model');run(**vars(p.parse_args()))
