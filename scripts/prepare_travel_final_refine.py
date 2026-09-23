#!/usr/bin/env python3
"""Compare the two strongest feasible development candidates on 24 x 2 pairs."""
import hashlib,time
from pathlib import Path
from run_travel_v5 import read,write,baseline
root=Path(__file__).resolve().parents[1]/'artifacts/travel_v5'
protocol=read(root/'protocol.json')
pool=[];evidence={}
for stage_name in ['diverse','lkh','lkh_special']:
    stage=root/stage_name
    assert read(stage/'status.json')['status']=='complete'
    assert read(stage/'independent_audit.json')['passed']
    spec=read(stage/'stage_protocol.json')
    assert spec['rows']==protocol['development'][:12] and spec['seeds']==[42]
    evidence[stage_name]={f:hashlib.sha256((stage/f).read_bytes()).hexdigest() for f in ['summary.json','stage_protocol.json','independent_audit.json']}
    for value in read(stage/'summary.json'):
        if value['config']['engine']=='baseline':continue
        if all(value['comparisons'][base]['feasible_losses']==0 and
            value['comparisons'][base]['median_runtime_ratio']<=.8 for base in ['old_balanced','old_reference']):
            pool.append(value)
pool.sort(key=lambda v:(-v['comparisons']['old_reference']['mean_travel_gain'],v['config']['name']))
assert len(pool)>=2
chosen=[v['config'] for v in pool[:2]]
out=root/'final_refine_spec.json'
if out.exists():raise FileExistsError(out)
write(out,dict(created_unix=time.time(),rows=protocol['development'][:24],seeds=[42,20260907],
    configs=[baseline('balanced'),baseline('reference')]+chosen,screen_evidence_sha256=evidence,
    ranking=[{'name':v['config']['name'],'reference_gain':v['comparisons']['old_reference']['mean_travel_gain']} for v in pool],
    rule='Top two by mean paired travel gain versus reference among zero-feasibility-loss candidates with median runtime ratio <=0.8 against both baselines. Exact ties use configuration name. Lock only from this fresh 24-route/two-seed stage under the existing eligibility rule; final benchmark unchanged.'))
print(chosen)
