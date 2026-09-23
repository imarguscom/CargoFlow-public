#!/usr/bin/env python3
"""Create development refinement and lock one candidate before benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from collections import defaultdict
from statistics import mean, median


def read(p):return json.loads(p.read_text())
def write(p,v):
    if p.exists():raise FileExistsError(p)
    p.write_text(json.dumps(v,indent=2)+'\n')


def refine(root):
    protocol=read(root/'protocol.json')
    summary=read(root/'screen/summary.json')
    assert read(root/'screen/independent_audit.json')['passed']
    candidates=[v for v in summary if v['config']['engine']!='baseline' and
                v['comparisons']['old_balanced']['feasible_losses']==0]
    top=sorted(candidates,key=lambda v:v['comparisons']['old_balanced']['mean_travel_gain'],reverse=True)[:2]
    configs=[v['config'] for v in summary if v['config']['engine']=='baseline']+[v['config'] for v in top]
    for v in top: configs.append({**v['config'],'name':v['config']['name']+'_3p8','seconds':3.8})
    best=top[0]['config']
    assert best['backend']=='0.13.4'
    for perturb,history in [(5,50),(10,100)]:
        configs.append({**best,'name':best['name']+f'_3p8_p{perturb}_h{history}',
                        'seconds':3.8,'max_perturbations':perturb,'history_length':history})
    spec={'rows':protocol['development'][:24],'configs':configs,'seeds':[42,20260907],
          'created_unix':time.time(),'screen_summary_sha256':hashlib.sha256((root/'screen/summary.json').read_bytes()).hexdigest(),
          'reason':'Pre-benchmark development adaptation recorded in docs/travel-speed-v5-protocol.md'}
    write(root/'refine_spec.json',spec)
    print(json.dumps({'routes':24,'seeds':2,'configs':[c['name'] for c in configs]}))


def lock(root, stage=None):
    stage=stage or (root/'polish_development' if (root/'polish_development').exists() else root/'refine')
    summary=read(stage/'summary.json')
    assert read(stage/'status.json')['status']=='complete'
    assert read(stage/'independent_audit.json')['passed']
    grouped=defaultdict(lambda:defaultdict(list))
    for p in (stage/'routes').glob('*.json'):
        for r in read(p):
            if r['solution']['feasible']:
                grouped[r['config']['name']][r['route_id']].append(r['solution']['audit']['metrics']['travel_seconds'])
    medians={name:median(mean(values) for values in rows.values()) for name,rows in grouped.items()}
    eligible=[]
    for v in summary:
        if v['config']['engine']=='baseline':continue
        if all(v['comparisons'][b]['feasible_losses']==0 and
               v['comparisons'][b]['mean_travel_gain']>0 and
               medians[v['config']['name']]<medians[b] and
               v['comparisons'][b]['median_runtime_ratio']<=.8 for b in ['old_balanced','old_reference']):
            eligible.append(v)
    if not eligible:
        write(root/'selection_rejected.json',{'created_unix':time.time(),'reason':'No refinement candidate reduced travel and runtime against both baselines.'})
        print('NO ELIGIBLE CANDIDATE');return
    best=max(eligible,key=lambda v:min(v['comparisons'][b]['mean_travel_gain'] for b in ['old_balanced','old_reference']))
    selected={'created_unix':time.time(),'candidate':best['config'],'development_comparisons':best['comparisons'],
              'selection_rule':'Maximise the smaller mean travel gain against both baselines, among eligible development candidates.',
              'selection_stage':str(stage),'timing_note':'Polished development records use component-sum estimates; final benchmark must measure complete fresh calls.',
              'refine_summary_sha256':hashlib.sha256((stage/'summary.json').read_bytes()).hexdigest()}
    stage_protocol=read(stage/'stage_protocol.json')
    selected['protocol_sha256']=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
    selected['source_hashes']={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for path in stage_protocol['input_source_hashes']
        if path.endswith(('.py','.cpp','.c','/LKH','.so'))}
    write(root/'locked_selection.json',selected)
    protocol=read(root/'protocol.json')
    base=[v['config'] for v in summary if v['config']['engine']=='baseline']
    write(root/'benchmark_spec.json',{'rows':protocol['benchmark'],'configs':base+[best['config']],
          'seeds':protocol['benchmark_seeds'],'locked_selection_sha256':hashlib.sha256((root/'locked_selection.json').read_bytes()).hexdigest()})
    print(json.dumps(selected,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['refine','lock']);p.add_argument('root',type=Path)
    p.add_argument('--stage',type=Path,help='Explicit audited development stage for candidate lock')
    a=p.parse_args()
    if a.mode=='refine':refine(a.root.resolve())
    else:lock(a.root.resolve(),a.stage.resolve() if a.stage else None)
