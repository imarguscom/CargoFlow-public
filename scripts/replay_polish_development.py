#!/usr/bin/env python3
"""Evaluate deterministic polishing of saved development candidate results.

This is component replay. Its additive time estimate is NOT a fresh end-to-
end benchmark; the final benchmark must execute solve_polished from inputs.
"""
import argparse
import copy
import shutil
from pathlib import Path
import time
from run_travel_v5 import read,write,sha,summarise,ROOT
import sys
sys.path.insert(0,str(ROOT/'src'))
from cargoflow.polished_search import polish_result


def main(root):
    base=root/'refine';out=root/'polish_development'
    if out.exists():raise FileExistsError(out)
    assert read(base/'status.json')['status']=='complete'
    assert read(base/'independent_audit.json')['passed']
    old=read(base/'stage_protocol.json')
    configs=copy.deepcopy(old['configs'])
    extras={}
    for c in configs:
        if c['engine']=='baseline':continue
        extras[c['name']]={'name':c['name']+'_2opt','engine':'polished','backend':c['backend'],
            'base_config':copy.deepcopy(c),'seconds':c['seconds']+.2,'polish_seconds':.15}
    configs+=list(extras.values())
    sources=[ROOT/'src/cargoflow'/n for n in ['solver.py','contract.py','baselines.py','travel_search.py','directed_polish.py','polished_search.py']]
    sources += [Path(__file__).resolve(),ROOT/'scripts/run_travel_v5.py']
    hashes={str(p):sha(p) for p in sources}
    for row in old['rows']:
        for k in ['instance_path','matrix_path']:hashes[row[k]]=sha(row[k])
    provenance={'base_stage':str(base),'base_records_sha256':{p.name:sha(p) for p in sorted((base/'routes').glob('*.json'))},
        'timing':'Original measured base time plus measured polish component; development estimate only. Final inference is a fresh complete solve.',
        'reused_original_records':len(old['rows'])*len(old['seeds'])*len(old['configs'])}
    write(out/'stage_protocol.json',{'created_unix':time.time(),'rows':old['rows'],'configs':configs,'seeds':old['seeds'],'input_source_hashes':hashes,'provenance':provenance})
    (out/'source_snapshot').mkdir()
    for p in sources:shutil.copy2(p,out/'source_snapshot'/p.name)
    write(out/'status.json',{'status':'running'})
    for n,row in enumerate(old['rows'],1):
        values=read(base/'routes'/f"{row['route_id']}.json")
        instance,matrix=read(row['instance_path']),read(row['matrix_path'])
        additional=[]
        for value in values:
            if value['config']['name'] in extras:
                config=extras[value['config']['name']]
                additional.append(polish_result(instance,matrix,value,config,seconds=.15))
        write(out/'routes'/f"{row['route_id']}.json",values+additional)
        print(f'polish replay {n}/{len(old["rows"])}',flush=True)
    summarise(out,old['rows'],configs,old['seeds'])
    assert all(sha(p)==v for p,v in hashes.items())
    assert all(sha(base/'routes'/k)==v for k,v in provenance['base_records_sha256'].items())
    write(out/'status.json',{'status':'complete','timing_is_development_estimate':True,'hashes_verified':len(hashes)})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();main(a.root.resolve())
