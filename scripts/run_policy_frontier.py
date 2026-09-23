#!/usr/bin/env python3
"""Frozen sequential study with independent stage checkpoints; never resume blindly."""
import argparse
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
from statistics import mean

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from run_policy_search import POLICIES
from select_policy_seed import choose_seed
from frontier_search import search
from frontier_bounds import bounds,objective_bound
from frontier_mip import solve_bound
from audit_policy_benchmark import replay
from cargoflow.policy_search import _score

RESEARCH=ROOT.parent
PREVIOUS=RESEARCH/'cargoflow-multiobjective-review-20260916/artifacts/benchmark50'
OLD=RESEARCH/'cargoflow/artifacts/benchmark_runtime/deps'
METHODS=['random_sa','granular_sa','adaptive_lns']
SEEDS=[42,271828]
CONFIGS=[dict(name='hgs12',backend='0.12.2',engine='native',seconds=3.8,neighbours=40),
    dict(name='ils13',backend='0.13.4',engine='native',seconds=3.8,neighbours=80),
    dict(name='ils14',backend='0.14.0',engine='native',seconds=3.8,neighbours=80),
    dict(name='ils_lkh',backend='lkh-3.0.13',engine='ils_lkh',seconds=3.8,initial_seconds=.5,candidates=5,move_type=5),
    dict(name='ortools_gls',backend='ortools-9.15.6755',engine='ortools_gls',seconds=3.8)]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.partial')
    tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)

class Worker:
    def __init__(self,backend,out):
        self.log=open(out/('worker_'+backend+'.log'),'w')
        command=[sys.executable,str(ROOT/'scripts/frontier_worker.py')] if backend=='0.14.0' else [sys.executable,str(ROOT/'scripts/run_travel_v5.py'),'worker','--backend',backend]
        self.proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,
            env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('worker startup failed: '+backend)
        self.ready=json.loads(line);assert self.ready['ready']==backend
    def solve(self,row,config,seed):
        self.proc.stdin.write(json.dumps(dict(row=row,config=config,seed=seed))+'\n');self.proc.stdin.flush()
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('worker exited '+str(self.proc.poll()))
        return json.loads(line)
    def close(self):
        if self.proc.poll() is None:self.proc.stdin.close();self.proc.wait(timeout=30)
        self.log.close()

def choose_rows(rows,count):
    eligible=[]
    for row in rows:
        instance=read(row['instance_path'])
        if sum(Decimal(str(n['demand_cm3'])) for n in instance['nodes'])<=Decimal(str(instance['vehicle_capacity_cm3'])):
            eligible.append((len(instance['nodes']),row['route_id'],row))
    eligible.sort();assert len(eligible)>=count
    return [eligible[round(i*(len(eligible)-1)/(count-1))][2] for i in range(count)]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);workers={};started=time.time()
    def progress(stage,done,total,**kwargs):
        record=dict(stage=stage,completed=done,total=total,pid=os.getpid(),elapsed_seconds=time.time()-started,**kwargs)
        write(out/'status.json',record);print(json.dumps(record),flush=True)
    try:
        previous=read(PREVIOUS/'protocol.json');rows=previous['rows']
        devbase=read(RESEARCH/'cargoflow-travel-v5/artifacts/travel_v5/protocol.json')['development']
        development=choose_rows(devbase,6);repeated=choose_rows(rows,10);mip_rows=choose_rows(rows,4)
        group=lambda r:(r['station_code'],r['date_YYYY_MM_DD'])
        assert {group(r) for r in development}.isdisjoint(group(r) for r in rows)
        sources=read(ROOT/'SOURCE_MANIFEST.json');hashes={str(ROOT/p):h for p,h in sources.items()}
        for row in rows+development:
            for key in ['instance_path','matrix_path']:hashes[row[key]]=sha(row[key])
        for row in rows:
            path=PREVIOUS/'routes'/(row['route_id']+'.json');hashes[str(path)]=sha(path)
        hashes[str(PREVIOUS/'protocol.json')]=sha(PREVIOUS/'protocol.json')
        for folder in [OLD,ROOT/'artifacts/runtime13/deps',ROOT/'artifacts/runtime14/deps',ROOT/'artifacts/runtime_ortools/deps']:
            for binary in folder.rglob('*.so'):hashes[str(binary)]=sha(binary)
        for p in [ROOT/'artifacts/runtime_frontier/policy_frontier.so',ROOT/'artifacts/runtime_lkh/LKH-3.0.13/LKH']:hashes[str(p)]=sha(p)
        for p,h in hashes.items():assert sha(p)==h,p
        for p in sources:
            target=out/'source_snapshot'/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
        protocol=dict(source_commit=(ROOT/'SOURCE_COMMIT').read_text().strip(),created_unix=started,
            rows=rows,development=development,repeated_seed_rows=repeated,mip_rows=mip_rows,policies=[asdict(p) for p in POLICIES],
            methods=METHODS,seeds=SEEDS,development_checkpoints=[.25,1,4,8],confirmation_checkpoints=[.25,1],seed_configs=CONFIGS,
            frozen_hashes=hashes,pid=os.getpid(),host=platform.node(),python=platform.python_version(),
            previous_protocol_sha256=sha(PREVIOUS/'protocol.json'),notes='Fixed-reference and fresh-single-seed studies are separate. No benchmark-driven parameter tuning; no SOTA claim for custom objectives.')
        write(out/'protocol.json',protocol)
        for backend in ['0.12.2','0.13.4','0.14.0','lkh-3.0.13','ortools-9.15.6755']:workers[backend]=Worker(backend,out)
        write(out/'workers.json',{k:dict(pid=w.proc.pid,ready=w.ready) for k,w in workers.items()})
        # Fresh development anchors, then all method/seed/policy trajectories.
        development_results=[]
        for index,row in enumerate(development):
            i,m=read(row['instance_path']),read(row['matrix_path'])
            oldconfig=dict(name='old_reference',engine='baseline',backend='0.12.2',budget=dict(iterations=5000))
            old=workers['0.12.2'].solve(row,oldconfig,42);ils=workers['0.13.4'].solve(row,CONFIGS[1],42)
            anchor=choose_seed(i,m,[('reference',old['solution']),('ils',ils['solution'])]);assert anchor['feasible']
            trials=[]
            for rng in SEEDS:
                for pi in range(4):
                    policy=POLICIES[(pi+index)%4]
                    for mi in range(3):
                        method=METHODS[(mi+index)%3]
                        trial=search(i,m,anchor['route'],policy,method,rng,[.25,1,4,8]);trials.append(trial)
                        write(out/'development'/row['route_id']/(f'{policy.name}_{method}_{rng}.json'),trial)
            record=dict(row=row,anchor=anchor,seed_candidates=[old,ils],trials=trials)
            write(out/'development'/(row['route_id']+'.json'),record);development_results.append(record)
            progress('development',index+1,len(development))
        selection={method:mean(t['checkpoints'][1]['score'] for r in development_results for t in r['trials'] if t['method']==method) for method in METHODS}
        selected=min(METHODS[1:],key=lambda k:selection[k])
        write(out/'selection.json',dict(selected=selected,mean_score_at_one_second=selection,source_commit=protocol['source_commit'],selection='Development only; targeted candidate minimum across all four modes and two seeds.'))
        progress('selection',1,1,selected=selected)
        # Full fixed-reference cohort, exact bounds, complete checkpoint routes.
        for index,row in enumerate(rows):
            original=read(PREVIOUS/'routes'/(row['route_id']+'.json'));i,m=read(row['instance_path']),read(row['matrix_path'])
            if not original['selected_seed']['feasible']:
                write(out/'confirmation'/(row['route_id']+'.json'),dict(row=row,status='capacity_infeasible',trials=[]));continue
            anchor=original['selected_seed']['route'];reference=replay(i,m,anchor);lower=bounds(i,m)
            for k in ['travel_seconds','energy_proxy','delivery_completion_sum_seconds']:assert Decimal(lower[k])<=reference[k]
            trials=[]
            for rng in SEEDS:
                for pi in range(4):
                    policy=POLICIES[(pi+index)%4]
                    methods=['random_sa',selected]
                    for method in (methods if index%2==0 else methods[::-1]):
                        trial=search(i,m,anchor,policy,method,rng,[.25,1]);trials.append(trial)
            record=dict(row=row,status='complete',anchor=anchor,reference={k:str(v) for k,v in reference.items()},lower_bounds=lower,
                objective_lower_bounds={p.name:str(objective_bound(lower,reference,p)) for p in POLICIES},trials=trials,
                previous_policies=[dict(name=e['name'],metrics=e['result']['metrics'],policy_seconds=e['policy_seconds'],single_mode_pipeline_seconds=e['single_mode_pipeline_seconds']) for e in original['policies']])
            write(out/'confirmation'/(row['route_id']+'.json'),record);progress('confirmation',index+1,len(rows))
        # Numerical full-route model with exact small-instance validation first.
        for index,row in enumerate(mip_rows):
            i,m=read(row['instance_path']),read(row['matrix_path']);anchor=read(PREVIOUS/'routes'/(row['route_id']+'.json'))['selected_seed']['route']
            results={}
            for policy in POLICIES:
                results[policy.name]=solve_bound(i,m,anchor,policy,seconds=10)
                write(out/'mip'/(row['route_id']+'.json'),dict(row=row,results=results))
            progress('mip',index+1,len(mip_rows))
        # SOTA-family travel benchmark and separately anchored latency pipeline.
        tasks=[(row,42) for row in rows]+[(row,271828) for row in repeated]
        for index,(row,rng) in enumerate(tasks):
            records=[];i,m=read(row['instance_path']),read(row['matrix_path'])
            for offset in range(len(CONFIGS)):
                config=CONFIGS[(offset+index)%len(CONFIGS)]
                record=workers[config['backend']].solve(row,config,rng)
                if record['solution']['feasible']:
                    metrics=replay(i,m,record['solution']['route'])
                    assert abs(float(metrics['travel_seconds'])-record['solution']['audit']['metrics']['travel_seconds'])<1e-7
                    record['independent_metrics']={k:float(v) for k,v in metrics.items()}
                records.append(record)
            pipeline=[]
            lkh=next(r for r in records if r['config']['name']=='ils_lkh')
            if rng==42 and lkh['solution']['feasible']:
                for policy in POLICIES:
                    result=search(i,m,lkh['solution']['route'],policy,selected,42,[1])
                    result['single_mode_pipeline_seconds']=lkh['end_to_end_seconds']+result['end_to_end_seconds'];pipeline.append(result)
            write(out/'seeds'/(row['route_id']+f'_{rng}.json'),dict(row=row,seed=rng,solvers=records,pipeline=pipeline))
            progress('seed_comparison',index+1,len(tasks))
        for worker in workers.values():worker.close()
        workers={}
        for p,h in hashes.items():assert sha(p)==h,p
        progress('complete',1,1)
        write(out/'exit.json',dict(returncode=0,finished_unix=time.time()))
    except BaseException as exc:
        progress('failed',0,0,error=repr(exc));write(out/'exit.json',dict(returncode=1,error=repr(exc)));traceback.print_exc();raise
    finally:
        for worker in workers.values():worker.close()

if __name__=='__main__':main()
