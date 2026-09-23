#!/usr/bin/env python3
"""Detached, sequential, atomic-checkpoint acceptance run on original50."""
import argparse
from dataclasses import asdict
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

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_policy_search import POLICIES
from select_policy_seed import choose_seed
from cargoflow.policy_search import search_policy

OLD=Path(os.environ.get('CARGOFLOW_BASELINE_DEPS', ROOT/'artifacts/benchmark_runtime/deps'))
DATA=Path(os.environ.get('CARGOFLOW_PREFERENCE_DATA', ROOT/'artifacts/preference_replay_v3'))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');tmp.replace(p)


class Worker:
    def __init__(self,backend,out):
        self.log=open(out/('worker_'+backend+'.log'),'w')
        self.proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_travel_v5.py'),'worker','--backend',backend],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        self.ready=json.loads(self.proc.stdout.readline());assert self.ready['ready']==backend
    def solve(self,row,config):
        self.proc.stdin.write(json.dumps(dict(row=row,config=config,seed=42))+'\n');self.proc.stdin.flush()
        line=self.proc.stdout.readline()
        if not line:raise RuntimeError('solver worker exited: '+str(self.proc.poll()))
        return json.loads(line)
    def close(self):
        if self.proc.poll() is None:self.proc.stdin.close();self.proc.wait(timeout=30)
        self.log.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'routes').mkdir()
    workers={};started=time.time()
    try:
        manifest=read(DATA/'manifest.json');by_id={r['route_id']:r for r in manifest}
        ids=[s.strip() for s in (ROOT/'configs/benchmark_routes_50.txt').read_text().splitlines() if s.strip()]
        assert len(ids)==len(set(ids))==50
        keys=['route_id','station_code','date_YYYY_MM_DD','instance_path','matrix_path']
        rows=[{k:by_id[i][k] for k in keys} for i in ids]
        sources=read(ROOT/'SOURCE_MANIFEST.json')
        hashes={str(ROOT/p):digest for p,digest in sources.items()}
        for r in rows:
            for k in ['instance_path','matrix_path']:hashes[r[k]]=sha(r[k])
        for folder in [OLD,ROOT/'artifacts/runtime13/deps']:
            for binary in (folder/'pyvrp').rglob('*.so'):hashes[str(binary)]=sha(binary)
        for p,digest in hashes.items():assert sha(p)==digest,p
        snapshot=out/'source_snapshot';snapshot.mkdir()
        for p in sources:
            target=snapshot/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/p,target)
        config_old=dict(name='old_reference',engine='baseline',backend='0.12.2',budget={'iterations':5000})
        config_ils=dict(name='ils13',engine='native',backend='0.13.4',seconds=3.8,neighbours=80)
        protocol=dict(created_unix=started,source_commit=(ROOT/'SOURCE_COMMIT').read_text().strip(),rows=rows,pilot_ids=(ROOT/'configs/benchmark_routes_pilot5.txt').read_text().splitlines(),seed=42,iterations=50000,policies=[asdict(p) for p in POLICIES],seed_configs=[config_old,config_ils],frozen_hashes=hashes,host=platform.node(),python=platform.python_version(),pid=os.getpid(),timing='Fresh old-reference and ILS, then policy calls. Imports preloaded; seed wall includes worker IO and seed audit. All-mode wall is recorded separately. Sequential workers, BLAS/OMP threads=1.')
        write(out/'protocol.json',protocol);write(out/'status.json',dict(status='running',completed=0,pid=os.getpid()))
        for backend in ['0.12.2','0.13.4']:workers[backend]=Worker(backend,out)
        write(out/'workers.json',{k:dict(pid=w.proc.pid,ready=w.ready) for k,w in workers.items()})
        for idx,row in enumerate(rows):
            route_start=time.perf_counter();generation_start=time.perf_counter()
            instance,matrix=read(row['instance_path']),read(row['matrix_path'])
            configs=[config_old,config_ils] if idx%2==0 else [config_ils,config_old]
            candidates={c['name']:workers[c['backend']].solve(row,c) for c in configs}
            selected=choose_seed(instance,matrix,[(k,candidates[k]['solution']) for k in ['old_reference','ils13']])
            seed_seconds=time.perf_counter()-generation_start
            entries={}
            for offset in [(idx+j)%4 for j in range(4)]:
                policy=POLICIES[offset];before=time.perf_counter()
                if selected['feasible']:
                    result=search_policy(instance,matrix,selected['route'],policy,iterations=50000,seed=42+offset)
                    status='complete'
                else:result=None;status='no_feasible_seed'
                seconds=time.perf_counter()-before
                entries[offset]=dict(name=policy.name,rng_seed=42+offset,status=status,result=result,policy_seconds=seconds,single_mode_pipeline_seconds=seed_seconds+seconds)
            record=dict(row=row,input_sha256={row[k]:hashes[row[k]] for k in ['instance_path','matrix_path']},seed_candidates=candidates,selected_seed=selected,seed_generation_wall_seconds=seed_seconds,policies=[entries[j] for j in range(4)],all_modes_wall_seconds=time.perf_counter()-route_start,host_load=os.getloadavg())
            write(out/'routes'/(row['route_id']+'.json'),record)
            write(out/'status.json',dict(status='running',completed=idx+1,pid=os.getpid()))
            print(f"completed {idx+1}/50 seed={selected['source']} feasible={selected['feasible']} wall={record['all_modes_wall_seconds']:.1f}s",flush=True)
        for w in workers.values():w.close()
        workers={}
        write(out/'status.json',dict(status='complete',completed=50,pid=os.getpid(),wall_seconds=time.time()-started))
        from audit_policy_benchmark import audit
        audit(out)
        write(out/'exit.json',dict(returncode=0,finished_unix=time.time()))
    except BaseException as exc:
        write(out/'status.json',dict(status='failed',error=repr(exc),completed=len(list((out/'routes').glob('*.json')))))
        write(out/'exit.json',dict(returncode=1,error=repr(exc),finished_unix=time.time()))
        traceback.print_exc();raise
    finally:
        for w in workers.values():w.close()


if __name__=='__main__':main()
