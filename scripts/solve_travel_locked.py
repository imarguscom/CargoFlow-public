#!/usr/bin/env python3
"""Run the benchmark-accepted travel solver on one fresh planning input pair.

Uses the same persistent-worker protocol as evaluation, with a one-shot worker
for this CLI. Reports both warm-call endpoint time and actual one-shot latency.
The default old PyVRP solver is unchanged. Requires completed accepted evidence.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--instance',type=Path,required=True)
    p.add_argument('--matrix',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--evidence-root',type=Path,default=ROOT/'artifacts/travel_v5')
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    lock=a.evidence_root/'locked_selection.json'
    decision=read(a.evidence_root/'benchmark/decision.json')
    audit=read(a.evidence_root/'benchmark/independent_audit.json')
    if not decision['accepted'] or not audit['passed']:
        raise RuntimeError('candidate has not passed the original benchmark gate')
    if sha(lock)!=decision['lock_sha256']:raise RuntimeError('candidate lock changed')
    config=read(lock)['candidate']
    if config!=decision['candidate']:raise RuntimeError('candidate config mismatch')
    evidence=read(a.evidence_root/'benchmark/stage_protocol.json')
    if sha(a.evidence_root/'benchmark/stage_protocol.json')!=audit['stage_protocol_sha256']:
        raise RuntimeError('benchmark protocol changed')
    # Only deployed solver sources and native runtimes must match; a fresh
    # application input is intentionally different from the benchmark data.
    for path,digest in evidence['input_source_hashes'].items():
        if path.endswith(('.py','.cpp','/LKH','.so','.c')) and sha(path)!=digest:
            raise RuntimeError('solver source/runtime changed: '+path)
    instance_path,matrix_path=a.instance.resolve(),a.matrix.resolve()
    before={str(v):sha(v) for v in [instance_path,matrix_path]}
    instance=read(instance_path)
    row=dict(route_id=instance['instance_id'],station_code=instance.get('station_code','application'),
        date_YYYY_MM_DD=instance['departure_time_utc'][:10],instance_path=str(instance_path),matrix_path=str(matrix_path))
    started=time.perf_counter()
    proc=subprocess.run([sys.executable,str(ROOT/'scripts/run_travel_v5.py'),'worker','--backend',config['backend']],
        input=json.dumps(dict(row=row,config=config,seed=a.seed))+'\n',capture_output=True,text=True,check=True,
        env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
    latency=time.perf_counter()-started
    messages=[json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    if len(messages)!=2 or messages[0].get('ready')!=config['backend']:
        raise RuntimeError('unexpected worker protocol')
    record=messages[1]
    if record['instance_sha256']!=before[str(instance_path)] or record['matrix_sha256']!=before[str(matrix_path)]:
        raise RuntimeError('planning inputs changed during solve')
    record.update(one_shot_worker_seconds=latency,lock_sha256=sha(lock),benchmark_decision_sha256=sha(a.evidence_root/'benchmark/decision.json'))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(feasible=record['solution']['feasible'],output=str(a.out.resolve()),
        warm_call_seconds=record['end_to_end_seconds'],one_shot_worker_seconds=latency)))

if __name__=='__main__':main()
