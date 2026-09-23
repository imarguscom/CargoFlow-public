#!/usr/bin/env python3
"""Continue a terminated frozen stage, preserving every completed route file."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from run_travel_v5 import ROOT,OLD_DEPS,read,write,sha,summarise

def alive(pid):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return bool(cmd) and (b'run_travel_v5.py' in cmd or b'resume_travel_v5.py' in cmd)
    except FileNotFoundError:return False

def main(stage):
    spec=read(stage/'stage_protocol.json');status=read(stage/'status.json')
    if status['status']=='complete':raise RuntimeError('completed stage must not rerun')
    if alive(status['pid']):raise RuntimeError('original stage still alive')
    # Do not create another solver while a prior worker survives.
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:cmd=(proc/'cmdline').read_bytes()
        except (OSError,PermissionError):continue
        if str(ROOT/'scripts/run_travel_v5.py').encode() in cmd:
            raise RuntimeError('original solver/worker still present: '+proc.name)
    assert all(sha(p)==h for p,h in spec['input_source_hashes'].items()), 'frozen input/source changed'
    lock=read(stage.parent/'locked_selection.json')
    assert all(sha(p)==h for p,h in lock['source_hashes'].items())
    rows,configs,seeds=spec['rows'],spec['configs'],spec['seeds']
    expected={(c['name'],s) for c in configs for s in seeds}
    by_name={c['name']:c for c in configs}
    preserved={}
    row_ids={r['route_id'] for r in rows}
    for path in sorted((stage/'routes').glob('*.json')):
        assert path.stem in row_ids
        records=read(path)
        assert len(records)==len(expected) and {(r['config']['name'],r['seed']) for r in records}==expected
        assert all(r['route_id']==path.stem and r['config']==by_name[r['config']['name']] for r in records)
        preserved[path.name]=sha(path)
    attempt=1
    while (stage/f'resume_{attempt}').exists():attempt+=1
    checkpoint=stage/f'resume_{attempt}';checkpoint.mkdir()
    write(checkpoint/'before.json',dict(previous_status=status,previous_failure=read(stage/'failure.json') if (stage/'failure.json').exists() else None,
        preserved_result_sha256=preserved,protocol_sha256=sha(stage/'stage_protocol.json'),lock_sha256=sha(stage.parent/'locked_selection.json'),
        helper_sha256=sha(__file__),pid=os.getpid(),started=time.time(),note='Original process and workers were absent; continue only missing route files with unchanged original order/config/seeds/source.'))
    write(stage/'status.json',dict(status='running',pid=os.getpid(),started=status['started'],resume_attempt=attempt,resumed=time.time()))
    workers={};code=1
    try:
        for backend in sorted({c['backend'] for c in configs}):
            log=open(checkpoint/f'worker_{backend}.log','w')
            proc=subprocess.Popen([sys.executable,str(ROOT/'scripts/run_travel_v5.py'),'worker','--backend',backend],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            ready=proc.stdout.readline()
            if not ready or json.loads(ready).get('ready')!=backend:raise RuntimeError('worker initialization failed: '+backend)
            workers[backend]=(proc,log)
        print(f'Resuming with {len(preserved)}/{len(rows)} complete route files preserved',flush=True)
        for idx,row in enumerate(rows):
            path=stage/'routes'/f"{row['route_id']}.json"
            if path.name in preserved:continue
            values=[]
            for seed_idx,seed in enumerate(seeds):
                offset=(idx+seed_idx)%len(configs)
                for config in configs[offset:]+configs[:offset]:
                    proc,_=workers[config['backend']]
                    proc.stdin.write(json.dumps(dict(row=row,config=config,seed=seed))+'\n');proc.stdin.flush()
                    line=proc.stdout.readline()
                    if not line:raise RuntimeError('worker exited: '+config['name'])
                    record=json.loads(line)
                    assert record['instance_sha256']==spec['input_source_hashes'][row['instance_path']]
                    assert record['matrix_sha256']==spec['input_source_hashes'][row['matrix_path']]
                    values.append(record)
            write(path,values)
            print(f'benchmark resumed {idx+1}/{len(rows)}',flush=True)
        assert all(sha(stage/'routes'/name)==h for name,h in preserved.items())
        assert all(sha(p)==h for p,h in spec['input_source_hashes'].items())
        summarise(stage,rows,configs,seeds)
        write(stage/'status.json',dict(status='complete',finished=time.time(),hashes_verified=len(spec['input_source_hashes']),
            resumed=True,preserved_routes=len(preserved),resume_attempt=attempt))
        write(checkpoint/'preservation.json',dict(passed=True,preserved_result_sha256=preserved,
            helper_sha256=sha(__file__),protocol_sha256=sha(stage/'stage_protocol.json'),lock_sha256=sha(stage.parent/'locked_selection.json')))
        subprocess.run([sys.executable,str(ROOT/'scripts/audit_travel_v5.py'),str(stage),'--lock',str(stage.parent/'locked_selection.json')],
            check=True,env={**os.environ,'PYTHONPATH':str(OLD_DEPS),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        code=0
    except BaseException as exc:
        write(checkpoint/'failure.json',dict(error=repr(exc),time=time.time()))
        raise
    finally:
        for proc,log in workers.values():
            if proc.poll() is None:
                try:proc.stdin.close()
                except BrokenPipeError:pass
                proc.wait(timeout=30)
            log.close()
        write(checkpoint/'exit.json',dict(returncode=code,finished=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',type=Path);a=p.parse_args();main(a.stage.resolve())
