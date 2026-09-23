#!/usr/bin/env python3
"""Serial, round-robin travel/runtime experiments on frozen planning inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
from statistics import median, mean

ROOT = Path(__file__).resolve().parents[1]
OLD_DEPS = Path(os.environ.get('CARGOFLOW_BASELINE_DEPS', ROOT/'artifacts/benchmark_runtime/deps'))
V3 = Path(os.environ.get('CARGOFLOW_PREFERENCE_DATA', ROOT/'artifacts/preference_replay_v3'))
V4 = Path(os.environ.get('CARGOFLOW_DEVELOPMENT_DATA', ROOT/'artifacts/softcost_v4'))


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def baseline(name):
    budgets = read(ROOT/'configs/solver_budgets.json')
    return {'name':'old_'+name, 'engine':'baseline', 'backend':'0.12.2', 'budget':budgets[name]}


def candidates():
    configs = []
    for name, backend, engine, neighbours in [
        ('hgs_native40','0.12.2','native',40),
        ('hgs_n16','0.12.2','native',16),
        ('hgs_n80','0.12.2','native',80),
        ('hgs_warm40','0.12.2','warm',40),
        ('ils_default50','0.13.4','native',50),
        ('ils_n20','0.13.4','native',20),
        ('ils_n80','0.13.4','native',80),
        ('ils_warm50','0.13.4','warm',50)]:
        configs.append(dict(name=name,backend=backend,engine=engine,neighbours=neighbours,seconds=3.0))
    return configs


def prepare(out):
    if out.exists(): raise FileExistsError(out)
    rows = read(V3/'manifest.json')
    by_id = {r['route_id']:r for r in rows}
    benchmark = [by_id[k.strip()] for k in (ROOT/'configs/benchmark_routes_50.txt').read_text().splitlines() if k.strip()]
    old = read(V4/'protocol.json')
    development = sorted(old['development'],key=lambda r: hashlib.sha256(('travel-v5-'+r['route_id']).encode()).hexdigest())
    def group(r): return (r['station_code'],r['date_YYYY_MM_DD'])
    assert {group(r) for r in development}.isdisjoint(group(r) for r in benchmark)
    keys = ['route_id','station_code','date_YYYY_MM_DD','instance_path','matrix_path']
    clean = lambda values: [{k:r[k] for k in keys} for r in values]
    protocol = {'created_unix':time.time(),'development':clean(development), 'benchmark':clean(benchmark),
        'screen_count':12,'refine_count':24,'seed_screen':[42],'seed_refine':[42,20260907],
        'benchmark_seeds':[42,20260907,271828], 'screen_candidates':candidates(),
        'goal':'Maintain baseline feasibility while reducing BOTH directed travel and end-to-end solver time relative to old balanced and reference.',
        'separation':'No actual sequences, zones, labels, old solutions or per-route tuned configurations are inference inputs. Development station-days are disjoint from the original 50-route benchmark.',
        'timing':'One task solver active at a time; persistent separate version workers; method order rotates by route and seed; import warmup excluded equally, input validation/matrix build/initialisation/audit included. Disk input loading excluded as in the old table. Record process CPU time and host load.',
        'selection':'Screen choose best two feasible candidates on mean paired travel vs old balanced; refine on first24 development routes/two seeds. Lock one candidate with zero baseline feasibility losses, mean and median travel reductions versus BOTH balanced/reference, and median end-to-end runtime ratio <=0.8 versus BOTH. No benchmark tuning.',
        'final_gate':'Original50 routes, three paired seeds averaged within route; zero baseline feasible losses; mean travel gain >=0.005 versus BOTH baselines; median travel reduction versus BOTH; lower95% station-day-cluster bootstrap bound >0 for mean travel gain versus BOTH; median paired end-to-end ratio <=0.8 versus BOTH. Preserve invalid capacity input as part of denominator.',
        'adaptation':'Development-only method changes allowed and recorded in separate stage protocols. Freeze all candidate code and configuration before the one final benchmark comparison.'}
    write(out/'protocol.json',protocol)
    print(json.dumps({'development':len(development),'benchmark':len(benchmark),'candidates':len(candidates())}))


def worker(backend):
    sys.path.insert(0,str(OLD_DEPS))
    if backend=='0.13.4' or backend.startswith(('ortools-','lkh-')): sys.path.insert(0,str(ROOT/'artifacts/runtime13/deps'))
    if backend.startswith('ortools-'):
        sys.path.insert(0,str(ROOT/'artifacts/runtime_ortools/deps'))
        sys.path.append(str(ROOT/'artifacts/runtime_ml/deps'))
    sys.path.insert(0,str(ROOT/'src'))
    from cargoflow.travel_search import solve_travel
    import pyvrp
    # Both versions import native search and baseline helpers before timing.
    import pyvrp.search, pyvrp.stop
    import cargoflow.baselines
    from importlib.metadata import version
    if backend.startswith('ortools-'):
        from cargoflow.ortools_search import solve_ortools
        assert 'ortools-'+version('ortools')==backend
    elif backend.startswith('lkh-'):
        from cargoflow.lkh_search import solve_lkh
        assert version('pyvrp')=='0.13.4'
    else:assert version('pyvrp')==backend
    print(json.dumps({'ready':backend,'path':pyvrp.__file__}),flush=True)
    for line in sys.stdin:
        task=json.loads(line)
        row=task['row']
        instance,matrix=read(row['instance_path']),read(row['matrix_path'])
        child_before=resource.getrusage(resource.RUSAGE_CHILDREN)
        before_cpu=time.process_time()+child_before.ru_utime+child_before.ru_stime
        load=os.getloadavg()
        if task['config']['engine'] in ['ortools_gls','ils_gls']:
            result=solve_ortools(instance,matrix,task['config'],task['seed'])
        elif task['config']['engine'] in ['lkh','ils_lkh']:
            result=solve_lkh(instance,matrix,task['config'],task['seed'])
        elif task['config']['engine']=='window_dp':
            from cargoflow.window_search import solve_window
            result=solve_window(instance,matrix,task['config'],task['seed'])
        elif task['config']['engine']=='polished':
            from cargoflow.polished_search import solve_polished
            result=solve_polished(instance,matrix,task['config'],task['seed'])
        else:
            result=solve_travel(instance,matrix,task['config'],task['seed'])
        child_after=resource.getrusage(resource.RUSAGE_CHILDREN)
        result.update(route_id=row['route_id'],station_code=row['station_code'],date=row['date_YYYY_MM_DD'],
            process_cpu_seconds=time.process_time()+child_after.ru_utime+child_after.ru_stime-before_cpu,load_before=list(load),
            load_after=list(os.getloadavg()),instance_sha256=sha(row['instance_path']),matrix_sha256=sha(row['matrix_path']))
        print(json.dumps(result,allow_nan=False),flush=True)


def summarise(out, rows, configs, seeds):
    records=[]
    for row in rows: records.extend(read(out/'routes'/f"{row['route_id']}.json"))
    values=[]
    for config in configs:
        name=config['name']; selected=[r for r in records if r['config']['name']==name]
        feasible=[r for r in selected if r['solution']['feasible']]
        value={'config':config,'records':len(selected),'feasible':len(feasible),
            'median_travel':median(r['solution']['audit']['metrics']['travel_seconds'] for r in feasible) if feasible else None,
            'median_end_to_end':median(r['end_to_end_seconds'] for r in selected),
            'median_cpu':median(r['process_cpu_seconds'] for r in selected),'comparisons':{}}
        for base_name in ['old_balanced','old_reference']:
            baseline_records={(r['route_id'],r['seed']):r for r in records if r['config']['name']==base_name}
            effects=[]; lost=0
            for r in selected:
                b=baseline_records.get((r['route_id'],r['seed']))
                if not b: continue
                lost+=int(b['solution']['feasible'] and not r['solution']['feasible'])
                if b['solution']['feasible'] and r['solution']['feasible']:
                    t=r['solution']['audit']['metrics']['travel_seconds']; bt=b['solution']['audit']['metrics']['travel_seconds']
                    effects.append({'route_id':r['route_id'],'seed':r['seed'],'station':r['station_code'],'date':r['date'],
                        'gain':1-t/bt,'delta':t-bt,'runtime_ratio':r['end_to_end_seconds']/b['end_to_end_seconds']})
            if effects:
                value['comparisons'][base_name]={'paired_records':len(effects),'feasible_losses':lost,
                    'mean_travel_gain':mean(x['gain'] for x in effects),'median_travel_gain':median(x['gain'] for x in effects),
                    'median_travel_delta':median(x['delta'] for x in effects),'median_runtime_ratio':median(x['runtime_ratio'] for x in effects)}
        values.append(value)
    write(out/'summary.json',values)
    for x in values: print(json.dumps(x),flush=True)
    return values


def run_stage(out, rows, configs, seeds):
    if (out/'status.json').exists(): raise FileExistsError('stage already started; preserve the existing session')
    sources=[ROOT/'src/cargoflow/solver.py',ROOT/'src/cargoflow/travel_search.py',Path(__file__).resolve()]
    sources += [ROOT/'src/cargoflow'/name for name in ['contract.py','baselines.py','directed_polish.py','polished_search.py','ortools_search.py','lkh_search.py','window_dp.py','window_search.py']
                if (ROOT/'src/cargoflow'/name).exists()]
    hashes={str(p):sha(p) for p in sources}
    if any(c['backend'].startswith('lkh-') for c in configs):
        for path in [ROOT/'artifacts/runtime_lkh/LKH-3.0.13/LKH',ROOT/'artifacts/runtime_lkh/LKH-3.0.13/SRC/LKHmain.c']:
            hashes[str(path)]=sha(path)
    if any(c['engine']=='window_dp' for c in configs):
        for path in [ROOT/'artifacts/runtime_window_dp/window_dp.so',ROOT/'native/window_dp.cpp']:
            hashes[str(path)]=sha(path)
    for row in rows:
        for key in ('instance_path','matrix_path'): hashes[row[key]]=sha(row[key])
    write(out/'stage_protocol.json',{'created_unix':time.time(),'rows':rows,'configs':configs,'seeds':seeds,'input_source_hashes':hashes})
    (out/'source_snapshot').mkdir()
    for path in sources: shutil.copy2(path,out/'source_snapshot'/path.name)
    write(out/'status.json',{'status':'running','pid':os.getpid(),'started':time.time()})
    workers={}
    try:
        for backend in sorted({c['backend'] for c in configs}):
            log=open(out/f'worker_{backend}.log','w')
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'worker','--backend',backend],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1,
                env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
            ready=proc.stdout.readline()
            if not ready: raise RuntimeError(f'worker {backend} failed; inspect log')
            assert json.loads(ready)['ready']==backend
            workers[backend]=(proc,log)
        started=time.time()
        for idx,row in enumerate(rows):
            results=[]
            for seed_idx,seed in enumerate(seeds):
                offset=(idx+seed_idx)%len(configs)
                ordered=configs[offset:]+configs[:offset]
                for config in ordered:
                    proc,_=workers[config['backend']]
                    proc.stdin.write(json.dumps({'row':row,'config':config,'seed':seed})+'\n');proc.stdin.flush()
                    line=proc.stdout.readline()
                    if not line: raise RuntimeError(f"worker {config['backend']} exited on {row['route_id']} {config['name']}")
                    record=json.loads(line)
                    assert record['instance_sha256']==hashes[row['instance_path']]
                    assert record['matrix_sha256']==hashes[row['matrix_path']]
                    results.append(record)
            write(out/'routes'/f"{row['route_id']}.json",results)
            print(f"{out.name} {idx+1}/{len(rows)} elapsed={time.time()-started:.1f}s",flush=True)
        summarise(out,rows,configs,seeds)
        assert all(sha(p)==h for p,h in hashes.items())
        write(out/'status.json',{'status':'complete','finished':time.time(),'hashes_verified':len(hashes)})
    except BaseException as exc:
        write(out/'failure.json',{'error':str(exc),'time':time.time()})
        raise
    finally:
        for proc,log in workers.values():
            proc.stdin.close()
            proc.wait(timeout=30)
            log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','worker','screen','custom'])
    p.add_argument('--output',type=Path)
    p.add_argument('--backend')
    p.add_argument('--stage-spec',type=Path)
    args=p.parse_args()
    if args.mode=='worker': worker(args.backend)
    elif args.mode=='prepare': prepare(args.output.resolve())
    elif args.mode=='screen':
        out=args.output.resolve(); protocol=read(out/'protocol.json')
        run_stage(out/'screen',protocol['development'][:12],[baseline('balanced'),baseline('reference')]+protocol['screen_candidates'],[42])
    else:
        spec=read(args.stage_spec)
        run_stage(args.output.resolve(),spec['rows'],spec['configs'],spec['seeds'])
