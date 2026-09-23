#!/usr/bin/env python3
"""Recompute complete saved routes and paired travel/runtime confirmation."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def replay(instance,matrix,route):
    nodes={n['id']:n for n in instance['nodes']}; depot=instance['depot_id']
    assert route[0]==route[-1]==depot
    assert Counter(route[1:-1])==Counter({k:1 for k in nodes if k!=depot})
    index={k:i for i,k in enumerate(matrix['node_ids'])}
    origin=datetime.fromisoformat(instance['departure_time_utc'].replace('Z','+00:00'))
    def bounds(n):
        tw=n['time_window']
        return None if tw is None else [Decimal(str((datetime.fromisoformat(tw[k].replace('Z','+00:00'))-origin).total_seconds())) for k in ['start_utc','end_utc']]
    db=bounds(nodes[depot]); departure_ok=db is None or db[0]<=0<=db[1]
    clock=travel=waiting=service=lateness=Decimal(0);violations=scaled=0
    for a,b in zip(route,route[1:]):
        arc=Decimal(str(matrix['travel_seconds'][index[a]][index[b]]))
        scaled+=int((arc*1000).to_integral_value(rounding=ROUND_CEILING))
        travel+=arc; arrival=clock+arc; window=bounds(nodes[b])
        start=max(arrival,window[0]) if window else arrival
        late=max(Decimal(0),start-window[1]) if window else Decimal(0)
        duration=Decimal(str(nodes[b]['service_seconds']))
        waiting+=start-arrival;service+=duration;lateness+=late
        violations+=int(late>0);clock=start+duration
    demand=sum(Decimal(str(nodes[k]['demand_cm3'])) for k in route[1:-1])
    excess=max(Decimal(0),demand-Decimal(str(instance['vehicle_capacity_cm3'])))
    metrics={'travel_seconds':float(travel),'waiting_seconds':float(waiting),'service_seconds':float(service),
        'elapsed_seconds':float(clock),'lateness_seconds':float(lateness),'time_window_violations':violations,
        'capacity_excess_cm3':float(excess),'demand_cm3':float(demand)}
    return metrics,scaled,departure_ok and violations==0 and excess==0


def audit(stage):
    write(stage/'independent_audit.json',{'passed':False,'state':'auditing'})
    assert read(stage/'status.json')['status']=='complete'
    protocol=read(stage/'stage_protocol.json')
    hashes=protocol['input_source_hashes']
    for path,expected in hashes.items():
        snapshot=stage/'source_snapshot'/Path(path).name
        target=snapshot if Path(path).suffix=='.py' and snapshot.exists() else Path(path)
        assert sha(target)==expected, str(target)
    configs={c['name']:c for c in protocol['configs']}
    expected_keys={(c,s) for c in configs for s in protocol['seeds']}
    assert {p.stem for p in (stage/'routes').glob('*.json')}=={r['route_id'] for r in protocol['rows']}
    records=[]
    for row in protocol['rows']:
        values=read(stage/'routes'/f"{row['route_id']}.json")
        assert len(values)==len(expected_keys)
        assert {(v['config']['name'],v['seed']) for v in values}==expected_keys
        instance,matrix=read(row['instance_path']),read(row['matrix_path'])
        for value in values:
            assert value['route_id']==row['route_id']
            assert value['station_code']==row['station_code'] and value['date']==row['date_YYYY_MM_DD']
            assert value['config']==configs[value['config']['name']]
            assert value['instance_sha256']==hashes[row['instance_path']]
            assert value['matrix_sha256']==hashes[row['matrix_path']]
            if value['config']['backend'].startswith('ortools-'):
                assert 'ortools-'+value['solution']['ortools_version']==value['config']['backend']
            elif value['config']['backend'].startswith('lkh-'):
                assert 'lkh-'+value['solution']['lkh_version']==value['config']['backend']
                if value['solution']['feasible']:
                    binary=[p for p in hashes if p.endswith('/LKH')][0]
                    assert value['solution']['binary_sha256']==hashes[binary]
            else:assert value['solution']['pyvrp_version']==value['config']['backend']
            if value['config']['engine']=='window_dp' and value['solution']['feasible']:
                library=[p for p in hashes if p.endswith('/window_dp.so')][0]
                assert value['solution']['window_dp_binary_sha256']==hashes[library]
            assert value['end_to_end_seconds']>=value['search_seconds']>=0
            solution=value['solution']
            if solution['route']:
                metrics,cost,feasible=replay(instance,matrix,solution['route'])
                assert all(abs(v-solution['audit']['metrics'][k])<1e-7 for k,v in metrics.items())
                if solution['feasible']:
                    assert feasible and cost==solution['objective_scaled']
            else: assert not solution['feasible']
            records.append(value)
    evidence={'passed':True,'records':len(records),'input_source_hashes_verified':len(hashes),
        'stage_protocol_sha256':sha(stage/'stage_protocol.json'),
        'result_hashes':{p.name:sha(p) for p in sorted((stage/'routes').glob('*.json'))},
        'auditor_sha256':sha(__file__),'notes':'Independent decimal schedule, coverage, capacity and directed scaled-cost replay. Timing values are recorded measurements, not independently rerun timings.'}
    write(stage/'independent_audit.json',evidence)
    return protocol,records


def confirm(stage,lock):
    import numpy as np
    protocol,records=audit(stage)
    locked=read(lock)
    assert locked['created_unix']<protocol['created_unix']
    candidate=locked['candidate']
    frozen=read(lock.parent/'protocol.json')
    assert sha(lock.parent/'protocol.json')==locked['protocol_sha256']
    assert len(protocol['rows'])==50 and protocol['rows']==frozen['benchmark']
    assert protocol['seeds']==frozen['benchmark_seeds']==[42,20260907,271828]
    assert set(c['name'] for c in protocol['configs'])=={'old_balanced','old_reference',candidate['name']}
    assert len(protocol['configs'])==3 and candidate in protocol['configs']
    bases={c['name']:c for c in protocol['configs'] if c['engine']=='baseline'}
    assert set(bases)=={'old_balanced','old_reference'}
    assert all(c['backend']=='0.12.2' for c in bases.values())
    assert {k:v for k,v in bases['old_balanced']['budget'].items() if k!='seeds'}==dict(runtime_seconds=5.0,max_iterations=10000,max_no_improvement=2500)
    assert {k:v for k,v in bases['old_reference']['budget'].items() if k!='seeds'}==dict(iterations=5000)
    assert all(protocol['input_source_hashes'][path]==digest for path,digest in locked['source_hashes'].items())
    index={(r['route_id'],r['seed'],r['config']['name']):r for r in records}
    comparisons={};accepted=True
    for base in ['old_balanced','old_reference']:
        effects=[];losses=0
        for row in protocol['rows']:
            seed_effects=[]
            for seed in protocol['seeds']:
                c=index[row['route_id'],seed,candidate['name']]; b=index[row['route_id'],seed,base]
                assert c.get('timing_mode')!='development_component_sum', 'final runtime must be measured on a fresh solve'
                losses+=int(b['solution']['feasible'] and not c['solution']['feasible'])
                if c['solution']['feasible'] and b['solution']['feasible']:
                    ct=c['solution']['audit']['metrics']['travel_seconds'];bt=b['solution']['audit']['metrics']['travel_seconds']
                    seed_effects.append([1-ct/bt, ct-bt, c['end_to_end_seconds']/b['end_to_end_seconds'],ct,bt,c['end_to_end_seconds'],b['end_to_end_seconds']])
            if seed_effects:
                effects.append({'route_id':row['route_id'],'station':row['station_code'],'date':row['date_YYYY_MM_DD'],
                    'mean_effect':np.mean(seed_effects,axis=0).tolist()})
        grouped=defaultdict(list)
        for e in effects: grouped[e['station'],e['date']].append(e['mean_effect'][0])
        sums=np.array([sum(g) for g in grouped.values()]);counts=np.array([len(g) for g in grouped.values()])
        rng=np.random.default_rng(20260909)
        sample=rng.integers(0,len(sums),size=(10000,len(sums)))
        ci=np.quantile(sums[sample].sum(axis=1)/counts[sample].sum(axis=1),[.025,.975]).tolist()
        values=np.array([e['mean_effect'] for e in effects])
        # The user's table reports median route travel, not the median of
        # paired percentage changes. Keep both, gate on the former.
        passed=bool(losses==0 and values[:,0].mean()>=.005 and np.median(values[:,3])<np.median(values[:,4]) and ci[0]>0 and np.median(values[:,2])<=.8)
        accepted &= passed
        comparisons[base]={'passed':passed,'paired_routes':len(effects),'station_days':len(grouped),'feasible_losses':losses,
            'mean_travel_reduction':float(values[:,0].mean()),'median_travel_reduction':float(np.median(values[:,0])),
            'travel_reduction_ci95':ci,'median_paired_travel_delta_seconds':float(np.median(values[:,1])),
            'median_paired_runtime_ratio':float(np.median(values[:,2])),
            'median_route_mean_candidate_travel':float(np.median(values[:,3])),
            'median_route_mean_baseline_travel':float(np.median(values[:,4])),
            'median_route_mean_candidate_runtime':float(np.median(values[:,5])),
            'median_route_mean_baseline_runtime':float(np.median(values[:,6])),
            'travel_better_routes':int((values[:,0]>0).sum()),'effects':effects}
    decision={'candidate':candidate,'accepted':accepted,'comparisons':comparisons,'lock_sha256':sha(lock),
        'scope':'Original 50 benchmark routes and fixed three seeds; mean effects within routes before grouping. Aggregate superiority does not guarantee every route improves.'}
    seed42={}
    for name in ['old_balanced','old_reference',candidate['name']]:
        values=[r for r in records if r['config']['name']==name and r['seed']==42 and r['solution']['feasible']]
        seed42[name]=dict(feasible=len(values),total=50,
            median_travel=median(r['solution']['audit']['metrics']['travel_seconds'] for r in values),
            median_end_to_end=median(r['end_to_end_seconds'] for r in values))
    decision['seed42_comparison']=seed42
    decision['original_reference_median_reproduced']=abs(seed42['old_reference']['median_travel']-11526.3)<.05
    write(stage/'decision.json',decision)
    print(json.dumps({**decision,'comparisons':{k:{a:b for a,b in v.items() if a!='effects'} for k,v in comparisons.items()}},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',type=Path);p.add_argument('--lock',type=Path)
    a=p.parse_args()
    if a.lock: confirm(a.stage.resolve(),a.lock.resolve())
    else:
        protocol,records=audit(a.stage.resolve()); print('AUDIT PASS',len(records))
