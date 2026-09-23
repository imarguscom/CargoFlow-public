#!/usr/bin/env python3
"""Independent Decimal replay of the complete multiobjective acceptance run."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from statistics import mean, median


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, value): Path(p).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replay(instance, matrix, route):
    """Does not import the production evaluator or solver auditor."""
    D=lambda x:Decimal(str(x))
    nodes={n['id']:n for n in instance['nodes']};depot=instance['depot_id']
    assert route[0]==route[-1]==depot
    assert Counter(route[1:-1])==Counter({k:1 for k in nodes if k!=depot})
    indexes={k:i for i,k in enumerate(matrix['node_ids'])}
    origin=datetime.fromisoformat(instance['departure_time_utc'].replace('Z','+00:00'))
    def window(n):
        v=n['time_window']
        return None if v is None else [D((datetime.fromisoformat(v[k].replace('Z','+00:00'))-origin).total_seconds()) for k in ['start_utc','end_utc']]
    capacity=D(instance['vehicle_capacity_cm3'])
    remaining=sum(D(n['demand_cm3']) for n in nodes.values())
    assert remaining<=capacity
    dw=window(nodes[depot]);assert dw is None or dw[0]<=0<=dw[1]
    clock=travel=waiting=energy=completion=Decimal(0)
    for a,b in zip(route,route[1:]):
        duration=D(matrix['travel_seconds'][indexes[a]][indexes[b]])
        energy+=duration*(1+(remaining/capacity if capacity else 0))
        arrival=clock+duration;tw=window(nodes[b]);start=max(arrival,tw[0]) if tw else arrival
        assert tw is None or start<=tw[1]
        clock=start+D(nodes[b]['service_seconds'])
        if b!=depot:completion+=clock
        waiting+=start-arrival;travel+=duration;remaining-=D(nodes[b]['demand_cm3'])
    assert remaining==0
    return dict(travel_seconds=travel,elapsed_seconds=clock,waiting_seconds=waiting,
                energy_proxy=energy,delivery_completion_sum_seconds=completion,preference_cost=Decimal(0))


def score(metrics,reference,policy):
    terms=[('time_weight','travel_seconds'),('energy_weight','energy_proxy'),
           ('completion_weight','delivery_completion_sum_seconds'),('preference_weight','preference_cost')]
    return sum(policy[w]*float(metrics[k])/(float(reference[k]) or 1.0) for w,k in terms if policy[w])/sum(policy[w] for w,k in terms)


def audit(root):
    import numpy as np
    root=Path(root);protocol=read(root/'protocol.json')
    assert read(root/'status.json')['status']=='complete'
    assert len(protocol['rows'])==50 and protocol['seed']==42 and protocol['iterations']==50000
    assert [p['name'] for p in protocol['policies']]==['fastest','delivery_speed','balanced','green']
    assert [p['max_travel_increase'] for p in protocol['policies']]==[0,.02,.02,.05]
    assert all(p['preference_weight']==0 for p in protocol['policies'])
    assert protocol['policies'][3]['max_energy_increase']==0
    for path,digest in protocol['frozen_hashes'].items():assert sha(path)==digest,path
    expected={r['route_id']+'.json' for r in protocol['rows']}
    assert {p.name for p in (root/'routes').glob('*.json')}==expected
    effects=defaultdict(list);infeasible=[];seed_sources=Counter();fallbacks=Counter();baseline_calls=0
    for row in protocol['rows']:
        value=read(root/'routes'/(row['route_id']+'.json'))
        assert value['row']==row
        assert value['input_sha256']=={p:protocol['frozen_hashes'][p] for p in [row['instance_path'],row['matrix_path']]}
        instance,matrix=read(row['instance_path']),read(row['matrix_path'])
        eligible=[]
        assert set(value['seed_candidates'])=={'old_reference','ils13'}
        for name,record in value['seed_candidates'].items():
            assert record['seed']==42 and record['route_id']==row['route_id']
            solution=record['solution'];baseline_calls+=1
            assert record['end_to_end_seconds']>=0
            if name=='old_reference':
                assert solution['pyvrp_version']=='0.12.2' and record['config']['budget']=={'iterations':5000}
            else:
                assert solution['pyvrp_version']=='0.13.4'
                assert record['config']['seconds']==3.8 and record['config']['neighbours']==80
            if solution['feasible']:
                metrics=replay(instance,matrix,solution['route'])
                assert abs(float(metrics['travel_seconds'])-solution['audit']['metrics']['travel_seconds'])<1e-7
                eligible.append((metrics['travel_seconds'],name,solution['route']))
        assert len(value['policies'])==4
        if not eligible:
            demand=sum(Decimal(str(n['demand_cm3'])) for n in instance['nodes'])
            capacity=Decimal(str(instance['vehicle_capacity_cm3']))
            assert demand>capacity,'Unexpected no-feasible-seed case'
            assert not value['selected_seed']['feasible']
            assert all(p['status']=='no_feasible_seed' and p['result'] is None for p in value['policies'])
            infeasible.append(dict(route_id=row['route_id'],reason='single-vehicle volume capacity',demand=str(demand),capacity=str(capacity)))
            continue
        # Preserve input candidate order to reproduce baseline-first ties.
        order={'old_reference':0,'ils13':1}
        chosen=min(eligible,key=lambda v:(v[0],order[v[1]]))
        assert value['selected_seed']['route']==chosen[2]
        assert value['selected_seed']['source']==chosen[1]
        seed_sources[chosen[1]]+=1
        reference=replay(instance,matrix,chosen[2])
        for idx,policy in enumerate(protocol['policies']):
            entry=value['policies'][idx]
            assert entry['name']==policy['name'] and entry['rng_seed']==42+idx
            result=entry['result'];assert result['official_audit']['feasible']
            assert result['search']['requested_iterations']==50000 and result['search']['seed']==42+idx
            assert result['policy']['max_travel_increase']==policy['max_travel_increase']
            assert result['policy']['max_energy_increase']==policy['max_energy_increase']
            assert result['policy']['weights']==dict(time=policy['time_weight'],energy_proxy=policy['energy_weight'],preference=policy['preference_weight'],delivery_completion=policy['completion_weight'])
            metrics=replay(instance,matrix,result['route'])
            for k,v in metrics.items():
                assert abs(float(v)-result['metrics'][k])<1e-6,(row['route_id'],policy['name'],k)
                assert abs(float(reference[k])-result['reference'][k])<1e-6
            assert metrics['travel_seconds']<=reference['travel_seconds']*(1+Decimal(str(policy['max_travel_increase'])))
            if policy['max_energy_increase'] is not None:
                assert metrics['energy_proxy']<=reference['energy_proxy']*(1+Decimal(str(policy['max_energy_increase'])))
            assert score(metrics,reference,policy)<=score(reference,reference,policy)+1e-12
            assert entry['policy_seconds']>=0 and value['seed_generation_wall_seconds']>=0
            assert abs(entry['single_mode_pipeline_seconds']-entry['policy_seconds']-value['seed_generation_wall_seconds'])<1e-7
            if result['search']['fallback_reason']:fallbacks[policy['name']]+=1
            changes={k:(float(metrics[k]/reference[k])-1 if reference[k] else 0.0) for k in ['travel_seconds','energy_proxy','delivery_completion_sum_seconds']}
            effects[policy['name']].append(dict(route_id=row['route_id'],station=row['station_code'],date=row['date_YYYY_MM_DD'],pilot=row['route_id'] in protocol['pilot_ids'],changes=changes,metrics={k:float(v) for k,v in metrics.items()},policy_seconds=entry['policy_seconds'],single_mode_pipeline_seconds=entry['single_mode_pipeline_seconds']))
    assert len(infeasible)==1
    summaries={}
    for policy in protocol['policies']:
        rows=effects[policy['name']];assert len(rows)==49
        item=dict(feasible=49,total=50,epsilon_violations=0,objective_regressions=0,final_fallbacks=fallbacks[policy['name']],median_policy_seconds=median(r['policy_seconds'] for r in rows),median_single_mode_pipeline_seconds=median(r['single_mode_pipeline_seconds'] for r in rows),changes={})
        for key in ['travel_seconds','energy_proxy','delivery_completion_sum_seconds']:
            groups=defaultdict(list)
            for r in rows:groups[r['station'],r['date']].append(r['changes'][key])
            sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
            rng=np.random.default_rng(20260916);sample=rng.integers(0,len(sums),size=(10000,len(sums)))
            ci=np.quantile(sums[sample].sum(axis=1)/counts[sample].sum(axis=1),[.025,.975]).tolist()
            values=[r['changes'][key] for r in rows]
            item['changes'][key]=dict(mean=mean(values),median=median(values),ci95=ci,min=min(values),max=max(values),better=sum(v < -1e-12 for v in values),worse=sum(v > 1e-12 for v in values),non_pilot_mean=mean(r['changes'][key] for r in rows if not r['pilot']))
        summaries[policy['name']]=item
    result=dict(passed=True,source_commit=protocol['source_commit'],routes=50,seed_solver_records=baseline_calls,policy_records=200,feasible_policy_records=196,seed_sources=dict(seed_sources),infeasible_inputs=infeasible,policies=summaries,scope='Original50 acceptance test with seed42; includes pilot5. Energy proxy is not measured fuel or physical energy.',protocol_sha256=sha(root/'protocol.json'),result_sha256={p.name:sha(p) for p in sorted((root/'routes').glob('*.json'))})
    write(root/'independent_audit.json',result);write(root/'paired_effects.json',dict(effects))
    print(json.dumps({k:v for k,v in result.items() if k!='result_sha256'},indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    audit(parser.parse_args().root)
