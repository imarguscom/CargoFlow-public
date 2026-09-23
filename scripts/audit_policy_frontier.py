"""Independent final audit and paired summaries for the frozen frontier study.

Uses the earlier independent Decimal route oracle, never the experimental
native evaluator. Integer assignment dual feasibility is checked explicitly.
"""
from collections import defaultdict
from decimal import Decimal,ROUND_FLOOR
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from statistics import mean,median
import numpy as np
from audit_policy_benchmark import replay

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def weighted(m,ref,p):
    pairs=[('travel_seconds','time_weight'),('delivery_completion_sum_seconds','completion_weight'),('energy_proxy','energy_weight')]
    return sum(p[w]*float(m[k])/(float(ref[k]) or 1) for k,w in pairs)/sum(p[w] for k,w in pairs)
def bootstrap(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[r['station'],r['date']].append(r[key])
    sums=np.array([sum(v) for v in groups.values()]);counts=np.array([len(v) for v in groups.values()])
    rng=np.random.default_rng(20260916);idx=rng.integers(0,len(sums),size=(10000,len(sums)))
    return np.quantile(sums[idx].sum(axis=1)/counts[idx].sum(axis=1),[.025,.975]).tolist()
def collapse(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[r['route_id']].append(r)
    return [dict(route_id=k,station=values[0]['station'],date=values[0]['date'],value=mean(v[key] for v in values)) for k,values in groups.items()]
def effect(rows,key):
    values=collapse(rows,key)
    return dict(routes=len(values),mean=mean(v['value'] for v in values),median=median(v['value'] for v in values),
        ci95=bootstrap(values,'value'),better=sum(v['value']< -1e-12 for v in values),worse=sum(v['value']>1e-12 for v in values))
def meta(row):return dict(route_id=row['route_id'],station=row['station_code'],date=row['date_YYYY_MM_DD'])

def audit(root):
    root=Path(root);protocol=read(root/'protocol.json');assert read(root/'exit.json')['returncode']==0
    for path,digest in protocol['frozen_hashes'].items():assert sha(path)==digest,path
    policies={p['name']:p for p in protocol['policies']};assert len(policies)==4
    assert [p['max_travel_increase'] for p in policies.values()]==[0,.02,.02,.05]
    assert all(p['preference_weight']==0 for p in policies.values())
    counts=defaultdict(int)
    def check_trial(instance,matrix,anchor,t):
        ref=replay(instance,matrix,anchor);p=policies[t['policy']];last=weighted(ref,ref,p)
        for c in t['checkpoints']:
            m=replay(instance,matrix,c['route']);counts['checkpoint_routes']+=1
            counts['fallbacks']+=int(c['fallback'])
            assert m['travel_seconds']<=ref['travel_seconds']*(1+Decimal(str(p['max_travel_increase'])))
            if p['max_energy_increase'] is not None:assert m['energy_proxy']<=ref['energy_proxy']*(1+Decimal(str(p['max_energy_increase'])))
            score=weighted(m,ref,p);assert score<=last+1e-12;last=score
            assert abs(score-c['score'])<1e-10
            for k,v in m.items():assert abs(float(v)-c['metrics'][k])<1e-6
        assert t['end_to_end_seconds']>=0
        return ref
    dev=[];dev_records=[]
    for row in protocol['development']:
        r=read(root/'development'/(row['route_id']+'.json'));assert r['row']==row
        i,m=read(row['instance_path']),read(row['matrix_path']);anchor=r['anchor']['route']
        assert len(r['seed_candidates'])==2
        eligible=[]
        for candidate in r['seed_candidates']:
            counts['development_seed_calls']+=1
            if candidate['solution']['feasible']:
                metric=replay(i,m,candidate['solution']['route'])
                eligible.append((metric['travel_seconds'],candidate['solution']['route']))
        assert eligible and anchor==min(eligible,key=lambda x:x[0])[1]
        assert len(r['trials'])==24
        assert {(t['method'],t['policy'],t['seed']) for t in r['trials']}=={(method,p,seed) for method in protocol['methods'] for p in policies for seed in protocol['seeds']}
        for t in r['trials']:
            check_trial(i,m,anchor,t)
            assert [c['budget'] for c in t['checkpoints']]==[.25,1,4,8]
            for c in t['checkpoints']:dev.append(dict(**meta(row),method=t['method'],policy=t['policy'],seed=t['seed'],budget=c['budget'],score=c['score'],valid_ratio=c['valid']/max(1,c['attempts']),evaluations=c['evaluations']))
        dev_records.append(r)
    candidates=['random_sa','granular_sa','adaptive_lns']
    scores={k:mean(r['score'] for r in dev if r['method']==k and r['budget']==1) for k in candidates}
    selected=min(candidates[1:],key=lambda k:scores[k]);selection=read(root/'selection.json')
    assert selected==selection['selected']
    for k,v in scores.items():assert abs(v-selection['mean_score_at_one_second'][k])<1e-12
    confirmation=[];bound_records=[];previous_lookup={};infeasible=0
    old_call_times={}
    for row in protocol['rows']:
        r=read(root/'confirmation'/(row['route_id']+'.json'));assert r['row']==row
        i,m=read(row['instance_path']),read(row['matrix_path'])
        if r['status']=='capacity_infeasible':
            assert sum(Decimal(str(v['demand_cm3'])) for v in i['nodes'])>Decimal(str(i['vehicle_capacity_cm3']))
            infeasible+=1;continue
        ref=replay(i,m,r['anchor']);assert len(r['trials'])==16
        assert {(t['method'],t['policy'],t['seed']) for t in r['trials']}=={(method,p,seed) for method in ['random_sa',selected] for p in policies for seed in protocol['seeds']}
        lower=r['lower_bounds'];scale=lower['scale'];n=len(m['node_ids'])
        costs=[[int((Decimal(str(x))*scale).to_integral_value(rounding=ROUND_FLOOR)) for x in a] for a in m['travel_seconds']]
        cert=lower['assignment_certificate'];u,v=cert['row_dual'],cert['column_dual'];perm=cert['permutation']
        assert sorted(perm)==list(range(n)) and all(j!=k for k,j in enumerate(perm))
        assert all(u[a]+v[b]<=costs[a][b] for a in range(n) for b in range(n) if a!=b)
        assignment=sum(costs[a][perm[a]] for a in range(n));assert assignment==sum(u)+sum(v)==lower['assignment_ticks']
        assert Decimal(lower['travel_seconds'])==Decimal(assignment)/scale
        dt=lower['shortest_travel_ticks'];assert dt[0]==0
        assert all(dt[b]<=dt[a]+costs[a][b] for a in range(n) for b in range(n))
        nodes={x['id']:x for x in i['nodes']};ids=m['node_ids'];capacity=Fraction(str(i['vehicle_capacity_cm3']))
        carried=sum(Fraction(str(nodes[ids[j]]['demand_cm3']))*dt[j]/scale for j in range(1,n))
        exact_energy=Fraction(assignment,scale)+(carried/capacity if capacity else 0)
        assert Fraction(lower['energy_proxy'])<=exact_energy
        # Recompute the latency relaxation from original times with Decimal.
        from datetime import datetime
        departure=datetime.fromisoformat(i['departure_time_utc'].replace('Z','+00:00'))
        early=[];service=[]
        for key in ids:
            node=nodes[key];tw=node['time_window']
            start=(datetime.fromisoformat(tw['start_utc'].replace('Z','+00:00'))-departure).total_seconds() if tw else 0
            early.append(int((Decimal(str(start))*scale).to_integral_value(rounding=ROUND_FLOOR)))
            service.append(int((Decimal(str(node['service_seconds']))*scale).to_integral_value(rounding=ROUND_FLOOR)))
        dc=lower['shortest_completion_ticks'];assert dc[0]==0
        assert all(dc[b]<=max(dc[a]+costs[a][b],early[b])+service[b] for a in range(n) for b in range(n))
        processing=sorted(min(costs[a][b] for a in range(n) if a!=b)+service[b] for b in range(1,n))
        dl=max(sum(dc[1:]),sum((n-1-k)*x for k,x in enumerate(processing)))
        assert Decimal(lower['delivery_completion_sum_seconds'])==Decimal(dl)/scale
        for p in policies.values():
            lb=weighted(lower,ref,p);assert abs(lb-float(r['objective_lower_bounds'][p['name']]))<1e-10
            old=next(v for v in r['previous_policies'] if v['name']==p['name'])
            best=min([t['checkpoints'][-1]['score'] for t in r['trials'] if t['policy']==p['name']]+[weighted(old['metrics'],ref,p)])
            assert lb<=best+1e-10
            bound_records.append(dict(**meta(row),policy=p['name'],lower_bound=lb,best=best,remaining_improvement_upper_fraction=(best-lb)/best))
        previous_lookup[row['route_id']]=r
        archived_paths=[path for path in protocol['frozen_hashes'] if path.endswith('/benchmark50/routes/'+row['route_id']+'.json')]
        assert len(archived_paths)==1
        archived=read(archived_paths[0])
        old_seed_calls=sum(v['end_to_end_seconds'] for v in archived['seed_candidates'].values())
        old_call_times[row['route_id']]={e['name']:old_seed_calls+e['policy_seconds'] for e in archived['policies']}
        for t in r['trials']:
            check_trial(i,m,r['anchor'],t);assert [c['budget'] for c in t['checkpoints']]==[.25,1]
            old=next(x for x in r['previous_policies'] if x['name']==t['policy'])
            for c in t['checkpoints']:
                p=policies[t['policy']]
                confirmation.append(dict(**meta(row),method=t['method'],policy=t['policy'],seed=t['seed'],budget=c['budget'],score=c['score'],
                    gain_vs_python=c['score']/weighted(old['metrics'],ref,p)-1,
                    travel_change=c['metrics']['travel_seconds']/float(ref['travel_seconds'])-1,
                    completion_change=c['metrics']['delivery_completion_sum_seconds']/float(ref['delivery_completion_sum_seconds'])-1,
                    energy_change=c['metrics']['energy_proxy']/float(ref['energy_proxy'])-1,
                    valid_ratio=c['valid']/max(1,c['attempts']),wrapper_seconds=t['end_to_end_seconds']))
    assert infeasible==1 and len(previous_lookup)==49
    solvers=[];pipelines=[];mips=[]
    bound_map={(r['route_id'],r['policy']):r for r in bound_records}
    def improve_upper_bound(row,metrics,source):
        reference=previous_lookup[row['route_id']]['reference']
        for name,p in policies.items():
            if metrics['travel_seconds']>Decimal(reference['travel_seconds'])*(1+Decimal(str(p['max_travel_increase']))):continue
            if p['max_energy_increase'] is not None and metrics['energy_proxy']>Decimal(reference['energy_proxy'])*(1+Decimal(str(p['max_energy_increase']))):continue
            score=weighted(metrics,reference,p);b=bound_map[row['route_id'],name]
            if score<b['best']:b['best']=score;b['best_source']=source
    tasks=[(r,42) for r in protocol['rows']]+[(r,271828) for r in protocol['repeated_seed_rows']]
    for row,seed in tasks:
        r=read(root/'seeds'/(row['route_id']+f'_{seed}.json'));assert r['row']==row and r['seed']==seed and len(r['solvers'])==5
        assert {x['config']['name'] for x in r['solvers']}=={x['name'] for x in protocol['seed_configs']}
        i,m=read(row['instance_path']),read(row['matrix_path'])
        for record in r['solvers']:
            assert record['config']==next(x for x in protocol['seed_configs'] if x['name']==record['config']['name'])
            assert record['seed']==seed
            s=record['solution'];counts['fresh_solver_calls']+=1
            entry=dict(**meta(row),name=record['config']['name'],seed=seed,feasible=s['feasible'],seconds=record['end_to_end_seconds'])
            if s['feasible']:
                metrics=replay(i,m,s['route']);assert abs(float(metrics['travel_seconds'])-record['independent_metrics']['travel_seconds'])<1e-7
                improve_upper_bound(row,metrics,record['config']['name'])
                entry['travel']=float(metrics['travel_seconds'])
            solvers.append(entry)
        lkh=next(x for x in r['solvers'] if x['config']['name']=='ils_lkh')
        expected=4 if seed==42 and lkh['solution']['feasible'] else 0;assert len(r['pipeline'])==expected
        for t in r['pipeline']:
            check_trial(i,m,lkh['solution']['route'],t)
            assert abs(t['single_mode_pipeline_seconds']-t['end_to_end_seconds']-lkh['end_to_end_seconds'])<1e-8
            c=t['checkpoints'][-1];oldrow=previous_lookup[row['route_id']];old=next(v for v in oldrow['previous_policies'] if v['name']==t['policy'])
            p=policies[t['policy']];oldref=oldrow['reference']
            exact=replay(i,m,c['route'])
            improve_upper_bound(row,exact,'fresh_single_seed_pipeline')
            old_t_ok=exact['travel_seconds']<=Decimal(oldref['travel_seconds'])*(1+Decimal(str(p['max_travel_increase'])))
            old_e_ok=p['max_energy_increase'] is None or exact['energy_proxy']<=Decimal(oldref['energy_proxy'])*(1+Decimal(str(p['max_energy_increase'])))
            old_comparable=old_call_times[row['route_id']][t['policy']]
            pipelines.append(dict(**meta(row),policy=t['policy'],seconds=t['single_mode_pipeline_seconds'],old_call_seconds=old_comparable,old_recorded_pipeline_seconds=old['single_mode_pipeline_seconds'],runtime_ratio=t['single_mode_pipeline_seconds']/old_comparable,
                travel_change=c['metrics']['travel_seconds']/old['metrics']['travel_seconds']-1,
                completion_change=c['metrics']['delivery_completion_sum_seconds']/old['metrics']['delivery_completion_sum_seconds']-1,
                energy_change=c['metrics']['energy_proxy']/old['metrics']['energy_proxy']-1,
                old_anchor_travel_guard_ok=old_t_ok,old_anchor_energy_guard_ok=old_e_ok))
    assert counts['fresh_solver_calls']==300
    for row in protocol['mip_rows']:
        r=read(root/'mip'/(row['route_id']+'.json'));assert set(r['results'])==set(policies)
        i,m=read(row['instance_path']),read(row['matrix_path']);original=previous_lookup[row['route_id']];ref=replay(i,m,original['anchor'])
        for name,result in r['results'].items():
            bound=result.get('numerical_lower_bound');known=1.
            if result.get('incumbent_audited'):
                metrics=replay(i,m,result['route']);known=weighted(metrics,ref,policies[name]);assert abs(known-result['independent_score'])<1e-8
                p=policies[name]
                assert metrics['travel_seconds']<=ref['travel_seconds']*(1+Decimal(str(p['max_travel_increase'])))
                if p['max_energy_increase'] is not None:assert metrics['energy_proxy']<=ref['energy_proxy']*(1+Decimal(str(p['max_energy_increase'])))
                improve_upper_bound(row,metrics,'scip_audited_incumbent')
            if bound is not None:assert bound<=known+1e-5
            mips.append(dict(**meta(row),policy=name,status=result['status'],bound=bound,audited_incumbent=result.get('incumbent_audited',False),seconds=result.get('total_seconds')))
    for b in bound_records:
        assert b['lower_bound']<=b['best']+1e-10
        b['remaining_improvement_upper_fraction']=(b['best']-b['lower_bound'])/b['best']
    for m in mips:
        if m['bound'] is not None:assert m['bound']<=bound_map[m['route_id'],m['policy']]['best']+1e-5
    summaries={}
    for name in policies:
        summaries[name]={}
        for method in ['random_sa',selected]:
            values=[r for r in confirmation if r['policy']==name and r['method']==method and r['budget']==1]
            summaries[name][method]={k:effect(values,k) for k in ['gain_vs_python','travel_change','completion_change','energy_change']}
            summaries[name][method].update(median_valid_ratio=median(r['valid_ratio'] for r in values),median_wrapper_seconds=median(r['wrapper_seconds'] for r in values))
    seed_summaries={}
    repeated_ids={r['route_id'] for r in protocol['repeated_seed_rows']}
    for config in protocol['seed_configs']:
        values=[r for r in solvers if r['name']==config['name'] and r['seed']==42];feasible=[r for r in values if r['feasible']]
        basemap={r['route_id']:r for r in solvers if r['name']=='ils_lkh' and r['seed']==42}
        paired=[dict(**r,travel_change=r['travel']/basemap[r['route_id']]['travel']-1) for r in feasible if basemap[r['route_id']]['feasible']]
        seed_summaries[config['name']]=dict(feasible=len(feasible),total=len(values),median_seconds=median(r['seconds'] for r in feasible) if feasible else None,
            median_travel=median(r['travel'] for r in feasible) if feasible else None,vs_ils_lkh=effect(paired,'travel_change') if paired else None)
        repeat=[r for r in solvers if r['name']==config['name'] and r['route_id'] in repeated_ids]
        base={(r['route_id'],r['seed']):r for r in solvers if r['name']=='ils_lkh'}
        pair=[dict(**r,travel_change=r['travel']/base[r['route_id'],r['seed']]['travel']-1) for r in repeat if r['feasible'] and base[r['route_id'],r['seed']]['feasible']]
        seed_summaries[config['name']]['repeated_subset']=dict(feasible=sum(r['feasible'] for r in repeat),total=len(repeat),vs_ils_lkh=effect(pair,'travel_change') if pair else None)
    pipeline_summary={}
    for name in policies:
        values=[r for r in pipelines if r['policy']==name]
        pipeline_summary[name]=dict(feasible=len(values),total=50,median_seconds=median(r['seconds'] for r in values),median_old_call_seconds=median(r['old_call_seconds'] for r in values),median_runtime_ratio=median(r['runtime_ratio'] for r in values),
            old_anchor_travel_violations=sum(not r['old_anchor_travel_guard_ok'] for r in values),old_anchor_energy_violations=sum(not r['old_anchor_energy_guard_ok'] for r in values),
            changes={k:effect(values,k) for k in ['travel_change','completion_change','energy_change']})
    saturation={}
    for method in candidates:
        saturation[method]={}
        for name in policies:
            values=[r for r in dev if r['method']==method and r['policy']==name]
            saturation[method][name]={str(b):mean(r['score'] for r in values if r['budget']==b) for b in [.25,1,4,8]}
    direct={}
    random_map={(r['route_id'],r['seed'],r['policy']):r for r in confirmation if r['method']=='random_sa' and r['budget']==1}
    for name in policies:
        values=[dict(**r,score_change_vs_random=r['score']/random_map[r['route_id'],r['seed'],name]['score']-1) for r in confirmation if r['method']==selected and r['policy']==name and r['budget']==1]
        direct[name]=effect(values,'score_change_vs_random')
    result=dict(passed=True,source_commit=protocol['source_commit'],selected_method=selected,counts=dict(counts),capacity_infeasible=1,
        development_scores=scores,development_saturation=saturation,confirmation=summaries,selected_vs_native_random=direct,seed_comparison=seed_summaries,pipeline=pipeline_summary,
        certified_gap_summary={name:dict(median=median(r['remaining_improvement_upper_fraction'] for r in bound_records if r['policy']==name),
            min=min(r['remaining_improvement_upper_fraction'] for r in bound_records if r['policy']==name),max=max(r['remaining_improvement_upper_fraction'] for r in bound_records if r['policy']==name)) for name in policies},
        numerical_mip_bounds=mips,scope='No global optimality or universal SOTA claim; same-anchor searches and changed-anchor fresh pipelines are separate.')
    write(root/'independent_audit.json',result)
    write(root/'derived_records.json',dict(development=dev,confirmation=confirmation,solvers=solvers,pipeline=pipelines,bounds=bound_records))
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);audit(p.parse_args().root)
