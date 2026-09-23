"""Native policy search with fixed-reference independent replay."""
import ctypes
import math
from decimal import Decimal
from pathlib import Path
import sys
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cargoflow.policy_search import RouteEvaluator, _exact_metrics, _score
from cargoflow.solver import audit_route
from audit_policy_benchmark import replay

LIB=None
def search(instance,matrix,reference_route,policy,method,seed,checkpoints,initial=None):
    global LIB
    started=perf_counter()
    if LIB is None:
        LIB=ctypes.CDLL(str(ROOT/'artifacts/runtime_frontier/policy_frontier.so'))
        doubles=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        LIB.policy_frontier.argtypes=[ctypes.c_int,*([doubles]*5),ctypes.c_double,integers,doubles,
            ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_uint64,ctypes.c_int,doubles,integers,doubles]
        LIB.policy_frontier.restype=ctypes.c_int
    assert policy.preference_weight==0 and method in ['random_sa','granular_sa','adaptive_lns']
    assert checkpoints and all(0<x<1000 for x in checkpoints) and sorted(set(checkpoints))==list(checkpoints)
    reference=replay(instance,matrix,reference_route)
    evaluator=RouteEvaluator(instance,matrix)
    ids=matrix['node_ids'];assert ids[0]==instance['depot_id'];n=len(ids)
    nodes={v['id']:v for v in instance['nodes']}
    early=[];late=[]
    for k in ids:
        w=evaluator.windows[k];early.append(w[0] if w else 0);late.append(w[1] if w else 1e100)
    arrays=[np.ascontiguousarray(v,dtype=np.float64) for v in [matrix['travel_seconds'],
        [nodes[k]['service_seconds'] for k in ids],[nodes[k]['demand_cm3'] for k in ids],early,late]]
    weights=[policy.time_weight,policy.completion_weight,policy.energy_weight]
    keys=['travel_seconds','delivery_completion_sum_seconds','energy_proxy']
    coef=np.array([w/(float(reference[k]) or 1)/sum(weights) for w,k in zip(weights,keys)],dtype=np.float64)
    tl=reference['travel_seconds']*(1+Decimal(str(policy.max_travel_increase)))
    el=None if policy.max_energy_increase is None else reference['energy_proxy']*(1+Decimal(str(policy.max_energy_increase)))
    initial=reference_route if initial is None else initial
    im=replay(instance,matrix,initial)
    eligible=im['travel_seconds']<=tl and (el is None or im['energy_proxy']<=el) and _score(im,reference,policy)<=1+1e-12
    rejected_initial=not eligible
    if not eligible:initial=reference_route
    r=np.array([evaluator.index[k] for k in initial],dtype=np.int32)
    output=np.zeros((len(checkpoints),n+1),dtype=np.int32);stats=np.zeros((len(checkpoints),7),dtype=np.float64)
    code=LIB.policy_frontier(n,*arrays,float(instance['vehicle_capacity_cm3']),r,coef,float(tl),float(el) if el is not None else 1e100,
        ['random_sa','granular_sa','adaptive_lns'].index(method),seed,len(checkpoints),np.array(checkpoints,dtype=np.float64),output,stats)
    assert code==0
    records=[];best=reference_route;bestm=reference
    for budget,indices,stat in zip(checkpoints,output,stats):
        route=[ids[i] for i in indices];fallback=False
        try:
            metrics=replay(instance,matrix,route)
            assert metrics['travel_seconds']<=tl and (el is None or metrics['energy_proxy']<=el)
            assert _score(metrics,reference,policy)<=_score(bestm,reference,policy)+1e-12
            best,bestm=route,metrics
        except AssertionError:fallback=True
        records.append(dict(budget=budget,route=best,metrics={k:float(v) for k,v in bestm.items()},score=_score(bestm,reference,policy),
            fallback=fallback,search_seconds=stat[0],attempts=int(stat[1]),valid=int(stat[2]),accepted=int(stat[3]),evaluations=int(stat[4])))
    return dict(method=method,seed=seed,policy=policy.name,reference={k:float(v) for k,v in reference.items()},
        rejected_initial=rejected_initial,checkpoints=records,end_to_end_seconds=perf_counter()-started)


def ensure_runtime():
    """Build locally on first use or when the C++ source is newer."""
    global LIB
    import os
    import shlex
    import subprocess
    import tempfile
    source = ROOT / 'native/policy_frontier.cpp'
    binary = ROOT / 'artifacts/runtime_frontier/policy_frontier.so'
    if binary.exists() and binary.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    if not compiler:
        raise ValueError('CXX must specify a C++ compiler')
    binary.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=binary.parent) as folder:
        output = Path(folder) / binary.name
        try:
            subprocess.run([*compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(output)], check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError('Native search needs a C++17 compiler; set CXX to its executable.') from exc
        output.replace(binary)
    LIB = None


def search_policy(instance, matrix, seed_route, policy, *, seconds=1.0, seed=42):
    """The supplied seed is BOTH the search start and protected reference."""
    started = perf_counter()
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds < 1000:
        raise ValueError('seconds must be finite and between 0 and 1000 (exclusive)')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError('seed must be an unsigned 64-bit integer')
    if policy.preference_weight:
        raise ValueError('Native policy search does not yet support personal preference costs')
    evaluator = RouteEvaluator(instance, matrix)
    seed_route = list(seed_route)
    seed_audit = audit_route(instance, matrix, seed_route)
    if not seed_audit['feasible']:
        raise ValueError('seed route must be feasible')
    reference = _exact_metrics(evaluator, seed_route, None)
    ensure_runtime()
    trial = search(instance, matrix, seed_route, policy, 'adaptive_lns', seed, [seconds])
    point = trial['checkpoints'][-1]
    route = point['route']
    official = audit_route(instance, matrix, route)
    fallback = 'native checkpoint rejected; protected reference retained' if point['fallback'] else None
    if not official['feasible']:
        fallback = 'final feasibility replay rejected candidate'
    else:
        metrics = _exact_metrics(evaluator, route, None)
        if metrics['travel_seconds'] > reference['travel_seconds'] * (1 + Decimal(str(policy.max_travel_increase))):
            fallback = 'exact travel epsilon exceeded'
        elif policy.max_energy_increase is not None and metrics['energy_proxy'] > reference['energy_proxy'] * (1 + Decimal(str(policy.max_energy_increase))):
            fallback = 'exact energy epsilon exceeded'
        elif _score(metrics, reference, policy) > _score(reference, reference, policy) + 1e-12:
            fallback = 'exact weighted objective worse than seed'
    if fallback:
        route, official, metrics = seed_route, seed_audit, reference
    return {
        'policy': {
            'name': policy.name, 'max_travel_increase': policy.max_travel_increase,
            'max_energy_increase': policy.max_energy_increase,
            'weights': dict(time=policy.time_weight, energy_proxy=policy.energy_weight,
                           preference=policy.preference_weight, delivery_completion=policy.completion_weight),
            'logits_T_D_E_P': policy.raw_logits, 'switches_T_D_E_P': policy.switches,
            'softmax_temperature': policy.temperature,
        },
        'route': route, 'reference_route': seed_route,
        'reference': {'feasible': True, **{k: float(v) for k, v in reference.items()}},
        'metrics': {'feasible': True, **{k: float(v) for k, v in metrics.items()}},
        'official_audit': official,
        'search': dict(method='adaptive_lns', requested_seconds=seconds,
                       search_seconds=point['search_seconds'], iterations=point['attempts'],
                       feasible_candidates=point['valid'], accepted_moves=point['accepted'],
                       objective_evaluations=point['evaluations'], seed=seed, fallback_reason=fallback),
        'policy_seconds': perf_counter() - started,
        'limitations': ['energy_proxy is not measured fuel, kWh, or CO2',
                        'personal preference costs are not enabled',
                        'the supplied seed defines the protection limits'],
    }
