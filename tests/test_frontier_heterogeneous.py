"""Post-freeze adversarial checks, first executed locally during the study."""
from copy import deepcopy
from decimal import Decimal
import itertools
from pathlib import Path
import random
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_solver import fixture
from audit_policy_benchmark import replay
from frontier_bounds import bounds
from frontier_search import search
from run_policy_search import POLICIES

@pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/'artifacts/runtime_frontier/policy_frontier.so').exists(),reason='optional frontier native runtime')
@pytest.mark.parametrize('seed',range(10))
def test_heterogeneous_loads_nonmetric_zero_arcs_and_windows(seed):
    rng=random.Random(500+seed);i,m=fixture()
    for name in ['E','F']:
        node=deepcopy(i['nodes'][1]);node['id']=name;i['nodes'].append(node)
    m['node_ids']=[v['id'] for v in i['nodes']]
    for node in i['nodes'][1:]:
        node.update(demand_cm3=rng.randrange(0,20)/8,service_seconds=rng.randrange(0,20)/4)
    i['vehicle_capacity_cm3']=sum(v['demand_cm3'] for v in i['nodes'])+1
    i['nodes'][2]['time_window']={'start_utc':'2018-07-27T16:00:05Z','end_utc':'2018-07-27T16:00:35Z'}
    m['travel_seconds']=[[0 if a==b else rng.randrange(0,50)/8 for b in range(6)] for a in range(6)]
    feasible=[]
    for middle in itertools.permutations(m['node_ids'][1:]):
        route=['D',*middle,'D']
        try:metrics=replay(i,m,route)
        except AssertionError:continue
        feasible.append((route,metrics))
    assert feasible
    lower=bounds(i,m)
    for route,metrics in feasible:
        for key in ['travel_seconds','energy_proxy','delivery_completion_sum_seconds']:
            assert Decimal(lower[key])<=metrics[key]
    anchor=max(feasible,key=lambda x:x[1]['travel_seconds'])[0]
    for policy in POLICIES:
        for method in ['random_sa','granular_sa','adaptive_lns']:
            trial=search(i,m,anchor,policy,method,seed,[.005,.01])
            assert trial['checkpoints'][-1]['score']<=1+1e-12
