import itertools
from decimal import Decimal
from pathlib import Path
import random
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from test_solver import fixture
from frontier_bounds import assignment,bounds,objective_bound
from audit_policy_benchmark import replay
from run_policy_search import POLICIES
native_available=pytest.mark.skipif(not (Path(__file__).resolve().parents[1]/'artifacts/runtime_frontier/policy_frontier.so').exists(),reason='optional frontier native experiment runtime')

def test_integer_assignment_certificate_matches_exhaustive():
    rng=random.Random(71)
    for n in range(2,7):
        matrix=[[rng.randrange(10000) for _ in range(n)] for _ in range(n)]
        cost,cert=assignment(matrix)
        assert cost==min(sum(matrix[i][p[i]] for i in range(n)) for p in itertools.permutations(range(n)))

@pytest.mark.parametrize('seed',range(8))
def test_bounds_below_every_feasible_permutation_with_nonmetric_arcs(seed):
    instance,matrix=fixture();rng=random.Random(seed)
    matrix['travel_seconds']=[[0 if i==j else rng.randrange(1,100)/7 for j in range(4)] for i in range(4)]
    instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:20Z','end_utc':'2018-07-27T20:00:00Z'}
    lower=bounds(instance,matrix)
    for order in itertools.permutations(['A','B','C']):
        metrics=replay(instance,matrix,['D',*order,'D'])
        for key in ['travel_seconds','energy_proxy','delivery_completion_sum_seconds']:
            assert Decimal(lower[key])<=metrics[key],(seed,key)
        for policy in POLICIES:assert objective_bound(lower,metrics,policy)<=1

@pytest.mark.parametrize('method',['random_sa','granular_sa','adaptive_lns'])
@pytest.mark.parametrize('policy',POLICIES,ids=lambda p:p.name)
@native_available
def test_native_policy_matches_exhaustive_oracle_and_exact_guards(method,policy):
    from frontier_search import search
    instance,matrix=fixture()
    matrix['travel_seconds']=[[0,9,3,8],[2,0,4,1],[4,7,0,6],[3,5,2,0]]
    reference_route=['D','A','B','C','D'];reference=replay(instance,matrix,reference_route)
    scores=[]
    from cargoflow.policy_search import _score
    for order in itertools.permutations(['A','B','C']):
        metrics=replay(instance,matrix,['D',*order,'D'])
        if metrics['travel_seconds']>reference['travel_seconds']*(1+Decimal(str(policy.max_travel_increase))):continue
        if policy.max_energy_increase is not None and metrics['energy_proxy']>reference['energy_proxy']:continue
        scores.append(_score(metrics,reference,policy))
    result=search(instance,matrix,reference_route,policy,method,42,[.01,.05])
    assert result['checkpoints'][-1]['score']==pytest.approx(min(scores))
    assert result['checkpoints'][1]['score']<=result['checkpoints'][0]['score']

@native_available
def test_native_rejects_changed_initial_outside_fixed_reference_budget():
    from frontier_search import search
    instance,matrix=fixture()
    result=search(instance,matrix,['D','A','B','C','D'],POLICIES[0],'adaptive_lns',42,[.01],initial=['D','C','B','A','D'])
    assert result['rejected_initial']

@native_available
def test_native_zero_capacity_zero_demand_single_customer():
    from frontier_search import search
    instance,matrix=fixture();instance['nodes']=instance['nodes'][:2]
    for node in instance['nodes']:node['demand_cm3']=0
    instance['vehicle_capacity_cm3']=0
    matrix['node_ids']=matrix['node_ids'][:2];matrix['travel_seconds']=[[0,0],[0,0]]
    result=search(instance,matrix,['D','A','D'],POLICIES[3],'adaptive_lns',42,[.01])
    assert result['checkpoints'][0]['metrics']['energy_proxy']==0


@native_available
def test_default_cli_uses_new_search_and_preserves_result_format(tmp_path):
    import json
    import subprocess
    instance, matrix = fixture()
    for name, value in [('instance', instance), ('matrix', matrix),
                        ('seed', {'route': ['D', 'A', 'B', 'C', 'D']})]:
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    output = tmp_path / 'results'
    subprocess.run([sys.executable, str(Path(__file__).resolve().parents[1] / 'scripts/run_policy_search.py'),
                    '--instance', str(tmp_path/'instance.json'), '--matrix', str(tmp_path/'matrix.json'),
                    '--seed-solution', str(tmp_path/'seed.json'), '--output-dir', str(output)],
                   check=True, capture_output=True, text=True)
    for policy in POLICIES:
        result = json.loads((output / (policy.name+'.json')).read_text())
        assert result['search']['method'] == 'adaptive_lns'
        assert result['search']['requested_seconds'] == 1
        assert result['official_audit']['feasible']
        assert result['reference_route'] == ['D', 'A', 'B', 'C', 'D']
        assert result['metrics']['travel_seconds'] <= result['reference']['travel_seconds'] * (1+policy.max_travel_increase)
    assert (output/'comparison.csv').is_file()


@pytest.mark.parametrize('failure', ['coverage', 'travel', 'energy'])
@native_available
def test_default_final_guards_keep_reference_on_bad_candidate(monkeypatch, failure):
    import frontier_search
    instance, matrix = fixture()
    reference = ['D', 'A', 'B', 'C', 'D']
    candidate = ['D', 'C', 'B', 'A', 'D']
    policy = POLICIES[0]
    if failure == 'coverage':
        candidate = ['D', 'A', 'D']
    if failure == 'energy':
        policy = POLICIES[3]
        matrix['travel_seconds'] = [[0 if i == j else 1 for j in range(4)] for i in range(4)]
        for node, demand in zip(instance['nodes'], [0, 2, 1, .25]):
            node['demand_cm3'] = demand
    point = dict(route=candidate, fallback=False, search_seconds=.01,
                 attempts=1, valid=1, accepted=1, evaluations=1)
    monkeypatch.setattr(frontier_search, 'search', lambda *a, **k: {'checkpoints': [point]})
    result = frontier_search.search_policy(instance, matrix, reference, policy, seconds=.01)
    assert result['route'] == reference
    assert result['official_audit']['feasible']
    assert result['search']['fallback_reason']


@pytest.mark.parametrize('seconds', [0, -1, float('nan'), float('inf'), True])
def test_default_rejects_invalid_search_budget(seconds):
    from frontier_search import search_policy
    instance, matrix = fixture()
    with pytest.raises(ValueError, match='seconds'):
        search_policy(instance, matrix, ['D', 'A', 'B', 'C', 'D'], POLICIES[0], seconds=seconds)
