from cargoflow.policy_search import Policy, RouteEvaluator, policy_from_logits, search_policy
from test_solver import fixture
import pytest


def test_load_aware_proxy_rewards_delivering_large_stop_earlier():
    instance, matrix = fixture()
    instance["nodes"][1]["demand_cm3"] = 1
    instance["nodes"][2]["demand_cm3"] = 5
    instance["nodes"][3]["demand_cm3"] = 1
    instance["vehicle_capacity_cm3"] = 10
    matrix["travel_seconds"] = [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]]
    evaluator = RouteEvaluator(instance, matrix)
    early = evaluator.evaluate(["D", "B", "A", "C", "D"])
    late = evaluator.evaluate(["D", "A", "C", "B", "D"])
    assert early["travel_seconds"] == late["travel_seconds"]
    assert early["energy_proxy"] < late["energy_proxy"]


def test_guarded_search_returns_officially_feasible_tour():
    instance, matrix = fixture()
    route = ["D", "A", "B", "C", "D"]
    result = search_policy(instance, matrix, route, Policy("test", 0.05, 0.5, 0.5), iterations=50)
    assert result["official_audit"]["feasible"]
    assert result["metrics"]["travel_seconds"] <= result["reference"]["travel_seconds"] * 1.05


def test_masked_softmax_turns_disabled_objectives_off():
    policy = policy_from_logits("test", 0.02, (0.0, 1.0, 3.0, 9.0), (True, True, False, False))
    assert policy.energy_weight == 0
    assert policy.preference_weight == 0
    assert abs(policy.time_weight + policy.completion_weight - 1) < 1e-12
    assert policy.completion_weight > policy.time_weight


@pytest.mark.parametrize('kind', ['duplicate', 'unknown', 'capacity', 'departure'])
def test_invalid_seed_is_rejected_before_search(kind):
    instance, matrix = fixture()
    route = ['D', 'A', 'B', 'C', 'D']
    if kind == 'duplicate': route = ['D', 'A', 'A', 'C', 'D']
    elif kind == 'unknown': route = ['D', 'A', 'X', 'C', 'D']
    elif kind == 'capacity': instance['vehicle_capacity_cm3'] = 1
    else:
        instance['nodes'][0]['time_window'] = {'start_utc': '2018-07-27T16:00:01Z', 'end_utc': '2018-07-27T17:00:00Z'}
    assert not RouteEvaluator(instance, matrix).evaluate(route)['feasible']
    with pytest.raises(ValueError, match='seed route'):
        search_policy(instance, matrix, route, Policy('test', 0, 1, 0), iterations=2)


def test_one_customer_needs_no_random_two_customer_move():
    instance, matrix = fixture()
    instance['nodes'] = instance['nodes'][:2]
    matrix['node_ids'] = matrix['node_ids'][:2]
    matrix['travel_seconds'] = [r[:2] for r in matrix['travel_seconds'][:2]]
    result = search_policy(instance, matrix, ['D', 'A', 'D'], Policy('test', 0, 1, 0), iterations=2)
    assert result['official_audit']['feasible']
    assert result['route'] == ['D', 'A', 'D']


def test_zero_cost_instance_does_not_divide_by_zero():
    instance, matrix = fixture()
    for n in instance['nodes']: n.update(demand_cm3=0, service_seconds=0, time_window=None)
    instance['vehicle_capacity_cm3'] = 0
    matrix['travel_seconds'] = [[0]*4 for _ in range(4)]
    result = search_policy(instance, matrix, ['D', 'A', 'B', 'C', 'D'], Policy('zero', 0, 1, 1), iterations=2)
    assert result['official_audit']['feasible']
    assert result['metrics']['travel_seconds'] == 0


def test_completion_means_service_finished():
    from cargoflow.solver import audit_route
    instance, matrix = fixture();route = ['D', 'A', 'B', 'C', 'D']
    expected = sum(s['departure_seconds'] for s in audit_route(instance, matrix, route)['schedule'][1:-1])
    assert RouteEvaluator(instance, matrix).evaluate(route)['delivery_completion_sum_seconds'] == expected


@pytest.mark.parametrize('cost', [None, [[0]], [[float('nan')]*4 for _ in range(4)], [[-1]*4 for _ in range(4)]])
def test_active_preference_requires_valid_cost_matrix(cost):
    instance, matrix = fixture()
    with pytest.raises(ValueError, match='preference'):
        search_policy(instance, matrix, ['D', 'A', 'B', 'C', 'D'], Policy('p', 0, 0, 0, preference_weight=1), preference_cost=cost, iterations=2)


def test_disabled_preference_does_not_read_supplied_matrix():
    instance, matrix = fixture();route = ['D', 'A', 'B', 'C', 'D'];p=Policy('off', 0, 1, 0)
    a=search_policy(instance,matrix,route,p,iterations=10)
    b=search_policy(instance,matrix,route,p,preference_cost=[[float('nan')]],iterations=10)
    assert a == b


def test_zero_reference_preference_still_penalizes_positive_candidate():
    from cargoflow.policy_search import _score
    ref=dict(travel_seconds=0,energy_proxy=0,delivery_completion_sum_seconds=0,preference_cost=0)
    p=Policy('p',0,0,0,preference_weight=1)
    assert _score({**ref,'preference_cost':2},ref,p) > _score(ref,ref,p)


def test_softmax_is_stable_for_large_equal_logits_and_small_temperature():
    p=policy_from_logits('stable',0,[1e308,1e308,0,0],[True,True,False,False],temperature=1e-308)
    assert p.time_weight == p.completion_weight == .5


def test_exact_final_guard_falls_back_if_fast_metrics_underestimate(monkeypatch):
    import cargoflow.policy_search as module
    instance,matrix=fixture();route=['D','A','B','C','D'];bad=['D','B','A','C','D']
    matrix['travel_seconds']=[[0,1,2,2],[2,0,1,2],[2,2,0,1],[1,2,2,0]]
    for node in instance['nodes']:node['time_window']=None
    evaluate=module.RouteEvaluator.evaluate
    def underestimate(self,path,preference_cost=None):
        result=evaluate(self,path,preference_cost)
        if list(path)==bad:result['travel_seconds']=1
        return result
    monkeypatch.setattr(module.RouteEvaluator,'evaluate',underestimate)
    monkeypatch.setattr(module,'_move',lambda current,rng:bad)
    result=search_policy(instance,matrix,route,Policy('guard',0,1,0),iterations=2)
    assert result['route']==route
    assert result['official_audit']['feasible']
    assert result['search']['fallback_reason']
