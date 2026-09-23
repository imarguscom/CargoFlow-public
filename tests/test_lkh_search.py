import itertools
import pytest
from importlib.metadata import version
from pathlib import Path

if version('pyvrp') != '0.13.4' or not (Path(__file__).resolve().parents[1]/'artifacts/runtime_lkh/LKH-3.0.13/LKH').is_file():
    pytest.skip('LKH tests require the isolated PyVRP 0.13.4 and built LKH runtime', allow_module_level=True)

from test_solver import fixture
from cargoflow.lkh_search import solve_lkh
from cargoflow.solver import audit_route


def test_lkh_directed_service_start_and_constant_service_cost():
    instance,matrix=fixture()
    instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:10Z','end_utc':'2018-07-27T16:00:10Z'}
    config=dict(name='test',backend='lkh-3.0.13',engine='lkh',seconds=.4)
    result=solve_lkh(instance,matrix,config)['solution']
    assert result['feasible'] and result['objective_scaled']==4500
    assert result['audit']['schedule'][1]['service_start_seconds']==10


@pytest.mark.parametrize('special',[False,True])
def test_lkh_nonmetric_windows_match_exhaustive_oracle(special):
    instance,matrix=fixture()
    matrix['travel_seconds']=[[0,20,20,1],[1,0,1,20],[20,1,0,20],[20,20,1,0]]
    instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:08Z','end_utc':'2018-07-27T16:00:10Z'}
    instance['nodes'][2]['time_window']={'start_utc':'2018-07-27T16:00:00Z','end_utc':'2018-07-27T16:00:05Z'}
    feasible=[]
    for perm in itertools.permutations(['A','B','C']):
        audit=audit_route(instance,matrix,['D',*perm,'D'])
        if audit['feasible']:feasible.append(audit['metrics']['travel_seconds'])
    assert feasible
    result=solve_lkh(instance,matrix,dict(name='test',backend='lkh-3.0.13',engine='lkh',seconds=.4,special=special))['solution']
    assert result['feasible'] and result['audit']['metrics']['travel_seconds']==min(feasible)


def test_lkh_cannot_hide_capacity_failure():
    instance,matrix=fixture();instance['vehicle_capacity_cm3']=3
    result=solve_lkh(instance,matrix,dict(name='test',backend='lkh-3.0.13',engine='lkh',seconds=.1))['solution']
    assert result['status']=='input_infeasible' and not result['feasible']


@pytest.mark.parametrize('engine',['lkh','ils_lkh'])
def test_one_customer_is_solved_exactly_without_native_minimum_size_failure(engine):
    instance,matrix=fixture()
    instance['nodes']=instance['nodes'][:2]
    matrix['node_ids']=matrix['node_ids'][:2]
    matrix['travel_seconds']=[row[:2] for row in matrix['travel_seconds'][:2]]
    result=solve_lkh(instance,matrix,dict(name='test',backend='lkh-3.0.13',engine=engine,seconds=.1))['solution']
    assert result['feasible'] and result['route']==['D','A','D']
    assert result['selected_phase']=='exact_small'
