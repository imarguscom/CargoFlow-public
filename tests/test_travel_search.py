from types import SimpleNamespace
from importlib.metadata import version
import numpy as np
import pytest
from test_solver import fixture
from cargoflow.solver import solve_instance
from cargoflow.travel_search import prepare_problem, solve_travel


def test_native_problem_matches_original_builder(monkeypatch):
    from pyvrp import Model
    instance,matrix=fixture()
    instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:10Z','end_utc':'2018-07-27T16:01:00Z'}
    matrix['travel_seconds'][0][1]=1.12501
    captured=[]
    def capture(self,**kwargs):
        captured.append(self.data())
        return SimpleNamespace(num_iterations=0,is_feasible=lambda:False,
                               best=SimpleNamespace(routes=lambda:[]))
    monkeypatch.setattr(Model,'solve',capture)
    solve_instance(instance,matrix,iterations=1)
    native,status=prepare_problem(instance,matrix)
    assert status is None
    old=captured[0]
    np.testing.assert_array_equal(native.distance_matrix(0),old.distance_matrix(0))
    np.testing.assert_array_equal(native.duration_matrix(0),old.duration_matrix(0))
    for i in range(native.num_locations):
        for key in ['x','y','tw_early','tw_late','name']:
            assert getattr(native.location(i),key)==getattr(old.location(i),key)
        if i:
            for key in ['delivery','pickup','service_duration','release_time','required']:
                assert getattr(native.location(i),key)==getattr(old.location(i),key)
    for key in ['capacity','tw_early','tw_late','start_late','start_depot','end_depot','unit_distance_cost','unit_duration_cost']:
        assert getattr(native.vehicle_type(0),key)==getattr(old.vehicle_type(0),key)


@pytest.mark.parametrize('engine',['native','warm'])
def test_directed_oracle_and_service_start_window(engine):
    instance,matrix=fixture()
    instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:10Z','end_utc':'2018-07-27T16:00:10Z'}
    c=dict(name='test',engine=engine,backend=version('pyvrp'),seconds=.08,neighbours=3)
    r=solve_travel(instance,matrix,c)['solution']
    assert r['feasible']
    assert r['route']==['D','A','B','C','D']
    assert r['objective_scaled']==4500
    assert r['audit']['schedule'][1]['service_start_seconds']==10
    assert r['audit']['schedule'][1]['departure_seconds']==10.125


def test_infeasible_capacity_is_preserved():
    instance,matrix=fixture(); instance['vehicle_capacity_cm3']=3
    c=dict(name='test',engine='native',backend=version('pyvrp'),seconds=.1)
    r=solve_travel(instance,matrix,c)['solution']
    assert not r['feasible'] and r['route'] is None and r['status']=='input_infeasible'


def test_multistart_retains_best_feasible_trajectory():
    from importlib.metadata import version
    if version('pyvrp') != '0.13.4': pytest.skip('ILS backend only')
    instance,matrix=fixture()
    config={'name':'test_multi','backend':'0.13.4','engine':'multistart',
            'seconds':.3,'restarts':3,'neighbours':20}
    record=solve_travel(instance,matrix,config,42)
    solution=record['solution']
    costs=[t['cost'] for t in solution['trajectories'] if t['feasible']]
    assert solution['feasible'] and solution['objective_scaled']==min(costs)==4500
    assert len(solution['trajectories'])==3
    assert len(set(t['seed'] for t in solution['trajectories']))==3
