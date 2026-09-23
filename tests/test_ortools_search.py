import pytest
from importlib.metadata import version

pytest.importorskip('ortools', reason='optional OR-Tools experiment runtime')
if version('pyvrp') != '0.13.4':
    pytest.skip('OR-Tools experiment uses the isolated PyVRP 0.13.4 runtime', allow_module_level=True)

from test_solver import fixture
from cargoflow.ortools_search import solve_ortools


@pytest.mark.parametrize('engine',['ortools_gls','ils_gls'])
def test_directed_cost_fixed_departure_and_service_start(engine):
    instance,matrix=fixture()
    instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:10Z','end_utc':'2018-07-27T16:00:10Z'}
    # Correctness oracle, not a cold-start 200ms performance requirement.
    config={'name':'test','backend':'ortools-9.15.6755','engine':engine,'seconds':1.0,'initial_seconds':.03}
    record=solve_ortools(instance,matrix,config)
    solution=record['solution']
    assert solution['feasible'] and solution['objective_scaled']==4500
    assert solution['audit']['schedule'][1]['service_start_seconds']==10
    assert solution['audit']['schedule'][1]['departure_seconds']==10.125
    assert record['end_to_end_seconds']>=record['search_seconds']


def test_ortools_cannot_hide_capacity_failure():
    instance,matrix=fixture();instance['vehicle_capacity_cm3']=3
    config={'name':'test','backend':'ortools-9.15.6755','engine':'ortools_gls','seconds':.1}
    solution=solve_ortools(instance,matrix,config)['solution']
    assert solution['status']=='input_infeasible' and not solution['feasible'] and solution['route'] is None
