from importlib.metadata import version
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
if version('pyvrp')!='0.14.0':pytest.skip('isolated 0.14.0 adapter',allow_module_level=True)
from test_solver import fixture
from frontier_pyvrp14 import solve

def test_latest_adapter_preserves_fixed_departure_and_service_start():
    i,m=fixture();i['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:10Z','end_utc':'2018-07-27T16:00:10Z'}
    r=solve(i,m,dict(seconds=.1),42)['solution']
    assert r['feasible'] and r['route']==['D','A','B','C','D']
    assert r['audit']['metrics']['travel_seconds']==4.5
    assert r['audit']['schedule'][1]['service_start_seconds']==10

def test_latest_adapter_capacity_failure():
    i,m=fixture();i['vehicle_capacity_cm3']=3
    r=solve(i,m,dict(seconds=.1),42)['solution']
    assert r['status']=='input_infeasible' and not r['feasible']
