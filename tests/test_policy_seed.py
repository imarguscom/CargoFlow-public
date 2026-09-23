import importlib.util
from pathlib import Path
from test_solver import fixture

spec=importlib.util.spec_from_file_location('policy_seed',Path(__file__).resolve().parents[1]/'scripts/select_policy_seed.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_no_feasible_seed_is_a_supported_outcome():
    instance,matrix=fixture()
    selected=module.choose_seed(instance,matrix,[('old',dict(feasible=False,route=None,audit=None))])
    assert selected['status']=='no_feasible_seed' and selected['route'] is None


def test_false_feasibility_and_stored_cost_cannot_bypass_replay():
    instance,matrix=fixture()
    bad=dict(feasible=True,route=['D','A','A','C','D'],audit={'metrics':{'travel_seconds':0}})
    good=dict(feasible=True,route=['D','A','B','C','D'],audit={'metrics':{'travel_seconds':999}})
    selected=module.choose_seed(instance,matrix,[('bad',bad),('good',good)])
    assert selected['source']=='good' and selected['travel_seconds']==4.5
