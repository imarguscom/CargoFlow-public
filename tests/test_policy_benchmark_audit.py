from decimal import Decimal
import importlib.util
from pathlib import Path
import pytest
from test_solver import fixture

spec=importlib.util.spec_from_file_location('policy_audit',Path(__file__).resolve().parents[1]/'scripts/audit_policy_benchmark.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_independent_replay_matches_hand_calculated_objectives():
    instance,matrix=fixture()
    m=audit.replay(instance,matrix,['D','A','B','C','D'])
    assert m['travel_seconds']==Decimal('4.5')
    assert m['energy_proxy']==Decimal('6.3984375')
    assert m['delivery_completion_sum_seconds']==Decimal('7.5')


@pytest.mark.parametrize('failure',['coverage','capacity','window'])
def test_independent_replay_rejects_physical_violations(failure):
    instance,matrix=fixture();route=['D','A','B','C','D']
    if failure=='coverage':route=['D','A','A','C','D']
    elif failure=='capacity':instance['vehicle_capacity_cm3']=1
    else:instance['nodes'][1]['time_window']={'start_utc':'2018-07-27T16:00:00Z','end_utc':'2018-07-27T16:00:01Z'}
    with pytest.raises(AssertionError):audit.replay(instance,matrix,route)
