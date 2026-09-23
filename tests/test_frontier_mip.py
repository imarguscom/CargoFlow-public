import itertools
import pytest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
pytest.importorskip('ortools',reason='optional SCIP experiment runtime')
from test_solver import fixture
from frontier_mip import solve_bound
from run_policy_search import POLICIES
from audit_policy_benchmark import replay
from cargoflow.policy_search import _score
from decimal import Decimal

@pytest.mark.parametrize('policy',POLICIES,ids=lambda p:p.name)
def test_full_mip_matches_exhaustive_objective(policy):
    i,m=fixture();m['travel_seconds']=[[0,9,3,8],[2,0,4,1],[4,7,0,6],[3,5,2,0]]
    route=['D','A','B','C','D'];ref=replay(i,m,route);scores=[]
    for middle in itertools.permutations(['A','B','C']):
        v=replay(i,m,['D',*middle,'D'])
        if v['travel_seconds']>ref['travel_seconds']*(1+Decimal(str(policy.max_travel_increase))):continue
        if policy.max_energy_increase is not None and v['energy_proxy']>ref['energy_proxy']:continue
        scores.append(_score(v,ref,policy))
    r=solve_bound(i,m,route,policy,seconds=2)
    assert r['status']==0 and r['incumbent_audited']
    assert r['independent_score']==pytest.approx(min(scores))
    assert r['numerical_lower_bound']==pytest.approx(min(scores))
