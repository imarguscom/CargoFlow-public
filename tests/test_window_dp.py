import itertools
import pytest
import numpy as np
from pathlib import Path

if not (Path(__file__).resolve().parents[1]/'artifacts/runtime_window_dp/window_dp.so').is_file():
    pytest.skip('optional block-DP native runtime is not built', allow_module_level=True)

from cargoflow.window_dp import improve_window_dp


def test_directed_block_dp_matches_full_permutation_oracle():
    rng=np.random.default_rng(512)
    travel=rng.integers(1,100,size=(8,8),dtype=np.int64);np.fill_diagonal(travel,0)
    original=travel.copy();route=[*range(8),0]
    cost=lambda r:sum(int(travel[a,b]) for a,b in zip(r,r[1:]))
    expected=min(cost([0,*p,0]) for p in itertools.permutations(range(1,8)))
    result,stats=improve_window_dp(route,travel,[0]*8,[0]*8,[10000]*8,7,.2)
    assert cost(result)==expected and stats['accepted_blocks']>0
    assert route==[*range(8),0]
    np.testing.assert_array_equal(travel,original)


def test_cheaper_but_infeasible_block_is_rejected():
    travel=np.array([[0,1,1,50],[20,0,5,1],[20,1,0,5],[1,20,20,0]])
    route=[0,1,2,3,0]
    # Node1 must start by 1. Unconstrained cheapest path starts at node2.
    result,_=improve_window_dp(route,travel,[0]*4,[0]*4,[100,1,100,100],3,.1)
    clock=0
    for a,b in zip(result,result[1:]):
        clock+=int(travel[a,b])
        if b==1:assert clock<=1
    assert sorted(result[1:-1])==[1,2,3]
    assert result==route  # Unique relaxed optimum violates node1's deadline.


def test_invalid_depot_cannot_reach_native_indexing():
    with pytest.raises(ValueError,match='complete depot tour'):
        improve_window_dp([9,1,2,9],np.ones((3,3),dtype=np.int64),[0]*3,[0]*3,[100]*3,2,.1)
