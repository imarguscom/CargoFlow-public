import numpy as np
from cargoflow.directed_polish import polish_indices, feasible


def test_one_pass_matches_full_directed_reversal_oracle():
    rng=np.random.default_rng(71);size=10
    travel=rng.integers(1,100,size=(size,size),dtype=np.int64);np.fill_diagonal(travel,0)
    route=list(range(size))+[0]
    service=np.zeros(size,dtype=np.int64);early=service.copy();late=np.full(size,100000,dtype=np.int64)
    def cost(r):return sum(int(travel[a,b]) for a,b in zip(r,r[1:]))
    best=cost(route)
    for i in range(1,size):
        for j in range(i+1,size):
            p=route.copy();p[i:j+1]=p[i:j+1][::-1];best=min(best,cost(p))
    original=travel.copy()
    improved,metrics=polish_indices(route,travel,service,early,late,seconds=1,max_passes=1)
    assert cost(improved)==best
    assert metrics['after_scaled']==best
    np.testing.assert_array_equal(travel,original)
    assert route==list(range(size))+[0]


def test_lower_cost_reversal_cannot_break_service_windows():
    size=6;travel=np.ones((size,size),dtype=np.int64);np.fill_diagonal(travel,0)
    route=list(range(size))+[0]
    for a,b in zip(route,route[1:]):travel[a,b]=10
    service=np.zeros(size,dtype=np.int64)
    early=np.arange(size,dtype=np.int64)*10;late=early.copy();late[0]=100
    assert feasible(np.array(route),travel,service,early,late)
    improved,metrics=polish_indices(route,travel,service,early,late,seconds=1)
    assert improved==route and metrics['moves']==0
