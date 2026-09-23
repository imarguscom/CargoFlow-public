"""Feasible directed 2-opt polishing; travel-only, bounded wall time.

This implements the standard 2-opt neighbourhood with the full asymmetric
internal reversal cost. PyVRP's SwapTails is inter-route and disabled for
single-vehicle problems; this is a separate experimental intra-route pass.
"""
from time import perf_counter
import numpy as np


def feasible(order, travel, service, earliest, latest):
    if not earliest[0] <= 0 <= latest[0]: return False
    # Recurrence for earliest service start with waiting, vectorised by the
    # cumulative amount of waiting that has been required so far.
    increments=travel[order[:-1],order[1:]]+service[order[:-1]]
    no_wait=np.cumsum(increments,dtype=np.int64)
    wait=np.maximum.accumulate(np.maximum(0,earliest[order[1:]]-no_wait))
    return bool(np.all(no_wait+wait<=latest[order[1:]]))


def polish_indices(order, travel, service, earliest, latest, *, seconds=.15, max_passes=100):
    started=perf_counter(); deadline=started+seconds
    route=np.array(order,dtype=np.int64,copy=True)
    travel=np.asarray(travel,dtype=np.int64)
    service=np.asarray(service,dtype=np.int64)
    earliest=np.asarray(earliest,dtype=np.int64);latest=np.asarray(latest,dtype=np.int64)
    size=len(travel)
    if route[0]!=0 or route[-1]!=0 or sorted(route[1:-1])!=list(range(1,size)):
        raise ValueError('expected one closed route covering all customers exactly once')
    if not feasible(route,travel,service,earliest,latest):
        raise ValueError('polishing requires an already feasible route')
    original=int(travel[route[:-1],route[1:]].sum());moves=0
    allowed=np.triu(np.ones((size-1,size-1),dtype=bool),k=1)
    for _ in range(max_passes):
        if perf_counter()>=deadline:break
        current=route[1:-1];before=route[:-2];after=route[2:]
        reversal=travel[current[1:],current[:-1]]-travel[current[:-1],current[1:]]
        prefix=np.r_[np.int64(0),np.cumsum(reversal,dtype=np.int64)]
        delta=(travel[np.ix_(before,current)]+travel[np.ix_(current,after)]
               -travel[before,current][:,None]-travel[current,after][None,:]
               +prefix[None,:]-prefix[:,None])
        pairs=np.argwhere(allowed & (delta<0))
        if not len(pairs):break
        ordering=np.argsort(delta[pairs[:,0],pairs[:,1]],kind='stable')
        improved=False
        for k in ordering:
            if perf_counter()>=deadline:break
            i,j=pairs[k]+1
            proposal=route.copy();proposal[i:j+1]=proposal[i:j+1][::-1]
            if feasible(proposal,travel,service,earliest,latest):
                route=proposal;moves+=1;improved=True;break
        if not improved:break
    final=int(travel[route[:-1],route[1:]].sum())
    assert final<=original
    return route.tolist(),{'moves':moves,'before_scaled':original,'after_scaled':final,'seconds':perf_counter()-started}
