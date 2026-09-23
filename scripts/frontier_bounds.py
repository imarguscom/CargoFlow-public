"""Exact integer assignment/shortest-path relaxations, no optimality overclaim."""
from decimal import Decimal, ROUND_FLOOR, localcontext
from fractions import Fraction
import heapq
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cargoflow.policy_search import RouteEvaluator

def assignment(a):
    """Hungarian primal/dual, arbitrary precision integers; certificate checked."""
    n=len(a);u=[0]*(n+1);v=[0]*(n+1);p=[0]*(n+1);way=[0]*(n+1)
    infinity=max(max(row) for row in a)*(n+1)+1
    for i in range(1,n+1):
        p[0]=i;j0=0;mins=[infinity]*(n+1);used=[False]*(n+1)
        while True:
            used[j0]=True;i0=p[j0];delta=infinity;j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cur=a[i0-1][j-1]-u[i0]-v[j]
                    if cur<mins[j]:mins[j]=cur;way[j]=j0
                    if mins[j]<delta:delta=mins[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:mins[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    permutation=[0]*n
    for j in range(1,n+1):permutation[p[j]-1]=j-1
    primal=sum(a[i][permutation[i]] for i in range(n));dual=sum(u[1:])+sum(v[1:])
    assert primal==dual and all(u[i+1]+v[j+1]<=a[i][j] for i in range(n) for j in range(n))
    return primal,dict(permutation=permutation,row_dual=u[1:],column_dual=v[1:])

def bounds(instance,matrix):
    D=lambda x:Decimal(str(x));scale=1000
    floor=lambda x:int((D(x)*scale).to_integral_value(rounding=ROUND_FLOOR))
    ids=matrix['node_ids'];n=len(ids);nodes={x['id']:x for x in instance['nodes']}
    assert ids[0]==instance['depot_id'] and n>1
    evaluator=RouteEvaluator(instance,matrix)
    t=[[floor(x) for x in row] for row in matrix['travel_seconds']]
    maxarc=max(max(row) for row in t);forbid=(maxarc+1)*(n+1)
    a=[[forbid if i==j else t[i][j] for j in range(n)] for i in range(n)]
    value,certificate=assignment(a)
    assert all(j!=i for i,j in enumerate(certificate['permutation']))
    service=[floor(nodes[k]['service_seconds']) for k in ids]
    early=[floor(evaluator.windows[k][0]) if evaluator.windows[k] else 0 for k in ids]
    def shortest(completion=False):
        d=[None]*n;d[0]=0;queue=[(0,0)]
        while queue:
            now,i=heapq.heappop(queue)
            if d[i]!=now:continue
            for j in range(n):
                cost=now+t[i][j]
                if completion:cost=max(cost,early[j])+service[j]
                if d[j] is None or cost<d[j]:d[j]=cost;heapq.heappush(queue,(cost,j))
        return d
    shortest_t=shortest();shortest_d=shortest(True)
    # Each customer's incoming edge plus service contributes to every later
    # completion. Sorting these lower processing times gives a second valid LB.
    processing=sorted(min(t[i][j] for i in range(n) if i!=j)+service[j] for j in range(1,n))
    d_ticks=max(sum(shortest_d[1:]),sum((n-1-k)*x for k,x in enumerate(processing)))
    capacity=D(instance['vehicle_capacity_cm3']);travel_lb=D(value)/scale
    weighted_drive=sum(Fraction(str(nodes[ids[j]]['demand_cm3']))*shortest_t[j]/scale for j in range(1,n))
    energy_fraction=Fraction(value,scale)+(weighted_drive/Fraction(capacity) if capacity else 0)
    with localcontext() as context:
        context.prec=60;context.rounding=ROUND_FLOOR
        energy_lb=Decimal(energy_fraction.numerator)/Decimal(energy_fraction.denominator)
    # Every delivery volume is carried over every arc preceding its delivery,
    # giving E = T + sum_j volume_j * driven_prefix_j / capacity.
    return dict(travel_seconds=str(travel_lb),delivery_completion_sum_seconds=str(D(d_ticks)/scale),energy_proxy=str(energy_lb),
        scale=scale,assignment_certificate=certificate,assignment_ticks=value,
        shortest_travel_ticks=shortest_t,shortest_completion_ticks=shortest_d,
        limitation='Certified relaxation bounds; ignoring subtours and joint windows may make gaps loose. Not a proof that the gap is achievable.')

def objective_bound(lower,reference,policy):
    D=lambda x:Decimal(str(x))
    pairs=[('travel_seconds',policy.time_weight),('delivery_completion_sum_seconds',policy.completion_weight),('energy_proxy',policy.energy_weight)]
    return sum(D(w)*D(lower[k])/(D(reference[k]) or 1) for k,w in pairs)/sum(D(w) for k,w in pairs)
