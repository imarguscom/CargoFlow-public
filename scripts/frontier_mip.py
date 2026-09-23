"""Optional SCIP full-tour model for numerical bounds on the same objectives.

Numerical solver dual bounds are labelled separately from exact-integer
assignment certificates. Load flows implement the volume-weighted energy proxy.
"""
from time import perf_counter
from decimal import Decimal
from frontier_bounds import bounds,objective_bound
from audit_policy_benchmark import replay
from cargoflow.policy_search import RouteEvaluator,_score

def solve_bound(instance,matrix,reference_route,policy,seconds=10):
    from ortools.linear_solver import pywraplp
    started=perf_counter();s=pywraplp.Solver.CreateSolver('SCIP')
    if s is None:return dict(status='SCIP_unavailable')
    s.SetNumThreads(1);s.SetTimeLimit(int(seconds*1000))
    s.SetSolverSpecificParametersAsString('randomization/randomseedshift = 42\nlimits/gap = 0.00001')
    reference=replay(instance,matrix,reference_route);ev=RouteEvaluator(instance,matrix)
    ids=matrix['node_ids'];n=len(ids);nodes={v['id']:v for v in instance['nodes']}
    travel=matrix['travel_seconds'];service=[float(nodes[k]['service_seconds']) for k in ids]
    capacity=float(instance['vehicle_capacity_cm3']);q=[float(nodes[k]['demand_cm3'])/(capacity or 1) for k in ids]
    tl=float(reference['travel_seconds'])*(1+policy.max_travel_increase)
    early=[max(0,ev.windows[k][0]) if ev.windows[k] else 0 for k in ids]
    horizon=max(early)+tl+sum(service)+1
    late=[min(horizon,ev.windows[k][1]) if ev.windows[k] else horizon for k in ids]
    arcs=[(i,j) for i in range(n) for j in range(n) if i!=j]
    x={(i,j):s.BoolVar(f'x{i}_{j}') for i,j in arcs}
    load={(i,j):s.NumVar(0,0 if j==0 else 1,f'f{i}_{j}') for i,j in arcs}
    finish={j:s.NumVar(early[j]+service[j],late[j]+service[j],f'c{j}') for j in range(1,n)}
    returned=s.NumVar(early[0],late[0], 'return')
    order={j:s.NumVar(1,n-1,f'o{j}') for j in range(1,n)}
    for i in range(n):
        s.Add(sum(x[i,j] for j in range(n) if i!=j)==1)
        s.Add(sum(x[j,i] for j in range(n) if i!=j)==1)
        outgoing=sum(load[i,j] for j in range(n) if i!=j)
        incoming=sum(load[j,i] for j in range(n) if i!=j)
        s.Add(outgoing-incoming==(sum(q) if i==0 else -q[i]))
    for i,j in arcs:
        s.Add(load[i,j]<=x[i,j])
        previous=0 if i==0 else finish[i]
        dest=returned if j==0 else finish[j]
        duration=float(travel[i][j])+service[j]
        bigm=horizon+max(service)+duration
        s.Add(dest>=previous+duration-bigm*(1-x[i,j]))
        if i and j:s.Add(order[j]>=order[i]+1-n*(1-x[i,j]))
    T=sum(float(travel[i][j])*x[i,j] for i,j in arcs)
    E=T+sum(float(travel[i][j])*load[i,j] for i,j in arcs)
    D=sum(finish.values())
    s.Add(T<=tl)
    if policy.max_energy_increase is not None:s.Add(E<=float(reference['energy_proxy'])*(1+policy.max_energy_increase))
    weights=[policy.time_weight,policy.completion_weight,policy.energy_weight]
    terms=[T,D,E];keys=['travel_seconds','delivery_completion_sum_seconds','energy_proxy']
    obj=sum(w*term/(float(reference[k]) or 1)/sum(weights) for w,term,k in zip(weights,terms,keys))
    s.Minimize(obj)
    index={k:i for i,k in enumerate(ids)};path=[index[k] for k in reference_route];used=set(zip(path,path[1:]))
    s.SetHint([x[e] for e in arcs],[1 if e in used else 0 for e in arcs])
    build=perf_counter()-started;status=s.Solve()
    result=dict(status=int(status),solver=s.SolverVersion(),model_build_seconds=build,total_seconds=perf_counter()-started,
        numerical_lower_bound=s.Objective().BestBound(),variables=s.NumVariables(),constraints=s.NumConstraints(),
        caveat='SCIP floating-point dual bound, subject to solver tolerances; separate from certified integer relaxation. Timeout is not optimality.')
    if status in [s.OPTIMAL,s.FEASIBLE]:
        nxt={i:j for (i,j),v in x.items() if v.solution_value()>.5};path=[0]
        while len(path)<=n:
            path.append(nxt[path[-1]])
            if path[-1]==0:break
        route=[ids[i] for i in path]
        try:
            metrics=replay(instance,matrix,route)
            assert metrics['travel_seconds']<=reference['travel_seconds']*(1+Decimal(str(policy.max_travel_increase)))
            if policy.max_energy_increase is not None:assert metrics['energy_proxy']<=reference['energy_proxy']*(1+Decimal(str(policy.max_energy_increase)))
            result.update(route=route,independent_score=_score(metrics,reference,policy),incumbent_audited=True)
        except (AssertionError,KeyError):result.update(incumbent_audited=False)
        result['solver_objective']=s.Objective().Value()
    return result
