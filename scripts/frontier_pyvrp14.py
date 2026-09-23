"""PyVRP 0.14.0 adapter preserving the frozen single-tour contract."""
from importlib.metadata import version
from time import perf_counter
import numpy as np
from cargoflow.solver import validate_pair,audit_route,_decimal,_ticks,_timestamp,SCALE

def solve(instance,matrix,config,seed):
    from pyvrp import Location,Client,Depot,VehicleType,ProblemData,SolveParams,solve as optimise
    from pyvrp.search import NeighbourhoodParams
    from pyvrp.stop import MaxRuntime
    assert version('pyvrp')=='0.14.0'
    started=perf_counter();validate_pair(instance,matrix);nodes=instance['nodes']
    result=dict(feasible=False,route=None,audit=None,pyvrp_version=version('pyvrp'),seed=seed,status='no_feasible_solution_found')
    def finish():return dict(solution=result,config=config,seed=seed,end_to_end_seconds=perf_counter()-started)
    if sum(_decimal(x['demand_cm3']) for x in nodes)>_decimal(instance['vehicle_capacity_cm3']):
        result['status']='input_infeasible';return finish()
    departure=_timestamp(instance['departure_time_utc']);windows=[]
    for node in nodes:
        tw=node['time_window']
        if tw:
            lo,hi=[int((_timestamp(tw[k])-departure).total_seconds()) for k in ['start_utc','end_utc']]
            if hi<0 or (node['kind']=='depot' and lo>0):result['status']='input_infeasible';return finish()
            windows.append((max(0,lo)*SCALE,hi*SCALE))
        else:windows.append(None)
    t=np.array([[_ticks(x) for x in row] for row in matrix['travel_seconds']],dtype=np.int64)
    service=[_ticks(x['service_seconds']) for x in nodes];loads=[_ticks(x['demand_cm3']) for x in nodes]
    cap=_ticks(instance['vehicle_capacity_cm3'],floor=True)
    if sum(loads)>cap:result['status']='rounding_infeasible';return finish()
    horizon=max((w[1] for w in windows if w),default=0)+sum(service)+len(nodes)*int(t.max())+SCALE
    assert max(horizon,cap,sum(loads))<2**44
    windows=[w or (0,horizon) for w in windows]
    locations=[Location(x=n['lng'],y=n['lat'],name=n['id']) for n in nodes]
    clients=[Client(location=i,delivery=[loads[i]],service_duration=service[i],tw_early=windows[i][0],tw_late=windows[i][1],required=True,name=nodes[i]['id']) for i in range(1,len(nodes))]
    depots=[Depot(location=0,tw_early=windows[0][0],tw_late=windows[0][1],name=nodes[0]['id'])]
    vehicles=[VehicleType(num_available=1,capacity=[cap],tw_early=0,start_late=0,tw_late=windows[0][1],unit_distance_cost=1,unit_duration_cost=0)]
    data=ProblemData(locations,clients,depots,vehicles,[t],[t])
    params=SolveParams(neighbourhood=NeighbourhoodParams(num_neighbours=config.get('neighbours',80)))
    found=optimise(data,stop=MaxRuntime(max(.001,config['seconds']-(perf_counter()-started))),seed=seed,params=params,collect_stats=False,display=False)
    routes=found.best.routes()
    if len(routes)==1:
        route=[nodes[0]['id']]+[nodes[a.idx+1]['id'] for a in routes[0] if a.is_client()]+[nodes[0]['id']]
        audit=audit_route(instance,matrix,route)
        result.update(route=route,audit=audit,feasible=bool(found.is_feasible() and audit['feasible']))
    result.update(status='feasible' if result['feasible'] else 'no_feasible_solution_found',iterations_run=found.num_iterations)
    return finish()
