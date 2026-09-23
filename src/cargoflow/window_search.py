"""Fresh ILS followed by bounded directed block DP within one service budget."""
import hashlib
from time import perf_counter
import numpy as np
from cargoflow.travel_search import solve_travel,prepare_problem
from cargoflow.solver import audit_route
from cargoflow.window_dp import improve_window_dp,LIBRARY


def solve_window(instance,matrix,config,seed=42):
    started=perf_counter()
    base=solve_travel(instance,matrix,dict(name='fresh_ils_initial',backend='0.13.4',engine='native',
          seconds=config['initial_seconds'],neighbours=80),seed)
    solution=base['solution']
    initial_travel=solution['audit']['metrics']['travel_seconds'] if solution['feasible'] else None
    dp_time=0;stats={}
    if solution['feasible']:
        data,status=prepare_problem(instance,matrix)
        n=data.num_locations
        lookup={key:i for i,key in enumerate(matrix['node_ids'])}
        route=[lookup[key] for key in solution['route']]
        services=[0]+[data.location(i).service_duration for i in range(1,n)]
        early=[data.location(i).tw_early for i in range(n)]
        late=[data.location(i).tw_late for i in range(n)]
        left=max(.001,config['seconds']-(perf_counter()-started))
        searching=perf_counter()
        indices,stats=improve_window_dp(route,data.duration_matrix(0),services,early,late,config['width'],left,seed)
        dp_time=perf_counter()-searching
        route=[matrix['node_ids'][i] for i in indices]
        audit=audit_route(instance,matrix,route)
        if audit['feasible'] and audit['metrics']['travel_seconds']<initial_travel:
            solution.update(route=route,audit=audit,objective_scaled=sum(int(data.distance_matrix(0)[a,b]) for a,b in zip(indices,indices[1:])))
        solution['window_dp_binary_sha256']=hashlib.sha256(LIBRARY.read_bytes()).hexdigest()
    solution.update(window_dp=stats,initial_travel_seconds=initial_travel)
    return dict(solution=solution,config=config,seed=seed,end_to_end_seconds=perf_counter()-started,
                search_seconds=base['search_seconds']+dp_time,timing_mode='measured_fresh_end_to_end')
