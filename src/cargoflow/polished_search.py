"""Compose a fresh travel solver with conservative directed 2-opt polishing."""
from copy import deepcopy
from time import perf_counter, process_time
import numpy as np
from cargoflow.solver import audit_route
from cargoflow.travel_search import prepare_problem, solve_travel
from cargoflow.directed_polish import polish_indices


def polish_result(instance,matrix,base_result,config,*,seconds=.15):
    """Refine a candidate result; preserves it if raw travel does not improve.

    Calling this on saved development results is component replay. Its summed
    timing must be labelled estimated; production uses solve_polished below.
    """
    started=perf_counter();cpu_started=process_time();record=deepcopy(base_result)
    record['config']=config
    record['polish']={'accepted':False,'moves':0,'input_route':base_result['solution']['route'],
        'before_travel_seconds':base_result['solution']['audit']['metrics']['travel_seconds'] if base_result['solution']['feasible'] else None}
    solution=record['solution']
    if solution['feasible']:
        data,status=prepare_problem(instance,matrix)
        assert status is None
        index={key:i for i,key in enumerate(matrix['node_ids'])}
        order=[index[k] for k in solution['route']]
        size=data.num_locations
        services=np.array([0]+[data.location(i).service_duration for i in range(1,size)],dtype=np.int64)
        early=np.array([data.location(i).tw_early for i in range(size)],dtype=np.int64)
        late=np.array([data.location(i).tw_late for i in range(size)],dtype=np.int64)
        improved,metrics=polish_indices(order,data.duration_matrix(0),services,early,late,seconds=max(.001,seconds-(perf_counter()-started)))
        record['polish'].update(metrics)
        if improved!=order:
            route=[matrix['node_ids'][i] for i in improved]
            check=audit_route(instance,matrix,route)
            old_travel=solution['audit']['metrics']['travel_seconds']
            if check['feasible'] and check['metrics']['travel_seconds']<old_travel:
                solution.update(route=route,audit=check,objective_scaled=metrics['after_scaled'])
                record['polish']['accepted']=True
    record['polish']['after_travel_seconds']=solution['audit']['metrics']['travel_seconds'] if solution['feasible'] else None
    elapsed=perf_counter()-started
    record['polish']['end_to_end_seconds']=elapsed
    record['end_to_end_seconds']=base_result['end_to_end_seconds']+elapsed
    if 'process_cpu_seconds' in base_result:
        record['process_cpu_seconds']=base_result['process_cpu_seconds']+process_time()-cpu_started
    record['timing_mode']='development_component_sum'
    return record


def solve_polished(instance,matrix,config,seed=42):
    started=perf_counter()
    assert config['engine']=='polished' and config['backend']==config['base_config']['backend']
    base=solve_travel(instance,matrix,config['base_config'],seed)
    remaining=config['seconds']-(perf_counter()-started)
    if remaining>0:
        result=polish_result(instance,matrix,base,config,seconds=min(config.get('polish_seconds',.15),remaining))
    else:
        result=base;result['config']=config;result['polish']={'accepted':False,'reason':'no remaining budget'}
    result['end_to_end_seconds']=perf_counter()-started
    result['timing_mode']='measured_fresh_end_to_end'
    return result
