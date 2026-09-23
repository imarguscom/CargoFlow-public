"""Travel-only OR-Tools GLS, optionally warm-started by a fresh PyVRP ILS.

Both are upstream algorithms. Direct transit matrices avoid Python callbacks
in the inner search. Original fixed-departure service-start constraints apply.
"""
import pyvrp  # Preload the native builder dependency, as in timed workers.
from importlib.metadata import version
from time import perf_counter
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from cargoflow.travel_search import prepare_problem, solve_travel
from cargoflow.solver import audit_route


def solve_ortools(instance,matrix,config,seed=42):
    started=perf_counter()
    installed=version('ortools')
    assert config['backend']=='ortools-'+installed
    warm=None;warm_search=0
    if config['engine']=='ils_gls':
        warm_config={'name':'planning_ils_initial','backend':'0.13.4','engine':'native',
                     'seconds':config.get('initial_seconds',.7),'neighbours':config.get('neighbours',80)}
        warm=solve_travel(instance,matrix,warm_config,seed)
        warm_search=warm['search_seconds']
    data,status=prepare_problem(instance,matrix)
    result={'schema_version':'cargoflow.solution.v0.1','instance_id':instance['instance_id'],
        'solver':'PyVRP+OR-Tools' if warm else 'OR-Tools','ortools_version':installed,'pyvrp_version':version('pyvrp'),
        'seed':seed,'feasible':False,'route':None,'audit':None,'status':status,'objective_scaled':None,
        'model':{'objective':'directed_travel_seconds','learned_costs':False,'fixed_departure':True,'time_windows':'service_start'}}
    if data is None:
        return {'solution':result,'config':config,'seed':seed,'end_to_end_seconds':perf_counter()-started,'search_seconds':warm_search}
    count=data.num_locations
    manager=pywrapcp.RoutingIndexManager(count,1,0)
    routing=pywrapcp.RoutingModel(manager)
    routing.solver().ReSeed(int(seed))
    travel=data.duration_matrix(0)
    services=[0]+[data.location(i).service_duration for i in range(1,count)]
    distance_index=routing.RegisterTransitMatrix(travel.tolist())
    routing.SetArcCostEvaluatorOfAllVehicles(distance_index)
    durations=[[int(travel[i,j])+services[i] for j in range(count)] for i in range(count)]
    duration_index=routing.RegisterTransitMatrix(durations)
    horizon=max(data.location(i).tw_late for i in range(count))
    routing.AddDimension(duration_index,horizon,horizon,True,'Time')
    dimension=routing.GetDimensionOrDie('Time')
    for i in range(1,count):
        location=data.location(i)
        dimension.CumulVar(manager.NodeToIndex(i)).SetRange(location.tw_early,location.tw_late)
    depot=data.location(0)
    dimension.CumulVar(routing.End(0)).SetRange(depot.tw_early,depot.tw_late)
    # Every customer is mandatory; single-vehicle delivery capacity is checked
    # before construction and again in the independent original-unit audit.
    params=pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy=getattr(routing_enums_pb2.FirstSolutionStrategy,config.get('first','PARALLEL_CHEAPEST_INSERTION'))
    params.local_search_metaheuristic=routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.guided_local_search_lambda_coefficient=config.get('gls_lambda',.1)
    params.log_search=False
    assignment=None
    initial=None
    if warm and warm['solution']['feasible']:
        lookup={key:i for i,key in enumerate(matrix['node_ids'])}
        route=[lookup[k] for k in warm['solution']['route'][1:-1]]
        routing.CloseModelWithParameters(params)
        initial=routing.ReadAssignmentFromRoutes([route],True)
    remaining=max(.001,config['seconds']-(perf_counter()-started))
    params.time_limit.FromMilliseconds(max(1,int(1000*remaining)))
    searching=perf_counter()
    if initial:
        assignment=routing.SolveFromAssignmentWithParameters(initial,params)
    else:
        assignment=routing.SolveWithParameters(params)
    search=perf_counter()-searching+warm_search
    result['ortools_status']=int(routing.status())
    if assignment:
        route=[];idx=routing.Start(0)
        while not routing.IsEnd(idx):
            route.append(matrix['node_ids'][manager.IndexToNode(idx)])
            idx=assignment.Value(routing.NextVar(idx))
        route.append(matrix['node_ids'][manager.IndexToNode(idx)])
        result.update(route=route,audit=audit_route(instance,matrix,route),objective_scaled=int(assignment.ObjectiveValue()))
        result['feasible']=result['audit']['feasible']
    if warm and warm['solution']['feasible'] and (not result['feasible'] or
        warm['solution']['audit']['metrics']['travel_seconds']<result['audit']['metrics']['travel_seconds']):
        for key in ['route','audit','objective_scaled','feasible']:result[key]=warm['solution'][key]
        result['selected_phase']='initial_ils'
    else:result['selected_phase']='ortools_gls'
    result['status']='feasible' if result['feasible'] else 'no_feasible_solution_found'
    result['solve_seconds']=search
    return {'solution':result,'config':config,'seed':seed,'end_to_end_seconds':perf_counter()-started,
            'search_seconds':search,'timing_mode':'measured_fresh_end_to_end',
            'initial_ils_end_to_end_seconds':warm['end_to_end_seconds'] if warm else 0}
