"""Isolated upstream LKH-3 TSPTW adapter for arbitrary directed travel matrices.

The solver adds origin service to each arc. Its tour objective differs from
travel by a route-independent constant; reported costs are recomputed from
original directed travel. No precedence closure assuming metricity is used.
"""
import pyvrp  # Preload native data dependency before the timed service call.
from itertools import permutations
from importlib.metadata import version
from pathlib import Path
import subprocess
import tempfile
from time import perf_counter
from cargoflow.travel_search import prepare_problem, solve_travel
from cargoflow.solver import audit_route

ROOT=Path(__file__).resolve().parents[2]
BINARY=ROOT/'artifacts/runtime_lkh/LKH-3.0.13/LKH'


def encode_problem(data):
    count=data.num_locations
    travel=data.duration_matrix(0)
    if max(int(travel.max()),max(data.location(i).service_duration for i in range(1,count)))>=2**28:
        raise ValueError('LKH int32 arc range exceeded')
    lines=['NAME : cargoflow','TYPE : TSPTW',f'DIMENSION : {count}',
           'EDGE_WEIGHT_TYPE : EXPLICIT','EDGE_WEIGHT_FORMAT : FULL_MATRIX','EDGE_WEIGHT_SECTION']
    lines += [' '.join(str(int(x)) for x in row) for row in travel]
    lines += ['TIME_WINDOW_SECTION']
    lines += [f'{i+1} {data.location(i).tw_early} {data.location(i).tw_late}' for i in range(count)]
    lines += ['SERVICE_TIME_SECTION','1 0']
    lines += [f'{i+1} {data.location(i).service_duration}' for i in range(1,count)]
    lines += ['DEPOT_SECTION','1','-1','EOF']
    return '\n'.join(lines)+'\n'


def decode_tour(text,count):
    body=text.split('TOUR_SECTION',1)[1].split('-1',1)[0]
    values=[int(v)-1 for v in body.split()]
    if sorted(values)!=list(range(count)):raise ValueError('invalid LKH coverage')
    idx=values.index(0)
    return values[idx:]+values[:idx]+[0]


def solve_lkh(instance,matrix,config,seed=42):
    started=perf_counter()
    assert config['backend']=='lkh-3.0.13'
    warm=None;warm_search=0
    if config['engine']=='ils_lkh' and len(instance['nodes'])>3:
        warm=solve_travel(instance,matrix,dict(name='fresh_ils_initial',backend='0.13.4',
            engine='native',seconds=config.get('initial_seconds',.5),neighbours=80),seed)
        warm_search=warm['search_seconds']
    data,status=prepare_problem(instance,matrix)
    result=dict(feasible=False,route=None,audit=None,status=status,objective_scaled=None,
        lkh_version='3.0.13',pyvrp_version=version('pyvrp'),seed=seed,
        model=dict(objective='directed_travel_seconds',learned_costs=False,fixed_departure=True,time_windows='service_start'))
    if data is None:
        return dict(solution=result,config=config,seed=seed,end_to_end_seconds=perf_counter()-started,search_seconds=warm_search)
    if data.num_locations<=3:
        # LKH requires at least three locations. At most two customers are
        # cheaper and exact to enumerate, preserving the integer contract.
        searching=perf_counter()
        for middle in permutations(range(1,data.num_locations)):
            indices=[0,*middle,0];clock=0;valid=True
            for a,b in zip(indices,indices[1:]):
                clock=max(clock+int(data.duration_matrix(0)[a,b]),data.location(b).tw_early)
                valid &= clock<=data.location(b).tw_late
                if b:clock+=data.location(b).service_duration
            if not valid:continue
            route=[matrix['node_ids'][i] for i in indices]
            audit=audit_route(instance,matrix,route)
            cost=sum(int(data.distance_matrix(0)[a,b]) for a,b in zip(indices,indices[1:]))
            if audit['feasible'] and (result['objective_scaled'] is None or cost<result['objective_scaled']):
                result.update(route=route,audit=audit,objective_scaled=cost,feasible=True)
        search=perf_counter()-searching
        result.update(status='feasible' if result['feasible'] else 'no_feasible_solution_found',
                      selected_phase='exact_small',solve_seconds=search,lkh_exit=None)
        return dict(solution=result,config=config,seed=seed,end_to_end_seconds=perf_counter()-started,
                    search_seconds=search,timing_mode='measured_fresh_end_to_end')
    if not BINARY.is_file():
        raise FileNotFoundError('Build LKH first: python scripts/build_lkh_runtime.py')
    with tempfile.TemporaryDirectory(prefix='cargoflow-lkh-') as folder:
        folder=Path(folder)
        (folder/'problem.tsp').write_text(encode_problem(data))
        lines=[f'PROBLEM_FILE = {folder}/problem.tsp',f'OUTPUT_TOUR_FILE = {folder}/out.tour',
            'PRECISION = 1','RUNS = 100000','MAX_TRIALS = 50',f'SEED = {seed}',
            'TRACE_LEVEL = 0',f'MAX_CANDIDATES = {config.get("candidates",5)}',
            f'MOVE_TYPE = {config.get("move_type",5)}','STOP_AT_OPTIMUM = NO']
        if config.get('special',False): lines += ['SPECIAL']
        if warm and warm['solution']['feasible']:
            lookup={key:i+1 for i,key in enumerate(matrix['node_ids'])}
            (folder/'initial.tour').write_text('TYPE : TOUR\nDIMENSION : '+str(data.num_locations)+'\nTOUR_SECTION\n'+
                '\n'.join(str(lookup[k]) for k in warm['solution']['route'][:-1])+'\n-1\nEOF\n')
            lines += [f'INITIAL_TOUR_FILE = {folder}/initial.tour','INITIAL_TOUR_FRACTION = 1']
        # LKH internal limits use CPU time. A wall timeout also bounds work on
        # a shared host; OUTPUT_TOUR_FILE checkpoints feasible improvements.
        remaining=max(.001,config['seconds']-(perf_counter()-started))
        lines += [f'TOTAL_TIME_LIMIT = {max(.001,remaining-.02):.6f}',f'TIME_LIMIT = {max(.001,remaining-.02):.6f}']
        (folder/'params.par').write_text('\n'.join(lines)+'\n')
        searching=perf_counter()
        try:
            proc=subprocess.run([str(BINARY),str(folder/'params.par')],capture_output=True,text=True,timeout=remaining)
            result.update(lkh_exit=proc.returncode,lkh_log=proc.stdout[-1500:]+proc.stderr[-1500:])
        except subprocess.TimeoutExpired:
            result.update(lkh_exit=None,lkh_wall_timeout=True)
        search=perf_counter()-searching+warm_search
        if (folder/'out.tour').exists():
            try:
                indices=decode_tour((folder/'out.tour').read_text(),data.num_locations)
                route=[matrix['node_ids'][i] for i in indices]
                audit=audit_route(instance,matrix,route)
                if audit['feasible']:
                    cost=sum(int(data.distance_matrix(0)[a,b]) for a,b in zip(indices,indices[1:]))
                    result.update(route=route,audit=audit,objective_scaled=cost,feasible=True)
            except (ValueError,IndexError):
                result['incomplete_checkpoint']=True
    if warm and warm['solution']['feasible'] and (not result['feasible'] or
        warm['solution']['audit']['metrics']['travel_seconds']<result['audit']['metrics']['travel_seconds']):
        for key in ['route','audit','objective_scaled','feasible']:result[key]=warm['solution'][key]
        result['selected_phase']='initial_ils'
    else:result['selected_phase']='lkh'
    result['status']='feasible' if result['feasible'] else 'no_feasible_solution_found'
    result['solve_seconds']=search
    return dict(solution=result,config=config,seed=seed,end_to_end_seconds=perf_counter()-started,
                search_seconds=search,timing_mode='measured_fresh_end_to_end')
