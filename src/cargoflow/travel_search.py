"""Travel-only search experiments with the original physical constraints.

The baseline implementation stays in solver.py. This adapter builds native
ProblemData once and tests public PyVRP search/initialisation interfaces.
PyVRP algorithms are upstream work, not a new CargoFlow routing algorithm.
"""
from importlib.metadata import version
import math
from time import perf_counter

from cargoflow.solver import (SCALE, _decimal, _ticks, _timestamp,
                             validate_pair, audit_route, solve_instance)


def prepare_problem(instance, matrix):
    """Equivalent to the original Model builder, using native matrix input."""
    import numpy as np
    from pyvrp import Client, Depot, VehicleType, ProblemData
    validate_pair(instance, matrix)
    nodes = instance["nodes"]
    total = sum((_decimal(n["demand_cm3"]) for n in nodes), _decimal(0))
    if total > _decimal(instance["vehicle_capacity_cm3"]):
        return None, "input_infeasible"
    departure = _timestamp(instance["departure_time_utc"])
    bounds = []
    for node in nodes:
        tw = node["time_window"]
        if tw:
            early, late = [int((_timestamp(tw[k])-departure).total_seconds())
                           for k in ("start_utc", "end_utc")]
            if late < 0 or (node["kind"] == "depot" and early > 0):
                return None, "input_infeasible"
            bounds.append((max(0, early)*SCALE, late*SCALE))
        else:
            bounds.append(None)
    # One exact decimal conversion, rather than converting identical distance
    # and duration matrices twice and creating N^2 Python Edge objects.
    travel = np.array([[_ticks(v) for v in row] for row in matrix["travel_seconds"]], dtype=np.int64)
    services = [_ticks(n["service_seconds"]) for n in nodes]
    loads = [_ticks(n["demand_cm3"]) for n in nodes]
    capacity = _ticks(instance["vehicle_capacity_cm3"], floor=True)
    if sum(loads) > capacity:
        return None, "rounding_infeasible"
    horizon = max((b[1] for b in bounds if b), default=0) + sum(services) + len(nodes)*int(travel.max()) + SCALE
    if max(horizon, capacity, sum(loads)) >= 2**44:
        raise ValueError("scaled data exceeds supported numerical range")
    db = bounds[0] or (0, horizon)
    depot = Depot(x=nodes[0]["lng"], y=nodes[0]["lat"], tw_early=db[0], tw_late=db[1], name=nodes[0]["id"])
    vehicle = VehicleType(num_available=1, capacity=[capacity], tw_early=0,
                          start_late=0, tw_late=db[1], unit_distance_cost=1,
                          unit_duration_cost=0)
    clients = []
    for i, n in enumerate(nodes[1:], 1):
        lo, hi = bounds[i] or (0, horizon)
        clients.append(Client(x=n["lng"], y=n["lat"], delivery=[loads[i]],
            service_duration=services[i], tw_early=lo, tw_late=hi, required=True, name=n["id"]))
    return ProblemData(clients, [depot], [vehicle], [travel], [travel]), None


def planning_initial_solutions(data, instance, matrix):
    from pyvrp import Solution
    from cargoflow.baselines import nearest_neighbor, deadline_greedy
    indexes = {key: i for i, key in enumerate(matrix["node_ids"])}
    values = []
    for fn in (nearest_neighbor, deadline_greedy):
        route = fn(instance, matrix)
        values.append(Solution(data, [[indexes[k] for k in route[1:-1]]]))
    return sorted(values, key=lambda s: (not s.is_feasible(), s.time_warp(), s.distance()))


def run_old_warm(data, params, initial, stop, seed):
    """Public HGS components; constructor recipes follow PyVRP 0.12.2 solve."""
    from pyvrp import (GeneticAlgorithm, PenaltyManager, Population,
                       RandomNumberGenerator, Solution)
    from pyvrp.diversity import broken_pairs_distance
    from pyvrp.crossover import ordered_crossover
    from pyvrp.search import LocalSearch, compute_neighbours
    rng = RandomNumberGenerator(seed=seed)
    local = LocalSearch(data, rng, compute_neighbours(data, params.neighbourhood))
    for cls in params.node_ops:
        if cls.supports(data): local.add_node_operator(cls(data))
    for cls in params.route_ops:
        if cls.supports(data): local.add_route_operator(cls(data))
    population = Population(broken_pairs_distance, params.population)
    penalty = PenaltyManager.init_from(data, params.penalty)
    initial = list(initial) + [Solution.make_random(data, rng)
        for _ in range(max(0, params.population.min_pop_size-len(initial)))]
    algorithm = GeneticAlgorithm(data, penalty, rng, population, local,
                                 ordered_crossover, initial, params.genetic)
    return algorithm.run(stop, collect_stats=False, display=False)


def solve_travel(instance, matrix, config, seed=42):
    """End-to-end time includes validation, matrix conversion, seed and audit."""
    started = perf_counter()
    installed = version("pyvrp")
    if installed != config["backend"]:
        raise ValueError(f"wrong solver backend: {installed} != {config['backend']}")
    if config['engine'] not in {'baseline','native','warm','multistart'}:
        raise ValueError('unknown search engine')
    if config["engine"] == "baseline":
        budget = config["budget"]
        result = solve_instance(instance, matrix, seed=seed,
            iterations=budget.get("iterations", budget.get("max_iterations", 5000)),
            runtime_seconds=budget.get("runtime_seconds"),
            max_iterations=budget.get("max_iterations"),
            max_no_improvement=budget.get("max_no_improvement"))
        return {"solution": result, "end_to_end_seconds": perf_counter()-started,
                "search_seconds": result["solve_seconds"], "config": config, "seed": seed}
    if isinstance(config['seconds'],bool) or not math.isfinite(config['seconds']) or config['seconds']<=0:
        raise ValueError('candidate seconds must be finite and positive')
    from pyvrp import solve, SolveParams
    from pyvrp.search import NeighbourhoodParams
    from pyvrp.stop import MaxRuntime
    data, status = prepare_problem(instance, matrix)
    prepared = perf_counter()
    result = {"feasible": False, "route": None, "audit": None, "status": status,
              "pyvrp_version": installed, "seed": seed, "objective_scaled": None,
              "model": {"objective": "directed_travel_seconds", "learned_costs": False,
                        "fixed_departure": True, "time_windows": "service_start"}}
    if data is None:
        return {"solution": result, "end_to_end_seconds": perf_counter()-started,
                "build_seconds": prepared-started, "search_seconds": 0, "config": config, "seed": seed}
    param_kwargs = {'neighbourhood':NeighbourhoodParams(num_neighbours=config.get('neighbours',40))}
    if installed=='0.13.4':
        from pyvrp import IteratedLocalSearchParams
        from pyvrp.search import PerturbationParams
        if 'history_length' in config:
            param_kwargs['ils']=IteratedLocalSearchParams(history_length=config['history_length'])
        if 'max_perturbations' in config:
            param_kwargs['perturbation']=PerturbationParams(min_perturbations=1,max_perturbations=config['max_perturbations'])
    params = SolveParams(**param_kwargs)
    initial = planning_initial_solutions(data, instance, matrix) if config["engine"] == "warm" else []
    # Fixed candidate service budget includes builder and initial solutions.
    remaining = max(.001, config["seconds"]-(perf_counter()-started))
    stop = MaxRuntime(remaining)
    searched = perf_counter()
    if installed == "0.12.2" and initial:
        found = run_old_warm(data, params, initial, stop, seed)
    elif config['engine'] == 'multistart':
        if installed != '0.13.4': raise ValueError('multistart requires PyVRP 0.13.4')
        restarts = config['restarts']
        if not isinstance(restarts, int) or restarts < 1: raise ValueError('invalid restarts')
        found = None
        trajectories = []
        for restart in range(restarts):
            left = config['seconds'] - (perf_counter() - started)
            if left <= 0: break
            each = left / (restarts - restart)
            trajectory_seed = (int(seed) + 104729 * restart) % (2**32)
            trial = solve(data, stop=MaxRuntime(each), seed=trajectory_seed,
                          params=params, collect_stats=False, display=False)
            trajectories.append({'seed':trajectory_seed,'feasible':trial.is_feasible(),
                                 'cost':int(trial.cost()) if trial.is_feasible() else None})
            if found is None or (trial.is_feasible() and
                (not found.is_feasible() or trial.cost() < found.cost())):
                found = trial
        if found is None:
            found = solve(data, stop=MaxRuntime(.001), seed=seed, params=params,
                          collect_stats=False, display=False)
        result['trajectories'] = trajectories
    else:
        kwargs = {"initial_solution": initial[0]} if initial else {}
        found = solve(data, stop=stop, seed=seed, params=params, collect_stats=False, display=False, **kwargs)
    search_time = perf_counter()-searched
    routes = found.best.routes()
    result.update(iterations_run=found.num_iterations, solver_feasible=found.is_feasible())
    if len(routes) == 1:
        nodes = instance["nodes"]
        route = [nodes[0]["id"]]+[nodes[i]["id"] for i in routes[0].visits()]+[nodes[0]["id"]]
        result["route"] = route
        result["audit"] = audit_route(instance, matrix, route)
    result["feasible"] = bool(found.is_feasible() and result["audit"] and result["audit"]["feasible"])
    result["status"] = "feasible" if result["feasible"] else "no_feasible_solution_found"
    result["objective_scaled"] = int(found.cost()) if found.is_feasible() else None
    result["solve_seconds"] = search_time
    return {"solution": result, "end_to_end_seconds": perf_counter()-started,
            "build_seconds": prepared-started, "search_seconds": search_time,
            "config": config, "seed": seed}
