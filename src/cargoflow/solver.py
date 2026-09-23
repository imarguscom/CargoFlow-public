"""Single-vehicle, closed-tour PyVRP baseline for the frozen v0.1 contract.

Only directed travel time contributes to the objective. Time windows apply
to service START. The vehicle departs at the supplied timestamp and returns
to the depot; these are explicit modelling choices, not learned quantities.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
from importlib.metadata import version
import platform
from time import perf_counter

from cargoflow.contract import validate_instance, validate_matrix

SCALE = 1000


def _decimal(value):
    return Decimal(str(value))


def _ticks(value, *, floor=False):
    return int((_decimal(value) * SCALE).to_integral_value(
        rounding=ROUND_FLOOR if floor else ROUND_CEILING))


def _timestamp(value):
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def validate_pair(instance, matrix):
    validate_instance(instance)
    validate_matrix(matrix)
    if instance["instance_id"] != matrix["instance_id"]:
        raise ValueError("instance_id differs between instance and matrix")
    if [node["id"] for node in instance["nodes"]] != matrix["node_ids"]:
        raise ValueError("matrix node_ids must match instance node order exactly")
    if len(instance["nodes"]) < 2:
        raise ValueError("at least one customer is required")


def _validate_cost_matrix(instance, matrix, objective_cost_seconds):
    """Validate and normalise an optional objective-distance matrix.

    The supplied matrix is intentionally separate from the contract's
    directed duration matrix.  It is used only as PyVRP's distance/objective
    value; the original duration matrix remains the timing source everywhere.
    """

    if objective_cost_seconds is None:
        return matrix["travel_seconds"]
    if not isinstance(objective_cost_seconds, list):
        raise ValueError("objective_cost_seconds must be a square matrix")
    n = len(matrix["node_ids"])
    if len(objective_cost_seconds) != n:
        raise ValueError("objective_cost_seconds must match matrix dimensions")
    result = []
    for i, row in enumerate(objective_cost_seconds):
        if not isinstance(row, list) or len(row) != n:
            raise ValueError("objective_cost_seconds must be a square matrix")
        values = []
        for j, value in enumerate(row):
            if isinstance(value, bool):
                raise ValueError("objective_cost_seconds values must be finite non-negative numbers")
            try:
                decimal = _decimal(value)
            except Exception as exc:
                raise ValueError("objective_cost_seconds values must be finite non-negative numbers") from exc
            if not decimal.is_finite() or decimal < 0:
                raise ValueError("objective_cost_seconds values must be finite non-negative numbers")
            if i == j and decimal != 0:
                raise ValueError("objective_cost_seconds diagonal must be zero")
            values.append(value)
        result.append(values)
    return result


def _stopping_criterion(*, max_iterations, runtime_seconds, max_no_improvement):
    """Build a PyVRP stop criterion from all requested limits.

    PyVRP has changed the constructor shape of ``MultipleCriteria`` between
    minor releases.  Keep the compatibility shim local so the public solver
    interface remains stable for the pinned release and nearby environments.
    """

    from pyvrp.stop import MaxIterations, MaxRuntime, MultipleCriteria, NoImprovement

    criteria = [MaxIterations(max_iterations)] if max_iterations is not None else []
    if runtime_seconds is not None:
        criteria.append(MaxRuntime(runtime_seconds))
    if max_no_improvement is not None:
        criteria.append(NoImprovement(max_no_improvement))
    if not criteria:
        # The legacy API always supplied an iteration limit. This branch is
        # retained for defensive use by callers passing only custom settings.
        return MaxIterations(1)
    if len(criteria) == 1:
        return criteria[0]
    try:
        return MultipleCriteria(criteria)
    except TypeError:
        return MultipleCriteria(*criteria)


def audit_route(instance, matrix, route):
    """Independently replay a tour using original decimal values, not PyVRP.

    Earliest possible service from the fixed departure is sufficient to
    check feasibility for this static, delivery-only single-vehicle model.
    """
    validate_pair(instance, matrix)
    nodes = {node["id"]: node for node in instance["nodes"]}
    depot = instance["depot_id"]
    expected = set(nodes) - {depot}
    errors = []
    if not isinstance(route, list) or any(not isinstance(x, str) for x in route):
        raise ValueError("route must be a list of node IDs")
    unknown = sorted(set(route) - set(nodes))
    counts = Counter(route[1:-1])
    missing = sorted(expected - set(counts))
    duplicates = sorted(k for k, count in counts.items() if count > 1)
    if len(route) < 3 or route[0] != depot or route[-1] != depot:
        errors.append("route must start and end at the depot")
    if depot in route[1:-1]:
        errors.append("intermediate depot visits are not allowed")
    if unknown or missing or duplicates:
        errors.append("customers must be covered exactly once with no unknown nodes")
    coverage = {"expected_customers": len(expected),
                "visited_unique_customers": len(set(counts) & expected),
                "missing": missing, "duplicates": duplicates, "unknown": unknown}
    if errors:
        return {"feasible": False, "errors": errors, "coverage": coverage,
                "metrics": None, "schedule": []}

    origin_time = _timestamp(instance["departure_time_utc"])
    indexes = {key: idx for idx, key in enumerate(matrix["node_ids"])}
    clock = travel = service = waiting = Decimal(0)
    demand = sum((_decimal(nodes[key]["demand_cm3"]) for key in route[1:-1]), Decimal(0))
    capacity = _decimal(instance["vehicle_capacity_cm3"])
    capacity_excess = max(Decimal(0), demand - capacity)
    if capacity_excess:
        errors.append("vehicle capacity exceeded")
    late_stops = 0
    late_total = Decimal(0)

    def bounds(node):
        tw = node["time_window"]
        if tw is None:
            return None
        return tuple(_decimal((_timestamp(tw[key]) - origin_time).total_seconds())
                     for key in ("start_utc", "end_utc"))

    depot_window = bounds(nodes[depot])
    if depot_window and not depot_window[0] <= 0 <= depot_window[1]:
        errors.append("fixed departure is outside depot window")

    def utc(seconds):
        return (origin_time + timedelta(seconds=float(seconds))).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")

    schedule = [{"node_id": depot, "arrival_seconds": 0.0,
                 "service_start_seconds": 0.0, "departure_seconds": 0.0,
                 "arrival_utc": utc(0), "service_start_utc": utc(0),
                 "departure_utc": utc(0), "travel_from_previous_seconds": 0.0,
                 "waiting_seconds": 0.0, "service_seconds": 0.0,
                 "lateness_seconds": 0.0, "remaining_load_cm3": float(demand)}]
    load = demand
    for previous, key in zip(route, route[1:]):
        arc = _decimal(matrix["travel_seconds"][indexes[previous]][indexes[key]])
        arrival = clock + arc
        window = bounds(nodes[key])
        start = max(arrival, window[0]) if window else arrival
        late = max(Decimal(0), start - window[1]) if window else Decimal(0)
        duration = _decimal(nodes[key]["service_seconds"])
        wait = start - arrival
        travel += arc
        service += duration
        waiting += wait
        late_stops += int(late > 0)
        late_total += late
        clock = start + duration
        load -= _decimal(nodes[key]["demand_cm3"])
        schedule.append({"node_id": key, "arrival_seconds": float(arrival),
                         "service_start_seconds": float(start), "departure_seconds": float(clock),
                         "arrival_utc": utc(arrival), "service_start_utc": utc(start),
                         "departure_utc": utc(clock), "travel_from_previous_seconds": float(arc),
                         "waiting_seconds": float(wait), "service_seconds": float(duration),
                         "lateness_seconds": float(late), "remaining_load_cm3": float(load)})
    if late_stops:
        errors.append("time window violations")
    return {"feasible": not errors, "errors": errors, "coverage": coverage,
            "metrics": {"travel_seconds": float(travel), "service_seconds": float(service),
                        "waiting_seconds": float(waiting), "elapsed_seconds": float(clock),
                        "demand_cm3": float(demand), "capacity_cm3": float(capacity),
                        "capacity_utilisation": float(demand / capacity) if capacity else None,
                        "capacity_excess_cm3": float(capacity_excess),
                        "time_window_violations": late_stops, "lateness_seconds": float(late_total)},
            "schedule": schedule}


def solve_instance(
    instance,
    matrix,
    *,
    seed=42,
    iterations=5000,
    runtime_seconds=None,
    max_iterations=None,
    max_no_improvement=None,
    objective_cost_seconds=None,
):
    """Return a reproducible-budget search result and independent audit.

    ``iterations=5000`` is the legacy interface and remains the default. New
    callers can combine a wall-clock limit, an iteration limit, and a
    no-improvement limit; whichever criterion fires first stops the search.
    Timing and feasibility always use the original directed duration matrix.
    The optional ``objective_cost_seconds`` matrix changes only PyVRP's
    distance objective.

    Search failure is not a proof of infeasibility or of optimality. Integer
    durations/demands round up, capacity/latest windows round down.
    """
    validate_pair(instance, matrix)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a non-negative 32-bit integer")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    if max_iterations is None:
        max_iterations = iterations
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if runtime_seconds is not None:
        if isinstance(runtime_seconds, bool) or not isinstance(runtime_seconds, (int, float)) or runtime_seconds <= 0:
            raise ValueError("runtime_seconds must be a positive number")
    if max_no_improvement is not None:
        if isinstance(max_no_improvement, bool) or not isinstance(max_no_improvement, int) or max_no_improvement < 1:
            raise ValueError("max_no_improvement must be a positive integer")
    nodes = instance["nodes"]
    departure = _timestamp(instance["departure_time_utc"])
    objective_cost = _validate_cost_matrix(instance, matrix, objective_cost_seconds)
    demand = sum((_decimal(n["demand_cm3"]) for n in nodes), Decimal(0))
    try:
        pyvrp_version = version("pyvrp")
    except Exception:
        # Input-only failures remain auditable even in an environment where
        # the optional solver wheel has not been installed yet.
        pyvrp_version = "unavailable"
    base = {"schema_version": "cargoflow.solution.v0.1", "instance_id": instance["instance_id"],
            "solver": "PyVRP", "pyvrp_version": pyvrp_version,
            "python_version": platform.python_version(), "platform": platform.platform(),
            "seed": seed, "iteration_limit": max_iterations,
            "runtime_limit_seconds": runtime_seconds,
            "max_no_improvement": max_no_improvement,
            "stop_conditions": {"runtime_seconds": runtime_seconds,
                                "max_iterations": max_iterations,
                                "max_no_improvement": max_no_improvement},
            "model": {"vehicles": 1, "closed_tour": True, "fixed_departure": True,
                      "time_windows": "service_start",
                      "objective": "directed_travel_seconds" if objective_cost_seconds is None else "objective_cost_seconds",
                      "time_ticks_per_second": SCALE, "load_ticks_per_cm3": SCALE,
                      "rounding": "ceil duration/service/demand; floor capacity/latest window",
                      "distance_attribute": "travel-time cost proxy; NOT metres" if objective_cost_seconds is None else "objective cost; NOT metres",
                      "duration_attribute": "original directed travel seconds",
                      "learned_costs": objective_cost_seconds is not None},
            "status": "input_infeasible", "feasible": False, "route": None,
            "audit": None, "diagnostics": [], "solve_seconds": 0.0}
    if demand > _decimal(instance["vehicle_capacity_cm3"]):
        base["diagnostics"] = ["total delivery demand exceeds one vehicle's capacity"]
        return base
    bounds = []
    for node in nodes:
        tw = node["time_window"]
        if tw:
            early, late = [int((_timestamp(tw[k]) - departure).total_seconds())
                           for k in ("start_utc", "end_utc")]
            if late < 0 or (node["kind"] == "depot" and early > 0):
                base["diagnostics"] = [f"{node['id']}: time window incompatible with fixed departure"]
                return base
            bounds.append((max(0, early) * SCALE, late * SCALE))
        else:
            bounds.append(None)
    # Timing/feasibility always use the original duration values. Only the
    # distance passed to PyVRP may be replaced by a learned or preference cost.
    travel = [[_ticks(v) for v in row] for row in matrix["travel_seconds"]]
    distance = [[_ticks(v) for v in row] for row in objective_cost]
    services = [_ticks(n["service_seconds"]) for n in nodes]
    loads = [_ticks(n["demand_cm3"]) for n in nodes]
    capacity = _ticks(instance["vehicle_capacity_cm3"], floor=True)
    if sum(loads) > capacity:
        base["status"] = "rounding_infeasible"
        base["diagnostics"] = ["conservative load rounding exceeds capacity; inspect source precision"]
        return base
    # Safe nonbinding horizon for unconstrained nodes: latest window + an
    # entire worst-arc tour + all service. No invented 24h shift constraint.
    horizon = max((b[1] for b in bounds if b), default=0) + sum(services) + len(nodes) * max(map(max, travel)) + SCALE
    if max(horizon, capacity, sum(loads)) >= 2**44:
        raise ValueError("scaled data exceeds supported numerical range")
    from pyvrp import Model
    model = Model()
    depot_bounds = bounds[0] or (0, horizon)
    locations = [model.add_depot(x=nodes[0]["lng"], y=nodes[0]["lat"],
                                 tw_early=depot_bounds[0], tw_late=depot_bounds[1], name=nodes[0]["id"])]
    model.add_vehicle_type(num_available=1, capacity=capacity,
                           tw_early=0, start_late=0, tw_late=depot_bounds[1],
                           unit_distance_cost=1, unit_duration_cost=0)
    for idx, node in enumerate(nodes[1:], 1):
        early, late = bounds[idx] or (0, horizon)
        locations.append(model.add_client(x=node["lng"], y=node["lat"],
                                          delivery=loads[idx], service_duration=services[idx],
                                          tw_early=early, tw_late=late, required=True, name=node["id"]))
    for i, origin in enumerate(locations):
        for j, destination in enumerate(locations):
            model.add_edge(origin, destination, distance=distance[i][j], duration=travel[i][j])
    started = perf_counter()
    result = model.solve(
        stop=_stopping_criterion(
            max_iterations=max_iterations,
            runtime_seconds=runtime_seconds,
            max_no_improvement=max_no_improvement,
        ),
        seed=seed,
        display=False,
        collect_stats=False,
    )
    base["solve_seconds"] = perf_counter() - started
    base["iterations_run"] = result.num_iterations
    base["solver_feasible"] = result.is_feasible()
    routes = result.best.routes()
    if len(routes) == 1:
        tour = [nodes[0]["id"]] + [nodes[i]["id"] for i in routes[0].visits()] + [nodes[0]["id"]]
        base["route"] = tour
        base["audit"] = audit_route(instance, matrix, tour)
    base["feasible"] = bool(result.is_feasible() and base["audit"] and base["audit"]["feasible"])
    base["status"] = "feasible" if base["feasible"] else "no_feasible_solution_found"
    base["objective_scaled"] = int(result.cost()) if result.is_feasible() else None
    if not base["feasible"]:
        base["diagnostics"] = ["No independently verified feasible solution in this search budget; this is not an infeasibility proof."]
    return base
