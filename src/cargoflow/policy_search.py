"""Guarded multi-objective post-search for a single delivery tour.

This module deliberately reports a load-aware *energy proxy*.  The Amazon
challenge data does not contain road distance, elevation, vehicle powertrain,
speed traces, or measured fuel/energy, so physical kWh, fuel, and CO2 cannot be
identified from the contract alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import math
import random
from typing import Any, Mapping, Sequence

from cargoflow.solver import audit_route, validate_pair


@dataclass(frozen=True)
class Policy:
    """A secondary-objective policy protected by a travel-time budget."""

    name: str
    max_travel_increase: float
    time_weight: float
    energy_weight: float
    preference_weight: float = 0.0
    completion_weight: float = 0.0
    max_energy_increase: float | None = None
    raw_logits: tuple[float, float, float, float] | None = None
    switches: tuple[bool, bool, bool, bool] | None = None
    temperature: float = 1.0

    def __post_init__(self) -> None:
        values = (self.max_travel_increase, self.time_weight, self.energy_weight, self.preference_weight, self.completion_weight)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("policy values must be finite and non-negative")
        if self.max_travel_increase > 1:
            raise ValueError("max_travel_increase must be a fraction between zero and one")
        if self.max_energy_increase is not None and (
            not math.isfinite(self.max_energy_increase) or not 0 <= self.max_energy_increase <= 1
        ):
            raise ValueError("max_energy_increase must be absent or a fraction between zero and one")
        if self.time_weight + self.energy_weight + self.preference_weight + self.completion_weight <= 0:
            raise ValueError("at least one objective weight must be positive")


def policy_from_logits(
    name: str,
    max_travel_increase: float,
    logits: Sequence[float],
    switches: Sequence[bool],
    *,
    temperature: float = 1.0,
    max_energy_increase: float | None = None,
) -> Policy:
    """Create interpretable objective weights with a masked softmax.

    The canonical order is travel, delivery completion, energy proxy, and
    historical preference. Disabled outputs receive exactly zero weight.
    """

    if len(logits) != 4 or len(switches) != 4:
        raise ValueError("logits and switches must contain T, D, E, and P")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    active = [idx for idx, enabled in enumerate(switches) if enabled]
    if not active:
        raise ValueError("at least one objective switch must be enabled")
    values = [float(value) for value in logits]
    if any(not math.isfinite(values[idx]) for idx in active):
        raise ValueError("active logits must be finite")
    offset = max(values[idx] for idx in active)
    exponentials = [0.0] * 4
    for idx in active:
        exponentials[idx] = math.exp((values[idx] - offset) / temperature)
    denominator = sum(exponentials)
    weights = [value / denominator for value in exponentials]
    return Policy(
        name=name,
        max_travel_increase=max_travel_increase,
        time_weight=weights[0],
        completion_weight=weights[1],
        energy_weight=weights[2],
        preference_weight=weights[3],
        max_energy_increase=max_energy_increase,
        raw_logits=tuple(values),
        switches=tuple(bool(value) for value in switches),
        temperature=temperature,
    )


def _timestamp(value: str) -> float:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


class RouteEvaluator:
    """Fast evaluator with an exact final check delegated to ``audit_route``."""

    def __init__(self, instance: Mapping[str, Any], matrix: Mapping[str, Any], *, load_factor: float = 1.0):
        validate_pair(instance, matrix)
        self.instance = instance
        self.matrix = matrix
        self.ids = list(matrix["node_ids"])
        self.index = {node_id: idx for idx, node_id in enumerate(self.ids)}
        self.travel = matrix["travel_seconds"]
        self.depot = str(instance["depot_id"])
        self.capacity = float(instance["vehicle_capacity_cm3"])
        self.load_factor = float(load_factor)
        if not math.isfinite(self.load_factor) or self.load_factor < 0:
            raise ValueError("load_factor must be finite and non-negative")
        departure = _timestamp(str(instance["departure_time_utc"]))
        self.demand: dict[str, float] = {}
        self.service: dict[str, float] = {}
        self.windows: dict[str, tuple[float, float] | None] = {}
        for node in instance["nodes"]:
            node_id = str(node["id"])
            self.demand[node_id] = float(node["demand_cm3"])
            self.service[node_id] = float(node["service_seconds"])
            window = node["time_window"]
            self.windows[node_id] = None if window is None else (
                _timestamp(window["start_utc"]) - departure,
                _timestamp(window["end_utc"]) - departure,
            )
        self.total_demand = sum(self.demand.values())
        self.customers = set(self.ids) - {self.depot}
        depot_window = self.windows[self.depot]
        self.input_feasible = (
            sum(Decimal(str(n['demand_cm3'])) for n in instance['nodes'])
            <= Decimal(str(instance['vehicle_capacity_cm3']))
            and (depot_window is None or depot_window[0] <= 0 <= depot_window[1])
        )

    def evaluate(self, route: Sequence[str], preference_cost: Sequence[Sequence[float]] | None = None) -> dict[str, float | bool]:
        if (not self.input_feasible or len(route) != len(self.ids) + 1
            or route[0] != self.depot or route[-1] != self.depot
            or set(route[1:-1]) != self.customers):
            return {"feasible": False}
        clock = travel_total = waiting = energy = preference = completion_sum = 0.0
        remaining = self.total_demand
        for origin, destination in zip(route, route[1:]):
            i, j = self.index[origin], self.index[destination]
            duration = float(self.travel[i][j])
            # Linear load-dependent proxy: empty-vehicle travel plus an
            # additional term proportional to the remaining load fraction.
            load_fraction = remaining / self.capacity if self.capacity else 0.0
            energy += duration * (1.0 + self.load_factor * load_fraction)
            if preference_cost is not None:
                preference += float(preference_cost[i][j])
            arrival = clock + duration
            window = self.windows[destination]
            start = max(arrival, window[0]) if window else arrival
            if window and start > window[1] + 1e-9:
                return {"feasible": False}
            waiting += start - arrival
            if destination != self.depot:
                completion_sum += start + self.service[destination]
            clock = start + self.service[destination]
            travel_total += duration
            remaining -= self.demand[destination]
        return {
            "feasible": True,
            "travel_seconds": travel_total,
            "elapsed_seconds": clock,
            "waiting_seconds": waiting,
            "energy_proxy": energy,
            "preference_cost": preference,
            "delivery_completion_sum_seconds": completion_sum,
        }


def _score(metrics: Mapping[str, float | bool], reference: Mapping[str, float | bool], policy: Policy) -> float:
    terms = [(policy.time_weight, 'travel_seconds'),
             (policy.energy_weight, 'energy_proxy'),
             (policy.preference_weight, 'preference_cost'),
             (policy.completion_weight, 'delivery_completion_sum_seconds')]
    # A zero reference uses one objective unit, rather than dividing by zero
    # or making every positive candidate indistinguishable from zero cost.
    return sum(weight * float(metrics[key]) / (float(reference[key]) or 1.0)
               for weight, key in terms if weight) / sum(weight for weight, _ in terms)


def _exact_metrics(evaluator, route, preference_cost):
    """Decimal objective replay used only for seed/final acceptance."""
    D = lambda value: Decimal(str(value))
    nodes = {n['id']: n for n in evaluator.instance['nodes']}
    remaining = sum(D(n['demand_cm3']) for n in nodes.values())
    capacity = D(evaluator.instance['vehicle_capacity_cm3'])
    clock = travel = waiting = energy = preference = completion = Decimal(0)
    for a, b in zip(route, route[1:]):
        i, j = evaluator.index[a], evaluator.index[b]
        duration = D(evaluator.travel[i][j])
        energy += duration * (1 + D(evaluator.load_factor) * (remaining / capacity if capacity else 0))
        if preference_cost is not None: preference += D(preference_cost[i][j])
        arrival = clock + duration
        window = evaluator.windows[b]
        start = max(arrival, D(window[0])) if window else arrival
        clock = start + D(nodes[b]['service_seconds'])
        if b != evaluator.depot: completion += clock
        waiting += start - arrival
        travel += duration
        remaining -= D(nodes[b]['demand_cm3'])
    return dict(travel_seconds=travel, elapsed_seconds=clock, waiting_seconds=waiting,
                energy_proxy=energy, preference_cost=preference,
                delivery_completion_sum_seconds=completion)


def _validate_preference(cost, n):
    try:
        valid = (cost is not None and len(cost) == n and all(len(row) == n for row in cost)
                 and all(not isinstance(v, bool) and math.isfinite(float(v)) and float(v) >= 0
                         for row in cost for v in row))
    except (TypeError, ValueError, OverflowError): valid = False
    if not valid: raise ValueError('active preference requires a finite non-negative square preference matrix')


def _move(route: Sequence[str], rng: random.Random) -> list[str]:
    customers = list(route[1:-1])
    n = len(customers)
    kind = rng.randrange(3)
    if kind == 0:
        source, target = rng.sample(range(n), 2)
        node = customers.pop(source)
        customers.insert(target, node)
    elif kind == 1:
        left, right = rng.sample(range(n), 2)
        customers[left], customers[right] = customers[right], customers[left]
    else:
        left, right = sorted(rng.sample(range(n), 2))
        customers[left : right + 1] = reversed(customers[left : right + 1])
    return [route[0], *customers, route[-1]]


def search_policy(
    instance: Mapping[str, Any],
    matrix: Mapping[str, Any],
    seed_route: Sequence[str],
    policy: Policy,
    *,
    preference_cost: Sequence[Sequence[float]] | None = None,
    iterations: int = 50_000,
    seed: int = 42,
    load_factor: float = 1.0,
) -> dict[str, Any]:
    """Run deterministic guarded ILS with three route neighbourhoods.

    The seed route defines the operational travel budget.  Candidate tours
    must remain feasible and within that budget before secondary objectives
    are considered.  Final feasibility is replayed with the project's official
    decimal auditor.
    """

    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    evaluator = RouteEvaluator(instance, matrix, load_factor=load_factor)
    seed_route = list(seed_route)
    seed_audit = audit_route(instance, matrix, seed_route)
    if not seed_audit['feasible']:
        raise ValueError("seed route must be feasible")
    if policy.preference_weight: _validate_preference(preference_cost, len(evaluator.ids))
    else: preference_cost = None
    exact_reference = _exact_metrics(evaluator, seed_route, preference_cost)
    reference = {'feasible': True, **{k: float(v) for k, v in exact_reference.items()}}
    limit = float(reference["travel_seconds"]) * (1.0 + policy.max_travel_increase)
    energy_limit = (
        float(reference["energy_proxy"]) * (1.0 + policy.max_energy_increase)
        if policy.max_energy_increase is not None else math.inf
    )
    rng = random.Random(seed)
    current = list(seed_route)
    current_metrics = reference
    current_score = _score(current_metrics, reference, policy)
    best = list(current)
    best_metrics = dict(current_metrics)
    best_score = current_score
    accepted = feasible_seen = 0
    temperature = 0.003
    executed = iterations if len(seed_route) > 3 else 0
    for step in range(executed):
        if step and step % 1000 == 0:
            current = list(best)
            current_metrics = dict(best_metrics)
            current_score = best_score
        candidate = _move(current, rng)
        metrics = evaluator.evaluate(candidate, preference_cost)
        if (
            not metrics.get("feasible")
            or float(metrics["travel_seconds"]) > limit + 1e-9
            or float(metrics["energy_proxy"]) > energy_limit + 1e-9
        ):
            continue
        feasible_seen += 1
        score = _score(metrics, reference, policy)
        cooling = max(1e-6, temperature * (1.0 - step / iterations))
        if score < current_score or rng.random() < math.exp((current_score - score) / cooling):
            current, current_metrics, current_score = candidate, metrics, score
            accepted += 1
        if score < best_score:
            best, best_metrics, best_score = candidate, dict(metrics), score
    official = audit_route(instance, matrix, best)
    fallback_reason = None
    if not official['feasible']:
        fallback_reason = 'final feasibility replay rejected candidate'
    else:
        exact_best = _exact_metrics(evaluator, best, preference_cost)
        if exact_best['travel_seconds'] > exact_reference['travel_seconds'] * (1 + Decimal(str(policy.max_travel_increase))):
            fallback_reason = 'exact travel epsilon exceeded'
        elif policy.max_energy_increase is not None and exact_best['energy_proxy'] > exact_reference['energy_proxy'] * (1 + Decimal(str(policy.max_energy_increase))):
            fallback_reason = 'exact energy epsilon exceeded'
        elif _score(exact_best, exact_reference, policy) > _score(exact_reference, exact_reference, policy) + 1e-12:
            fallback_reason = 'exact weighted objective worse than seed'
    if fallback_reason:
        best, official, exact_best = seed_route, seed_audit, exact_reference
    best_metrics = {'feasible': True, **{k: float(v) for k, v in exact_best.items()}}
    return {
        "policy": {
            "name": policy.name,
            "max_travel_increase": policy.max_travel_increase,
            "max_energy_increase": policy.max_energy_increase,
            "weights": {
                "time": policy.time_weight,
                "energy_proxy": policy.energy_weight,
                "preference": policy.preference_weight,
                "delivery_completion": policy.completion_weight,
            },
            "logits_T_D_E_P": policy.raw_logits,
            "switches_T_D_E_P": policy.switches,
            "softmax_temperature": policy.temperature,
        },
        "route": best,
        "reference": reference,
        "metrics": best_metrics,
        "official_audit": official,
        "search": {"iterations": executed, "requested_iterations": iterations,
                   "feasible_candidates": feasible_seen, "accepted_moves": accepted,
                   "seed": seed, "fallback_reason": fallback_reason},
        "limitations": [
            "energy_proxy is comparative and is not physical fuel, kWh, or CO2",
            "preference is disabled unless an out-of-sample preference-cost matrix is supplied",
            "the seed-route travel value is a heuristic reference, not a proven global optimum",
        ],
    }


__all__ = ["Policy", "RouteEvaluator", "policy_from_logits", "search_policy"]
