"""Deterministic, constraint-agnostic route construction heuristics.

The two constructors in this module deliberately use only the frozen route
instance and the directed travel-time matrix.  They return a closed tour;
``audit_route`` remains the single source of truth for feasibility and route
metrics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

from cargoflow.solver import validate_pair


def _seconds(value: str) -> Decimal:
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return Decimal(str(parsed.timestamp()))


def _window_seconds(node: dict[str, Any], departure: Decimal) -> tuple[Decimal, Decimal] | None:
    window = node.get("time_window")
    if window is None:
        return None
    return (
        _seconds(window["start_utc"]) - departure,
        _seconds(window["end_utc"]) - departure,
    )


def _matrix_index(matrix: dict[str, Any]) -> dict[str, int]:
    return {node_id: index for index, node_id in enumerate(matrix["node_ids"])}


def nearest_neighbor(instance: dict[str, Any], matrix: dict[str, Any]) -> list[str]:
    """Construct a nearest-neighbour closed tour from directed durations.

    At every step the smallest *outgoing* duration from the current node is
    selected.  Node IDs are the deterministic tie-breaker.
    """

    validate_pair(instance, matrix)
    depot = instance["depot_id"]
    nodes = {node["id"]: node for node in instance["nodes"]}
    index = _matrix_index(matrix)
    unvisited = set(nodes) - {depot}
    route = [depot]
    while unvisited:
        current = route[-1]
        current_index = index[current]
        next_node = min(
            unvisited,
            key=lambda node_id: (matrix["travel_seconds"][current_index][index[node_id]], node_id),
        )
        route.append(next_node)
        unvisited.remove(next_node)
    route.append(depot)
    return route


def deadline_greedy(instance: dict[str, Any], matrix: dict[str, Any]) -> list[str]:
    """Construct a deterministic earliest-deadline, minimum-lateness tour.

    Candidates that can start service by their latest service-start time are
    ordered by deadline, travel-plus-waiting, and ID.  If none can be on time,
    candidates are ordered by projected lateness, travel-plus-waiting, and ID.
    Service time is included in the evolving clock but not in the candidate's
    travel-plus-waiting tie-breaker.
    """

    validate_pair(instance, matrix)
    depot = instance["depot_id"]
    nodes = {node["id"]: node for node in instance["nodes"]}
    index = _matrix_index(matrix)
    departure = _seconds(instance["departure_time_utc"])
    clock = Decimal(0)
    unvisited = set(nodes) - {depot}
    route = [depot]

    while unvisited:
        current = route[-1]
        current_index = index[current]
        candidates: list[tuple[str, Decimal, Decimal, Decimal, Decimal]] = []
        for node_id in unvisited:
            node = nodes[node_id]
            travel = Decimal(str(matrix["travel_seconds"][current_index][index[node_id]]))
            arrival = clock + travel
            window = _window_seconds(node, departure)
            start = max(arrival, window[0]) if window else arrival
            wait_plus_travel = start - clock
            lateness = max(Decimal(0), start - window[1]) if window else Decimal(0)
            deadline = window[1] if window else Decimal("Infinity")
            candidates.append((node_id, lateness, deadline, wait_plus_travel, travel))

        on_time = [candidate for candidate in candidates if candidate[1] == 0]
        if on_time:
            node_id, *_ = min(on_time, key=lambda candidate: (candidate[2], candidate[3], candidate[0]))
        else:
            node_id, *_ = min(candidates, key=lambda candidate: (candidate[1], candidate[3], candidate[0]))
        selected = nodes[node_id]
        travel = Decimal(str(matrix["travel_seconds"][current_index][index[node_id]]))
        arrival = clock + travel
        window = _window_seconds(selected, departure)
        start = max(arrival, window[0]) if window else arrival
        clock = start + Decimal(str(selected["service_seconds"]))
        route.append(node_id)
        unvisited.remove(node_id)

    route.append(depot)
    return route


__all__ = ["deadline_greedy", "nearest_neighbor"]
