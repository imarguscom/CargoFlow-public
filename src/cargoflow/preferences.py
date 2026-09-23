"""Leakage-safe historical-transition preference modelling helpers.

The model is intentionally a soft edge preference model.  It does not predict
an entire route and it never changes the duration matrix used for timing and
feasibility.  The small public helpers in this module are also useful for
auditing feature construction independently of LightGBM installation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import ceil
from typing import Any, Iterable, Mapping, Sequence


# Keep this list explicit: a review can mechanically check that no historical
# order, raw node identifier, post-delivery value, or derived route score is
# admitted to the model input.
FEATURE_NAMES = (
    "travel_seconds",
    "origin_rel_lat",
    "origin_rel_lng",
    "destination_rel_lat",
    "destination_rel_lng",
    "origin_demand_cm3",
    "destination_demand_cm3",
    "origin_service_seconds",
    "destination_service_seconds",
    "origin_window_start_remaining_seconds",
    "origin_window_end_remaining_seconds",
    "destination_window_start_remaining_seconds",
    "destination_window_end_remaining_seconds",
    "station_code",
    "weekday",
    "departure_hour",
    "route_size",
    "capacity_utilisation",
    "same_zone",
)

DEFAULT_NEGATIVE_SAMPLING_SEED = 20260907
_UINT64_MASK = (1 << 64) - 1


def _mix_uint64(value: int) -> int:
    """Stable integer mixer used by the bounded negative sampler."""

    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (value ^ (value >> 31)) & _UINT64_MASK


def _stable_edge_seed(seed: int, instance_id: str, origin_id: str, destination_id: str) -> int:
    value = int(seed) & _UINT64_MASK
    for text in (instance_id, origin_id, destination_id):
        for byte in text.encode("utf-8"):
            value = _mix_uint64(value ^ byte)
    return value


def _sample_negative_ids(
    candidates: Sequence[str],
    count: int,
    *,
    seed: int,
    instance_id: str,
    origin_id: str,
    destination_id: str,
) -> list[str]:
    """Uniformly sample without replacement using only explicit integer math."""

    pool = list(candidates)
    state = _stable_edge_seed(seed, instance_id, origin_id, destination_id)
    for index in range(count):
        state = _mix_uint64(state + index)
        remaining = len(pool) - index
        swap_index = index + state % remaining
        pool[index], pool[swap_index] = pool[swap_index], pool[index]
    return pool[:count]


class PreferenceModel:
    """Pickle-friendly wrapper carrying the categorical station mapping."""

    def __init__(self, estimator: Any, station_codes: Mapping[str, int]):
        self.estimator = estimator
        self.station_codes = dict(station_codes)

    def _frame(self, rows: Any):
        import pandas as pd

        frame = rows.copy() if hasattr(rows, "copy") else pd.DataFrame(rows, columns=FEATURE_NAMES)
        if "station_code" not in frame:
            frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
        frame["station_code"] = frame["station_code"].map(
            lambda value: self.station_codes.get(str(value), -1) if not isinstance(value, (int, float)) else value
        ).astype("int32")
        categories = getattr(getattr(self.estimator, "booster_", None), "pandas_categorical", None)
        frame["station_code"] = pd.Categorical(
            frame["station_code"], categories=categories[0] if categories else None
        )
        return frame

    def predict_proba(self, rows: Any):
        return self.estimator.predict_proba(self._frame(rows))

    def predict(self, rows: Any):
        return self.estimator.predict(self._frame(rows))

_FORBIDDEN_FEATURE_TERMS = (
    "history",
    "position",
    "next",
    "sequence",
    "route_score",
    "actual",
    "post_delivery",
    "node_id",
)


def _departure(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _relative_seconds(value: str, departure: datetime) -> float:
    parsed = _departure(value)
    return (parsed - departure).total_seconds()


def _window_values(node: Mapping[str, Any], departure: datetime) -> tuple[float, float]:
    window = node.get("time_window")
    if window is None:
        # A wide finite sentinel is easier for tree models than NaN and makes
        # the absence of a window explicit and deterministic.
        return (0.0, 1.0e9)
    return (
        _relative_seconds(window["start_utc"], departure),
        _relative_seconds(window["end_utc"], departure),
    )


def _metadata_value(metadata: Mapping[str, Any] | None, key: str, default: Any) -> Any:
    if metadata is None:
        return default
    return metadata.get(key, default)


def _same_zone(
    metadata: Mapping[str, Any] | None,
    origin_id: str,
    destination_id: str,
    depot_id: str,
) -> int:
    """Compare auxiliary stop zones without exposing the zone value itself."""

    if origin_id == depot_id or destination_id == depot_id:
        return 0
    zones = _metadata_value(metadata, "zone_by_stop", None)
    if not isinstance(zones, Mapping):
        return 0
    origin_zone = zones.get(origin_id)
    destination_zone = zones.get(destination_id)
    if not isinstance(origin_zone, str) or not origin_zone:
        return 0
    if not isinstance(destination_zone, str) or not destination_zone:
        return 0
    return int(origin_zone == destination_zone)


def edge_features(
    instance: Mapping[str, Any],
    matrix: Mapping[str, Any],
    origin_id: str,
    destination_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, float | int | str]:
    """Build one planning-time feature row for a directed edge.

    ``origin_id`` and ``destination_id`` are lookup arguments only and are
    never returned as model features.
    """

    node_ids = matrix["node_ids"]
    if origin_id not in node_ids or destination_id not in node_ids:
        raise ValueError("edge endpoints must be present in matrix.node_ids")
    if origin_id == destination_id:
        raise ValueError("self edges are not preference examples")
    nodes = {node["id"]: node for node in instance["nodes"]}
    origin = nodes[origin_id]
    destination = nodes[destination_id]
    depot = nodes[instance["depot_id"]]
    departure = _departure(instance["departure_time_utc"])
    origin_start, origin_end = _window_values(origin, departure)
    destination_start, destination_end = _window_values(destination, departure)
    origin_index = node_ids.index(origin_id)
    destination_index = node_ids.index(destination_id)
    demand = sum(float(node["demand_cm3"]) for node in instance["nodes"])
    capacity = float(instance["vehicle_capacity_cm3"])
    route_size = len(instance["nodes"]) - 1
    result: dict[str, float | int | str] = {
        "travel_seconds": float(matrix["travel_seconds"][origin_index][destination_index]),
        "origin_rel_lat": float(origin["lat"]) - float(depot["lat"]),
        "origin_rel_lng": float(origin["lng"]) - float(depot["lng"]),
        "destination_rel_lat": float(destination["lat"]) - float(depot["lat"]),
        "destination_rel_lng": float(destination["lng"]) - float(depot["lng"]),
        "origin_demand_cm3": float(origin["demand_cm3"]),
        "destination_demand_cm3": float(destination["demand_cm3"]),
        "origin_service_seconds": float(origin["service_seconds"]),
        "destination_service_seconds": float(destination["service_seconds"]),
        "origin_window_start_remaining_seconds": origin_start,
        "origin_window_end_remaining_seconds": origin_end,
        "destination_window_start_remaining_seconds": destination_start,
        "destination_window_end_remaining_seconds": destination_end,
        "station_code": str(_metadata_value(metadata, "station_code", "unknown")),
        "weekday": departure.weekday(),
        "departure_hour": departure.hour,
        "route_size": route_size,
        "capacity_utilisation": demand / capacity if capacity else 0.0,
        "same_zone": _same_zone(metadata, origin_id, destination_id, instance["depot_id"]),
    }
    if tuple(result) != FEATURE_NAMES:
        raise AssertionError("preference feature schema drifted")
    return result


def feature_names() -> tuple[str, ...]:
    return FEATURE_NAMES


def assert_feature_schema(features: Mapping[str, Any]) -> None:
    names = tuple(features)
    if names != FEATURE_NAMES:
        raise ValueError("preference features do not match the planning-time schema")
    lowered = " ".join(names).lower()
    if any(term in lowered for term in _FORBIDDEN_FEATURE_TERMS):
        raise ValueError("preference features contain a forbidden historical field")


def historical_route_from_positions(actual: Mapping[str, Any], depot_id: str) -> list[str]:
    """Convert a labelled historical position map into a closed route."""

    order = actual.get("actual") if isinstance(actual, Mapping) else None
    if not isinstance(order, Mapping):
        raise ValueError("historical record has no actual position map")
    customers = sorted(
        ((position, node_id) for node_id, position in order.items() if node_id != depot_id),
        key=lambda item: item[0],
    )
    return [depot_id, *(node_id for _, node_id in customers), depot_id]


def transition_examples(
    instance: Mapping[str, Any],
    matrix: Mapping[str, Any],
    historical_route: Sequence[str],
    *,
    metadata: Mapping[str, Any] | None = None,
    negatives_per_positive: int = 3,
    sampling_seed: int = DEFAULT_NEGATIVE_SAMPLING_SEED,
) -> list[dict[str, Any]]:
    """Create positives plus exactly three deterministic uniform negatives.

    Negative destinations are sampled without replacement from every legal
    destination, excluding the true next stop and self.  The stable seed is
    mixed with the instance and edge lookup keys; those keys never enter the
    feature row.  No full origin-destination matrix is expanded beyond the
    fixed three examples per positive transition.
    """

    if negatives_per_positive != 3:
        raise ValueError("the preference contract requires exactly three negatives per positive")
    node_ids = list(matrix["node_ids"])
    route = list(historical_route)
    if len(route) < 3 or route[0] != instance["depot_id"] or route[-1] != instance["depot_id"]:
        raise ValueError("historical_route must be a closed route")
    customers = set(node_ids) - {instance["depot_id"]}
    if set(route[1:-1]) != customers or len(route[1:-1]) != len(customers):
        raise ValueError("historical_route must cover each customer exactly once")
    examples: list[dict[str, Any]] = []
    for origin_id, destination_id in zip(route, route[1:]):
        positive = edge_features(instance, matrix, origin_id, destination_id, metadata=metadata)
        assert_feature_schema(positive)
        examples.append({"features": positive, "label": 1})
        candidates = sorted(candidate for candidate in node_ids if candidate != origin_id and candidate != destination_id)
        if len(candidates) < negatives_per_positive:
            raise ValueError("route does not have three valid negative destinations for every positive")
        negative_ids = _sample_negative_ids(
            candidates,
            negatives_per_positive,
            seed=sampling_seed,
            instance_id=str(instance.get("instance_id", "")),
            origin_id=str(origin_id),
            destination_id=str(destination_id),
        )
        for negative_id in negative_ids:
            negative = edge_features(instance, matrix, origin_id, negative_id, metadata=metadata)
            assert_feature_schema(negative)
            examples.append({"features": negative, "label": 0})
    return examples


def _predict_score(model: Any, rows: list[dict[str, Any]]) -> list[float]:
    values = [list(row[name] for name in FEATURE_NAMES) for row in rows]
    if hasattr(model, "predict_proba"):
        try:
            predictions = model.predict_proba(values)
        except (TypeError, ValueError):
            # LightGBM models trained with a categorical station field expect
            # a dataframe; tiny test doubles generally accept the plain list.
            import pandas as pd
            frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
            categories = getattr(getattr(model, "booster_", None), "pandas_categorical", None)
            frame["station_code"] = pd.Categorical(
                frame["station_code"],
                categories=categories[0] if categories else None,
            )
            predictions = model.predict_proba(frame)
        return [float(item[1]) for item in predictions]
    try:
        predictions = model.predict(values)
    except (TypeError, ValueError):
        import pandas as pd
        frame = pd.DataFrame(rows, columns=FEATURE_NAMES)
        categories = getattr(getattr(model, "booster_", None), "pandas_categorical", None)
        frame["station_code"] = pd.Categorical(
            frame["station_code"],
            categories=categories[0] if categories else None,
        )
        predictions = model.predict(frame)
    return [float(value) for value in predictions]


def preference_cost_matrix(
    instance: Mapping[str, Any],
    matrix: Mapping[str, Any],
    model: Any,
    *,
    lambda_: float,
    metadata: Mapping[str, Any] | None = None,
) -> list[list[float | int]]:
    """Turn per-origin model scores into rank-quantile soft costs.

    ``lambda_=0`` returns an exact copy of the original matrix, which makes
    the pure-PyVRP comparison a true control rather than a numerically similar
    reimplementation.
    """

    if isinstance(lambda_, bool) or not isinstance(lambda_, (int, float)) or not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between zero and one")
    original = matrix["travel_seconds"]
    if lambda_ == 0:
        return [list(row) for row in original]
    node_ids = list(matrix["node_ids"])
    result: list[list[float | int]] = [[0 for _ in node_ids] for _ in node_ids]
    for i, origin_id in enumerate(node_ids):
        destinations = [node_id for node_id in node_ids if node_id != origin_id]
        rows = [edge_features(instance, matrix, origin_id, destination_id, metadata=metadata) for destination_id in destinations]
        scores = _predict_score(model, rows)
        ranked = sorted(
            zip(destinations, scores),
            key=lambda pair: (-pair[1], pair[0]),
        )
        denominator = max(1, len(ranked))
        quantile = {destination_id: (denominator - rank) / denominator for rank, (destination_id, _) in enumerate(ranked)}
        for destination_id in destinations:
            j = node_ids.index(destination_id)
            duration = original[i][j]
            result[i][j] = float(duration) * (1.0 + float(lambda_) * (1.0 - quantile[destination_id]))
    return result


def isolate_station_days(
    candidates: Iterable[Mapping[str, Any]],
    benchmark_route_ids: set[str] | frozenset[str],
    *,
    excluded_station_days: Iterable[tuple[str, str]] = (),
    validation_fraction: float = 0.2,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], set[tuple[str, str]]]:
    """Split candidate routes by station-day, excluding protected groups.

    Each candidate must provide ``route_id``, ``station_code``, and
    ``date_YYYY_MM_DD`` metadata.  The last 20% of dates per station are
    validation groups (at least one group when a station has two or more
    groups); benchmark and caller-provided station-days are excluded from both
    training and validation.
    """

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    rows = list(candidates)
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = {}
    excluded: set[tuple[str, str]] = {
        (str(station), str(date)) for station, date in excluded_station_days
    }
    for row in rows:
        route_id = str(row["route_id"])
        station = str(row["station_code"])
        date = str(row["date_YYYY_MM_DD"])
        key = (station, date)
        if route_id in benchmark_route_ids:
            excluded.add(key)
        groups.setdefault(station, {}).setdefault(date, []).append(row)
    train: list[Mapping[str, Any]] = []
    validation: list[Mapping[str, Any]] = []
    for station in sorted(groups):
        dates = sorted(groups[station])
        eligible = [date for date in dates if (station, date) not in excluded]
        if len(eligible) > 1:
            n_validation = max(1, ceil(len(eligible) * validation_fraction))
        else:
            n_validation = 0
        validation_dates = set(eligible[-n_validation:]) if n_validation else set()
        for date in eligible:
            (validation if date in validation_dates else train).extend(groups[station][date])
    train.sort(key=lambda row: (str(row["station_code"]), str(row["date_YYYY_MM_DD"]), str(row["route_id"])))
    validation.sort(key=lambda row: (str(row["station_code"]), str(row["date_YYYY_MM_DD"]), str(row["route_id"])))
    return train, validation, excluded


def select_validation_routes(candidates: Iterable[Mapping[str, Any]], limit: int = 50) -> list[Mapping[str, Any]]:
    """Select a deterministic station/date/scale-stratified cohort.

    Stations are round-robined after each station's date and customer-count
    strata are interleaved.  This guarantees one route per available station
    before filling the remaining slots and never consults route order or
    solver-derived fields.
    """

    if limit < 1:
        return []
    by_station: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        station = str(row.get("station_code", ""))
        by_station.setdefault(station, []).append(row)

    station_queues: dict[str, list[Mapping[str, Any]]] = {}
    for station in sorted(by_station):
        by_date: dict[str, list[Mapping[str, Any]]] = {}
        for row in by_station[station]:
            by_date.setdefault(str(row.get("date_YYYY_MM_DD", "")), []).append(row)
        date_groups = {
            date: sorted(
                rows,
                key=lambda row: (
                    int(row.get("customer_count", 0)),
                    str(row.get("route_id", "")),
                ),
            )
            for date, rows in by_date.items()
        }
        queue: list[Mapping[str, Any]] = []
        dates = sorted(date_groups)
        for index in range(max((len(rows) for rows in date_groups.values()), default=0)):
            for date in dates:
                rows = date_groups[date]
                if index < len(rows):
                    queue.append(rows[index])
        station_queues[station] = queue

    selected: list[Mapping[str, Any]] = []
    pointers = {station: 0 for station in station_queues}
    while len(selected) < limit:
        added = False
        for station in sorted(station_queues):
            index = pointers[station]
            queue = station_queues[station]
            if index >= len(queue):
                continue
            selected.append(queue[index])
            pointers[station] = index + 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def select_lambda(evaluations: Iterable[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    """Select the largest-validity preference candidate by edge consistency.

    Each evaluation row must contain ``lambda``, ``feasible_routes``,
    ``median_travel_seconds``, ``time_window_violations``, and
    ``historical_edge_consistency``.  The zero row is the fail-safe control.
    Non-zero candidates are eligible only when they do not reduce feasibility,
    do not worsen median travel by more than one percent, do not add time
    window violations, and improve historical edge consistency over the zero
    control.  Ties use the smallest lambda for determinism.
    """

    rows = [dict(row) for row in evaluations]
    zero = next((row for row in rows if float(row.get("lambda", 0)) == 0.0), None)
    if zero is None:
        raise ValueError("lambda=0 control evaluation is required")
    zero_feasible = int(zero["feasible_routes"])
    zero_travel = float(zero["median_travel_seconds"])
    zero_violations = int(zero["time_window_violations"])
    eligible: list[dict[str, Any]] = []
    for row in rows:
        value = float(row["lambda"])
        if value == 0.0:
            continue
        travel_ratio = float(row["median_travel_seconds"]) / zero_travel if zero_travel else 1.0
        gain = float(row["historical_edge_consistency"]) - float(zero["historical_edge_consistency"])
        if (
            int(row["feasible_routes"]) >= zero_feasible
            and travel_ratio <= 1.01
            and int(row["time_window_violations"]) <= zero_violations
            and gain > 0.0
        ):
            row["travel_ratio_vs_zero"] = travel_ratio
            row["edge_consistency_gain_vs_zero"] = gain
            eligible.append(row)
    if not eligible:
        return 0.0, {"selected_lambda": 0.0, "reason": "no non-zero candidate passed feasibility/travel/window/consistency guards", "eligible": []}
    selected = max(eligible, key=lambda row: (float(row["edge_consistency_gain_vs_zero"]), -float(row["lambda"])))
    return float(selected["lambda"]), {"selected_lambda": float(selected["lambda"]), "reason": "maximum historical edge-consistency gain among guarded candidates", "eligible": eligible}


__all__ = [
    "FEATURE_NAMES",
    "assert_feature_schema",
    "edge_features",
    "feature_names",
    "historical_route_from_positions",
    "isolate_station_days",
    "preference_cost_matrix",
    "select_validation_routes",
    "select_lambda",
    "transition_examples",
]
