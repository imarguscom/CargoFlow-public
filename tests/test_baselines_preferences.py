from __future__ import annotations

import sys

import pytest

from cargoflow.baselines import deadline_greedy, nearest_neighbor
from cargoflow.preferences import (
    FEATURE_NAMES,
    edge_features,
    isolate_station_days,
    preference_cost_matrix,
    _sample_negative_ids,
    select_lambda,
    select_validation_routes,
    transition_examples,
)


def fixture():
    ids = ["D", "A", "B", "C", "E", "F"]
    nodes = [{"id": key, "kind": "depot" if key == "D" else "customer",
              "lat": 0, "lng": index, "demand_cm3": 0 if key == "D" else 1,
              "service_seconds": 0 if key == "D" else 1, "time_window": None}
             for index, key in enumerate(ids)]
    instance = {"schema_version": "cargoflow.instance.v0.1", "instance_id": "test",
                "source": "amazon_lmrrc2021", "depot_id": "D", "vehicle_capacity_cm3": 5,
                "departure_time_utc": "2018-07-27T16:00:00Z", "nodes": nodes}
    matrix = {"schema_version": "cargoflow.matrix.v0.1", "instance_id": "test",
              "node_ids": ids,
              "travel_seconds": [[0 if left == right else 1 for right in ids] for left in ids]}
    return instance, matrix


def test_baselines_are_deterministic_and_closed():
    instance, matrix = fixture()
    expected = ["D", "A", "B", "C", "E", "F", "D"]
    assert nearest_neighbor(instance, matrix) == expected
    assert nearest_neighbor(instance, matrix) == expected
    assert deadline_greedy(instance, matrix) == expected


def test_features_have_no_history_or_raw_identifier():
    instance, matrix = fixture()
    features = edge_features(instance, matrix, "D", "A", metadata={"station_code": "S1", "zone_by_stop": {"A": "z1", "B": "z1"}})
    assert tuple(features) == FEATURE_NAMES
    assert not any("id" in name.lower() or "next" in name.lower() for name in features)
    assert features["station_code"] == "S1"


def test_same_zone_uses_only_auxiliary_stop_metadata():
    instance, matrix = fixture()
    metadata = {"zone_by_stop": {"A": "z1", "B": "z1", "C": "z2"}}
    assert edge_features(instance, matrix, "A", "B", metadata=metadata)["same_zone"] == 1
    assert edge_features(instance, matrix, "B", "C", metadata=metadata)["same_zone"] == 0
    assert edge_features(instance, matrix, "C", "E", metadata=metadata)["same_zone"] == 0
    assert edge_features(instance, matrix, "D", "A", metadata={"zone_by_stop": {"D": "z1", "A": "z1"}})["same_zone"] == 0


def test_fixed_three_negatives_per_positive():
    instance, matrix = fixture()
    route = ["D", "A", "B", "C", "E", "F", "D"]
    examples = transition_examples(instance, matrix, route)
    repeated = transition_examples(instance, matrix, route)
    assert examples == repeated
    assert len(examples) == 24
    assert sum(example["label"] for example in examples) == 6
    # For D -> A the legal destinations are B,C,E,F; the sampled set is not
    # the dictionary-order prefix B,C,E.
    sampled = {example["features"]["destination_rel_lng"] for example in examples[1:4]}
    assert sampled != {2.0, 3.0, 4.0}


def test_negative_sampler_uses_stable_integer_mixing():
    candidates = ["A", "B", "C", "D", "E", "F"]
    first = _sample_negative_ids(candidates, 3, seed=20260907, instance_id="i", origin_id="o", destination_id="d")
    second = _sample_negative_ids(candidates, 3, seed=20260907, instance_id="i", origin_id="o", destination_id="d")
    assert first == second
    assert len(set(first)) == 3


class _Model:
    def predict_proba(self, rows):
        return [[0.2, 0.8] for _ in rows]


def test_lambda_zero_is_exact_original_and_nonzero_changes_cost():
    instance, matrix = fixture()
    model = _Model()
    assert preference_cost_matrix(instance, matrix, model, lambda_=0) == matrix["travel_seconds"]
    assert preference_cost_matrix(instance, matrix, model, lambda_=0.05) != matrix["travel_seconds"]


def test_station_day_isolation_excludes_benchmark_group_and_uses_late_validation():
    rows = [
        {"route_id": "r1", "station_code": "S", "date_YYYY_MM_DD": "2020-01-01"},
        {"route_id": "r2", "station_code": "S", "date_YYYY_MM_DD": "2020-01-02"},
        {"route_id": "r3", "station_code": "S", "date_YYYY_MM_DD": "2020-01-03"},
        {"route_id": "r4", "station_code": "S", "date_YYYY_MM_DD": "2020-01-04"},
    ]
    train, validation, excluded = isolate_station_days(rows, {"r2"})
    assert {row["route_id"] for row in train} == {"r1", "r3"}
    assert {row["route_id"] for row in validation} == {"r4"}
    assert excluded == {("S", "2020-01-02")}
    train_groups = {(row["station_code"], row["date_YYYY_MM_DD"]) for row in train}
    validation_groups = {(row["station_code"], row["date_YYYY_MM_DD"]) for row in validation}
    assert train_groups.isdisjoint(validation_groups)


def test_validation_selection_covers_all_stations_deterministically():
    rows = [
        {"route_id": f"{station}-{index}", "station_code": station,
         "date_YYYY_MM_DD": f"2020-01-{index + 1:02d}", "customer_count": index + 1}
        for station in [f"S{index:02d}" for index in range(17)]
        for index in range(4)
    ]
    first = select_validation_routes(rows, limit=50)
    second = select_validation_routes(list(reversed(rows)), limit=50)
    assert first == second
    assert len(first) == 50
    assert {row["station_code"] for row in first} == {f"S{index:02d}" for index in range(17)}


def test_lambda_selection_falls_back_when_guards_fail():
    selected, report = select_lambda([
        {"lambda": 0, "feasible_routes": 10, "median_travel_seconds": 100, "time_window_violations": 1, "historical_edge_consistency": 0.2},
        {"lambda": 0.05, "feasible_routes": 9, "median_travel_seconds": 100, "time_window_violations": 0, "historical_edge_consistency": 0.9},
    ])
    assert selected == 0
    assert report["eligible"] == []


def test_lambda_selection_falls_back_when_consistency_does_not_improve():
    selected, report = select_lambda([
        {"lambda": 0, "feasible_routes": 10, "median_travel_seconds": 100, "time_window_violations": 1, "historical_edge_consistency": 0.8},
        {"lambda": 0.05, "feasible_routes": 10, "median_travel_seconds": 100, "time_window_violations": 1, "historical_edge_consistency": 0.79},
    ])
    assert selected == 0
    assert report["eligible"] == []


def test_lambda_zero_preserves_same_seed_route_and_audit():
    pytest.importorskip("pyvrp")
    from cargoflow.solver import audit_route, solve_instance

    instance, matrix = fixture()
    plain = solve_instance(instance, matrix, seed=42, iterations=20)
    objective = preference_cost_matrix(instance, matrix, _Model(), lambda_=0.0)
    controlled = solve_instance(instance, matrix, seed=42, iterations=20, objective_cost_seconds=objective)
    assert plain["route"] == controlled["route"]
    assert audit_route(instance, matrix, plain["route"]) == audit_route(instance, matrix, controlled["route"])
