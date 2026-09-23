"""Small exhaustive oracle, constraint failures, and the converter boundary."""
from copy import deepcopy
from itertools import permutations
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cargoflow.solver import audit_route, solve_instance
from test_convert_amazon_route import _write_fixture, converter


def fixture():
    nodes = [{"id": key, "kind": "depot" if key == "D" else "customer",
              "lat": 0, "lng": idx, "demand_cm3": 0 if key == "D" else 1.125,
              "service_seconds": 0 if key == "D" else 0.125, "time_window": None}
             for idx, key in enumerate(["D", "A", "B", "C"])]
    instance = {"schema_version": "cargoflow.instance.v0.1", "instance_id": "test",
                "source": "amazon_lmrrc2021", "depot_id": "D", "vehicle_capacity_cm3": 4,
                "departure_time_utc": "2018-07-27T16:00:00Z", "nodes": nodes}
    matrix = {"schema_version": "cargoflow.matrix.v0.1", "instance_id": "test",
              "node_ids": ["D", "A", "B", "C"],
              "travel_seconds": [[0, 1.125, 9, 9], [9, 0, 1.125, 9],
                                 [9, 9, 0, 1.125], [1.125, 9, 9, 0]]}
    return instance, matrix


def test_asymmetric_optimum_matches_exhaustive_oracle_and_repeats():
    instance, matrix = fixture()
    candidates = [audit_route(instance, matrix, ["D", *order, "D"])
                  for order in permutations(["A", "B", "C"])]
    optimum = min(a["metrics"]["travel_seconds"] for a in candidates if a["feasible"])
    result = solve_instance(instance, matrix, iterations=100)
    repeated = solve_instance(instance, matrix, iterations=100)
    assert result["feasible"]
    assert result["route"] == repeated["route"] == ["D", "A", "B", "C", "D"]
    assert result["audit"]["metrics"]["travel_seconds"] == optimum == 4.5
    assert result["objective_scaled"] == 4500
    assert result["audit"]["metrics"]["service_seconds"] == 0.375
    assert result["audit"]["metrics"]["elapsed_seconds"] == 4.875
    assert result["audit"]["schedule"][-1]["remaining_load_cm3"] == 0


def test_waiting_and_service_start_window_not_completion():
    instance, matrix = fixture()
    instance["nodes"][1]["time_window"] = {"start_utc": "2018-07-27T16:00:10Z", "end_utc": "2018-07-27T16:00:10Z"}
    result = solve_instance(instance, matrix, iterations=100)
    assert result["feasible"]
    a = result["audit"]["schedule"][1]
    assert a["service_start_seconds"] == 10
    assert a["departure_seconds"] == 10.125
    assert a["waiting_seconds"] == 8.875
    # Objective excludes waiting and service, despite both affecting time.
    assert result["objective_scaled"] == 4500


def test_impossible_capacity_has_no_fake_solution():
    instance, matrix = fixture()
    instance["vehicle_capacity_cm3"] = 3
    result = solve_instance(instance, matrix)
    assert result["status"] == "input_infeasible"
    assert result["route"] is None
    assert not result["feasible"]


def test_impossible_window_search_failure_is_not_a_proof():
    instance, matrix = fixture()
    instance["nodes"][1]["time_window"] = {"start_utc": "2018-07-27T16:00:00Z", "end_utc": "2018-07-27T16:00:00Z"}
    result = solve_instance(instance, matrix, iterations=100)
    assert not result["feasible"]
    assert result["status"] == "no_feasible_solution_found"


@pytest.mark.parametrize("route", [["D", "A", "B", "D"], ["D", "A", "A", "B", "C", "D"],
                                   ["D", "A", "B", "C", "X", "D"], ["A", "B", "C", "D"],
                                   ["D", "A", "D", "B", "C", "D"]])
def test_audit_rejects_bad_coverage_or_depot(route):
    instance, matrix = fixture()
    assert not audit_route(instance, matrix, route)["feasible"]


def test_audit_detects_lateness_capacity_and_includes_return():
    instance, matrix = fixture()
    instance["vehicle_capacity_cm3"] = 1
    instance["nodes"][0]["time_window"] = {"start_utc": "2018-07-27T16:00:00Z", "end_utc": "2018-07-27T16:00:04Z"}
    audit = audit_route(instance, matrix, ["D", "A", "B", "C", "D"])
    assert not audit["feasible"]
    assert audit["metrics"]["capacity_excess_cm3"] == 2.375
    assert audit["metrics"]["time_window_violations"] == 1
    assert audit["metrics"]["lateness_seconds"] == 0.875


@pytest.mark.parametrize("field", ["instance_id", "node_ids"])
def test_pair_mismatch_rejected(field):
    instance, matrix = fixture()
    matrix[field] = "wrong" if field == "instance_id" else ["D", "B", "A", "C"]
    with pytest.raises(ValueError, match="instance_id|node order"):
        solve_instance(instance, matrix)


def test_past_window_and_precision_guard():
    instance, matrix = fixture()
    instance["nodes"][1]["time_window"] = {"start_utc": "2018-07-26T16:00:00Z", "end_utc": "2018-07-26T17:00:00Z"}
    assert solve_instance(instance, matrix)["status"] == "input_infeasible"
    instance, matrix = fixture()
    instance["vehicle_capacity_cm3"] = 3.3751
    instance["nodes"][1]["demand_cm3"] = 1.1251
    assert solve_instance(instance, matrix)["status"] == "rounding_infeasible"


def test_converter_output_runs_without_schema_changes(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    out = tmp_path / "processed"
    converter.convert(raw, out)
    instance = json.loads((out / "instance.json").read_text())
    matrix = json.loads((out / "matrix.json").read_text())
    before = deepcopy((instance, matrix))
    result = solve_instance(instance, matrix, iterations=100)
    assert result["feasible"]
    assert result["audit"]["coverage"]["visited_unique_customers"] == 2
    assert (instance, matrix) == before


def test_cli_saved_result_audit_and_current_input_replay(tmp_path):
    instance, matrix = fixture()
    for name, value in (("instance", instance), ("matrix", matrix)):
        (tmp_path / f"{name}.json").write_text(json.dumps(value))
    script = Path(__file__).parents[1] / "scripts" / "solve_route.py"
    command = [sys.executable, str(script), "--instance", str(tmp_path / "instance.json"),
               "--matrix", str(tmp_path / "matrix.json"), "--output-dir", str(tmp_path / "run"),
               "--iterations", "100"]
    assert subprocess.run(command, capture_output=True).returncode == 0
    solution = tmp_path / "run" / "solution.json"
    audit_command = command + ["--check-solution", str(solution)]
    assert subprocess.run(audit_command, capture_output=True).returncode == 0
    saved = json.loads(solution.read_text())
    original_route = saved["route"]
    saved["route"] = ["D", "A", "A", "C", "D"]
    solution.write_text(json.dumps(saved))
    assert subprocess.run(audit_command, capture_output=True).returncode == 2
    assert not json.loads((tmp_path / "run" / "audit.json").read_text())["feasible"]
    saved["route"] = original_route
    solution.write_text(json.dumps(saved))
    matrix["travel_seconds"][0][1] = 10
    (tmp_path / "matrix.json").write_text(json.dumps(matrix))
    replay = subprocess.run(audit_command, capture_output=True, text=True)
    assert replay.returncode == 0
    updated = json.loads((tmp_path / "run" / "audit.json").read_text())
    assert updated["audit"]["metrics"]["travel_seconds"] == 13.375
