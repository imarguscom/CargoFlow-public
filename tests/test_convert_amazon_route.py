from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from cargoflow.contract import validate_instance, validate_matrix


SCRIPT = Path(__file__).parents[1] / "scripts" / "convert_amazon_route.py"
SPEC = importlib.util.spec_from_file_location("convert_amazon_route", SCRIPT)
assert SPEC and SPEC.loader
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _write_fixture(root: Path) -> Path:
    inputs = root / "almrrc2021-data-training" / "model_build_inputs"
    inputs.mkdir(parents=True)
    route_id = converter.DEFAULT_ROUTE_ID
    route = {"station_code": "DLA3", "date_YYYY_MM_DD": "2018-07-27", "departure_time_utc": "16:02:10", "executor_capacity_cm3": 1000.0, "stops": {"VE": {"lat": 1.0, "lng": 2.0, "type": "Station"}, "B": {"lat": 1.1, "lng": 2.1, "type": "Dropoff"}, "A": {"lat": 1.2, "lng": 2.2, "type": "Dropoff"}}}
    packages = {"A": {"p1": {"scan_status": "DELIVERED", "time_window": {"start_time_utc": "2018-07-27 16:00:00", "end_time_utc": "2018-07-27 18:00:00"}, "planned_service_time_seconds": 2.5, "dimensions": {"depth_cm": 2, "height_cm": 3, "width_cm": 4}}}, "B": {"p2": {"scan_status": "DELIVERED", "time_window": {"start_time_utc": float("nan"), "end_time_utc": float("nan")}, "planned_service_time_seconds": 1, "dimensions": {"depth_cm": 1, "height_cm": 2, "width_cm": 3}}, "p3": {"scan_status": "DELIVERED", "time_window": {"start_time_utc": "2018-07-27 16:30:00", "end_time_utc": "2018-07-27 19:00:00"}, "planned_service_time_seconds": 1, "dimensions": {"depth_cm": 1, "height_cm": 1, "width_cm": 1}}}}
    travel = {"VE": {"VE": 0, "A": 2.2, "B": 3.3}, "A": {"VE": 2.1, "A": 0, "B": 4.4}, "B": {"VE": 3.1, "A": 4.1, "B": 0}}
    actual = {"actual": {"VE": 0, "A": 2, "B": 1}}
    for name, value in (("route_data", {route_id: route}), ("package_data", {route_id: packages}), ("travel_times", {route_id: travel}), ("actual_sequences", {route_id: actual})):
        (inputs / f"{name}.json").write_text(json.dumps(value, allow_nan=True), encoding="utf-8")
    return inputs.parent.parent


def test_convert_fixture_and_validators(tmp_path, capsys):
    raw = _write_fixture(tmp_path / "raw")
    output = tmp_path / "processed"
    converter.convert(raw, output)
    instance = json.loads((output / "instance.json").read_text())
    matrix = json.loads((output / "matrix.json").read_text())
    assert validate_instance(instance) is True
    assert validate_matrix(matrix) is True
    assert [node["id"] for node in instance["nodes"]] == ["VE", "A", "B"]
    assert instance["nodes"][1]["demand_cm3"] == 24
    assert instance["nodes"][1]["service_seconds"] == 2.5
    assert instance["nodes"][1]["time_window"] == {"start_utc": "2018-07-27T16:00:00Z", "end_utc": "2018-07-27T18:00:00Z"}
    assert instance["nodes"][2]["time_window"] == {"start_utc": "2018-07-27T16:30:00Z", "end_utc": "2018-07-27T19:00:00Z"}
    assert matrix["travel_seconds"][0][1] == 2.2
    assert matrix["travel_seconds"][1][0] == 2.1
    assert "actual" not in instance and "actual" not in matrix
    assert sorted(path.name for path in output.iterdir()) == ["instance.json", "matrix.json"]
    assert capsys.readouterr().out.splitlines() == [
        f"route_id={converter.DEFAULT_ROUTE_ID}",
        "nodes=3 customers=2",
        "matrix=3x3",
        "customers_with_time_window=2",
    ]


def test_converter_rejects_empty_time_window_intersection(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    package_path = raw / "almrrc2021-data-training" / "model_build_inputs" / "package_data.json"
    packages = json.loads(package_path.read_text())
    route_packages = packages[converter.DEFAULT_ROUTE_ID]["B"]
    route_packages["p2"]["time_window"] = {"start_time_utc": "2018-07-27 16:00:00", "end_time_utc": "2018-07-27 17:00:00"}
    route_packages["p3"]["time_window"] = {"start_time_utc": "2018-07-27 18:00:00", "end_time_utc": "2018-07-27 19:00:00"}
    package_path.write_text(json.dumps(packages), encoding="utf-8")
    with pytest.raises(converter.ConversionError, match="empty time-window intersection"):
        converter.convert(raw, tmp_path / "processed")


def test_validator_rejects_non_finite_matrix_value():
    with pytest.raises(ValueError, match="finite"):
        validate_matrix({"schema_version": "cargoflow.matrix.v0.1", "instance_id": "x", "node_ids": ["VE"], "travel_seconds": [[float("nan")]]})


def test_converter_discovers_non_ve_station_depot(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    inputs = raw / "almrrc2021-data-training" / "model_build_inputs"
    route_path = inputs / "route_data.json"
    routes = json.loads(route_path.read_text())
    route = routes[converter.DEFAULT_ROUTE_ID]
    route["stops"]["D0"] = route["stops"].pop("VE")
    route_path.write_text(json.dumps(routes))

    travel_path = inputs / "travel_times.json"
    travel_data = json.loads(travel_path.read_text())
    travel = travel_data[converter.DEFAULT_ROUTE_ID]
    travel["D0"] = travel.pop("VE")
    for row in travel.values():
        row["D0"] = row.pop("VE")
    travel_path.write_text(json.dumps(travel_data))

    actual_path = inputs / "actual_sequences.json"
    actual_data = json.loads(actual_path.read_text())
    order = actual_data[converter.DEFAULT_ROUTE_ID]["actual"]
    order["D0"] = order.pop("VE")
    actual_path.write_text(json.dumps(actual_data))

    output = tmp_path / "processed"
    converter.convert(raw, output)
    instance = json.loads((output / "instance.json").read_text())
    matrix = json.loads((output / "matrix.json").read_text())
    assert instance["depot_id"] == "D0"
    assert instance["nodes"][0]["id"] == "D0"
    assert matrix["node_ids"][0] == "D0"


def test_converter_normalises_source_self_travel_to_zero(tmp_path):
    raw = _write_fixture(tmp_path / "raw")
    travel_path = raw / "almrrc2021-data-training" / "model_build_inputs" / "travel_times.json"
    travel_data = json.loads(travel_path.read_text())
    travel_data[converter.DEFAULT_ROUTE_ID]["A"]["A"] = 0.2
    travel_path.write_text(json.dumps(travel_data))
    output = tmp_path / "processed"
    converter.convert(raw, output)
    matrix = json.loads((output / "matrix.json").read_text())
    assert matrix["travel_seconds"][1][1] == 0
