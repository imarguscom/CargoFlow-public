from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from test_convert_amazon_route import _write_fixture


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_application_inference_constructs_without_actual_sequence(tmp_path):
    pytest.importorskip("pyvrp")
    raw = _write_fixture(tmp_path / "raw")
    inputs = raw / "almrrc2021-data-training" / "model_build_inputs"
    route_id = "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77"
    output = tmp_path / "inference"
    module = _load_script("infer_application.py")
    result = module.run(
        route_id,
        inputs / "route_data.json",
        inputs / "package_data.json",
        inputs / "travel_times.json",
        output,
        iterations=10,
    )
    assert result["pure_pyvrp"] is not None
    assert (output / "instance.json").is_file()
    assert (output / "matrix.json").is_file()


def test_history_evaluation_requires_explicit_truth_path(tmp_path):
    module = _load_script("evaluate_history.py")
    route_list = tmp_path / "routes.txt"
    route_list.write_text("route-1\n")
    with pytest.raises(OSError):
        module.run(tmp_path / "missing-new_actual_sequences.json", route_list, tmp_path / "processed", tmp_path / "out.csv")


def test_application_metadata_reads_station_and_stop_zones():
    module = _load_script("infer_application.py")
    route = {
        "station_code": "S1",
        "stops": {
            "D": {"type": "Station"},
            "A": {"type": "Dropoff", "zone_id": "z1"},
            "B": {"type": "Dropoff", "zone_id": "z1"},
            "C": {"type": "Dropoff"},
        },
    }
    assert module._route_metadata(route) == {
        "station_code": "S1",
        "zone_by_stop": {"A": "z1", "B": "z1"},
    }
