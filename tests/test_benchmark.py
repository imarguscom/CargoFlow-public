from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from test_convert_amazon_route import _write_fixture, converter


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


selector = _load_script("select_benchmark_routes")
benchmark = _load_script("run_benchmark")


def _route(station: str, customers: int):
    stops = {"D": {"type": "Station"}}
    stops.update({f"C{index}": {"type": "Dropoff"} for index in range(customers)})
    return {
        "station_code": station,
        "executor_capacity_cm3": 1000,
        "date_YYYY_MM_DD": "2018-01-01",
        "departure_time_utc": "12:00:00",
        "stops": stops,
    }


def test_selection_is_deterministic_and_covers_every_station():
    routes = {
        f"route-{station}-{count}": _route(station, count)
        for station in ("A", "B", "C")
        for count in range(1, 5)
    }
    selected, cuts, eligible = selector.select_routes(routes, sample_size=6, seed=7)
    reordered = dict(reversed(list(routes.items())))
    repeated, repeated_cuts, _ = selector.select_routes(reordered, sample_size=6, seed=7)
    assert selected == repeated
    assert cuts == repeated_cuts
    assert eligible == 12
    assert len({item.route_id for item in selected}) == 6
    assert {item.station_code for item in selected} == {"A", "B", "C"}
    assert {station: sum(item.station_code == station for item in selected) for station in "ABC"} == {"A": 2, "B": 2, "C": 2}


def test_selection_rejects_sample_too_small_for_station_coverage():
    routes = {f"route-{station}": _route(station, 2) for station in "ABC"}
    with pytest.raises(ValueError, match="cover every station"):
        selector.select_routes(routes, sample_size=2)


def test_two_route_benchmark_writes_results_summary_and_resumes(tmp_path, monkeypatch):
    raw = _write_fixture(tmp_path / "raw")
    inputs = raw / "almrrc2021-data-training" / "model_build_inputs"
    first = converter.DEFAULT_ROUTE_ID
    second = "RouteID_second"
    for filename in benchmark.RAW_FILES.values():
        path = inputs / filename
        records = json.loads(path.read_text())
        records[second] = deepcopy(records[first])
        path.write_text(json.dumps(records, allow_nan=True))
    route_list = tmp_path / "routes.txt"
    route_list.write_text(f"{first}\n{second}\n")
    results = benchmark.run(
        raw,
        route_list,
        tmp_path / "processed",
        tmp_path / "outputs",
        seed=42,
        iterations=100,
    )
    assert len(results) == 2
    assert all(row["feasible"] is True for row in results)
    assert (tmp_path / "outputs" / "results.csv").is_file()
    summary = json.loads((tmp_path / "outputs" / "summary.json").read_text())
    assert summary["selected_routes"] == 2
    assert summary["solver_feasible_routes"] == 2
    assert summary["status_counts"] == {"feasible": 2}
    assert summary["solve_seconds_total"] >= 0
    assert summary["solve_seconds_median"] >= 0
    assert "wall_seconds" not in summary

    monkeypatch.setattr(
        benchmark,
        "solve_instance",
        lambda *_args, **_kwargs: pytest.fail("cached solutions should not be solved again"),
    )
    resumed = benchmark.run(
        raw,
        route_list,
        tmp_path / "processed",
        tmp_path / "outputs",
        seed=42,
        iterations=100,
        resume=True,
    )
    assert len(resumed) == 2
    assert all(row["feasible"] is True for row in resumed)
    with pytest.raises(ValueError, match="configuration differs"):
        benchmark.run(
            raw,
            route_list,
            tmp_path / "processed",
            tmp_path / "outputs",
            seed=43,
            iterations=100,
            resume=True,
        )
