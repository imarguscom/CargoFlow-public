from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from test_convert_amazon_route import _write_fixture


def _load_comparison():
    path = Path(__file__).parents[1] / "scripts" / "run_comparison.py"
    spec = importlib.util.spec_from_file_location("run_comparison_resilience", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mid_route_failure_replaces_route_with_complete_error_rows(tmp_path, monkeypatch):
    raw = _write_fixture(tmp_path / "raw")
    route_id = "RouteID_00143bdd-0a6b-49ec-bb35-36593d303e77"
    route_list = tmp_path / "routes.txt"
    route_list.write_text(route_id + "\n")
    budgets = tmp_path / "budgets.json"
    budgets.write_text(json.dumps({
        "reference": {"iterations": 5, "seeds": [42]},
        "fast": {"iterations": 5, "seeds": [42, 43, 44]},
        "balanced": {"iterations": 5, "seeds": [42, 43, 44]},
    }))
    module = _load_comparison()

    def fail(*args, **kwargs):
        raise RuntimeError("synthetic mid-route failure")

    monkeypatch.setattr(module, "solve_instance", fail)
    output = tmp_path / "comparison"
    summary = module.run(
        raw,
        route_list,
        tmp_path / "processed",
        output,
        budgets_path=budgets,
    )
    rows = list(csv.DictReader((output / "comparison.csv").open()))
    records = [json.loads(line) for line in (output / "solutions.jsonl").read_text().splitlines()]
    assert summary["rows"] == 10
    assert len(rows) == 10
    assert len(records) == 10
    assert len({(row["route_id"], row["method"], row["config"], row["seed"]) for row in rows}) == 10
    assert {row["status"] for row in rows} == {"input_error"}
    assert all(row["error"] == "synthetic mid-route failure" for row in rows)
