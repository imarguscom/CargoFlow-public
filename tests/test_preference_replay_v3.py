from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _load_runner():
    path = Path(__file__).parents[1] / "scripts" / "run_preference_replay_v3.py"
    spec = importlib.util.spec_from_file_location("preference_replay_v3_tests", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(tmp_path: Path) -> argparse.Namespace:
    route_ids = [f"holdout-{index:02d}" for index in range(13)]
    holdout = {
        "name": "application_replay_holdout_13",
        "source": "application-format holdout",
        "route_ids": route_ids,
        "station_days": [[f"S{index:02d}", f"2020-01-{index + 1:02d}"] for index in range(12)],
    }
    holdout_path = tmp_path / "holdout.json"
    holdout_path.write_text(json.dumps(holdout), encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]\n", encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.txt"
    benchmark_path.write_text("benchmark\n", encoding="utf-8")
    budgets_path = tmp_path / "budgets.json"
    budgets_path.write_text("{}\n", encoding="utf-8")
    preference_path = tmp_path / "preference.json"
    preference_path.write_text(json.dumps({"lambda_candidates": [0.0], "negative_sampling_seed": 1}), encoding="utf-8")
    return argparse.Namespace(
        manifest=manifest_path,
        benchmark_routes=benchmark_path,
        holdout_config=holdout_path,
        raw_dir=tmp_path / "raw",
        application_route_list=None,
        application_route=tmp_path / "new_route.json",
        application_package=tmp_path / "new_package.json",
        application_travel=tmp_path / "new_travel.json",
        application_actual=tmp_path / "actual.json",
        budgets=budgets_path,
        preference_config=preference_path,
        output_dir=tmp_path / "run",
    )


def _patch_pipeline(monkeypatch, runner, *, comparison_error: bool):
    validation_rows = [
        {
            "route_id": f"validation-{index:02d}",
            "station_code": f"S{index % 17:02d}",
            "date_YYYY_MM_DD": f"2020-02-{index + 1:02d}",
        }
        for index in range(50)
    ]
    expected_counts = {
        "manifest": 6112,
        "train": 3791,
        "validation_pool": 1302,
        "excluded": 1019,
        "benchmark_station_days": 46,
        "application_holdout_station_days": 12,
        "protected_station_days": 57,
    }
    monkeypatch.setattr(runner, "_git_state", lambda: {"commit": "test", "clean": True})
    monkeypatch.setattr(
        runner,
        "_split_manifest",
        lambda *_args, **_kwargs: (
            [{"route_id": "train"}],
            validation_rows,
            {"rows": [], "counts": expected_counts, "asserts": {"all": True}},
        ),
    )
    monkeypatch.setattr(runner, "select_validation_routes", lambda rows, limit=50: list(rows)[:limit])
    monkeypatch.setattr(runner, "_fit_and_save", lambda *_args, **_kwargs: object())

    def fake_validation_rows(_model, rows, lambdas, *, seed):
        return [
            {
                "route_id": row["route_id"],
                "lambda": lambda_,
                "historical_route": ["depot", "customer", "depot"],
                "route": ["depot", "customer", "depot"],
                "audit": {"feasible": True, "metrics": {"travel_seconds": 1, "time_window_violations": 0}},
                "solution": {"seed": seed},
            }
            for lambda_ in lambdas
            for row in rows
        ]

    monkeypatch.setattr(runner, "_validation_solution_rows", fake_validation_rows)
    monkeypatch.setattr(runner, "_aggregate_validation", lambda *_args, **_kwargs: [{
        "lambda": 0.0,
        "feasible_routes": 50,
        "median_travel_seconds": 1,
        "time_window_violations": 0,
        "historical_edge_consistency": 1,
    }])
    monkeypatch.setattr(runner, "select_lambda", lambda _rows: (0.0, {"selected_lambda": 0.0}))

    def comparison_run(*_args, **_kwargs):
        if comparison_error:
            raise RuntimeError("synthetic comparison failure")
        return {"rows": 500}

    monkeypatch.setattr(runner, "_load_script", lambda *_args, **_kwargs: SimpleNamespace(run=comparison_run))


def test_comparison_failure_marks_run_failed_and_never_complete(tmp_path, monkeypatch):
    runner = _load_runner()
    args = _args(tmp_path)
    _patch_pipeline(monkeypatch, runner, comparison_error=True)

    with pytest.raises(RuntimeError, match="synthetic comparison failure"):
        runner.run(args)

    record = json.loads((args.output_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "failed"
    assert record["status"] != "complete"
    assert "synthetic comparison failure" in record["error"]


def test_success_marks_complete_only_after_comparison(tmp_path, monkeypatch):
    runner = _load_runner()
    args = _args(tmp_path)
    _patch_pipeline(monkeypatch, runner, comparison_error=False)

    result = runner.run(args)

    assert result["status"] == "complete"
    record = json.loads((args.output_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "complete"
    assert record["comparison_summary"] == {"rows": 500}
    assert record["replay_decision"] == "not_run_gate_zero"
    assert (args.output_dir / "application_route_ids.txt").read_text(encoding="utf-8").count("\n") == 13


def test_validation_aggregation_requires_exact_locked_keys(tmp_path):
    runner = _load_runner()
    path = tmp_path / "validation.jsonl"
    rows = []
    for lambda_ in (0.0, 0.05):
        for route_id in ("a", "b"):
            rows.append({
                "route_id": route_id,
                "lambda": lambda_,
                "historical_route": ["d", "a", "d"],
                "route": ["d", "a", "d"],
                "audit": {"feasible": True, "metrics": {"travel_seconds": 1, "time_window_violations": 0}},
            })
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert len(runner._aggregate_validation(path, expected_route_ids=["a", "b"], expected_lambdas=[0.0, 0.05])) == 2

    path.write_text("".join(json.dumps(row) + "\n" for row in rows + [rows[0]]), encoding="utf-8")
    with pytest.raises(AssertionError, match="duplicate route/lambda"):
        runner._aggregate_validation(path, expected_route_ids=["a", "b"], expected_lambdas=[0.0, 0.05])


def test_split_manifest_rejects_duplicate_and_missing_holdout_ids():
    runner = _load_runner()
    row = {"route_id": "r1", "station_code": "S1", "date_YYYY_MM_DD": "2020-01-01"}
    with pytest.raises(AssertionError, match="unique"):
        runner._split_manifest([row, dict(row)], set(), {"route_ids": [], "station_days": []})
    missing_holdout_ids = ["missing"] + [f"holdout-{index:02d}" for index in range(12)]
    with pytest.raises(AssertionError, match="missing"):
        runner._split_manifest([row], set(), {"route_ids": missing_holdout_ids, "station_days": []})


def test_split_manifest_rejects_holdout_route_count_and_group_overlap(monkeypatch):
    runner = _load_runner()
    holdout_ids = [f"holdout-{index:02d}" for index in range(13)]
    manifest = [
        {"route_id": route_id, "station_code": "S0", "date_YYYY_MM_DD": "2020-01-01"}
        for route_id in holdout_ids
    ]
    with pytest.raises(AssertionError, match="exactly 13 unique"):
        runner._split_manifest(manifest, set(), {"route_ids": holdout_ids[:-1], "station_days": []})

    train_row = {"route_id": "train", "station_code": "S1", "date_YYYY_MM_DD": "2020-02-01"}
    validation_row = {"route_id": "validation", "station_code": "S1", "date_YYYY_MM_DD": "2020-02-01"}
    manifest.extend([train_row, validation_row])
    monkeypatch.setattr(runner, "isolate_station_days", lambda *_args, **_kwargs: ([train_row], [validation_row], set()))
    with pytest.raises(AssertionError, match="station-day groups overlap"):
        runner._split_manifest(manifest, set(), {"route_ids": holdout_ids, "station_days": []})


def test_train_helper_cli_is_disabled():
    script = Path(__file__).parents[1] / "scripts" / "train_preference_model.py"
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 2
    assert "Direct CLI disabled" in result.stderr
    assert "run_preference_replay_v3.py" in result.stderr
