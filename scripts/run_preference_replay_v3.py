#!/usr/bin/env python3
"""Run the v3 replay-safe preference experiment in one isolated directory.

The command writes only ignored evidence under ``--output-dir``.  The
application route/package/travel records are read for blind inference; the
application sequence file is passed only to the explicit history evaluator
after blind outputs are complete.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import pickle
import platform
import subprocess
import sys
from statistics import median
from typing import Any, Iterable, Mapping
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

from cargoflow.preferences import (  # noqa: E402
    DEFAULT_NEGATIVE_SAMPLING_SEED,
    FEATURE_NAMES,
    historical_route_from_positions,
    isolate_station_days,
    preference_cost_matrix,
    select_lambda,
    select_validation_routes,
)
from cargoflow.solver import audit_route, solve_instance  # noqa: E402
from train_preference_model import _fit_lightgbm_routes  # noqa: E402


def _load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=lambda _value: None)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")


def _route_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result = [value for value in values if value and not value.startswith("#")]
    if len(result) != len(set(result)):
        raise ValueError(f"{path} contains duplicate route IDs")
    return result


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "station_code": row.get("station_code", "unknown"),
        "zone_by_stop": row.get("zone_by_stop", {}),
    }


def _load_script(name: str, module_name: str):
    path = SCRIPTS / name
    spec = importlib.util.spec_from_file_location(module_name, path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_state() -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    return {"commit": commit, "clean": not bool(status.strip())}


def _short_error(exc: BaseException) -> str:
    """Return a compact lifecycle error suitable for the run record."""

    detail = str(exc).strip().replace("\n", " ")
    if len(detail) > 240:
        detail = detail[:237] + "..."
    return f"{type(exc).__name__}: {detail}" if detail else type(exc).__name__


def _mark_run_failed(output: Path, exc: BaseException) -> None:
    """Make an interrupted/failed run distinguishable from a completed run."""

    run_path = output / "run.json"
    if not run_path.exists():
        return
    try:
        record = _load_json(run_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return
    if record.get("status") != "running":
        return
    record.update({
        "status": "failed",
        "error": _short_error(exc),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    })
    _write_json(run_path, record)


def _split_manifest(
    manifest: list[dict[str, Any]],
    benchmark_ids: set[str],
    holdout: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_ids = [str(row["route_id"]) for row in manifest]
    if len(manifest_ids) != len(set(manifest_ids)):
        raise AssertionError("manifest route IDs must be unique")
    manifest_id_set = set(manifest_ids)
    holdout_route_values = [str(value) for value in holdout["route_ids"]]
    if len(holdout_route_values) != 13 or len(set(holdout_route_values)) != 13:
        raise AssertionError("application replay holdout must contain exactly 13 unique route IDs")
    holdout_ids = set(holdout_route_values)
    missing_holdout_ids = holdout_ids - manifest_id_set
    if missing_holdout_ids:
        raise AssertionError("application holdout route IDs missing from manifest")
    missing_benchmark_ids = set(benchmark_ids) - manifest_id_set
    if missing_benchmark_ids:
        raise AssertionError("benchmark route IDs missing from manifest")
    holdout_groups = {(str(pair[0]), str(pair[1])) for pair in holdout["station_days"]}
    benchmark_groups = {
        (str(row["station_code"]), str(row["date_YYYY_MM_DD"]))
        for row in manifest
        if str(row["route_id"]) in benchmark_ids
    }
    train, validation, excluded = isolate_station_days(
        manifest,
        benchmark_ids,
        excluded_station_days=holdout_groups,
    )
    train_ids = {str(row["route_id"]) for row in train}
    validation_ids = {str(row["route_id"]) for row in validation}
    excluded_ids = manifest_id_set - train_ids - validation_ids
    if train_ids & validation_ids or train_ids & excluded_ids or validation_ids & excluded_ids:
        raise AssertionError("train, validation, and excluded route partitions overlap")
    if train_ids | validation_ids | excluded_ids != manifest_id_set:
        raise AssertionError("train, validation, and excluded route partitions are incomplete")
    train_groups = {(str(row["station_code"]), str(row["date_YYYY_MM_DD"])) for row in train}
    validation_groups = {(str(row["station_code"]), str(row["date_YYYY_MM_DD"])) for row in validation}
    if train_groups & validation_groups:
        raise AssertionError("train and validation station-day groups overlap")
    if train_ids & holdout_ids or validation_ids & holdout_ids:
        raise AssertionError("application holdout route IDs crossed into model partitions")
    if train_groups & holdout_groups or validation_groups & holdout_groups:
        raise AssertionError("application holdout station-days crossed into model partitions")
    if train_groups & benchmark_groups or validation_groups & benchmark_groups:
        raise AssertionError("benchmark station-days crossed into model partitions")
    if len(excluded) != len(benchmark_groups | holdout_groups):
        raise AssertionError("protected station-day count is inconsistent")
    split_rows: list[dict[str, Any]] = []
    for row in manifest:
        group = (str(row["station_code"]), str(row["date_YYYY_MM_DD"]))
        if group in benchmark_groups:
            split = "excluded_benchmark_group"
        elif group in holdout_groups:
            split = "excluded_application_holdout_group"
        elif str(row["route_id"]) in train_ids:
            split = "train"
        elif str(row["route_id"]) in validation_ids:
            split = "validation_pool"
        else:
            raise AssertionError(f"route {row['route_id']} has no split")
        split_rows.append({
            "route_id": row["route_id"],
            "split": split,
            "station_code": row["station_code"],
            "date_YYYY_MM_DD": row["date_YYYY_MM_DD"],
        })
    counts = {
        "manifest": len(manifest),
        "train": len(train),
        "validation_pool": len(validation),
        "excluded": len(manifest) - len(train) - len(validation),
        "benchmark_station_days": len(benchmark_groups),
        "application_holdout_station_days": len(holdout_groups),
        "protected_station_days": len(benchmark_groups | holdout_groups),
        "application_holdout_routes_in_manifest": len(holdout_ids & manifest_id_set),
    }
    return train, validation, {
        "rows": split_rows,
        "counts": counts,
        "asserts": {
            "train_holdout_route_disjoint": not bool(train_ids & holdout_ids),
            "validation_holdout_route_disjoint": not bool(validation_ids & holdout_ids),
            "train_holdout_group_disjoint": not bool(train_groups & holdout_groups),
            "validation_holdout_group_disjoint": not bool(validation_groups & holdout_groups),
            "train_benchmark_group_disjoint": not bool(train_groups & benchmark_groups),
            "validation_benchmark_group_disjoint": not bool(validation_groups & benchmark_groups),
            "train_validation_station_day_disjoint": not bool(train_groups & validation_groups),
            "manifest_ids_unique": len(manifest_ids) == len(manifest_id_set),
            "holdout_ids_present": holdout_ids <= manifest_id_set,
            "train_validation_excluded_partition": (
                not bool(train_ids & validation_ids)
                and not bool(train_ids & excluded_ids)
                and not bool(validation_ids & excluded_ids)
                and train_ids | validation_ids | excluded_ids == manifest_id_set
            ),
        },
    }


def _fit_and_save(rows: list[dict[str, Any]], path: Path, metadata_path: Path, *, role: str, seed: int, sampling_seed: int) -> Any:
    model, memory = _fit_lightgbm_routes(rows, seed=seed, sampling_seed=sampling_seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    _write_json(metadata_path, {
        "role": role,
        "feature_names": list(FEATURE_NAMES),
        "routes": len(rows),
        "seed": seed,
        "negative_sampling_seed": sampling_seed,
        "memory": memory,
    })
    return model


def _validation_solution_rows(model: Any, rows: list[dict[str, Any]], lambdas: list[float], *, seed: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for lambda_ in lambdas:
        for row in rows:
            instance = _load_json(Path(row["instance_path"]))
            matrix = _load_json(Path(row["matrix_path"]))
            actual = _load_json(Path(row["actual_path"]))
            historical_route = historical_route_from_positions(actual, instance["depot_id"])
            objective = preference_cost_matrix(instance, matrix, model, lambda_=lambda_, metadata=_row_metadata(row))
            solution = solve_instance(
                instance,
                matrix,
                seed=seed,
                iterations=5000,
                objective_cost_seconds=objective,
            )
            result.append({
                "route_id": row["route_id"],
                "station_code": row["station_code"],
                "date_YYYY_MM_DD": row["date_YYYY_MM_DD"],
                "lambda": lambda_,
                "solver_config": {"seed": seed, "iterations": 5000},
                "historical_route": historical_route,
                "route": solution.get("route"),
                "audit": solution.get("audit"),
                "solution": solution,
            })
    return result


def _aggregate_validation(
    path: Path,
    *,
    expected_route_ids: Iterable[str] | None = None,
    expected_lambdas: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[float, list[dict[str, Any]]] = {}
    seen_keys: set[tuple[str, float]] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            lambda_ = float(record["lambda"])
            key = (str(record["route_id"]), lambda_)
            if key in seen_keys:
                raise AssertionError("validation evidence contains duplicate route/lambda key")
            seen_keys.add(key)
            grouped.setdefault(lambda_, []).append(record)
    if expected_lambdas is not None:
        expected_lambda_set = {float(value) for value in expected_lambdas}
        if set(grouped) != expected_lambda_set:
            raise AssertionError("validation evidence lambda set differs from locked candidates")
    expected_ids = {str(value) for value in expected_route_ids} if expected_route_ids is not None else None
    evaluations: list[dict[str, Any]] = []
    for lambda_, rows in sorted(grouped.items()):
        row_ids = {str(row["route_id"]) for row in rows}
        if expected_ids is not None and row_ids != expected_ids:
            raise AssertionError(f"lambda {lambda_} does not cover the locked validation routes")
        if len(rows) != len(row_ids) or (expected_ids is not None and len(rows) != len(expected_ids)):
            raise AssertionError(f"lambda {lambda_} has an invalid validation row count")
        travel: list[float] = []
        consistency: list[float] = []
        feasible = 0
        violations = 0
        for row in rows:
            audit = row.get("audit") or {}
            if audit.get("feasible"):
                feasible += 1
            metrics = audit.get("metrics") or {}
            if metrics.get("travel_seconds") is not None:
                travel.append(float(metrics["travel_seconds"]))
            violations += int(metrics.get("time_window_violations", 0))
            route = row.get("route")
            historical = row["historical_route"]
            if route:
                historical_edges = set(zip(historical, historical[1:]))
                route_edges = set(zip(route, route[1:]))
                consistency.append(len(historical_edges & route_edges) / len(historical_edges))
        evaluations.append({
            "lambda": lambda_,
            "feasible_routes": feasible,
            "median_travel_seconds": median(travel) if travel else float("inf"),
            "time_window_violations": violations,
            "historical_edge_consistency": median(consistency) if consistency else 0.0,
        })
    return evaluations


def _write_blind_jsonl(raw_dir: Path, output: Path, route_ids: list[str], *, label: str) -> None:
    rows: list[dict[str, Any]] = []
    for route_id in route_ids:
        payload = _load_json(raw_dir / route_id / "application_inference.json")
        for method in ("pure_pyvrp", "learned"):
            rows.append({"route_id": route_id, "method": method, "label": label, "solution": payload[method]})
    _write_jsonl(output, rows)


def _run_history_evaluation(actual: Path, route_list: Path, processed: Path, output_dir: Path, *, stem: str) -> list[dict[str, Any]]:
    output_csv = output_dir / f"{stem}.csv"
    evaluator = SCRIPTS / "evaluate_history.py"
    subprocess.run([
        sys.executable, str(evaluator),
        "--actual-sequences", str(actual),
        "--route-list", str(route_list),
        "--processed-dir", str(processed),
        "--output", str(output_csv),
    ], cwd=ROOT, check=True)
    with output_csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows


def _application_decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pure = {row["route_id"]: row for row in rows if row["method"] == "pure_pyvrp"}
    learned = {row["route_id"]: row for row in rows if row["method"] == "learned"}
    if len(pure) != 13 or len(learned) != 13:
        raise AssertionError("application history evaluation must contain 13 pure and 13 learned rows")
    deltas = [float(learned[key]["travel_seconds"]) - float(pure[key]["travel_seconds"]) for key in pure]
    learned_feasible = sum(row["feasible"] == "True" for row in learned.values())
    return {
        "routes": 13,
        "learned_feasible": learned_feasible,
        "learned_time_window_violations": sum(int(float(row["time_window_violations"] or 0)) for row in learned.values()),
        "paired_median_travel_delta": median(deltas),
        "paired_better": sum(delta < 0 for delta in deltas),
        "paired_equal": sum(delta == 0 for delta in deltas),
        "paired_worse": sum(delta > 0 for delta in deltas),
        "accepted": learned_feasible == 13 and median(deltas) <= 0,
    }


def _run_impl(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir
    if (output / "run.json").exists():
        raise FileExistsError(f"refusing to overwrite existing v3 run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    git = _git_state()
    budgets = _load_json(args.budgets)
    pref_config = _load_json(args.preference_config)
    holdout = _load_json(args.holdout_config)
    application_route_list = args.application_route_list
    if application_route_list is None:
        application_route_list = output / "application_route_ids.txt"
        application_route_list.write_text("".join(f"{route_id}\n" for route_id in holdout["route_ids"]), encoding="utf-8")
    route_ids = _route_ids(application_route_list)
    if route_ids != list(holdout["route_ids"]):
        raise AssertionError("application route list does not match tracked holdout config")
    run_record: dict[str, Any] = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "python": platform.python_version(),
        "inputs": {
            "manifest": str(args.manifest),
            "benchmark_routes": str(args.benchmark_routes),
            "holdout_config": str(args.holdout_config),
            "raw_dir": str(args.raw_dir),
            "application_route_list": str(application_route_list),
            "application_route": str(args.application_route),
            "application_package": str(args.application_package),
            "application_travel": str(args.application_travel),
            "application_actual": str(args.application_actual),
        },
        "holdout": {
            "name": holdout.get("name", "application_replay_holdout_13"),
            "source": holdout.get("source", "application-format holdout"),
            "route_ids": list(holdout["route_ids"]),
            "station_days": [list(pair) for pair in holdout["station_days"]],
        },
        "config": {"budgets": budgets, "preference": pref_config, "seeds": {"solver": 42, "negative_sampling": int(pref_config.get("negative_sampling_seed", DEFAULT_NEGATIVE_SAMPLING_SEED))}},
        "outputs": {},
    }
    _write_json(output / "run.json", run_record)

    manifest = _load_json(args.manifest)
    benchmark_ids = set(_route_ids(args.benchmark_routes))
    train_rows, validation_rows, split = _split_manifest(manifest, benchmark_ids, holdout)
    expected = {"manifest": 6112, "train": 3791, "validation_pool": 1302, "excluded": 1019, "benchmark_station_days": 46, "application_holdout_station_days": 12, "protected_station_days": 57}
    if any(split["counts"].get(key) != value for key, value in expected.items()):
        raise AssertionError(f"v3 split counts differ from expected: {split['counts']}")
    if not all(split["asserts"].values()):
        raise AssertionError("v3 split disjointness assertion failed")
    _write_json(output / "split.json", split)
    locked_validation = select_validation_routes(validation_rows, limit=50)
    if len(locked_validation) != 50 or len({row["station_code"] for row in locked_validation}) != 17:
        raise AssertionError("v3 locked validation cohort is incomplete")
    (output / "validation_routes_50.txt").write_text("".join(row["route_id"] + "\n" for row in locked_validation), encoding="utf-8")

    seed = int(pref_config.get("model_seed", 42))
    sampling_seed = int(pref_config.get("negative_sampling_seed", DEFAULT_NEGATIVE_SAMPLING_SEED))
    models = output / "models"
    validation_model = _fit_and_save(
        [dict(row) for row in train_rows],
        models / "validation_model.pkl",
        models / "validation_model_metadata.json",
        role="validation_model", seed=seed, sampling_seed=sampling_seed,
    )
    validation_rows_jsonl = output / "validation_solutions.jsonl"
    validation_solution_rows = _validation_solution_rows(
        validation_model,
        [dict(row) for row in locked_validation],
        [float(value) for value in pref_config["lambda_candidates"]],
        seed=42,
    )
    _write_jsonl(validation_rows_jsonl, validation_solution_rows)
    evaluations = _aggregate_validation(
        validation_rows_jsonl,
        expected_route_ids=[row["route_id"] for row in locked_validation],
        expected_lambdas=[float(value) for value in pref_config["lambda_candidates"]],
    )
    selected_lambda, selection = select_lambda(evaluations)
    _write_json(output / "lambda_selection.json", {
        "evaluations": evaluations,
        "selected_lambda": selected_lambda,
        "selection": selection,
    })
    run_record["outputs"].update({
        "split": str(output / "split.json"),
        "validation_routes": str(output / "validation_routes_50.txt"),
        "validation_model": str(models / "validation_model.pkl"),
        "validation_solutions": str(validation_rows_jsonl),
        "lambda_selection": str(output / "lambda_selection.json"),
    })

    app_dir = output / "application_replay"
    replay_decision = "not_run_gate_zero"
    if selected_lambda != 0.0:
        refit_model = _fit_and_save(
            [dict(row) for row in train_rows + validation_rows],
            models / "refit_model.pkl",
            models / "refit_model_metadata.json",
            role="refit_model", seed=seed, sampling_seed=sampling_seed,
        )
        run_record["outputs"]["refit_model"] = str(models / "refit_model.pkl")
        infer = _load_script("infer_application.py", "infer_application_v3")
        raw_outputs = app_dir / "blind_raw"
        infer.run_batch(
            args.application_route,
            args.application_package,
            args.application_travel,
            raw_outputs,
            route_ids=route_ids,
            model_path=models / "refit_model.pkl",
            lambda_=selected_lambda,
            seed=42,
            iterations=5000,
        )
        _write_blind_jsonl(raw_outputs, app_dir / "blind_solutions.jsonl", route_ids, label="learned_candidate")
        learned_rows = _run_history_evaluation(args.application_actual, application_route_list, raw_outputs, app_dir, stem="history_evaluation")
        _write_jsonl(app_dir / "history_evaluation.jsonl", learned_rows)
        decision = _application_decision(learned_rows)
        _write_json(app_dir / "replay_decision.json", decision)
        replay_decision = "accepted_nonzero" if decision["accepted"] else "rejected_fallback_lambda0"
        if not decision["accepted"]:
            fallback_raw = app_dir / "fallback_lambda0_raw"
            infer.run_batch(
                args.application_route,
                args.application_package,
                args.application_travel,
                fallback_raw,
                route_ids=route_ids,
                model_path=models / "refit_model.pkl",
                lambda_=0.0,
                seed=42,
                iterations=5000,
            )
            _write_blind_jsonl(fallback_raw, app_dir / "fallback_blind_solutions.jsonl", route_ids, label="lambda0_fallback")
            fallback_rows = _run_history_evaluation(args.application_actual, application_route_list, fallback_raw, app_dir, stem="fallback_history_evaluation")
            _write_jsonl(app_dir / "fallback_history_evaluation.jsonl", fallback_rows)
    comparison = _load_script("run_comparison.py", "run_comparison_v3")
    comparison_output = output / "comparison"
    comparison_summary = comparison.run(
        args.raw_dir,
        args.benchmark_routes,
        comparison_output / "processed",
        comparison_output,
        budgets_path=args.budgets,
    )
    run_record = _load_json(output / "run.json")
    run_record["outputs"]["comparison"] = str(comparison_output)
    run_record["comparison_summary"] = comparison_summary
    run_record.update({
        "status": "complete",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "selected_lambda": selected_lambda,
        "replay_decision": replay_decision,
    })
    _write_json(output / "run.json", run_record)
    return run_record


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Run the pipeline and persist a failed lifecycle state on exceptions."""

    try:
        return _run_impl(args)
    except BaseException as exc:
        _mark_run_failed(args.output_dir, exc)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--benchmark-routes", required=True, type=Path)
    parser.add_argument("--holdout-config", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--application-route-list", type=Path,
                        help="Optional route list; defaults to the tracked holdout cohort")
    parser.add_argument("--application-route", required=True, type=Path)
    parser.add_argument("--application-package", required=True, type=Path)
    parser.add_argument("--application-travel", required=True, type=Path)
    parser.add_argument("--application-actual", required=True, type=Path)
    parser.add_argument("--budgets", required=True, type=Path)
    parser.add_argument("--preference-config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, AssertionError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
