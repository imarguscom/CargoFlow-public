#!/usr/bin/env python3
"""Internal helper for the leakage-safe LightGBM edge-preference model.

The public command is ``scripts/run_preference_replay_v3.py``.  This module
keeps importable fitting functions for that runner; its direct CLI is
disabled so a caller cannot bypass the v3 holdout and selection gate.
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys
from typing import Any, Mapping
from statistics import median
try:
    import resource
except ModuleNotFoundError:  # Windows has no POSIX resource module.
    resource = None
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cargoflow.preferences import (  # noqa: E402
    DEFAULT_NEGATIVE_SAMPLING_SEED,
    FEATURE_NAMES,
    PreferenceModel,
    historical_route_from_positions,
    isolate_station_days,
    preference_cost_matrix,
    select_validation_routes,
    select_lambda,
    transition_examples,
)
from cargoflow.solver import audit_route, solve_instance  # noqa: E402


def _route_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _row_metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "station_code": row.get("station_code", "unknown"),
        "zone_by_stop": row.get("zone_by_stop", {}),
    }


def _examples(rows: list[dict[str, Any]], *, sampling_seed: int = DEFAULT_NEGATIVE_SAMPLING_SEED) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        instance = _read_json(Path(row["instance_path"]))
        matrix = _read_json(Path(row["matrix_path"]))
        actual = _read_json(Path(row["actual_path"]))
        route = historical_route_from_positions(actual, instance["depot_id"])
        output.extend(transition_examples(instance, matrix, route, metadata=_row_metadata(row), sampling_seed=sampling_seed))
    return output


def _fit_lightgbm(examples: list[dict[str, Any]], *, seed: int = 42) -> Any:
    try:
        import lightgbm as lgb
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("LightGBM training requires the optional cargoflow[ml] dependency") from exc
    frame = pd.DataFrame([example["features"] for example in examples], columns=FEATURE_NAMES)
    frame["station_code"] = frame["station_code"].astype("category")
    labels = [int(example["label"]) for example in examples]
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(frame, labels, categorical_feature=["station_code"])
    return model


def _fit_lightgbm_routes(
    rows: list[dict[str, Any]],
    *,
    seed: int = 42,
    sampling_seed: int = DEFAULT_NEGATIVE_SAMPLING_SEED,
) -> tuple[Any, dict[str, Any]]:
    """Fit from a disk-backed float32 matrix, avoiding Python example buildup."""

    try:
        import lightgbm as lgb
        import numpy as np
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("LightGBM training requires the optional cargoflow[ml] dependency") from exc
    if not rows:
        raise ValueError("no routes available for training")
    station_codes = {station: index for index, station in enumerate(sorted({str(row.get("station_code", "unknown")) for row in rows}))}
    counts: list[int] = []
    for row in rows:
        instance = _read_json(Path(row["instance_path"]))
        counts.append(4 * len(instance["nodes"]))
    total = sum(counts)
    with tempfile.TemporaryDirectory(prefix="cargoflow-preference-") as tmp:
        feature_path = Path(tmp) / "features.float32"
        label_path = Path(tmp) / "labels.uint8"
        features = np.memmap(feature_path, mode="w+", dtype="float32", shape=(total, len(FEATURE_NAMES)))
        labels = np.memmap(label_path, mode="w+", dtype="uint8", shape=(total,))
        cursor = 0
        for row, expected in zip(rows, counts):
            instance = _read_json(Path(row["instance_path"]))
            matrix = _read_json(Path(row["matrix_path"]))
            actual = _read_json(Path(row["actual_path"]))
            route = historical_route_from_positions(actual, instance["depot_id"])
            examples = transition_examples(instance, matrix, route, metadata=_row_metadata(row), sampling_seed=sampling_seed)
            if len(examples) != expected:
                raise ValueError(f"route {row['route_id']} produced {len(examples)} examples; expected {expected}")
            for example in examples:
                for column, name in enumerate(FEATURE_NAMES):
                    value = example["features"][name]
                    features[cursor, column] = station_codes[str(value)] if name == "station_code" else float(value)
                labels[cursor] = int(example["label"])
                cursor += 1
        features.flush()
        labels.flush()
        frame = pd.DataFrame(features, columns=FEATURE_NAMES, copy=False)
        frame["station_code"] = frame["station_code"].astype("category")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=200, learning_rate=0.05,
            num_leaves=31, random_state=seed, verbosity=-1,
        )
        model.fit(frame, labels, categorical_feature=["station_code"])
    return PreferenceModel(model, station_codes), {
        "examples": total,
        "feature_storage": "float32 disk-backed memmap",
        "station_codes": len(station_codes),
        "peak_rss_units": (
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if resource is not None else None
        ),
        "peak_rss_unit": "bytes on macOS; kilobytes on Linux; unavailable on Windows",
    }


def _evaluate_lambda(model: Any, rows: list[dict[str, Any]], lambda_: float) -> dict[str, Any]:
    """Evaluate one cost weight on the locked validation routes."""

    feasible = 0
    travel: list[float] = []
    violations = 0
    consistency: list[float] = []
    for row in rows:
        instance = _read_json(Path(row["instance_path"]))
        matrix = _read_json(Path(row["matrix_path"]))
        actual = _read_json(Path(row["actual_path"]))
        historical_route = historical_route_from_positions(actual, instance["depot_id"])
        historical = audit_route(instance, matrix, historical_route)
        objective = preference_cost_matrix(
            instance,
            matrix,
            model,
            lambda_=lambda_,
            metadata=_row_metadata(row),
        )
        solution = solve_instance(
            instance,
            matrix,
            seed=42,
            iterations=5000,
            objective_cost_seconds=objective,
        )
        audit = solution.get("audit") or {}
        metrics = audit.get("metrics") or {}
        if audit.get("feasible"):
            feasible += 1
        if metrics.get("travel_seconds") is not None:
            travel.append(float(metrics["travel_seconds"]))
        violations += int(metrics.get("time_window_violations", 0))
        if solution.get("route"):
            historical_edges = set(zip(historical_route, historical_route[1:]))
            candidate_edges = set(zip(solution["route"], solution["route"][1:]))
            consistency.append(len(historical_edges & candidate_edges) / len(historical_edges))
    return {
        "lambda": lambda_,
        "feasible_routes": feasible,
        "median_travel_seconds": median(travel) if travel else float("inf"),
        "time_window_violations": violations,
        "historical_edge_consistency": median(consistency) if consistency else 0.0,
    }


def train_from_manifest(
    manifest_path: Path,
    benchmark_routes_path: Path,
    validation_routes_path: Path,
    model_path: Path,
    *,
    metadata_path: Path | None = None,
    seed: int = 42,
    sampling_seed: int = DEFAULT_NEGATIVE_SAMPLING_SEED,
    lambda_candidates: list[float] | None = None,
) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, list):
        raise ValueError("manifest must be a JSON array")
    benchmark_ids = _route_ids(benchmark_routes_path)
    train_rows, validation_rows, excluded_groups = isolate_station_days(manifest, benchmark_ids)
    locked_validation = select_validation_routes(validation_rows, limit=50)
    validation_routes_path.parent.mkdir(parents=True, exist_ok=True)
    validation_routes_path.write_text(
        "".join(f"{row['route_id']}\n" for row in locked_validation), encoding="utf-8"
    )
    if not train_rows:
        raise ValueError("station-day isolation left no training routes")
    model, train_memory = _fit_lightgbm_routes([dict(row) for row in train_rows], seed=seed, sampling_seed=sampling_seed)
    lambda_candidates = lambda_candidates or [0.0, 0.02, 0.05, 0.10, 0.15]
    locked_validation = locked_validation[:50]
    lambda_report: dict[str, Any]
    if locked_validation:
        evaluations = [_evaluate_lambda(model, [dict(row) for row in locked_validation], value) for value in lambda_candidates]
        locked_lambda, lambda_report = select_lambda(evaluations)
    else:
        locked_lambda = 0.0
        lambda_report = {"selected_lambda": 0.0, "reason": "no validation routes available", "eligible": []}
    # After selecting the weight, fit the deployable model with every
    # non-benchmark route in the isolated train + validation partition.
    final_rows = [dict(row) for row in train_rows + validation_rows]
    model, refit_memory = _fit_lightgbm_routes(final_rows, seed=seed, sampling_seed=sampling_seed)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with model_path.open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "model": "LightGBM",
        "feature_names": list(FEATURE_NAMES),
        "training_routes": len(train_rows),
        "validation_routes": len(locked_validation),
        "excluded_benchmark_station_days": [list(group) for group in sorted(excluded_groups)],
        "negative_examples_per_positive": 3,
        "seed": seed,
        "negative_sampling_seed": sampling_seed,
        "lambda_candidates": lambda_candidates,
        "locked_lambda": locked_lambda,
        "lambda_selection": lambda_report,
        "refit_routes": len(final_rows),
        "training_memory": train_memory,
        "refit_memory": refit_memory,
    }
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> int:
    print(
        "Direct CLI disabled: use scripts/run_preference_replay_v3.py; "
        "this module is an internal fitting helper.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
