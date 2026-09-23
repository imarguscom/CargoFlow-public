#!/usr/bin/env python3
"""Run the fixed Amazon route cohort through the vanilla PyVRP baseline."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cargoflow.solver import audit_route, solve_instance  # noqa: E402

CONVERTER_PATH = ROOT / "scripts" / "convert_amazon_route.py"
SPEC = importlib.util.spec_from_file_location("convert_amazon_route", CONVERTER_PATH)
if not SPEC or not SPEC.loader:
    raise RuntimeError(f"cannot load converter: {CONVERTER_PATH}")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

RAW_FILES = {
    "route": "route_data.json",
    "packages": "package_data.json",
    "travel": "travel_times.json",
    "actual": "actual_sequences.json",
}
RESULT_FIELDS = [
    "route_id",
    "station_code",
    "customer_count",
    "status",
    "feasible",
    "historical_feasible",
    "solver_travel_seconds",
    "historical_travel_seconds",
    "travel_improvement_percent",
    "elapsed_seconds",
    "time_window_violations",
    "lateness_seconds",
    "capacity_utilisation",
    "solve_seconds",
    "error",
]


def _route_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    route_ids = [value for value in values if value]
    if not route_ids or len(route_ids) != len(set(route_ids)):
        raise ValueError("route list must contain unique, non-empty route IDs")
    return route_ids


def _load_selected(path: Path, route_ids: list[str]) -> dict[str, Any]:
    print(f"loading {path.name}", flush=True)
    with path.open(encoding="utf-8") as stream:
        all_records = json.load(stream, parse_constant=lambda _value: None)
    if not isinstance(all_records, dict):
        raise ValueError(f"{path.name} must be a top-level object")
    missing = sorted(set(route_ids) - set(all_records))
    if missing:
        raise ValueError(f"{path.name} is missing selected route {missing[0]}")
    selected = {route_id: all_records[route_id] for route_id in route_ids}
    del all_records
    gc.collect()
    return selected


def _historical_route(actual: dict[str, Any], depot_id: str) -> list[str]:
    order = actual["actual"]
    customers = sorted(
        ((position, node_id) for node_id, position in order.items() if node_id != depot_id),
        key=lambda item: item[0],
    )
    return [depot_id, *(node_id for _, node_id in customers), depot_id]


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _previous_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in ("feasible", "historical_feasible"):
            if row[field] in {"True", "False"}:
                row[field] = row[field] == "True"
    return {row["route_id"]: row for row in rows}


def run(
    raw_dir: Path,
    route_list: Path,
    processed_dir: Path,
    output_dir: Path,
    *,
    seed: int,
    iterations: int,
    resume: bool = False,
) -> list[dict[str, Any]]:
    route_ids = _route_ids(route_list)
    inputs = converter._input_dir(raw_dir)
    records = {
        label: _load_selected(inputs / filename, route_ids)
        for label, filename in RAW_FILES.items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    previous = _previous_rows(output_dir / "results.csv") if resume else {}
    if previous:
        summary_path = output_dir / "summary.json"
        if not summary_path.is_file():
            raise ValueError("resume requires the previous summary.json")
        previous_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            previous_summary.get("seed") != seed
            or previous_summary.get("iteration_limit") != iterations
        ):
            raise ValueError("resume configuration differs from the previous run")
        if set(previous) != set(route_ids):
            raise ValueError("resume route list differs from the previous run")
    results: list[dict[str, Any]] = []
    for index, route_id in enumerate(route_ids, 1):
        route_dir = processed_dir / route_id
        previous_row = previous.get(route_id)
        if (
            previous_row
            and previous_row["status"] != "input_error"
            and (route_dir / "solution.json").is_file()
        ):
            results.append(previous_row)
            print(
                f"[{index}/{len(route_ids)}] {route_id}: {previous_row['status']} (cached)",
                flush=True,
            )
            continue
        row: dict[str, Any] = {field: "" for field in RESULT_FIELDS}
        row["route_id"] = route_id
        route = records["route"][route_id]
        row["station_code"] = route.get("station_code", "")
        try:
            converter.convert_records(
                route_id,
                route,
                records["packages"][route_id],
                records["travel"][route_id],
                records["actual"][route_id],
                route_dir,
                emit_summary=False,
            )
            instance = json.loads((route_dir / "instance.json").read_text(encoding="utf-8"))
            matrix = json.loads((route_dir / "matrix.json").read_text(encoding="utf-8"))
            row["customer_count"] = len(instance["nodes"]) - 1
            historical = audit_route(
                instance,
                matrix,
                _historical_route(records["actual"][route_id], instance["depot_id"]),
            )
            solution = solve_instance(instance, matrix, seed=seed, iterations=iterations)
            _write_json(route_dir / "solution.json", solution)
            row["status"] = solution["status"]
            row["feasible"] = solution["feasible"]
            row["historical_feasible"] = historical["feasible"]
            solver_metrics = (solution.get("audit") or {}).get("metrics") or {}
            historical_metrics = historical.get("metrics") or {}
            solver_travel = solver_metrics.get("travel_seconds")
            historical_travel = historical_metrics.get("travel_seconds")
            row["solver_travel_seconds"] = solver_travel if solver_travel is not None else ""
            row["historical_travel_seconds"] = historical_travel if historical_travel is not None else ""
            if solution["feasible"] and historical["feasible"] and historical_travel:
                row["travel_improvement_percent"] = round(
                    100 * (historical_travel - solver_travel) / historical_travel, 6
                )
            for field in (
                "elapsed_seconds",
                "time_window_violations",
                "lateness_seconds",
                "capacity_utilisation",
            ):
                row[field] = solver_metrics.get(field, "")
            row["solve_seconds"] = solution.get("solve_seconds", "")
        except (OSError, ValueError, TypeError, KeyError, converter.ConversionError) as exc:
            row["status"] = "input_error"
            row["feasible"] = False
            row["error"] = str(exc)
        results.append(row)
        print(f"[{index}/{len(route_ids)}] {route_id}: {row['status']}", flush=True)

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(results)
    improvements = [
        float(row["travel_improvement_percent"])
        for row in results
        if row["travel_improvement_percent"] != ""
    ]
    solve_times = [
        float(row["solve_seconds"])
        for row in results
        if row["solve_seconds"] != ""
    ]
    summary = {
        "selected_routes": len(route_ids),
        "status_counts": dict(sorted(Counter(row["status"] for row in results).items())),
        "solver_feasible_routes": sum(row["feasible"] is True for row in results),
        "historical_feasible_routes": sum(
            row["historical_feasible"] is True for row in results
        ),
        "paired_feasible_routes": len(improvements),
        "median_travel_improvement_percent": median(improvements) if improvements else None,
        "solve_seconds_total": sum(solve_times),
        "solve_seconds_median": median(solve_times) if solve_times else None,
        "seed": seed,
        "iteration_limit": iterations,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--route-list", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse rows with saved solutions and retry prior input errors",
    )
    args = parser.parse_args()
    try:
        run(
            args.raw_dir,
            args.route_list,
            args.processed_dir,
            args.output_dir,
            seed=args.seed,
            iterations=args.iterations,
            resume=args.resume,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
