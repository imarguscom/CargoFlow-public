#!/usr/bin/env python3
"""Compare historical, deterministic heuristics, and budgeted PyVRP runs.

This runner intentionally has its own loading and row-writing path so a
comparison cannot silently inherit resume state from ``run_benchmark.py``.
It writes only aggregate/route-level outputs; converted inputs and solver
artifacts belong in the caller-specified ignored directories.
"""

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
from time import perf_counter
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cargoflow.baselines import deadline_greedy, nearest_neighbor  # noqa: E402
from cargoflow.solver import audit_route, solve_instance  # noqa: E402

CONVERTER_PATH = ROOT / "scripts" / "convert_amazon_route.py"
SPEC = importlib.util.spec_from_file_location("convert_amazon_route_comparison", CONVERTER_PATH)
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

FIELDS = [
    "route_id", "station_code", "method", "config", "seed", "customer_count",
    "status", "feasible", "historical_feasible", "travel_seconds", "service_seconds",
    "waiting_seconds", "elapsed_seconds", "lateness_seconds", "capacity_utilisation",
    "time_window_violations", "travel_change_percent", "historical_edge_consistency",
    "construction_seconds", "search_seconds", "end_to_end_seconds", "error",
]


def _route_ids(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    route_ids = [value for value in values if value and not value.startswith("#")]
    if not route_ids or len(route_ids) != len(set(route_ids)):
        raise ValueError("route list must contain unique, non-empty route IDs")
    return route_ids


def _load_selected(path: Path, route_ids: list[str]) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream, parse_constant=lambda _value: None)
    if not isinstance(records, dict):
        raise ValueError(f"{path.name} must be a top-level object")
    missing = sorted(set(route_ids) - set(records))
    if missing:
        raise ValueError(f"{path.name} is missing selected route {missing[0]}")
    selected = {route_id: records[route_id] for route_id in route_ids}
    del records
    gc.collect()
    return selected


def _historical_route(actual: dict[str, Any], depot_id: str) -> list[str]:
    order = actual["actual"]
    customers = sorted(((position, node_id) for node_id, position in order.items() if node_id != depot_id), key=lambda item: item[0])
    return [depot_id, *(node_id for _, node_id in customers), depot_id]


def _edge_consistency(historical: list[str], route: list[str] | None) -> float | None:
    if not route:
        return None
    historical_edges = set(zip(historical, historical[1:]))
    route_edges = set(zip(route, route[1:]))
    return len(historical_edges & route_edges) / len(historical_edges) if historical_edges else None


def _empty_row(route_id: str, station_code: str, method: str, config: str, seed: int | str, count: int) -> dict[str, Any]:
    row = {field: "" for field in FIELDS}
    row.update({"route_id": route_id, "station_code": station_code, "method": method,
               "config": config, "seed": seed, "customer_count": count,
               "feasible": False, "historical_feasible": False})
    return row


def _metrics_row(row: dict[str, Any], audit: dict[str, Any] | None, historical: dict[str, Any], *, route: list[str] | None, construction: float, search: float, end_to_end: float) -> None:
    row["construction_seconds"] = construction
    row["search_seconds"] = search
    row["end_to_end_seconds"] = end_to_end
    row["historical_feasible"] = historical["feasible"]
    row["historical_edge_consistency"] = _edge_consistency(
        historical["route"], route
    ) if "route" in historical else None
    if not audit:
        return
    row["feasible"] = audit["feasible"]
    metrics = audit.get("metrics") or {}
    for field in ("travel_seconds", "service_seconds", "waiting_seconds", "elapsed_seconds", "lateness_seconds", "capacity_utilisation", "time_window_violations"):
        row[field] = metrics.get(field, "")
    historical_metrics = historical.get("metrics") or {}
    if metrics.get("travel_seconds") is not None and historical_metrics.get("travel_seconds"):
        row["travel_change_percent"] = round(100 * (metrics["travel_seconds"] - historical_metrics["travel_seconds"]) / historical_metrics["travel_seconds"], 6)


def _method_specs(budgets: dict[str, Any]) -> list[tuple[str, str, int, dict[str, Any]]]:
    specs = [("pyvrp", "reference", int(budgets["reference"].get("seeds", [42])[0]), budgets["reference"])]
    for config in ("fast", "balanced"):
        for seed in budgets[config]["seeds"]:
            specs.append(("pyvrp", config, int(seed), budgets[config]))
    return specs


def run(raw_dir: Path, route_list: Path, processed_dir: Path, output_dir: Path, *, budgets_path: Path) -> dict[str, Any]:
    route_ids = _route_ids(route_list)
    inputs = converter._input_dir(raw_dir)
    records = {label: _load_selected(inputs / filename, route_ids) for label, filename in RAW_FILES.items()}
    budgets = json.loads(budgets_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    solution_records: list[dict[str, Any]] = []
    for number, route_id in enumerate(route_ids, 1):
        route_record = records["route"][route_id]
        station = str(route_record.get("station_code", ""))
        route_dir = processed_dir / route_id
        route_rows: list[dict[str, Any]] = []
        route_solution_records: list[dict[str, Any]] = []
        try:
            converter.convert_records(route_id, route_record, records["packages"][route_id], records["travel"][route_id], records["actual"][route_id], route_dir, emit_summary=False)
            instance = json.loads((route_dir / "instance.json").read_text(encoding="utf-8"))
            matrix = json.loads((route_dir / "matrix.json").read_text(encoding="utf-8"))
            historical_route = _historical_route(records["actual"][route_id], instance["depot_id"])
            historical = audit_route(instance, matrix, historical_route)
            historical["route"] = historical_route
            count = len(instance["nodes"]) - 1

            historical_row = _empty_row(route_id, station, "historical", "observed", "", count)
            historical_row["status"] = "feasible" if historical["feasible"] else "infeasible"
            _metrics_row(
                historical_row,
                historical,
                historical,
                route=historical_route,
                construction=0.0,
                search=0.0,
                end_to_end=0.0,
            )
            historical_row["historical_edge_consistency"] = 1.0
            historical_row["travel_change_percent"] = 0.0
            route_rows.append(historical_row)
            route_solution_records.append({
                "route_id": route_id, "station_code": station, "method": "historical",
                "config": "observed", "seed": "", "status": historical_row["status"],
                "route": historical_route, "audit": historical,
            })

            for method, constructor in (("nearest_neighbor", nearest_neighbor), ("deadline_greedy", deadline_greedy)):
                method_started = perf_counter()
                candidate_route = constructor(instance, matrix)
                construction = perf_counter() - method_started
                candidate_audit = audit_route(instance, matrix, candidate_route)
                row = _empty_row(route_id, station, method, "heuristic", "", count)
                row["status"] = "feasible" if candidate_audit["feasible"] else "infeasible"
                _metrics_row(row, candidate_audit, historical, route=candidate_route, construction=construction, search=0.0, end_to_end=perf_counter() - method_started)
                route_rows.append(row)
                route_solution_records.append({
                    "route_id": route_id, "station_code": station, "method": method,
                    "config": "heuristic", "seed": "", "status": row["status"],
                    "route": candidate_route, "audit": candidate_audit,
                })

            for method, config, seed, budget in _method_specs(budgets):
                method_started = perf_counter()
                solution = solve_instance(instance, matrix, seed=seed, iterations=int(budget.get("iterations", budget.get("max_iterations", 5000))), runtime_seconds=budget.get("runtime_seconds"), max_iterations=budget.get("max_iterations"), max_no_improvement=budget.get("max_no_improvement"))
                end_to_end = perf_counter() - method_started
                row = _empty_row(route_id, station, method, config, seed, count)
                row["status"] = solution["status"]
                _metrics_row(row, solution.get("audit"), historical, route=solution.get("route"), construction=0.0, search=solution.get("solve_seconds", 0.0), end_to_end=end_to_end)
                route_rows.append(row)
                route_solution_records.append({
                    "route_id": route_id, "station_code": station, "method": method,
                    "config": config, "seed": seed, "status": row["status"],
                    "route": solution.get("route"), "audit": solution.get("audit"),
                    "solution": solution,
                })
        except (OSError, ValueError, TypeError, KeyError, converter.ConversionError, RuntimeError) as exc:
            count = int(route_record.get("stops", {}).__len__() - 1) if isinstance(route_record.get("stops"), dict) else ""
            route_rows = []
            route_solution_records = []
            for method, config, seed, _budget in [("historical", "observed", "", None), ("nearest_neighbor", "heuristic", "", None), ("deadline_greedy", "heuristic", "", None), *_method_specs(budgets)]:
                row = _empty_row(route_id, station, method, config, seed, count)
                row["status"] = "input_error"
                row["error"] = str(exc)
                route_rows.append(row)
                route_solution_records.append({
                    "route_id": route_id, "station_code": station, "method": method,
                    "config": config, "seed": seed, "status": "input_error",
                    "route": None, "audit": None, "error": str(exc),
                })
        if len(route_rows) != 10 or len({(row["route_id"], row["method"], row["config"], str(row["seed"])) for row in route_rows}) != 10:
            raise RuntimeError(f"comparison route {route_id} did not produce the fixed method set")
        rows.extend(route_rows)
        solution_records.extend(route_solution_records)
        print(f"[{number}/{len(route_ids)}] {route_id}", flush=True)

    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "solutions.jsonl").open("w", encoding="utf-8") as stream:
        for record in solution_records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    method_summary: dict[str, Any] = {}
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        improvements = [float(row["travel_change_percent"]) for row in selected if row["travel_change_percent"] != ""]
        method_summary[method] = {
            "rows": len(selected),
            "route_rows": len({row["route_id"] for row in selected}),
            "feasible_routes": sum(row["feasible"] is True for row in selected),
            "median_travel_change_percent": median(improvements) if improvements else None,
            "median_end_to_end_seconds": median([float(row["end_to_end_seconds"]) for row in selected if row["end_to_end_seconds"] != ""]) if any(row["end_to_end_seconds"] != "" for row in selected) else None,
        }
    summary = {"selected_routes": len(route_ids), "rows": len(rows), "methods": method_summary, "budgets": budgets}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--route-list", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--budgets", type=Path, default=ROOT / "configs" / "solver_budgets.json")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.raw_dir, args.route_list, args.processed_dir, args.output_dir, budgets_path=args.budgets), indent=2))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
