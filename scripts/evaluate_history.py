#!/usr/bin/env python3
"""Evaluate saved application inference routes against historical order.

This is deliberately the only post-hoc application command that reads the
actual-sequence record.  It reports per-route audit metrics and makes no
population-level generalisation.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cargoflow.solver import audit_route  # noqa: E402

CONVERTER_PATH = ROOT / "scripts" / "convert_amazon_route.py"
SPEC = importlib.util.spec_from_file_location("convert_amazon_route_history", CONVERTER_PATH)
if not SPEC or not SPEC.loader:
    raise RuntimeError(f"cannot load converter: {CONVERTER_PATH}")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _load_selected(path: Path, route_ids: list[str]) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream, parse_constant=lambda _value: None)
    if not isinstance(records, dict):
        raise ValueError(f"{path.name} must be a top-level object")
    missing = sorted(set(route_ids) - set(records))
    if missing:
        raise ValueError(f"{path.name} is missing selected route {missing[0]}")
    return {route_id: records[route_id] for route_id in route_ids}


def _historical_route(actual: dict[str, Any], depot_id: str) -> list[str]:
    order = actual["actual"]
    return [depot_id, *(node_id for _, node_id in sorted((position, node_id) for node_id, position in order.items() if node_id != depot_id)), depot_id]


def run(actual_sequences: Path, route_list: Path, processed_dir: Path, output_csv: Path) -> list[dict[str, Any]]:
    route_ids = [line.strip() for line in route_list.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    # Application truth is intentionally an explicit argument. In particular,
    # do not fall back to model_build_inputs/actual_sequences.json: that is a
    # different cohort and would silently invalidate the post-hoc comparison.
    actuals = _load_selected(actual_sequences, route_ids)
    rows: list[dict[str, Any]] = []
    for route_id in route_ids:
        route_dir = processed_dir / route_id
        instance = json.loads((route_dir / "instance.json").read_text(encoding="utf-8"))
        matrix = json.loads((route_dir / "matrix.json").read_text(encoding="utf-8"))
        historical = audit_route(instance, matrix, _historical_route(actuals[route_id], instance["depot_id"]))
        outputs = []
        for name in ("application_inference.json", "solution.json"):
            path = route_dir / name
            if path.is_file():
                outputs.append((name, json.loads(path.read_text(encoding="utf-8"))))
        if not outputs:
            outputs = [("historical_only", {})]
        for name, payload in outputs:
            candidates = payload.items() if name == "application_inference.json" else [("solution", payload)]
            for method, solution in candidates:
                if not isinstance(solution, dict) or not isinstance(solution.get("route"), list):
                    continue
                audit = audit_route(instance, matrix, solution["route"])
                metrics = audit.get("metrics") or {}
                rows.append({"route_id": route_id, "method": method, "historical_feasible": historical["feasible"], "feasible": audit["feasible"], "travel_seconds": metrics.get("travel_seconds", ""), "lateness_seconds": metrics.get("lateness_seconds", ""), "time_window_violations": metrics.get("time_window_violations", ""), "historical_travel_seconds": (historical.get("metrics") or {}).get("travel_seconds", "")})
        if not outputs or outputs == [("historical_only", {})]:
            rows.append({"route_id": route_id, "method": "historical", "historical_feasible": historical["feasible"], "feasible": historical["feasible"], "travel_seconds": (historical.get("metrics") or {}).get("travel_seconds", ""), "lateness_seconds": (historical.get("metrics") or {}).get("lateness_seconds", ""), "time_window_violations": (historical.get("metrics") or {}).get("time_window_violations", ""), "historical_travel_seconds": (historical.get("metrics") or {}).get("travel_seconds", "")})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ["route_id", "method", "historical_feasible", "feasible", "travel_seconds", "lateness_seconds", "time_window_violations", "historical_travel_seconds"]
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actual-sequences", required=True, type=Path,
                        help="Explicit model_score_inputs/new_actual_sequences.json")
    parser.add_argument("--route-list", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps({"rows": len(run(args.actual_sequences, args.route_list, args.processed_dir, args.output))}, indent=2))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
