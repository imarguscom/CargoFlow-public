#!/usr/bin/env python3
"""Run application inference from a new route, package, and travel record.

This command intentionally has no option for an application actual-sequence
file.  It constructs the canonical instance without historical order and runs
pure PyVRP plus the optional learned-cost PyVRP result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cargoflow.preferences import preference_cost_matrix  # noqa: E402
from cargoflow.solver import solve_instance  # noqa: E402

CONVERTER_PATH = ROOT / "scripts" / "convert_amazon_route.py"
SPEC = importlib.util.spec_from_file_location("convert_amazon_route_application", CONVERTER_PATH)
if not SPEC or not SPEC.loader:
    raise RuntimeError(f"cannot load converter: {CONVERTER_PATH}")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


def _load(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        # The Amazon application inputs encode missing windows as bare NaN;
        # normalize those sentinels to JSON null before contract conversion.
        return json.load(stream, parse_constant=lambda _value: None)


def _unwrap(value: object, route_id: str, label: str) -> object:
    if isinstance(value, dict) and route_id in value and isinstance(value[route_id], dict):
        return value[route_id]
    if isinstance(value, dict):
        return value
    raise ValueError(f"{label} must be a route object or a route-keyed object")


def _route_metadata(route: dict[str, object]) -> dict[str, object]:
    stops = route.get("stops")
    zone_by_stop: dict[str, str] = {}
    if isinstance(stops, dict):
        for stop_id, stop in stops.items():
            if isinstance(stop, dict):
                zone = stop.get("zone_id")
                if isinstance(zone, str) and zone:
                    zone_by_stop[str(stop_id)] = zone
    station = route.get("station_code")
    return {
        "station_code": station if isinstance(station, str) and station else "unknown",
        "zone_by_stop": zone_by_stop,
    }


def run(
    route_id: str,
    route_path: Path,
    package_path: Path,
    travel_path: Path,
    output_dir: Path,
    *,
    model_path: Path | None = None,
    lambda_: float = 0.0,
    station_code: str | None = None,
    seed: int = 42,
    iterations: int = 5000,
) -> dict[str, object]:
    route = _unwrap(_load(route_path), route_id, "new_route")
    packages = _unwrap(_load(package_path), route_id, "package")
    travel = _unwrap(_load(travel_path), route_id, "travel")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(route, dict):
        raise ValueError("new_route must be an object")
    metadata = _route_metadata(route)
    if station_code and station_code != "unknown":
        metadata["station_code"] = station_code
    converter.convert_records(route_id, route, packages, travel, None, output_dir, emit_summary=False)
    instance = _load(output_dir / "instance.json")
    matrix = _load(output_dir / "matrix.json")
    pure = solve_instance(instance, matrix, seed=seed, iterations=iterations)
    result: dict[str, object] = {"pure_pyvrp": pure, "learned": None, "lambda": lambda_}
    if model_path is not None:
        with model_path.open("rb") as stream:
            model = pickle.load(stream)
        objective = preference_cost_matrix(instance, matrix, model, lambda_=lambda_, metadata=metadata)
        result["learned"] = solve_instance(instance, matrix, seed=seed, iterations=iterations, objective_cost_seconds=objective)
    (output_dir / "application_inference.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def run_batch(
    route_path: Path,
    package_path: Path,
    travel_path: Path,
    output_dir: Path,
    *,
    route_ids: list[str] | None = None,
    model_path: Path | None = None,
    lambda_: float = 0.0,
    station_code: str | None = None,
    seed: int = 42,
    iterations: int = 5000,
) -> dict[str, object]:
    """Run all route keys in the application score-input files.

    These files are intentionally separate from the historical score-input
    sequence file.  A route-list can lock the intended 13-route cohort;
    otherwise the common route keys are used in deterministic order.
    """

    route_records = _load(route_path)
    package_records = _load(package_path)
    travel_records = _load(travel_path)
    if not all(isinstance(value, dict) for value in (route_records, package_records, travel_records)):
        raise ValueError("application input files must be route-keyed objects")
    common = set(route_records) & set(package_records) & set(travel_records)
    selected = route_ids or sorted(common)
    missing = sorted(set(selected) - common)
    if missing:
        raise ValueError(f"application input files are missing route {missing[0]}")
    results: dict[str, object] = {}
    for route_id in selected:
        route_dir = output_dir / route_id
        results[route_id] = run(
            route_id,
            route_path,
            package_path,
            travel_path,
            route_dir,
            model_path=model_path,
            lambda_=lambda_,
            station_code=station_code,
            seed=seed,
            iterations=iterations,
        )
    (output_dir / "batch_summary.json").write_text(
        json.dumps({"route_ids": selected, "results": results}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return {"route_ids": selected, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id")
    parser.add_argument("--route-list", type=Path,
                        help="Optional newline-delimited application route IDs")
    parser.add_argument("--new-route", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--travel", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=0.0)
    parser.add_argument("--station-code")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5000)
    args = parser.parse_args()
    try:
        if args.route_id:
            result = run(args.route_id, args.new_route, args.package, args.travel, args.output_dir, model_path=args.model, lambda_=args.lambda_, station_code=args.station_code, seed=args.seed, iterations=args.iterations)
        else:
            selected = None
            if args.route_list:
                selected = [line.strip() for line in args.route_list.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
            result = run_batch(args.new_route, args.package, args.travel, args.output_dir, route_ids=selected, model_path=args.model, lambda_=args.lambda_, station_code=args.station_code, seed=args.seed, iterations=args.iterations)
        print(json.dumps(result, indent=2))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
