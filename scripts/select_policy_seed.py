from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cargoflow.solver import audit_route


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the shorter feasible route from the baseline and ILS results."
    )
    parser.add_argument("--route-list", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--ils-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def travel_seconds(solution: dict) -> float:
    return float(solution["audit"]["metrics"]["travel_seconds"])


def choose_seed(instance, matrix, candidates):
    """Compare only independently replayed feasible candidates."""
    eligible = []
    for source, solution in candidates:
        route = solution.get('route')
        if not solution.get('feasible') or not route:
            continue
        audit = audit_route(instance, matrix, route)
        if audit['feasible']:
            eligible.append((audit['metrics']['travel_seconds'], source, route, audit))
    if not eligible:
        return dict(feasible=False, status='no_feasible_seed', route=None,
                    source=None, travel_seconds=None, audit=None)
    cost, source, route, audit = min(eligible, key=lambda v: v[0])
    return dict(feasible=True, status='feasible', route=route, source=source,
                travel_seconds=cost, audit=audit)


def main() -> None:
    args = parse_args()
    route_ids = [
        line.strip()
        for line in args.route_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=False)

    for route_id in route_ids:
        baseline = json.loads(
            (args.processed_dir / route_id / "solution.json").read_text(encoding="utf-8")
        )
        ils_result = json.loads(
            (args.ils_dir / f"{route_id}.json").read_text(encoding="utf-8")
        )
        ils = ils_result["solution"]
        route_dir = args.processed_dir / route_id
        instance = json.loads((route_dir / 'instance.json').read_text(encoding='utf-8'))
        matrix = json.loads((route_dir / 'matrix.json').read_text(encoding='utf-8'))
        payload = choose_seed(instance, matrix, [('pyvrp_0.12.2_baseline', baseline), ('pyvrp_0.13.4_ils', ils)])
        destination = args.output_dir / f"{route_id}.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(route_id, payload["source"], payload["travel_seconds"], flush=True)


if __name__ == "__main__":
    main()
