from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cargoflow.travel_search import solve_travel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the PyVRP 0.13.x ILS configuration on a fixed route list."
    )
    parser.add_argument("--route-list", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3.8)
    parser.add_argument("--neighbours", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    route_ids = [
        line.strip()
        for line in args.route_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "name": f"pyvrp_ils_{args.seconds:g}s_n{args.neighbours}",
        "backend": "0.13.4",
        "engine": "native",
        "seconds": args.seconds,
        "neighbours": args.neighbours,
    }

    for route_id in route_ids:
        route_dir = args.processed_dir / route_id
        instance = json.loads((route_dir / "instance.json").read_text(encoding="utf-8"))
        matrix = json.loads((route_dir / "matrix.json").read_text(encoding="utf-8"))
        result = solve_travel(instance, matrix, config, seed=args.seed)
        destination = args.output_dir / f"{route_id}.json"
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
        solution = result["solution"]
        metrics = (solution.get("audit") or {}).get("metrics") or {}
        print(
            route_id,
            solution["status"],
            f"travel_seconds={metrics.get('travel_seconds')}",
            f"solve_seconds={result['end_to_end_seconds']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
