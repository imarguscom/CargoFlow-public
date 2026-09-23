#!/usr/bin/env python3
"""Solve a CargoFlow instance, or independently audit a saved tour."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cargoflow.solver import audit_route, solve_instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--runtime-seconds", type=float)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-no-improvement", type=int)
    parser.add_argument("--objective-cost", type=Path,
                        help="Optional JSON square matrix used only for PyVRP distance cost")
    parser.add_argument("--check-solution", type=Path,
                        help="Replay a saved route without running the solver")
    args = parser.parse_args()
    try:
        instance = json.loads(args.instance.read_text(encoding="utf-8"))
        matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
        if args.check_solution:
            saved = json.loads(args.check_solution.read_text(encoding="utf-8"))
            if saved.get("instance_id") != instance["instance_id"]:
                raise ValueError("saved solution instance_id differs")
            audit = audit_route(instance, matrix, saved["route"])
            result = {"instance_id": instance["instance_id"], "feasible": audit["feasible"],
                      "status": "audit_passed" if audit["feasible"] else "audit_failed", "audit": audit}
            filename = "audit.json"
        else:
            objective = json.loads(args.objective_cost.read_text(encoding="utf-8")) if args.objective_cost else None
            result = solve_instance(
                instance,
                matrix,
                seed=args.seed,
                iterations=args.iterations,
                runtime_seconds=args.runtime_seconds,
                max_iterations=args.max_iterations,
                max_no_improvement=args.max_no_improvement,
                objective_cost_seconds=objective,
            )
            filename = "solution.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / filename).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        schedule = (result.get("audit") or {}).get("schedule", [])
        # Write even on failure, so a previous run's schedule cannot be mistaken
        # for this run's result when reusing an output directory.
        with (args.output_dir / "schedule.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            if schedule:
                writer = csv.DictWriter(handle, fieldnames=list(schedule[0]))
                writer.writeheader()
                writer.writerows(schedule)
        print(json.dumps({"status": result["status"], "feasible": result["feasible"],
                          "metrics": (result.get("audit") or {}).get("metrics"),
                          "diagnostics": result.get("diagnostics", [])}, indent=2))
        return 0 if result["feasible"] else 2
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
