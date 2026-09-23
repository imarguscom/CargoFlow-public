#!/usr/bin/env python3
"""Run default adaptive policy search from a fixed audited seed solution."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cargoflow.policy_search import policy_from_logits  # noqa: E402


POLICIES = (
    policy_from_logits("fastest", 0.00, (0.0, 0.0, 0.0, 0.0), (True, False, False, False)),
    policy_from_logits("delivery_speed", 0.02, (0.0, 1.098612289, 0.0, 0.0), (True, True, False, False)),
    policy_from_logits("balanced", 0.02, (-1.049822124, -1.386294361, -0.916290732, 0.0), (True, True, True, False)),
    policy_from_logits("green", 0.05, (-2.302585093, 0.0, -0.105360516, 0.0), (True, False, True, False), max_energy_increase=0.00),
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--seed-solution", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1.0, help="native search budget per policy (default: 1)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.seconds < 1000:
        parser.error('--seconds must be finite and between 0 and 1000 (exclusive)')
    if not 0 <= args.seed < 2**64 - len(POLICIES) + 1:
        parser.error('--seed and policy offsets must fit unsigned 64-bit integers')
    instance, matrix, solution = map(read_json, (args.instance, args.matrix, args.seed_solution))
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if not solution.get('route'):
        payload = {'status': 'no_feasible_seed', 'feasible': False, 'route': None,
                   'policies': [p.name for p in POLICIES]}
        (args.output_dir/'status.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        print(json.dumps(payload))
        return 0
    results = []
    from frontier_search import search_policy
    for offset, policy in enumerate(POLICIES):
        started = perf_counter()
        result = search_policy(
            instance,
            matrix,
            solution["route"],
            policy,
            seconds=args.seconds,
            seed=args.seed + offset,
        )
        result['policy_seconds'] = perf_counter() - started
        results.append(result)
        (args.output_dir / f"{policy.name}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    reference = results[0]["reference"]
    def change(value, base):
        return 100 * (value/base-1) if base else (0.0 if value == 0 else None)
    with (args.output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "policy", "feasible", "travel_seconds", "travel_change_pct",
            "elapsed_seconds", "waiting_seconds", "energy_proxy", "energy_proxy_change_pct",
            "delivery_completion_sum_seconds", "delivery_completion_change_pct",
        ))
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]
            writer.writerow({
                "policy": result["policy"]["name"],
                "feasible": result["official_audit"]["feasible"],
                "travel_seconds": metrics["travel_seconds"],
                "travel_change_pct": change(metrics["travel_seconds"], reference["travel_seconds"]),
                "elapsed_seconds": metrics["elapsed_seconds"],
                "waiting_seconds": metrics["waiting_seconds"],
                "energy_proxy": metrics["energy_proxy"],
                "energy_proxy_change_pct": change(metrics["energy_proxy"], reference["energy_proxy"]),
                "delivery_completion_sum_seconds": metrics["delivery_completion_sum_seconds"],
                "delivery_completion_change_pct": change(metrics["delivery_completion_sum_seconds"], reference["delivery_completion_sum_seconds"]),
            })
    print(json.dumps({
        result["policy"]["name"]: {
            "travel_seconds": result["metrics"]["travel_seconds"],
            "energy_proxy": result["metrics"]["energy_proxy"],
            "feasible": result["official_audit"]["feasible"],
        }
        for result in results
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
