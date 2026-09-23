#!/usr/bin/env python3
"""Generate an audited ILS+LKH seed using a separate PyVRP 0.13.4 environment."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--instance', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=3.8)
    parser.add_argument('--initial-seconds', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (math.isfinite(args.seconds) and math.isfinite(args.initial_seconds)
            and 0 < args.initial_seconds < args.seconds):
        parser.error('Budgets must satisfy 0 < --initial-seconds < --seconds')
    if not 1 <= args.seed <= 2147483647:
        parser.error('--seed must be between 1 and 2147483647')
    try:
        installed = version('pyvrp')
    except PackageNotFoundError:
        installed = None
    if installed != '0.13.4':
        python = ROOT/'artifacts/runtime_ils/bin/python'
        if args.worker or not python.is_file():
            parser.error('Create artifacts/runtime_ils with Python venv and install pyvrp==0.13.4; see data/README.md')
        return subprocess.run([str(python), str(Path(__file__).resolve()),
                               *sys.argv[1:], '--worker']).returncode
    from cargoflow.lkh_search import solve_lkh
    instance = json.loads(args.instance.read_text())
    matrix = json.loads(args.matrix.read_text())
    config = dict(name='ils_lkh_c5', backend='lkh-3.0.13', engine='ils_lkh',
                  seconds=args.seconds, initial_seconds=args.initial_seconds,
                  candidates=5, move_type=5)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result = solve_lkh(instance, matrix, config, seed=args.seed)
    solution = result['solution']
    (args.output_dir/'solution.json').write_text(json.dumps(solution, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=solution['status'], feasible=solution['feasible'],
                         metrics=(solution.get('audit') or {}).get('metrics'),
                         solver_seconds=result['end_to_end_seconds']), indent=2))
    return 0 if solution['feasible'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
