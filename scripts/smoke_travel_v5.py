#!/usr/bin/env python3
"""Exercise the accepted CLI once and independently replay its fresh result."""
import json
import subprocess
import sys
from pathlib import Path

from audit_travel_v5 import read, replay, sha

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/travel_v5'


def main():
    decision = read(ART / 'benchmark/decision.json')
    assert decision['accepted'], 'The locked candidate has not passed.'
    name = decision['candidate']['name']
    rows = read(ART / 'benchmark/stage_protocol.json')['rows']
    row = next(row for row in rows if any(
        r['config']['name'] == name and r['seed'] == 42 and r['solution']['feasible']
        for r in read(ART / 'benchmark/routes' / (row['route_id'] + '.json'))
    ))
    folder = ART / 'application_smoke'
    folder.mkdir(exist_ok=False)
    output = folder / 'fresh_result.json'
    cmd = [sys.executable, str(ROOT / 'scripts/solve_travel_locked.py'),
           '--instance', row['instance_path'], '--matrix', row['matrix_path'],
           '--out', str(output)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    (folder / 'stdout.log').write_text(proc.stdout)
    (folder / 'stderr.log').write_text(proc.stderr)
    proc.check_returncode()
    result = read(output)
    metrics, objective, feasible = replay(read(row['instance_path']),
                                         read(row['matrix_path']),
                                         result['solution']['route'])
    assert feasible and result['solution']['feasible']
    assert objective == result['solution']['objective_scaled']
    assert all(abs(v - result['solution']['audit']['metrics'][k]) < 1e-7
               for k, v in metrics.items())
    proof = dict(passed=True, route_id=row['route_id'], seed=42,
                 fresh_result_sha256=sha(output), metrics=metrics,
                 warm_call_seconds=result['end_to_end_seconds'],
                 one_shot_worker_seconds=result['one_shot_worker_seconds'],
                 note='Packaging smoke on one existing benchmark input; a fresh solve, not a new generalization experiment.')
    (folder / 'independent_smoke.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
