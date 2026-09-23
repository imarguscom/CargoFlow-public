"""Isolated latest-version worker; other pinned workers remain unchanged."""
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'artifacts/runtime14/deps'),str(ROOT/'artifacts/benchmark_runtime/deps'),str(ROOT/'src')]
from frontier_pyvrp14 import solve
from importlib.metadata import version
assert version('pyvrp')=='0.14.0'
print(json.dumps(dict(ready='0.14.0',version=version('pyvrp'))),flush=True)
for line in sys.stdin:
    task=json.loads(line);row=task['row']
    instance=json.loads(Path(row['instance_path']).read_text());matrix=json.loads(Path(row['matrix_path']).read_text())
    record=solve(instance,matrix,task['config'],task['seed'])
    record.update(route_id=row['route_id'],host_load=os.getloadavg())
    print(json.dumps(record,allow_nan=False),flush=True)
