#!/usr/bin/env python3
"""Check adapters and document the small-instance-only pre-lock source change."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
from collections import defaultdict
from statistics import mean,median
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'artifacts/travel_v5'
tests=['test_travel_search.py','test_ortools_search.py','test_lkh_search.py',
       'test_window_dp.py','test_directed_polish.py','test_travel_audit.py']
p=subprocess.run([sys.executable,'-m','pytest',*['tests/'+f for f in tests],'-q'],cwd=ROOT,capture_output=True,text=True)
if (out/'release_tests.log').exists():
    attempt=1
    while (out/f'release_tests_attempt{attempt}.log').exists():attempt+=1
    shutil.copy2(out/'release_tests.log',out/f'release_tests_attempt{attempt}.log')
(out/'release_tests.log').write_text(p.stdout+p.stderr)
print(p.stdout+p.stderr,flush=True)
if p.returncode:raise SystemExit(p.returncode)
spec=json.loads((out/'final_refine/stage_protocol.json').read_text())
sizes=[len(json.loads(Path(r['instance_path']).read_text())['nodes']) for r in spec['rows']]
assert min(sizes)>3, 'revalidate development timing if exact-small branch affects development rows'
changed=[]
for path,digest in spec['input_source_hashes'].items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=digest:changed.append(path)
assert all(Path(path).name=='lkh_search.py' for path in changed)
groups=defaultdict(lambda:defaultdict(list))
for path in (out/'final_refine/routes').glob('*.json'):
    for r in json.loads(path.read_text()):
        if r['solution']['feasible']:groups[r['config']['name']][r['route_id']].append(r['solution']['audit']['metrics']['travel_seconds'])
medians={name:median(mean(values) for values in rows.values()) for name,rows in groups.items()}
result=dict(tests_passed=True,tests=tests,development_min_locations=min(sizes),development_max_locations=max(sizes),
    changed_sources_since_development=changed,exact_small_branch_does_not_affect_development=True,median_route_mean_travel=medians)
(out/'release_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
