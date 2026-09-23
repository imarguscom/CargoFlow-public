#!/usr/bin/env python3
"""Verify a relocated evidence archive without editing its original files."""
import argparse,contextlib,hashlib,importlib.util,io,json
from pathlib import Path
from decimal import Decimal
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evidence',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    evidence=a.evidence.resolve();mapping=json.loads((evidence/'path_map.json').read_text())
    spec=importlib.util.spec_from_file_location('independent_travel_audit',Path(__file__).with_name('audit_travel_v5.py'))
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    def resolve(path):return evidence/mapping[str(path)] if str(path) in mapping else Path(path)
    auditor.read=lambda p:json.loads(resolve(p).read_text())
    auditor.sha=lambda p:hashlib.sha256(resolve(p).read_bytes()).hexdigest()
    captured={}
    auditor.write=lambda p,value:captured.__setitem__(str(Path(p).relative_to(evidence)),value)
    total=0;stages={}
    for stage in sorted((evidence/'travel_v5').iterdir()):
        if not stage.is_dir() or not (stage/'stage_protocol.json').exists():continue
        _,records=auditor.audit(stage);total+=len(records);stages[stage.name]=len(records)
    with contextlib.redirect_stdout(io.StringIO()):
        auditor.confirm(evidence/'travel_v5/benchmark',evidence/'travel_v5/locked_selection.json')
    original=json.loads((evidence/'travel_v5/benchmark/decision.json').read_text())
    repeated=captured['travel_v5/benchmark/decision.json']
    assert original['accepted']==repeated['accepted']
    assert original['candidate']==repeated['candidate']
    def same(a,b):
        if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
        if isinstance(a,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
        if isinstance(a,(int,float)) and not isinstance(a,bool):return abs(a-b)<1e-10
        return a==b
    assert same(original,repeated), 'statistical recomputation differs'
    recovery=[]
    for folder in sorted((evidence/'travel_v5/benchmark').glob('resume_*')):
        if not folder.is_dir() or not (folder/'before.json').exists():continue
        before=json.loads((folder/'before.json').read_text())
        assert auditor.sha(evidence/'travel_v5/benchmark/stage_protocol.json')==before['protocol_sha256']
        assert auditor.sha(evidence/'travel_v5/locked_selection.json')==before['lock_sha256']
        assert auditor.sha(evidence/'tools/resume_travel_v5.py')==before['helper_sha256']
        assert all(auditor.sha(evidence/'travel_v5/benchmark/routes'/name)==digest
                   for name,digest in before['preserved_result_sha256'].items())
        recovery.append(dict(attempt=folder.name,preserved_files_verified=len(before['preserved_result_sha256'])))
    capacity_failures=[]
    for row in json.loads((evidence/'travel_v5/protocol.json').read_text())['benchmark']:
        instance=auditor.read(row['instance_path'])
        demand=sum(Decimal(str(n['demand_cm3'])) for n in instance['nodes'])
        capacity=Decimal(str(instance['vehicle_capacity_cm3']))
        if demand>capacity:
            capacity_failures.append(dict(route_id=row['route_id'],demand_cm3=str(demand),capacity_cm3=str(capacity)))
    a.out.parent.mkdir(exist_ok=True,parents=True)
    if a.out.exists():raise FileExistsError(a.out)
    a.out.write_text(json.dumps(dict(passed=True,stage_records=stages,total_records_including_reused=total,
        final_records=stages['benchmark'],recovery_preservation=recovery,capacity_overflow_routes=capacity_failures,decision_matches=True,accepted=original['accepted'],
        note='Independent original-unit replay and paired statistics from copied planning inputs; original evidence files were not rewritten.'),indent=2)+'\n')
    print(a.out.read_text())

if __name__=='__main__':main()
