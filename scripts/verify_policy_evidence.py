#!/usr/bin/env python3
"""Read-only local recomputation using the archive's frozen independent auditor."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists()
    root=a.evidence.resolve();mapping=json.loads((root/'path_map.json').read_text())
    file=root/'benchmark50/source_snapshot/scripts/audit_policy_benchmark.py'
    spec=importlib.util.spec_from_file_location('frozen_policy_auditor',file);auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    def resolve(p):return root/mapping[str(p)] if str(p) in mapping else Path(p)
    auditor.read=lambda p:json.loads(resolve(p).read_text())
    auditor.sha=lambda p:hashlib.sha256(resolve(p).read_bytes()).hexdigest()
    captured={};auditor.write=lambda p,v:captured.__setitem__(Path(p).name,v)
    with contextlib.redirect_stdout(io.StringIO()):result=auditor.audit(root/'benchmark50')
    original=json.loads((root/'benchmark50/independent_audit.json').read_text())
    def same(x,y):
        if isinstance(x,dict):return x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,list):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
        if isinstance(x,(int,float)) and not isinstance(x,bool):return abs(x-y)<1e-10
        return x==y
    assert same(result,original),'Independent statistical result differs'
    assert same(captured['paired_effects.json'],json.loads((root/'benchmark50/paired_effects.json').read_text()))
    preflight=json.loads((root/'preflight/results.json').read_text())
    assert len(preflight)==2 and all(r['returncode']==0 for r in preflight)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    result=dict(passed=True,statistics_match=True,source_commit=result['source_commit'],routes=result['routes'],seed_solver_records=result['seed_solver_records'],policy_records=result['policy_records'],feasible_policy_records=result['feasible_policy_records'],evidence_not_modified=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(a.out.read_text())


if __name__=='__main__':main()
