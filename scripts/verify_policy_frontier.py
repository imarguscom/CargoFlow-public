"""Local read-only replay of the sealed frontier study."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert not a.out.exists();mapping=json.loads((root/'path_map.json').read_text())
    sys.path.insert(0,str(root/'verification'))
    file=root/'verification/audit_policy_frontier.py';spec=importlib.util.spec_from_file_location('archived_frontier_audit',file)
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    def resolve(p):return root/mapping[str(p)] if str(p) in mapping else Path(p)
    auditor.read=lambda p:json.loads(resolve(p).read_text())
    auditor.sha=lambda p:hashlib.sha256(resolve(p).read_bytes()).hexdigest()
    captured={};auditor.write=lambda p,v:captured.__setitem__(Path(p).name,v)
    with contextlib.redirect_stdout(io.StringIO()):result=auditor.audit(root/'frontier')
    def same(x,y):
        if isinstance(x,dict):return isinstance(y,dict) and x.keys()==y.keys() and all(same(x[k],y[k]) for k in x)
        if isinstance(x,list):return len(x)==len(y) and all(same(a,b) for a,b in zip(x,y))
        if isinstance(x,(int,float)) and not isinstance(x,bool):return abs(x-y)<=1e-9*max(1,abs(x))
        return x==y
    assert same(result,json.loads((root/'frontier/independent_audit.json').read_text()))
    assert same(captured['derived_records.json'],json.loads((root/'frontier/derived_records.json').read_text()))
    output=dict(passed=True,statistics_match=True,evidence_not_modified=True,source_commit=result['source_commit'],counts=result['counts'])
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(output,indent=2)+'\n');print(a.out.read_text())

if __name__=='__main__':main()
