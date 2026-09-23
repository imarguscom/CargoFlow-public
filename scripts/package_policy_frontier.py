"""Seal completed frontier evidence, inputs and frozen source/runtime files."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert not a.out.exists()
    read=lambda p:json.loads(p.read_text());sha=lambda b:hashlib.sha256(b).hexdigest()
    assert read(root/'exit.json')['returncode']==0 and read(root/'independent_audit.json')['passed']
    protocol=read(root/'protocol.json');mapping={}
    with tarfile.open(a.out,'w:gz',compresslevel=3) as tar:
        def put(name,data):
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
        for path in sorted(root.rglob('*')):
            if path.is_file():tar.add(path,arcname='frontier/'+str(path.relative_to(root)),recursive=False)
        for index,(source,digest) in enumerate(protocol['frozen_hashes'].items()):
            path=Path(source);data=path.read_bytes();assert sha(data)==digest,source
            candidates=[p for p in (root/'source_snapshot').rglob(path.name) if str(path).endswith('/'+str(p.relative_to(root/'source_snapshot'))) and sha(p.read_bytes())==digest]
            if candidates:
                assert len(candidates)==1;mapping[source]='frontier/'+str(candidates[0].relative_to(root))
            else:
                name=f'frozen_inputs/{index}_{path.name}';put(name,data);mapping[source]=name
        for path in sorted((root.parent/'preflight').glob('*')):
            if path.is_file():tar.add(path,arcname='preflight/'+path.name,recursive=False)
        auditor=Path(__file__).with_name('audit_policy_frontier.py')
        put('verification/audit_policy_frontier.py',auditor.read_bytes())
        put('verification/audit_policy_benchmark.py',Path(__file__).with_name('audit_policy_benchmark.py').read_bytes())
        put('path_map.json',json.dumps(mapping,indent=2).encode())
    manifest=dict(source_commit=protocol['source_commit'],sha256=sha(a.out.read_bytes()),bytes=a.out.stat().st_size,
                  planning_input_count=2*(len(protocol['rows'])+len(protocol['development'])),auditor_sha256=sha(auditor.read_bytes()))
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
