#!/usr/bin/env python3
"""Seal completed policy evidence and copied planning inputs, never raw Git data."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();assert not a.out.exists()
    read=lambda path:json.loads(path.read_text())
    sha=lambda b:hashlib.sha256(b).hexdigest()
    assert read(root/'exit.json')['returncode']==0 and read(root/'independent_audit.json')['passed']
    protocol=read(root/'protocol.json');mapping={}
    with tarfile.open(a.out,'w:gz',compresslevel=6) as tar:
        def put(name,data):
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
        for path in sorted(root.rglob('*')):
            if path.is_file():tar.add(path,arcname='benchmark50/'+str(path.relative_to(root)),recursive=False)
        for idx,(source,digest) in enumerate(protocol['frozen_hashes'].items()):
            path=Path(source);data=path.read_bytes();assert sha(data)==digest,source
            snapshot_candidates=[]
            for snapshot in (root/'source_snapshot').rglob(path.name):
                relative=snapshot.relative_to(root/'source_snapshot')
                if str(path).endswith('/'+str(relative)) and sha(snapshot.read_bytes())==digest:snapshot_candidates.append(snapshot)
            if snapshot_candidates:
                assert len(snapshot_candidates)==1
                mapping[source]='benchmark50/'+str(snapshot_candidates[0].relative_to(root))
            else:
                name=f'frozen_inputs/{idx}_{path.name}';mapping[source]=name;put(name,data)
        preflight=root.parent/'preflight'
        for path in sorted(preflight.glob('*')):
            if path.is_file():tar.add(path,arcname='preflight/'+path.name,recursive=False)
        put('path_map.json',json.dumps(mapping,indent=2).encode())
    manifest=dict(sha256=sha(a.out.read_bytes()),bytes=a.out.stat().st_size,source_commit=protocol['source_commit'],planning_inputs=len(protocol['rows'])*2,full_results=50)
    a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
