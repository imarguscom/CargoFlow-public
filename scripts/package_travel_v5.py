#!/usr/bin/env python3
"""Package immutable planning inputs, completed stages and native provenance."""
import hashlib,io,json,platform,subprocess,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/travel_v5'

def sha_bytes(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    assert read(ART/'benchmark/status.json')['status']=='complete'
    assert read(ART/'benchmark/independent_audit.json')['passed']
    target=ROOT/'artifacts/travel_v5_evidence.tgz'
    if target.exists():raise FileExistsError(target)
    native={};inputs={};mapping={}
    for stage in ART.iterdir():
        if not stage.is_dir() or not (stage/'stage_protocol.json').exists():continue
        assert read(stage/'status.json')['status']=='complete'
        for path,digest in read(stage/'stage_protocol.json')['input_source_hashes'].items():
            if path.endswith('.json'):
                if path in inputs:assert inputs[path]==digest
                inputs[path]=digest
            elif path.endswith(('/LKH','.so','.cpp','.c')):
                if path in native:assert native[path]==digest
                native[path]=digest
    # Snapshot actual environment; neither compiler nor CPU probing runs
    # until the benchmark and independent audit have completely finished.
    environment=dict(platform=platform.platform(),python=platform.python_version(),hostname=platform.node(),created_unix=time.time())
    try:
        environment['compiler']=subprocess.check_output(['cc','--version'],text=True).splitlines()[0]
        environment['cpu_model']=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
        import os
        environment['cpu_affinity']=sorted(os.sched_getaffinity(0))
    except (OSError,StopIteration,AttributeError):pass
    environment['pyvrp_extension_sha256']={}
    old_deps=ROOT/'artifacts/benchmark_runtime/deps'
    for deps in [old_deps,ROOT/'artifacts/runtime13/deps']:
        for binary in sorted((deps/'pyvrp').rglob('*.so')):
            environment['pyvrp_extension_sha256'][str(binary)]=sha_bytes(binary.read_bytes())
    with tarfile.open(target,'w:gz',compresslevel=6) as tar:
        def put(name,data):
            info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
        for path,digest in sorted(inputs.items()):
            data=Path(path).read_bytes();assert sha_bytes(data)==digest
            # The path's parent is the unique route directory in the frozen DTO corpus.
            name='planning_inputs/'+Path(path).parent.name+'/'+Path(path).name
            assert name not in mapping.values()
            mapping[path]=name;put(name,data)
        for idx,(path,digest) in enumerate(sorted(native.items())):
            data=Path(path).read_bytes();assert sha_bytes(data)==digest
            name=f'native/{idx}_{Path(path).name}';mapping[path]=name;put(name,data)
        for path in sorted(ART.rglob('*')):
            if path.is_file():tar.add(path,arcname='travel_v5/'+str(path.relative_to(ART)),recursive=False)
        for path in sorted((ROOT/'scripts').glob('*travel*v5*.py')):
            tar.add(path,arcname='tools/'+path.name,recursive=False)
        for filename in ['solve_travel_locked.py','check_travel_release.py','build_lkh_runtime.py','build_window_dp.py','prove_lkh_nonmetric_patch.py']:
            tar.add(ROOT/'scripts'/filename,arcname='tools/'+filename,recursive=False)
        for path in [ROOT/'artifacts/runtime_lkh/provenance.json',ROOT/'artifacts/runtime_lkh/build_provenance.json',
                     ROOT/'artifacts/runtime_lkh/LKH-3.0.13.tgz',ROOT/'artifacts/runtime_window_dp/build_provenance.json',
                     ROOT/'artifacts/runtime_ortools/wheels/provenance.json']:
            if path.exists():tar.add(path,arcname='runtime_provenance/'+str(path.relative_to(ROOT/'artifacts')),recursive=False)
        for folder in [ROOT/'artifacts/runtime13',ROOT/'artifacts/runtime13/wheels']:
            for path in sorted(folder.glob('*.json')):
                tar.add(path,arcname='runtime_provenance/'+str(path.relative_to(ROOT/'artifacts')),recursive=False)
        put('path_map.json',json.dumps(mapping,indent=2).encode())
        put('environment.json',json.dumps(environment,indent=2).encode())
    info=dict(archive=str(target),sha256=sha_bytes(target.read_bytes()),bytes=target.stat().st_size,
              planning_files=len(inputs),native_files=len(native),environment=environment)
    target.with_suffix('.manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
