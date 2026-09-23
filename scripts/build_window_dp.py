#!/usr/bin/env python3
import hashlib,json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'artifacts/runtime_window_dp';out.mkdir(exist_ok=True,parents=True)
source=root/'native/window_dp.cpp';binary=out/'window_dp.so'
subprocess.run(['c++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(binary)],check=True)
(out/'build_provenance.json').write_text(json.dumps({'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
 'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'command':'c++ -O3 -std=c++17 -shared -fPIC native/window_dp.cpp -o artifacts/runtime_window_dp/window_dp.so'},indent=2)+'\n')
