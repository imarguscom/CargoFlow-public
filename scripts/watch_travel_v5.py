#!/usr/bin/env python3
"""Stream a detached stage log; a disconnected watcher never owns the solve."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('exit_marker',type=Path);a=p.parse_args()
offset=0
while True:
    if a.log.exists():
        with a.log.open() as stream:
            stream.seek(offset);new=stream.read();offset=stream.tell()
        if new:print(new,end='',flush=True)
    if a.exit_marker.exists():
        code=json.loads(a.exit_marker.read_text())['returncode']
        raise SystemExit(code)
    time.sleep(30)
