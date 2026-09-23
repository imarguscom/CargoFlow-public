import json,subprocess,tarfile,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'artifacts/runtime_lkh'
source=root/'LKH-3.0.13'
proof=root/'nonmetric_proof';proof.mkdir(exist_ok=False)
with tarfile.open(root/'LKH-3.0.13.tgz') as tar:
 (proof/'original_main.c').write_bytes(tar.extractfile('LKH-3.0.13/SRC/LKHmain.c').read())
subprocess.run(['cc','-O3','-fcommon','-DTWO_LEVEL_TREE','-I'+str(source/'SRC/INCLUDE'),'-c',str(proof/'original_main.c'),'-o',str(proof/'original_main.o')],check=True)
objects=[str(p) for p in (source/'SRC/OBJ').glob('*.o') if p.name!='LKHmain.o']
subprocess.run(['cc','-o',str(proof/'LKH_original'),*objects,str(proof/'original_main.o'),'-lm'],check=True)
(proof/'problem.tsp').write_text('NAME : nonmetric-proof\nTYPE : TSPTW\nDIMENSION : 4\nEDGE_WEIGHT_TYPE : EXPLICIT\nEDGE_WEIGHT_FORMAT : FULL_MATRIX\nEDGE_WEIGHT_SECTION\n0 20000 20000 1000\n1000 0 1000 20000\n20000 1000 0 20000\n20000 20000 1000 0\nTIME_WINDOW_SECTION\n1 0 1000000\n2 8000 10000\n3 0 5000\n4 0 1000000\nSERVICE_TIME_SECTION\n1 0\n2 125\n3 125\n4 125\nDEPOT_SECTION\n1\n-1\nEOF\n')
results={}
for label,binary in [('original',proof/'LKH_original'),('patched',source/'LKH')]:
 out=proof/(label+'.tour')
 params=proof/(label+'.par')
 params.write_text(f'PROBLEM_FILE = {proof}/problem.tsp\nOUTPUT_TOUR_FILE = {out}\nPRECISION = 1\nRUNS = 10\nMAX_TRIALS = 20\nTRACE_LEVEL = 0\nTOTAL_TIME_LIMIT = .2\n')
 r=subprocess.run([str(binary),str(params)],capture_output=True,text=True,timeout=2)
 results[label]=dict(exit=r.returncode,feasible_tour_written=out.exists(),log=r.stdout+r.stderr,tour=out.read_text() if out.exists() else None)
(proof/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
assert not results['original']['feasible_tour_written'] and results['patched']['feasible_tour_written']
