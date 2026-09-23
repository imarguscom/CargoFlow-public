#!/usr/bin/env python3
"""Freeze development-only alternatives after single-trajectory ILS plateau."""
from pathlib import Path
from run_travel_v5 import read,write,baseline
root=Path(__file__).resolve().parents[1]/'artifacts/travel_v5'
protocol=read(root/'protocol.json')
configs=[baseline('balanced'),baseline('reference')]
for first,tag in [('PARALLEL_CHEAPEST_INSERTION','insert'),('PATH_CHEAPEST_ARC','arc')]:
    configs.append(dict(name='gls_'+tag,backend='ortools-9.15.6755',engine='ortools_gls',seconds=3.8,first=first,gls_lambda=.1))
for lam in [.1,.2]:
    configs.append(dict(name='ils_gls_'+str(lam).replace('.','p'),backend='ortools-9.15.6755',engine='ils_gls',seconds=3.8,initial_seconds=.7,neighbours=80,gls_lambda=lam))
for neighbours in [20,80]:
    for restarts in [4,8]:
        configs.append(dict(name=f'ils_n{neighbours}_restart{restarts}',backend='0.13.4',engine='multistart',seconds=3.8,neighbours=neighbours,restarts=restarts))
out=root/'diverse_spec.json'
if out.exists():raise FileExistsError(out)
write(out,dict(rows=protocol['development'][:12],seeds=[42],configs=configs,
    rationale='Single ILS trajectory and feasible directed 2-opt plateau near reference. Test upstream GLS and independent ILS restarts under the same total 3.8-second service budget, with fresh paired old baselines. No final benchmark inputs used.',
    next_selection='Retain at most two configurations with zero baseline feasible losses and greatest mean paired travel reduction vs reference for 24 development routes and two seeds. Require lower median route travel and positive mean gain versus both baselines before final lock. Final 0.5% gain/positive CI/20% time gate unchanged.'))
print(out)
