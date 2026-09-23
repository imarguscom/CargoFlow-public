#!/usr/bin/env python3
"""Freeze constrained variable-depth search comparisons on development only."""
from pathlib import Path
from run_travel_v5 import read,write,baseline
root=Path(__file__).resolve().parents[1]/'artifacts/travel_v5'
protocol=read(root/'protocol.json')
configs=[baseline('balanced'),baseline('reference')]
for engine in ['lkh','ils_lkh']:
    for candidates in [5,20]:
        configs.append(dict(name=f'{engine}_special_c{candidates}',backend='lkh-3.0.13',engine=engine,
                            seconds=3.8,initial_seconds=.5,candidates=candidates,move_type=5,special=True))
for width in [10,12]:
    for initial in [2.5,3.0]:
        configs.append(dict(name=f'ils_dp{width}_init{initial}',backend='0.13.4',engine='window_dp',seconds=3.8,initial_seconds=initial,width=width))
out=root/'lkh_special_spec.json'
if out.exists():raise FileExistsError(out)
write(out,dict(rows=protocol['development'][:12],seeds=[42],configs=configs,
    rationale='Test the author-report SPECIAL preset with/without a fresh short ILS initial route: constrained 5-opt, double-bridge kicks, population 10, no recursive higher-order swaps. Disable only metric-dependent precedence closure; original directed arcs and service-start constraints retained. Includes fresh process startup, TSPLIB construction and audit in 3.8 seconds. Also test fixed-endpoint block DP after 2.5/3-second fresh ILS, widths 10/12, accepting only globally feasible travel improvements.',
    next_selection='At most two zero-feasibility-loss candidates ranked by mean paired travel reduction vs reference move to 24 development routes and two seeds. Keep strict final gain/CI/time/feasibility gate and original50 holdout untouched.'))
print(out)
