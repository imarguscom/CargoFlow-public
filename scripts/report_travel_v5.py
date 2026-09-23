#!/usr/bin/env python3
"""Create a Chinese result report from sealed, audited benchmark artifacts."""
import argparse,json
from collections import defaultdict
from pathlib import Path
from statistics import mean,median


def read(p):return json.loads(p.read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('artifacts',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();root=a.artifacts.resolve();out=a.out.resolve();out.mkdir(exist_ok=True,parents=True)
    protocol=read(root/'protocol.json');decision=read(root/'benchmark/decision.json');audit=read(root/'benchmark/independent_audit.json')
    assert audit['passed']
    records=[r for f in (root/'benchmark/routes').glob('*.json') for r in read(f)]
    groups=defaultdict(lambda:defaultdict(list))
    for r in records:groups[r['config']['name']][r['route_id']].append(r)
    candidate=decision['candidate']['name'];names=['old_balanced','old_reference',candidate]
    label={'old_balanced':'PyVRP balanced 0.12.2','old_reference':'PyVRP reference 0.12.2',candidate:candidate}
    table=[];plot_values=[]
    for name in names:
        rows=groups[name];feasible=[v for v in rows.values() if all(r['solution']['feasible'] for r in v)]
        travel=median(mean(r['solution']['audit']['metrics']['travel_seconds'] for r in v) for v in feasible)
        runtime=median(mean(r['end_to_end_seconds'] for r in v) for v in feasible)
        calls=sum(r['solution']['feasible'] for v in rows.values() for r in v)
        seed42=[r['solution']['audit']['metrics']['travel_seconds'] for v in rows.values() for r in v if r['seed']==42 and r['solution']['feasible']]
        table.append(f'| {label[name]} | {len(feasible)}/{len(rows)} | {calls}/{sum(map(len,rows.values()))} | {travel:,.1f} | {runtime:.3f} | {median(seed42):,.1f} |')
        plot_values.append(runtime)
    rows=[]
    for base,v in decision['comparisons'].items():
        ci=v['travel_reduction_ci95']
        rows.append(f"| {label[base]} | {100*v['mean_travel_reduction']:.3f}% | [{100*ci[0]:.3f}%, {100*ci[1]:.3f}%] | {100*(1-v['median_paired_runtime_ratio']):.1f}% | {v['feasible_losses']} | {'通过' if v['passed'] else '未通过'} |")
    original_table=[]
    for name in names:
        v=decision['seed42_comparison'][name]
        original_table.append(f"| {label[name]} | {v['feasible']}/{v['total']} | {v['median_travel']:,.1f} | {v['median_end_to_end']:.3f} |")
    paired_counts=[]
    index={(r['route_id'],r['seed'],r['config']['name']):r for r in records}
    for base in names[:2]:
        counts=dict(better=0,equal=0,worse=0)
        for r in records:
            if r['config']['name']!=candidate:continue
            b=index[r['route_id'],r['seed'],base]
            if not(r['solution']['feasible'] and b['solution']['feasible']):continue
            delta=r['solution']['audit']['metrics']['travel_seconds']-b['solution']['audit']['metrics']['travel_seconds']
            counts['equal' if abs(delta)<1e-7 else 'better' if delta<0 else 'worse']+=1
        paired_counts.append(f"相对 {label[base]} 的 147 个可行配对调用中，{counts['better']} 次更短、{counts['equal']} 次相同、{counts['worse']} 次更长。")
    delivery_note=''
    if (out/'delivery_check.json').exists() and (out/'local_verification.json').exists():
        delivered=read(out/'delivery_check.json');verified=read(out/'local_verification.json')
        assert delivered['passed'] and verified['passed'] and verified['decision_matches']
        delivery_note=f"本地离线复核再次通过全部 {verified['final_records']} 条最终结果，并精确复现统计判定；包含开发及组件重放的 {verified['total_records_including_reused']} 条记录也逐条通过约束检查。发布前针对适配器、反例与审核器的测试为 {delivered['release_tests']}。实际入口完成了一次新求解，调用耗时 {delivered['warm_call_seconds']:.3f} 秒，包含一次性进程启动为 {delivered['one_shot_worker_seconds']:.3f} 秒；这只是入口验证的一个样本，不是冷启动性能对照。锁定核心提交为 `{delivered['source_commit_at_lock']}`。\n\n证据包 SHA-256：`{delivered['archive_sha256']}`。详见 [本地复核]({out/'local_verification.json'})、[交付检查]({out/'delivery_check.json'})、[使用方式]({out/'USAGE.md'})。"
    stages=[]
    for name in ['screen','refine','diverse','lkh','lkh_special','final_refine']:
        stage=root/name
        if not (stage/'summary.json').exists():continue
        spec=read(stage/'stage_protocol.json');av=read(stage/'independent_audit.json');assert av['passed']
        for v in read(stage/'summary.json'):
            if v['config']['engine']=='baseline':continue
            c=v['comparisons']['old_reference']
            stages.append(f"| {name} | {len(spec['rows'])} × {len(spec['seeds'])} | {v['config']['name']} | {v['feasible']}/{v['records']} | {100*c['mean_travel_gain']:.3f}% | {v['median_end_to_end']:.3f} |")
    (out/'development_trials.md').write_text('# 开发集完整尝试记录\n\n正的百分比表示行驶时间下降。不同阶段样本数不同，不能直接混合排名。所有记录来自新求解；此前的 2-opt 组件重放单列保存，没有计作新实验。\n\n| 阶段 | 路线 × 种子 | 配置 | 可行调用 | 相对 reference 平均行驶时间下降 | 中位求解秒 |\n|---|---:|---|---:|---:|---:|\n'+'\n'.join(stages)+'\n')
    recovery_note=''
    recovery=list((root/'benchmark').glob('resume_*/before.json'))
    if recovery:
        counts=[len(read(p)['preserved_result_sha256']) for p in recovery]
        recovery_note=f'正式测试曾因连接中断而停止，随后按完整路线检查点续跑（首次保留 {counts[0]} 条）。已完成结果的逐文件哈希、锁定配置与源码均保持不变；续跑沿用原始的路线、种子和方法次序。中断记录与校验保留在 benchmark/resume_* 中。'
    status='通过预设的全部验收条件' if decision['accepted'] else '未通过预设的全部验收条件'
    text=f'''# CargoFlow：行驶时间与求解速度对照结果

**结论：{status}。** 本次目标是保持可行率，同时超过原 PyVRP balanced 和 reference 的行驶时间与求解速度。候选为 `{candidate}`。

## 与用户原表直接比较：相同 50 条路线、seed=42

| 方法 | 可行路线 | 中位行驶秒 | 本次中位求解秒 |
|---|---:|---:|---:|
{chr(10).join(original_table)}

reference 的行驶时间中位数精确复现原表的 11,526.3 秒（校验标志：`{decision['original_reference_median_reproduced']}`）。运行时间使用本次同机实测，原表为约 16.63 秒。候选使用 PyVRP 0.13.4 的 ILS 产生初始解，再由 LKH 在总预算 3.8 秒内改进。

## 三个种子的复核

每条路线运行种子 42、20260907、271828，先在路线内取平均，再计算路线中位数。可行路线要求三个种子全部可行。行驶时间不含服务与等待；求解时间包含验证、建模、初始化、搜索与结果检查，预加载依赖和读取原始输入的时间按原基线口径排除。中位数按各方法全部种子可行的路线计算。

| 方法 | 可行路线 | 可行调用 | 中位行驶秒 | 中位求解秒 | seed42 中位行驶秒 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

容量超过单车上限的原始输入仍保留在分母中。没有增加车辆、放宽时间窗、丢弃客户或修改有向行驶矩阵。

## 逐路线配对结论

百分比先在同一路线、同一种子下相对基线计算，再在路线内平均；因此不同于“两列中位数相减”的百分比。95% 区间按站点日期整组重采样 10,000 次。

| 相对基线 | 平均行驶时间下降 | 95% 区间 | 中位配对求解时间下降 | 新增不可行调用 | 验收 |
|---|---:|---:|---:|---:|---|
{chr(10).join(rows)}

{' '.join(paired_counts)}

![配对行驶收益与求解耗时]({(out/'comparison.png').resolve()})

验收要求同时满足：零可行性损失、平均配对行驶时间至少下降 0.5%、95% 区间下界大于零、路线中位行驶时间下降、配对求解时间中位比值不超过 0.8。统计范围仅限本批 50 条路线，不能保证每条路线都更短，也不能作为其他地区数据的泛化证明。

## 方法与筛选

使用开发集测试了旧版 HGS 参数、原生建模、新版 ILS、初始化与多次重启、引导局部搜索、可行 2-opt、LKH 及论文推荐设置、固定端点分块动态规划。开发站点日期与原始 50 条测试路线隔离。先按预先写下的规则筛选，再在 24 条开发路线、两个种子上复核；配置锁定后才运行此处的原始 50 条路线。[完整开发结果](development_trials.md)。

最终配置：

```json
{json.dumps(decision['candidate'],indent=2)}
```

PyVRP、OR-Tools、LKH 的算法贡献属于其作者。本次工作是约束适配、预算管理、候选组合与验证。LKH 3.0.13 的 TSPTW 代价包含恒定的服务时间，报告重新计算纯行驶代价；其依赖三角不等式的先后顺序推断被关闭，并用非度量有向反例对照穷举。LKH 许可限学术与非商业使用，不能据此直接作为商业部署依赖。原应用默认 PyVRP 求解器保持原样。

## 证据与复现

最终独立检查通过 {audit['records']} 条结果：重新计算访问覆盖、容量、固定出发时间、服务开始时间窗、原始行驶时间及整数目标。输入、源码和原生二进制哈希随阶段保存。每个最终结果都来自新输入求解，未读取缓存路线；开发阶段的组件重放估计没有用于本次速度结论。

{recovery_note}

{delivery_note}

主要证据：[最终判定]({root/'benchmark/decision.json'})、[独立审核]({root/'benchmark/independent_audit.json'})、`benchmark/routes/`、`locked_selection.json` 和各阶段 `stage_protocol.json`。配置好实验输入和运行时后，可使用 `scripts/solve_travel_locked.py --instance ... --matrix ... --out ...`；该入口检查锁定配置和源码哈希，并分别报告求解调用时间与一次性进程启动后的实际延迟。本地证据包含 Linux 原生文件，仅用于复核，不能直接当作 macOS 运行时。

## 主要参考

- [PyVRP 0.13.4 发布记录](https://github.com/PyVRP/PyVRP/releases/tag/v0.13.4)。
- [OR-Tools 路由搜索设置](https://developers.google.com/optimization/routing/routing_options)。
- [Helsgaun 2017：LKH-3 技术报告](https://forskning.ruc.dk/en/publications/an-extension-of-the-lin-kernighan-helsgaun-tsp-solver-for-constra/)。
- [Held 与 Karp 1962：动态规划](https://epubs.siam.org/doi/10.1137/0110015)。
'''
    (out/'REPORT.md').write_text(text)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
    comparisons=[decision['comparisons'][b] for b in names[:2]]
    gains=np.array([100*v['mean_travel_reduction'] for v in comparisons]);ci=np.array([v['travel_reduction_ci95'] for v in comparisons])*100
    ax[0].errorbar(gains,[1,0],xerr=np.array([gains-ci[:,0],ci[:,1]-gains]),fmt='o',capsize=5,color='#2463A6',markersize=8)
    ax[0].axvline(0,color='#a0a0a0',linewidth=1);ax[0].set_yticks([1,0],['vs Balanced','vs Reference']);ax[0].set_ylim(-.5,1.5)
    ax[0].set_xlabel('Paired driving time reduction (%)');ax[0].set_title('Route means and cluster 95% intervals')
    bars=ax[1].bar(['Balanced','Reference','Candidate'],plot_values,color=['#9CA3AF','#6B7280','#2463A6'],width=.6)
    ax[1].bar_label(bars,fmt='%.2f s',padding=4);ax[1].set_ylim(0,max(plot_values)*1.2)
    ax[1].set_ylabel('Median solver time (seconds)');ax[1].set_title('Fresh calls on the same host')
    fig.savefig(out/'comparison.png',dpi=180);fig.savefig(out/'comparison.pdf');plt.close(fig)
    print(out/'REPORT.md')

if __name__=='__main__':main()
