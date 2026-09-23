"""Generate a result-derived report and standalone saturation figure."""
import argparse
import json
import os
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.evidence;out=a.out;out.mkdir(parents=True,exist_ok=True)
    r=json.loads((root/'frontier/independent_audit.json').read_text());assert r['passed']
    assert json.loads((out/'local_verification.json').read_text())['passed']
    names={'fastest':'纯行驶对照','delivery_speed':'快速送达','balanced':'均衡','green':'绿色代理'}
    selected=r['selected_method'];comparison=[];native=[];pipeline=[];bounds=[];seeds=[];saturation=[]
    pct=lambda x:f'{100*x:+.3f}%'
    for name,label in names.items():
        v=r['confirmation'][name][selected]
        comparison.append(f"| {label} | {pct(v['gain_vs_python']['mean'])} | {pct(v['travel_change']['mean'])} | {pct(v['completion_change']['mean'])} | {pct(v['energy_change']['mean'])} | {v['median_wrapper_seconds']:.3f} |")
        d=r['selected_vs_native_random'][name]
        native.append(f"| {label} | {pct(d['mean'])} | [{pct(d['ci95'][0])}, {pct(d['ci95'][1])}] | {d['better']}/{d['worse']} |")
        v=r['pipeline'][name]
        pipeline.append(f"| {label} | {v['feasible']}/50 | {v['median_seconds']:.3f} | {v['median_old_call_seconds']:.3f} | {pct(v['changes']['travel_change']['mean'])} | {pct(v['changes']['completion_change']['mean'])} | {pct(v['changes']['energy_change']['mean'])} | {v['old_anchor_travel_violations']} | {v['old_anchor_energy_violations']} |")
        b=r['certified_gap_summary'][name]
        bounds.append(f"| {label} | {100*b['median']:.2f}% | {100*b['min']:.2f}%–{100*b['max']:.2f}% |")
        for method,modes in r['development_saturation'].items():
            values=modes[name]
            saturation.append(f"| {label} | {method} | "+' | '.join(f"{100*(1-values[str(b)]):.3f}%" for b in [.25,1,4,8])+" |")
    for name,v in r['seed_comparison'].items():
        paired=v['vs_ils_lkh'];repeat=v['repeated_subset'];ci=paired['ci95'] if paired else [0,0]
        seeds.append(f"| {name} | {v['feasible']}/{v['total']} | {v['median_seconds']:.3f} | {pct(paired['mean']) if paired else 'NA'} | [{pct(ci[0])}, {pct(ci[1])}] | {repeat['feasible']}/{repeat['total']} | {pct(repeat['vs_ils_lkh']['mean']) if repeat['vs_ils_lkh'] else 'NA'} |")
    mip=[]
    for policy,label in names.items():
        rows=[x for x in r['numerical_mip_bounds'] if x['policy']==policy]
        optimal=sum(x['status']==0 for x in rows);finite=[x['bound'] for x in rows if x['bound'] is not None and x['bound']>=0]
        interval=f'{min(finite):.6f}–{max(finite):.6f}' if finite else '未形成非负数值界'
        mip.append(f"| {label} | {optimal}/{len(rows)} | {interval} |")
    report=f'''# 固定参考路线的优化空间与当前强算法对比

实测源码：`{r['source_commit']}`；开发集选择的方法：**{selected}**。
本轮是独立实验，未修改 main、原始基线或生产多目标策略。
所有完整结果已在实验环境中 独立重算，并在本地从复制输入复算一致。
结果解释与采用建议见 [本轮判断](DECISION.md)。

## 实验规模与可解释范围

- 6 条与 original50 站点日期分离的开发路线；3 种搜索方法、4 种策略、2 个随机种子，记录 0.25/1/4/8 秒曲线。
- 原始 50 条固定参考路线确认；1 条本身超容量，49 条进入两种方法、四种策略、两个种子的搜索对比。
- 五种旅行时间求解器、共同 3.8 秒调用预算：50 条 seed42，另 10 条元数据选出的路线重复 seed271828，对比阶段共 {r['counts']['fresh_solver_calls']} 次新求解，另有 {r['counts']['development_seed_calls']} 次开发参考求解。
- 全部可行路线的可验证下界，另 4 条完整路线 × 4 策略的 SCIP 模型，每个最多 10 秒求解，模型构建另计。
- 独立审核了 **{r['counts']['checkpoint_routes']} 条检查点路线**，数值候选回退记录 **{r['counts']['fallbacks']}**。

这些是当前版本在本数据上的对比，不构成“全球 SOTA”的证明。
本轮未使用历史路线作为规划输入。绿色仍是体积加权行驶时间代理；个人偏好关闭。

## 当前强求解器：等预算的新求解

下表以 ILS+LKH 为逐路线配对参照，负值表示行驶更短。
各方法均包含建模、求解及审核的调用时间，排除依赖预加载和文件读取。
质量比较仅使用双方都可行的路线，因此必须同时阅读可行数，不能忽略失败。
第二种子仅在预先按规模选出的 10 条上重复，不冒充完整 50 条多种子结果。

| 求解器 | seed42 可行数 | 中位调用秒 | 配对平均行驶变化 | 站点日期 bootstrap 95% 区间 | 重复子集可行数 | 子集两种子配对平均变化 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(seeds)}

PyVRP 0.14.0 是查询时观察到的最新稳定版；0.12.2 使用 HGS，0.13.4/0.14.0 使用 ILS。
OR-Tools GLS 是独立工程参考。ILS+LKH 使用之前已验证的研究实现，保留 LKH 学术/非商业限制。
这五者原生优化的是旅行时间，不能把此表理解为它们已优化我们的动态能耗代理和客户完成指标。

## 同一参考路线、同一保护范围：新策略搜索

固定此前验收时每条路线的 seed、权重和 epsilon；先对两个 RNG 种子取路线平均，再汇总。
策略用 1 秒原生搜索；最后一列包括准备、搜索和精确审核的完整策略调用。
“目标变化”对比已保存的原 Python 50,000 次迭代结果；后三项对比共同参考路线。
旧 Python 结果与耗时来自前轮同机验收存档，未在本轮重新运行；新方法之间的等预算对比则在本轮完成。

| 策略 | 加权目标相对旧 Python | 行驶变化 | 客户完成时刻和变化 | 能耗代理变化 | 中位策略调用秒 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(comparison)}

为了分离 C++ 加速和搜索方法改进，下面在同一原生评价器、同一 1 秒预算下，
直接比较开发集选中的方法与随机 SA。负值表示选中方法的加权目标更低。

| 策略 | 平均加权目标变化 | 配对 95% 区间 | 改善/退步路线数 |
|---|---:|---:|---:|
{chr(10).join(native)}

原生 random-SA 是同类算法移植，RNG 和循环退火调度与旧 Python 不完全相同；
不能把两者全部差异归因于编译语言。所有方法保留同样的最终精确保护。

## 一次 ILS+LKH 种子替代双种子流程

这部分确实重新生成种子，再运行策略，不把旧路线当成免费缓存。
新 epsilon 相对新种子；可行率、目标变化和对旧参考保护的跨越分别报告。
均值仅比较新旧都可行的配对路线，对应旧模式输出；时间统一为种子求解调用之和加策略调用，旧流程的
文件读取/进程通信/种子选择外围开销不计入这张速度表。
旧流程耗时来自前轮同机存档，因此速度比是同机、非同期估计；不能视为控制服务器负载后的精确加速比。

| 策略 | 可行数 | 新中位秒 | 旧中位秒 | 行驶变化 | 完成时刻和变化 | 能耗代理变化 | 超旧参考行驶限的路线数 | 超旧绿色代理限的路线数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(pipeline)}

最后两列非零不表示新策略违反自身 epsilon，而是证明更换种子改变了可行域。
它们必须暴露；如果业务要求始终相对旧参考守住限额，就不能直接把新流程当作等价替换。

## 搜索是否饱和

下表是开发集的平均加权目标改善（相对固定 seed，越大越好），同一次搜索的累积检查点。
只有 6 条开发路线、两个种子，曲线趋平仅能说明这些方法和预算下的经验饱和。

| 策略 | 方法 | 0.25 秒 | 1 秒 | 4 秒 | 8 秒 |
|---|---|---:|---:|---:|---:|
{chr(10).join(saturation)}

![开发集饱和曲线](saturation.png)

## 能否证明接近最优

采用向下取整的毫秒成本，对无自环的指派松弛验证整数原始/对偶证书；
再结合有向最短路和客户服务时间下界。能耗满足
`E = T + sum(volume_j × delivery前累计行驶时间_j) / capacity`，
因此可以构造相应下界。它没有删除原问题约束来声称获得可行解：松弛仅用于下界。

设已知可行目标为 U、下界为 L，表中 `(U-L)/U` 是“剩余可能改善”的安全外界。
**大差距不代表有这么多可实现收益；它也可能仅仅说明下界太松。**

| 策略 | 下界差距中位数 | 路线范围 |
|---|---:|---:|
{chr(10).join(bounds)}

SCIP 完整路线模型使用弧选择、顺序、完成时刻及载荷流；小实例已与穷举最优值一致。
完整实例的数值界受求解器容差约束，和上面的整数证书分开记录。

| 策略 | 10 秒内数值模型报告最优的实例数 | 数值目标下界范围（seed 目标为 1） |
|---|---:|---:|
{chr(10).join(mip)}

达到时间上限不能证明最优；即使模型报告最优，也需结合数值容差和原单位审核。

## 复核与来源

正式冻结版本完整测试：PyVRP 0.13.4 环境 **116 passed**（0.14 专用模块单独验证）；
PyVRP 0.14.0 适配 **2 passed**。源码、原始输入、已安装原生库和全部输出哈希保存在证据归档。
报告只提交汇总与验证材料；比赛输入和完整客户路线留在研究证据目录。

参见 [冻结协议](../policy-frontier-protocol.md)、[算法与文献范围](../policy-frontier-literature.md)、
[近期神经方法与精确方法的适配核查](../policy-frontier-recent-work.md)、
[机器可读汇总](summary.json)、[本地复核](local_verification.json)。
'''
    (out/'REPORT.md').write_text(report)
    (out/'summary.json').write_text(json.dumps(r,indent=2)+'\n')
    cache=Path(__file__).resolve().parents[1]/'artifacts/matplotlib';cache.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for axis,(policy,_) in zip(axes.flat,names.items()):
        for method,modes in r['development_saturation'].items():
            budgets=[.25,1,4,8];axis.plot(budgets,[100*(1-modes[policy][str(b)]) for b in budgets],marker='o',label=method)
        axis.set_xscale('log');axis.set_xticks([.25,1,4,8],labels=['0.25','1','4','8']);axis.set_title(policy)
        axis.set_xlabel('Cumulative search seconds');axis.set_ylabel('Mean objective improvement (%)');axis.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Fixed-reference development trajectories (6 routes, 2 RNG seeds)')
    fig.savefig(out/'saturation.png',dpi=170);plt.close(fig)
    print(out/'REPORT.md')

if __name__=='__main__':main()
