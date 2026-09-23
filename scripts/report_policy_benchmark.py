#!/usr/bin/env python3
"""Derive a portable, Chinese acceptance report without raw customer routes."""
import argparse
import json
from pathlib import Path
from statistics import median


def main():
    p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    root=a.evidence.resolve();out=a.out;out.mkdir(parents=True,exist_ok=True)
    audit=json.loads((root/'benchmark50/independent_audit.json').read_text());assert audit['passed']
    verification=json.loads((out/'local_verification.json').read_text());assert verification['passed']
    names={'fastest':'最短行驶对照','delivery_speed':'快速送达','balanced':'均衡','green':'绿色代理'}
    rows=[];intervals=[]
    for name,v in audit['policies'].items():
        c=v['changes']
        rows.append(f"| {names[name]} | {v['feasible']}/{v['total']} | {100*c['travel_seconds']['mean']:+.3f}% | {100*c['delivery_completion_sum_seconds']['mean']:+.3f}% | {100*c['energy_proxy']['mean']:+.3f}% | {v['median_policy_seconds']:.2f} | {v['median_single_mode_pipeline_seconds']:.2f} |")
        for field,title in [('travel_seconds','行驶'),('delivery_completion_sum_seconds','客户完成时刻和'),('energy_proxy','能耗代理')]:
            ci=c[field]['ci95'];intervals.append(f"| {names[name]} | {title} | [{100*ci[0]:+.3f}%, {100*ci[1]:+.3f}%] | {100*c[field]['non_pilot_mean']:+.3f}% |")
    public={k:v for k,v in audit.items() if k not in ['result_sha256','infeasible_inputs']}
    public['infeasible_input_count']=len(audit['infeasible_inputs'])
    (out/'summary.json').write_text(json.dumps(public,indent=2)+'\n')
    report=f'''# 多目标策略：审查及原始 50 条路线验收

**结论：四种策略均通过预先冻结的功能、约束及 epsilon 验收。**
源码测试提交：`{audit['source_commit']}`。被审查的原始提交为 `107b508`。
本结果评估修复后的版本，不把合作者的 5 条先导结果作为完整验收。

## 50 条完整结果

同一条路线从新运行的 PyVRP 0.12.2 reference（5,000 次迭代）和
PyVRP 0.13.4 ILS（3.8 秒、80 邻居）中选择较短的可行种子。
四种策略共享这条种子，各运行 50,000 次迭代；种子分别为 42–45。
百分比先逐路线相对种子计算，再平均；负值表示下降。

| 模式 | 可行路线 | 行驶时间变化 | 客户完成时刻和变化 | 能耗代理变化 | 中位策略耗时/秒 | 中位单模式总耗时/秒 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

单模式总耗时包括两次新种子求解、选择审核及该策略搜索。预加载依赖
不计入；策略耗时单独列出，不能拿它冒充完整规划耗时，也不能沿用
之前 ILS+LKH 的 3.83 秒结论。本次复现的是合作者的双种子流程，
并未把初始化偷偷换成 LKH。

原始 50 条中的一条输入本身超过单车容量，明确保留为不可行。
其余 49 条的四种策略全部可行，共 196/200 个策略结果可行，
没有新增不可行、没有遗漏客户或重复客户、没有违反时间窗或 epsilon。
完整保留了 100 次种子求解和 200 个策略结果。

快速送达在全部 49 条可行路线降低客户完成时刻和，但平均增加行驶
时间；绿色代理在 47 条降低、2 条持平，其改善中位数仅约 0.031%。
均衡模式的能耗代理均值略增，不能称作同时改善所有目标。
这是可用的目标权衡模块，尚不是同时更短、更快的算法升级。

## 指标含义与限制

- 能耗代理为 `sum(travel_seconds * (1 + remaining_volume / capacity))`，
  使用剩余货物体积，系数固定为 1。它不是货物重量、燃油升数、kWh
  或 CO₂，也没有车辆/道路标定。绿色模式结果只能称为代理指标下降。
- 客户完成时刻和使用服务**结束**时刻。原先导代码使用开始服务时刻；
  本次纠正了定义并保持原先导文件不变，其旧百分比不能直接混用。
- 四个预设均关闭个人历史偏好，没有输入个人行为模型。它们是业务
  目标权衡模式。偏好矩阵可选接口已测试，仍是静态矩阵接口。
- 各模式以同一启发式种子为参考，不保证全局最优。绿色代理允许行驶
  时间增加最多 5% 并要求代理不增加；快速送达/均衡的行驶上限为 2%；
  最短行驶对照为 0%。各模式最终加权目标不得劣于种子。
- 一次固定随机种子的接受测试包含原先导 5 条，不是新的独立泛化
  测试。下表同时报告去除先导后的可行路线平均变化。

## 配对不确定性

按站点日期整组 bootstrap 10,000 次，固定种子 20260916。

| 模式 | 指标 | 全部可行路线均值的 95% 区间 | 非先导可行路线平均变化 |
|---|---|---:|---:|
{chr(10).join(intervals)}

## 审查修复与验证

原三个测试通过；补充的 15 个对抗用例失败后，完成了以下修复：

1. 对种子进行覆盖、容量和固定出发时间的正式审核；不再信任保存的
   `feasible` 标签或代价作为种子选择的充分条件。
2. 最终使用 Decimal 重算成本、epsilon 与加权目标；不通过时返回
   已审核的种子，避免快速浮点评估误差突破保护。
3. 修复单客户移动采样、零参考值除零与极端 logits 的 softmax。
4. 关闭偏好时完全不读取偏好矩阵；开启时必须提供合法矩阵。
5. 修复无可行解/超容量输入的批量流程，保留结果分母并报告状态。
6. 将客户完成指标纠正为服务结束时刻，并明确代理指标的限制。

实验环境中原基线及新策略测试 **68 passed**，新版求解器适配测试
**21 passed**，合计 **89 项**。最终 50 条结果由独立实现重新计算
覆盖、容量、固定出发、服务开始时间窗、三个目标与 epsilon，全部通过；
本地又从复制输入复算，统计结果一致。原基线求解器、约束、50 条路线
及预算配置没有修改。

最终提交还在隔离目录进行了完整测试收集：默认 0.12.2 环境
**76 passed, 4 skipped**（仅缺少可选运行时/新版接口的测试跳过）；
完整 0.13.4 实验环境 **89 passed, 0 skipped**。
详见 [最终集成测试及源码身份](integration.json)。

参考：[冻结协议](../multiobjective-validation-protocol.md)、
[机器可读摘要](summary.json)、[本地复核](local_verification.json)。
完整逐路线证据保存在研究任务目录；Git 中不提交原始比赛输入。
'''
    (out/'REPORT.md').write_text(report)
    print(out/'REPORT.md')


if __name__=='__main__':main()
