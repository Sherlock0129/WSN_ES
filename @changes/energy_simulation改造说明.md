# energy_simulation 改造说明

本文档基于 src/core/energy_simulation_change（git diff 文本）对 energy_simulation.py 的改动进行梳理，并结合现状作出行为层面的解读与测试建议。

## 一、变更概览
- 新增主动模式参数 active_transfer_interval，并向被动传能管理器（PassiveManager）透传。
- 在每个时间步引入/恢复“信息层估算”与“定期上报”的处理：
  - 周期性调用 periodic_info_collector.step(nodes, t)（若启用）。
  - 恢复 scheduler.nim.estimate_all_nodes(current_time=t) 的周期性执行，用于对未上报节点进行能量估算。
- 信息层的“超时/陈旧检测与强制上报”改为每个时间步都执行，不再依赖“是否本轮触发传能”。
- 执行传能后，先应用确定性能量变更，再做一次能量估算，保证信息层与能量层一致。
- 扩展反馈详情 feedback_details，聚合本轮选中计划的关键统计，并落盘到统计信息中。

---

## 二、关键代码改动点

1) EnergySimulation.__init__ 参数扩展
```py
class EnergySimulation:
    def __init__(..., predictive_window=60, active_transfer_interval=60):
        ...
        self.passive_manager = PassiveTransferManager(
            ..., predictive_window=predictive_window,
            active_transfer_interval=active_transfer_interval
        )
```
- 新增 active_transfer_interval（主动触发间隔，单位：分钟），并传给被动传能管理器。
- 影响：PassiveManager 可基于该参数实现“定时主动触发”的策略（即便为被动管理器命名，也可能支持混合模式）。

2) 时间步循环的前置处理顺序
- 仍保留 ADCR 链路层 step：
  - `if self.network.adcr_link: self.network.adcr_link.step(t)`
- 新增（或启用）定期上报收集器 step：
  - `if self.network.periodic_info_collector: self.network.periodic_info_collector.step(self.network.nodes, t)`
- 恢复信息层周期性估算：
  - 将原先禁用的“批量更新节点信息”的块改为：
    - `self.scheduler.nim.estimate_all_nodes(current_time=t)`
  - 含义：每个时间步都会对未上报节点进行能量估算（包含衰减与太阳能采集等）。

3) 触发被动传能前后的信息层一致性
- 触发判定：`should_trigger, trigger_reason = self.passive_manager.should_trigger_transfer(t, self.network)`。
- 当 should_trigger 为 True 时：
  - 生成传能计划 plans 并执行（细节略）。
  - 在将计划交给物理层后：
    - 先调用 `nim.apply_energy_transfer_changes(plans, current_time=t)`，将“确定性”的能量变化应用到信息层。
    - 再次调用 `nim.estimate_all_nodes(current_time=t)`，对其余节点进行估算，确保同一时间步结束时，信息层对所有节点都有一致、最新的认知。

4) 强制上报逻辑位置调整（重要）
- 旧逻辑：仅在“本轮触发了传能”的分支里、且启用了 PathCollector 的机会主义转发时，检查并强制上报超时节点。
- 新逻辑：与是否触发传能无关，“每个时间步末尾”都会执行一次信息层的检查：
```py
path_collector = getattr(self.network, 'path_info_collector', None)
oc_enabled = path_collector is not None and getattr(path_collector, 'enable_opportunistic_info_forwarding', False)
max_wait_time = getattr(path_collector, 'max_wait_time', 10) if path_collector else 10
forced_count = self.scheduler.nim.check_timeout_and_force_report(
    current_time=t,
    max_wait_time=max_wait_time,
    path_collector=path_collector if oc_enabled else None,
    network=self.network
)
```
- 打印从“[超时强制上报]”统一为“[信息层强制上报]”。
- 影响：
  - 强制上报不再依赖“本轮是否调度”，提升信息层新鲜度的下界保证；
  - 与 physical_center 中的增强（支持 AoI 陈旧触发、force_report_on_stale）协同更好。

5) 反馈细节增强
```py
feedback_score, feedback_details = self.scheduler.compute_network_feedback_score(...)
# 追加聚合指标（若 details 为 dict）
feedback_details['K_used'] = self.K
feedback_details['trigger_reason'] = trigger_reason or ''
# 对本轮已选中/执行的 plans 聚合：数量、总/均值时长、总/均值 aoi_cost、信息增益、计划分数、传输量等
```
- 稳健性：用 try/except 包裹，不影响主流程；record_feedback_score 会把 details 全量落盘。

---

## 三、行为变化与影响评估
- 信息层时序更清晰：
  - 每步“先估算（基于上一刻的上报/传能状态）、后判定是否调度”。
  - 若发生传能：先落地确定性能量变更，再做估算补齐其余节点。
- 信息新鲜度与可靠性提高：
  - 强制上报不再依赖调度触发，避免“长时间未调度导致信息层一直陈旧”。
  - 与 NodeInfoManager 的新逻辑配合：estimate_all_nodes 会跳过“刚通过传能更新”的节点，防止覆盖精确信息；apply_energy_transfer_changes 不再刷新 AoI/到达时间，语义更准确。
- 定期上报链路接入：
  - periodic_info_collector.step 每步执行，使定期上报与估算/强制上报形成互补。
- 可观测性增强：
  - 反馈详情包含计划级别的聚合指标，便于离线分析与调参。

---

## 四、兼容性与配置建议
- 确保 scheduler.nim 暴露 estimate_all_nodes 与 apply_energy_transfer_changes；与 physical_center 的实现保持一致。
- 如需启用定期上报：初始化 network.periodic_info_collector 并保证其对外暴露 step(nodes, t)。
- PassiveManager 若引入 active_transfer_interval，请在配置中设置合适的间隔（例如 60 分钟）。
- 机会主义信息转发：
  - 若使用 PathCollector 的 oc 功能，请确保 enable_opportunistic_info_forwarding=True 且配置 max_wait_time。

---

## 五、测试建议
1) 基线回归
- 无 ADCR/无 PathCollector/无 PeriodicCollector：确认仿真可运行，estimate_all_nodes 每步执行，统计正常。

2) 与 NodeInfoManager 的交互
- 在执行传能的时间步：
  - 先调用 apply_energy_transfer_changes，再 estimate_all_nodes；
  - 验证“刚传能节点”的能量未被估算覆盖、AoI 维持语义（不被误置 0）。

3) 强制上报
- 禁用/启用 oc 分别测试：
  - oc 禁用：check_timeout_and_force_report 仍被调用，但不经由 PathCollector 路径；
  - oc 启用：超时/陈旧达到阈值时有“[信息层强制上报]”输出，且统计正确。

4) 定期上报
- 启用 periodic_info_collector：
  - 确认其 step 每步被调用，能与估算、强制上报协同工作，统计口径正确（见 physical_center 的新增统计字段）。

5) 反馈详情
- 触发多条计划时，检查 record_feedback_score 落盘的 details 含有新增聚合字段，数值与 plans 一致。

---

## 六、注意点
- 注释中的步骤编号（Step 1.6 / 1.7）在 diff 中出现了顺序上的小不一致，实际执行顺序以代码为准：ADCR -> PeriodicCollector -> estimate_all_nodes。
- 每步既估算一次，调度触发后又再估算一次，属于“先全量、后精化”的设计；配合“跳过刚传能节点”不会出现覆盖问题，如仍有性能顾虑可在后续做条件收敛。
- 若 PathCollector 与 PeriodicCollector 共享同一 VC，则信息层的强制上报、定期上报与估算的耦合要在 VC 层保持一致性（详见 physical_center 改造说明）。

