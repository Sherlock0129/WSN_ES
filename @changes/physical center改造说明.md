# physical_center 改造说明

本文档聚焦 src/info_collection/physical_center.py（NodeInfoManager）在本次提交中的全部改动与影响。内容来源于 physical_center_change.txt 的差异摘要并经整理。

## 一、改造目标
- 与新引入的 PeriodicCollector 对齐统计口径与接口。
- 修复/优化能量估算与传能状态之间的相互影响，保证“真实到达”与“能量传输”的语义分离。
- 增强信息新鲜度（AoI）管理，支持基于 AoI 的强制上报策略。

## 二、关键改动

1) 初始化参数新增
- 新增 force_report_on_stale: bool = False，记录到 self.force_report_on_stale。
  - 用于开启“基于 AoI 的陈旧强制上报”能力。

2) 信息传输能耗与计数统计扩展（加入 PeriodicCollector）
- info_transmission_energy 加入键 periodic_collector。
- info_transmission_stats 新增：
  - total_periodic_collector_energy
  - periodic_collector_transmission_count
- add_info_transmission_energy(transmission_type=...) 与 record_transmission(...) 均支持 "periodic_collector"。
- reset()/get_info_transmission_statistics()/log_info_transmission_statistics() 同步适配：
  - 统计结构、日志输出包含 periodic_collector 项；
  - node_breakdown 使用 dict.get(key, 0.0) 防止缺键报错；
  - Top10 输出新增“定期收集器消耗(J)”列。

3) 能量估算流程更稳健，避免覆盖“刚参与传能”的精确信息
- 在周期性估算中，若节点满足：
  - is_estimated == False 且 arrival_time == current_time
  - 判定为“刚通过 apply_energy_transfer_changes 更新能量”，其能量为精确值；仅更新 t 与 AoI，不进行能量估算。
- 新增 skipped_count 统计，并将日志从“估算 X 个节点”改为“估算 X 个节点，跳过 Y 个刚更新的节点”。

4) 传能应用不再刷新“到达/AoI”
- apply_energy_transfer_changes(...) 现在仅：
  - 更新 energy
  - 标记 is_estimated = False
- 不再刷新 arrival_time / record_time / aoi / t。
  - 这些字段只在“真实上报/真实到达”（ADCR/PathCollector/PeriodicCollector/强制上报）时更新。

5) 强制上报逻辑增强
- 原有：仅对等待队列中的超时节点进行强制上报并放行。
- 现有：
  - 对 timeout_nodes 成功执行后，显式调用 update_node_info(...)，视为“真实到达”，刷新 arrival_time/record_time/aoi 等。
  - 新增基于 AoI 的“陈旧强制上报”（受 force_report_on_stale 控制）：
    - 遍历未处理节点，若 AoI ≥ max_wait_time，则按当前节点状态触发 update_node_info()；
    - 打印 "AoI超限强制上报" 日志。
  - 返回值改为 forced_total（真实成功强制上报的数量），而非 len(timeout_nodes)。

## 三、行为变化
- 估算与传能解耦：
  - 传能不再“刷新信息到达时间”，因此不会意外将 AoI 归零；
  - 刚传能的节点在同一时刻内不会被估算覆盖其精确能量。
- 统计更全面：
  - 能耗与传输计数包含 periodic_collector，日志/报表信息更完整。
- 强制上报更可靠：
  - 超时/陈旧两条路径都会最终形成一次“真实到达”，数据面与统计面一致。

## 四、配置与使用建议
- 若需开启基于 AoI 的“陈旧强制上报”：
  - 构造 NodeInfoManager（或 VirtualCenter 工厂）时传入 force_report_on_stale=True。
- 在与上层（PathCollector/PeriodicCollector/ADCR）协同时：
  - 上报成功后应调用 update_node_info(...) 以刷新到达时间与 AoI，保持统计与数据一致。

## 五、测试建议
- 估算与传能交互：
  - 在传能完成当刻运行估算，确认该节点能量保持精确值，AoI 不被误置 0。
- 强制上报：
  - 超时路径触发后，arrival_time/aoi 正确刷新，forced_total 计数正确；
  - force_report_on_stale=True 时，满足 AoI 阈值的节点会被触发并记录日志。
- 统计输出：
  - get_info_transmission_statistics() 返回结构包含 periodic_collector；
  - log_info_transmission_statistics() 的总计与 Top10 列表包含 periodic_collector 列，且与累积能耗一致。

## 六、兼容性与注意点
- 若外部代码依赖“传能会刷新到达时间/AoI”的旧行为，需要相应调整逻辑。
- 传输类型字符串请使用 "adcr" / "path_collector" / "periodic_collector" 三者之一。
- 若后续将 NodeInfoManager 替换为工厂创建的 VirtualCenter，请确保工厂暴露 force_report_on_stale 透传参数。
