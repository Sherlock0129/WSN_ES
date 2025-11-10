# 效率得分-10分问题修复

## 修复时间
2025-11-10

## 问题描述

用户报告："效率分数出现过-10分"

从可视化图表可以看到，Efficiency Score (20%) 确实出现了约-10分的情况。

## 根本原因分析

### 效率得分计算公式

```python
efficiency_score = (efficiency - 0.5) * 0.2 * 100
```

当 `efficiency = 0` 时：
```
efficiency_score = (0 - 0.5) * 0.2 * 100 = -10
```

### 问题根源：统计方式错误

**旧的统计方式**（第69行）：
```python
# ❌ 从plans计算sent_total
sent_total = sum(p.get("duration", 1) * p["donor"].E_char for p in plans)
```

**问题场景**：
1. 调度器生成了传输计划（plans）
2. 执行 `execute_energy_transfer(plans)`
3. 某些donor能量不足，传输被跳过（`continue`）
4. 这些被跳过的传输**不会更新** `transferred_history`
5. 但在统计时，**仍然从plans计算sent_total**
6. 结果：`sent_total > 0`（从plans统计），但 `delivered_total = 0`（没实际传输）
7. 效率 = 0 / sent_total = 0%，得分 = -10

### 具体例子

```python
# 场景：3个传输计划，但所有donor能量不足

# 执行前
plans = [plan1, plan2, plan3]  # 每个duration=5, E_char=300

# 执行中（network.py）
for plan in plans:
    if donor.current_energy < total_consumption:
        print("[警告] Donor能量不足，跳过传输")
        continue  # 跳过！transferred_history没有更新

# 统计时（旧方式）
sent_total = sum(p.get("duration", 1) * p["donor"].E_char for p in plans)
# = 5*300 + 5*300 + 5*300 = 4500 J  ❌ 错误！实际没有发送

delivered_total = 0  # 没有实际传输

efficiency = 0 / 4500 = 0%
efficiency_score = -10分
```

## 修复方案

### 核心改进：从实际历史记录统计

不再从plans（计划）统计，而是从transferred_history（实际执行历史）统计：

```python
# ✅ 从transferred_history获取实际传输的能量
post_transferred_total = sum(sum(n.transferred_history) for n in network.nodes)
sent_total = max(0.0, post_transferred_total - pre_transferred_total)
```

**优势**：
- ✅ 只统计实际执行的传输
- ✅ 被跳过的传输不会被统计
- ✅ sent_total和delivered_total完全一致
- ✅ 效率计算准确

## 修复内容

### 1. 修改energy_simulation.py

**添加pre_transferred_total获取**（第141行）：
```python
pre_transferred_total = sum(sum(n.transferred_history) for n in self.network.nodes)
```

**修改函数调用**（第230行）：
```python
stats = self.stats.compute_step_stats(plans, pre_energies, pre_received_total, pre_transferred_total, self.network)
```

### 2. 修改simulation_stats.py

**修改函数签名**（第42-43行）：
```python
def compute_step_stats(self, plans: List[Dict], pre_energies: np.ndarray, 
                      pre_received_total: float, pre_transferred_total: float, network) -> Dict[str, float]:
```

**修改统计计算**（第68-72行）：
```python
# 本轮各 donor 实际下发的能量
# 应该从transferred_history获取实际传输的能量，而不是从plans计算
# 因为plans中的传输可能因能量不足而被跳过
post_transferred_total = sum(sum(n.transferred_history) for n in network.nodes)
sent_total = max(0.0, post_transferred_total - pre_transferred_total)  # 本轮实际发送
```

### 3. 修改lookahead.py（K值前瞻优化）

**修改_compute_stats_for_network函数**（第4-14行）：
```python
def _compute_stats_for_network(net, plans, pre_energies, pre_received_total, pre_transferred_total):
    # ...
    post_transferred_total = sum(sum(n.transferred_history) for n in net.nodes)
    sent_total = max(0.0, post_transferred_total - pre_transferred_total)
    # ...
```

**修改调用处**（第33行和第60-61行）：
```python
pre_transferred_total = sum(sum(n.transferred_history) for n in net_copy.nodes)
# ...
pre_std, post_std, delivered_total, total_loss = _compute_stats_for_network(
    net_copy, plans, pre_energies, pre_received_total, pre_transferred_total
)
```

## 修复效果对比

### 修复前

| 场景 | sent_total | delivered_total | efficiency | score |
|-----|-----------|----------------|-----------|-------|
| **3个计划，全部跳过** | 4500 J | 0 J | 0% | -10分 |
| **3个计划，2个跳过** | 4500 J | 1200 J | 27% | -4.6分 |

**问题**：sent_total包含了未执行的计划，导致效率被低估

### 修复后

| 场景 | sent_total | delivered_total | efficiency | score |
|-----|-----------|----------------|-----------|-------|
| **3个计划，全部跳过** | 0 J | 0 J | 0% | -10分* |
| **3个计划，2个跳过** | 1500 J | 1200 J | 80% | +6分 |

**\*注**：全部跳过时仍然是-10分，但这是合理的（没有传输）

## 影响分析

### 修复前的问题

1. **统计不准确**：包含未执行的传输
2. **效率被低估**：分母虚高（包含跳过的传输）
3. **自适应误判**：基于错误数据调整V参数
4. **可视化误导**：图表显示虚假的低效率

### 修复后的改进

1. ✅ **统计准确**：只统计实际执行的传输
2. ✅ **效率真实**：反映实际传输效率
3. ✅ **自适应正确**：基于准确数据调整
4. ✅ **可视化真实**：图表反映真实情况

## 现在效率为0的情况

修复后，如果看到efficiency = 0（得分-10），只可能是：

### 正常情况 ✅

1. **没有传输计划**：网络已均衡，sent = 0, delivered = 0
2. **所有传输被跳过**：所有donor能量不足，sent = 0, delivered = 0
3. **节点全部锁定**：Duration调度器节点锁定，sent = 0, delivered = 0

### 不再出现的异常情况 ❌

~~1. **有计划但未执行**：sent > 0, delivered = 0~~ （已修复）
~~2. **统计不一致**：sent虚高~~ （已修复）

## 测试验证

### 验证点

1. ✅ 无linter错误
2. ✅ 函数签名修改正确
3. ✅ 所有调用处已更新
4. ✅ lookahead.py已同步修改

### 预期效果

- 效率得分更加准确
- 不再出现虚假的低效率（因统计错误导致）
- 自适应机制基于真实数据调整
- 图表反映真实的传输效率

## 相关文件

### 修改的文件
- `src/core/energy_simulation.py` - 添加pre_transferred_total
- `src/core/simulation_stats.py` - 修改统计方式
- `src/dynamic_k/lookahead.py` - 同步修改K值前瞻

### 相关文档
- `docs/zero_efficiency_fix.md` - 效率为0的原因分析
- `docs/duration_stat_bug_fix.md` - Duration统计bug修复
- `docs/efficiency_minus_ten_fix.md` - 本文档

## 总结

这次修复解决了一个根本性的统计问题：

**修复前**：从计划（plans）统计发送能量
- ❌ 包含未执行的传输
- ❌ 效率被低估
- ❌ 自适应误判

**修复后**：从实际历史（transferred_history）统计发送能量
- ✅ 只统计实际执行的传输
- ✅ 效率准确反映真实情况
- ✅ 自适应基于真实数据

现在如果看到-10分的效率得分，说明真的没有传输发生（而不是因为统计错误）。这样的反馈信息才是真实有用的！

---

**版本历史**:
- v1.0 (2025-11-10): 初始修复，从transferred_history统计sent_total

