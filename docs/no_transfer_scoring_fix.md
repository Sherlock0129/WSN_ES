# 没有传输时的评分修复

## 修复时间
2025-11-10

## 问题描述

用户提出关键问题：**"理论上不应该出现效率低于10%的传输，为什么图中会出现效率分=-10的点？"**

这是一个非常好的观察，揭示了评分逻辑的不合理之处。

## 问题分析

### 两种不同的"效率"

#### 1. 路径效率 (eta) - 计划阶段
```python
eta = self._path_eta(path)  # 单条路径的能量衰减效率
# 效率低于10%的传输直接放弃
if eta < 0.1:
    continue  # 在计划阶段过滤
```

- **定义**：单条传输路径的能量衰减效率
- **范围**：0-100%
- **过滤**：< 10% 的路径被放弃

#### 2. 整体传输效率 (efficiency) - 统计阶段
```python
sent = stats.get('sent_total', 0.0)      # 本时间步所有传输的总发送
delivered = stats.get('delivered_total', 0.0)  # 本时间步所有传输的总接收
efficiency = delivered / sent  # 整体统计效率
```

- **定义**：整个时间步所有传输的总体效率
- **范围**：0-100%
- **评分**：影响自适应机制

### 关键区别

| 项目 | 路径效率 (eta) | 整体效率 (efficiency) |
|-----|---------------|---------------------|
| **作用阶段** | 计划阶段 | 统计阶段 |
| **作用对象** | 单条路径 | 所有传输 |
| **过滤阈值** | < 10% 被放弃 | 无过滤 |
| **影响** | 决定是否传输 | 决定反馈分数 |

## 为什么会出现 efficiency = 0%？

即使所有单条路径的 eta 都 > 10%，整体 efficiency 仍可能为 0：

### 情况1：没有生成传输计划
```python
# 原因：网络能量已均衡
receivers = [n for n in nodes if Q[n] > 0]  # Q[n]=0 → receivers=[]
# 结果：plans = []
# 统计：sent = 0, delivered = 0
# 效率：efficiency = 0 (sent=0时的保护值)
```

### 情况2：所有路径效率都 < 10%
```python
# 原因：网络节点距离太远
for d in donors:
    eta = self._path_eta(path)  # 例如 eta = 0.08
    if eta < 0.1:  # True
        continue  # 所有路径都被过滤
# 结果：plans = []
# 统计：sent = 0, delivered = 0
# 效率：efficiency = 0
```

### 情况3：所有传输因能量不足被跳过
```python
# 原因：所有donor能量不足
for plan in plans:
    if donor.current_energy < total_consumption:
        continue  # 跳过传输
# 结果：实际没有传输
# 统计：sent = 0, delivered = 0 (从transferred_history统计)
# 效率：efficiency = 0
```

### 情况4：节点全部锁定
```python
# 原因：使用DurationAware调度器，所有节点正在传输
nodes = self._filter_unlocked_nodes(nodes, t)  # nodes = []
# 结果：plans = []
# 统计：sent = 0, delivered = 0
# 效率：efficiency = 0
```

## 原有评分逻辑的问题

### 旧逻辑（有问题）
```python
if sent > 0:
    efficiency = delivered / sent
else:
    efficiency = 0.0  # 没有传输时设为0

efficiency_score = (efficiency - 0.5) * 0.2 * 100
# 当 efficiency=0 时，score = (0-0.5)*0.2*100 = -10分
```

**问题**：
- ❌ 没有传输时，给 **-10分（惩罚）**
- ❌ 但"没有传输"通常是**正常情况**
- ❌ 不应该惩罚正常的网络状态

### 为什么不合理？

| 没有传输的原因 | 是否应该惩罚？ | 旧逻辑 | 合理性 |
|--------------|--------------|-------|-------|
| **网络已均衡** | ❌ 否 | -10分 | ❌ 不合理 |
| **所有路径eta<10%** | ❌ 否（主动过滤） | -10分 | ❌ 不合理 |
| **节点全部锁定** | ❌ 否（正在传输） | -10分 | ❌ 不合理 |
| **Donor能量不足** | ❌ 否（能量保护） | -10分 | ❌ 不合理 |

**结论**：这些都是正常或保护性的行为，不应该被惩罚！

## 修复方案

### 新逻辑（合理）
```python
if sent > 0:
    # 有传输：计算效率并评分
    efficiency = delivered / sent
    efficiency_score = (efficiency - 0.5) * 0.2 * 100
else:
    # 没有传输：给中性分数（0分）
    # 这是正常情况，不应该惩罚
    efficiency = 0.0
    efficiency_score = 0.0  # ✅ 中性分数
```

### 修复效果

| 场景 | 旧逻辑 | 新逻辑 | 改进 |
|-----|-------|-------|------|
| **网络均衡，无传输** | -10分 | **0分** | ✅ 合理 |
| **所有路径eta<10%** | -10分 | **0分** | ✅ 合理 |
| **节点锁定，无传输** | -10分 | **0分** | ✅ 合理 |
| **有传输，效率80%** | +6分 | +6分 | ✅ 不变 |
| **有传输，效率20%** | -6分 | -6分 | ✅ 不变 |

## 对自适应机制的影响

### 修复前
```python
# 场景：网络均衡，无传输
efficiency_score = -10分
total_score = balance(2) + survival(0) + efficiency(-10) + energy(0) = -8分

# 自适应机制误判：认为效率太低，可能调整V
if avg_feedback < -2.0:
    if efficiency_score < -2.0:  # True (-10 < -2.0)
        self.V = min(self.V_max, self.V * 1.1)  # 增大V（错误！）
```

**问题**：网络均衡是好事，但被误判为"效率低"，触发错误调整

### 修复后
```python
# 场景：网络均衡，无传输
efficiency_score = 0分  # 中性
total_score = balance(2) + survival(0) + efficiency(0) + energy(0) = 2分

# 自适应机制正确判断：网络状态良好
if avg_feedback > 0:  # True
    # 不触发调整，保持当前V值
```

**改进**：不再误判，自适应机制基于真实情况调整

## 修复内容

**位置**：`src/scheduling/schedulers.py` 第133-148行

**关键改动**：
```python
if sent > 0:
    # 有传输：正常评分
    efficiency = delivered / sent
    efficiency_score = (efficiency - 0.5) * 0.2 * 100
else:
    # 没有传输：中性分数
    efficiency = 0.0
    efficiency_score = 0.0  # ✅ 改为0分（原来是-10分）
```

## 现在的行为

修复后，效率得分的含义：

| 得分范围 | 含义 | 触发条件 |
|---------|------|---------|
| **+10分** | 效率极高（100%） | 有传输，且效率完美 |
| **0分** | 中性 | **没有传输** OR **效率50%** |
| **-10分** | 效率极低（0%） | **有传输但完全失败** |

**重要**：现在 -10分 只在"有传输但完全失败"时出现，这才是真正应该警惕的情况！

## 预期效果

### 可视化图表
- ✅ 不再出现虚假的 -10分（没有传输时）
- ✅ 只在真正低效时才出现负分
- ✅ 曲线更加平滑，反映真实情况

### 自适应机制
- ✅ 不再因"没有传输"而误判
- ✅ 更准确地识别真正的效率问题
- ✅ V参数调整更加合理

### 日志输出
- ✅ 当 efficiency_score=0 时，可能是没有传输（正常）
- ✅ 当 efficiency_score=-10 时，一定是有传输但完全失败（异常）

## 相关修复

这是一系列效率相关修复的最后一步：

1. ✅ **efficiency_score_fix.md** - 修复效率分段映射（已回退）
2. ✅ **duration_stat_bug_fix.md** - 修复duration统计bug
3. ✅ **zero_efficiency_fix.md** - 添加能量检查和诊断
4. ✅ **efficiency_minus_ten_fix.md** - 从transferred_history统计
5. ✅ **no_transfer_scoring_fix.md** - 本修复

## 总结

这次修复解决了一个逻辑不合理的问题：

**问题根源**：
- 将"没有传输"（正常情况）视为"效率低"（异常情况）
- 给予惩罚性的 -10分

**修复方案**：
- 区分"没有传输"和"传输失败"
- 没有传输时给 0分（中性），而不是 -10分（惩罚）

**核心理念**：
> "没有传输"不等于"传输失败"，应该区别对待

现在：
- ✅ 效率得分更加合理
- ✅ 自适应机制判断更准确
- ✅ 图表反映真实的网络状态

---

**版本历史**:
- v1.0 (2025-11-10): 修复"没有传输"时的评分逻辑，从-10分改为0分

