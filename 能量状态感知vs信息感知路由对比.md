# 能量状态感知 vs 信息感知路由对比

## 核心区别

### 1. **能量状态感知（Energy State Aware）**

**关注点**: 节点的**能量水平**

**目的**: 保护低能量节点，避免其因为频繁转发而耗尽能量

**工作原理**:
- 检查节点的**当前能量 / 电池容量**
- **低能量节点** → 增加路由代价（惩罚，避免选择）
- **高能量节点** → 正常代价
- **太阳能节点** → 降低代价（奖励，优先选择）

**参数说明**:
```python
enable_energy_state_aware: bool = False  # 是否启用
low_energy_threshold: float = 0.2        # 能量比例 < 0.2 = 低能量
medium_energy_threshold: float = 0.5     # 能量比例 < 0.5 = 中等能量
low_energy_penalty: float = 1.5          # 低能量节点代价 × 1.5（增加50%代价）
medium_energy_penalty: float = 1.2       # 中等能量节点代价 × 1.2（增加20%代价）
solar_bonus: float = 0.9                 # 太阳能节点代价 × 0.9（降低10%代价）
```

**计算示例**:
```
节点A: 能量比例 = 0.15 (< 0.2) → 惩罚系数 = 1.5
节点B: 能量比例 = 0.4 (< 0.5) → 惩罚系数 = 1.2
节点C: 能量比例 = 0.8 (≥ 0.5) → 惩罚系数 = 1.0
节点D: 能量比例 = 0.6 + 有太阳能 → 惩罚系数 = 1.0 × 0.9 = 0.9

最终代价: C_final = C_original × penalty
```

---

### 2. **信息感知路由（Info-Aware Routing）**

**关注点**: 节点携带的**信息量**

**目的**: 优先选择有未上报信息的节点，实现信息"搭便车"传输

**工作原理**:
- 检查节点的**info_volume**（累积的信息量，单位：bits）
- 检查节点的**info_is_reported**（是否已上报）
- **有未上报信息** → 降低路由代价（奖励，优先选择）
- **无信息或已上报** → 正常代价

**参数说明**:
```python
enable_info_aware_routing: bool = False  # 是否启用
info_reward_factor: float = 0.2          # 信息奖励强度（0~1）
max_info_wait_time: int = 10             # 最大等待时间（分钟）
min_info_volume_threshold: int = 1       # 最小信息量阈值
```

**计算示例**:
```
节点A: info_volume = 0 → info_bonus = 1.0（无奖励）
节点B: info_volume = 50000 bits → 归一化 = 0.05 → info_bonus = 1.0 - (0.2 × 0.05) = 0.99
节点C: info_volume = 100000 bits → 归一化 = 0.1 → info_bonus = 1.0 - (0.2 × 0.1) = 0.98
节点D: info_volume = 500000 bits → 归一化 = 0.5 → info_bonus = 1.0 - (0.2 × 0.5) = 0.9

最终代价: C_final = C_original × info_bonus
```

---

## 对比表格

| 特性 | 能量状态感知 | 信息感知路由 |
|------|-------------|-------------|
| **关注对象** | 节点能量水平 | 节点信息量 |
| **判断依据** | 当前能量 / 电池容量 | info_volume, info_is_reported |
| **主要目的** | 保护低能量节点 | 实现信息搭便车 |
| **代价调整** | 低能量节点**增加**代价 | 有信息节点**降低**代价 |
| **优先级** | 高能量 > 中能量 > 低能量 | 有信息 > 无信息 |
| **太阳能节点** | 有奖励（降低代价） | 无特殊处理 |
| **依赖机制** | 独立运行 | 依赖机会主义信息传递 |
| **适用场景** | 能量均衡、延长网络寿命 | 减少信息上报能耗 |

---

## 同时启用的效果

两个功能可以**同时启用**，效果叠加：

```python
# 最终代价计算流程：
C_v = C_original  # 原始路由代价

# 1. 能量状态感知调整
if energy_state_aware:
    energy_penalty = get_energy_state_penalty(node)
    C_v *= energy_penalty  # 可能增加或减少

# 2. 信息感知路由调整
if info_aware_routing:
    if node has un-reported info_volume:
        info_bonus = 1.0 - (info_reward_factor * normalized_volume)
        C_v *= info_bonus  # 降低代价

# 最终评分
score = efficiency × (1.0 / C_v)
```

**示例**:
```
节点X:
  - 能量比例 = 0.3（中等能量）
  - info_volume = 100000 bits（未上报）
  - 原始代价 = 100

计算过程:
  C_v = 100
  C_v *= 1.2  # 中等能量惩罚 = 120
  C_v *= 0.98  # 信息奖励 = 117.6

最终评分 = efficiency × (1.0 / 117.6)
```

---

## 使用建议

### 场景1: 仅启用能量状态感知
```python
enable_energy_state_aware = True
enable_info_aware_routing = False
```
**适用**: 关注网络能量均衡，延长网络寿命

### 场景2: 仅启用信息感知路由
```python
enable_energy_state_aware = False
enable_info_aware_routing = True
```
**适用**: 重点关注减少信息上报能耗，网络能量充足

### 场景3: 同时启用（推荐）
```python
enable_energy_state_aware = True
enable_info_aware_routing = True
```
**适用**: 平衡能量管理和信息传输效率

---

## 注意事项

1. **能量状态感知**：
   - 降低低能量节点被选中的概率（保护它们）
   - 但可能导致路由路径变长或效率降低

2. **信息感知路由**：
   - 需要先启用机会主义信息传递机制
   - 信息量会在路径完成后累积
   - 有超时保护机制（避免信息丢失）

3. **同时启用时**：
   - 低能量但有信息的节点：惩罚和奖励叠加
   - 需要根据网络特性调整参数，避免冲突

