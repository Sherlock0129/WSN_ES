# 反馈分数数据提取诊断指南

## 问题

用户发现 "Dimensional Scores Over Time" 图表中所有线条颜色相同，怀疑是**数据提取**环节出现问题。

---

## 诊断改进

### 新增详细调试信息

在 `plot_feedback_scores()` 方法中添加了全面的数据诊断：

```python
# 1. 打印第一条记录的结构
print(f"First record keys: {list(self.feedback_scores[0].keys())}")
print(f"First record: {self.feedback_scores[0]}")

# 2. 检查每个维度的数据范围和唯一值
print(f"Balance scores - min: {min(balance_scores):.2f}, max: {max(balance_scores):.2f}")
print(f"  Unique values: {len(set(balance_scores))}, First 5: {balance_scores[:5]}")

# 3. 检查数组引用是否相同
print(f"balance == survival: {balance_scores is survival_scores}")
```

---

## 运行诊断

### 步骤1：运行仿真

```bash
python src/sim/refactored_main.py
```

### 步骤2：查看调试输出

你会看到类似这样的详细信息：

```
[Feedback Scores Debug Info]
Total records: 8

First record keys: ['time_step', 'total_score', 'balance_score', 'survival_score', 'efficiency_score', 'energy_score', 'impact']
First record: {'time_step': 60, 'total_score': -1.23, 'balance_score': 0.85, 'survival_score': 0.0, 'efficiency_score': -2.10, 'energy_score': -0.08, 'impact': '中性（影响很小）'}

Balance scores - min: -2.15, max: 3.42, avg: 0.85
  Unique values: 8, First 5: [0.85, 1.20, -0.45, 2.30, -2.15]

Survival scores - min: 0.00, max: 0.00, avg: 0.00
  Unique values: 1, First 5: [0.0, 0.0, 0.0, 0.0, 0.0]

Efficiency scores - min: -8.50, max: 2.30, avg: -2.10
  Unique values: 8, First 5: [-2.10, -8.50, 1.20, 2.30, -3.45]

Energy scores - min: -0.35, max: 0.12, avg: -0.08
  Unique values: 7, First 5: [-0.08, -0.12, 0.05, 0.12, -0.35]

Data arrays are same object?
  balance == survival: False
  balance == efficiency: False
  balance == energy: False

  Plotted Balance Score (40%) with color #2E86AB
  Plotted Survival Score (30%) with color #A23B72
  Plotted Efficiency Score (20%) with color #F18F01
  Plotted Energy Level Score (10%) with color #C73E1D
```

---

## 诊断分析

### 情况1：数据正常提取

**特征：**
- ✅ First record 包含所有必要的键
- ✅ 每个维度有多个唯一值（Unique values > 1）
- ✅ 数组不是同一个对象（all False）

**结论：** 数据提取正常，问题在**可视化**环节（颜色设置）。

**解决：** 已通过强制重置 matplotlib 配置和显式颜色设置修复。

---

### 情况2：所有分数都相同

**特征：**
- ❌ 所有维度的 min == max
- ❌ Unique values == 1
- ❌ First 5 所有值相同

**示例：**
```
Balance scores - min: 2.50, max: 2.50, avg: 2.50
  Unique values: 1, First 5: [2.5, 2.5, 2.5, 2.5, 2.5]
```

**原因：** 网络状态完全稳定，没有变化。

**解决：** 
- 检查网络是否在运行能量传输
- 增加仿真时间
- 调整网络参数增加动态性

---

### 情况3：某些分数始终为0

**特征：**
- ⚠️ Survival scores 全为 0（min=0, max=0）
- ⚠️ Unique values: 1

**示例：**
```
Survival scores - min: 0.00, max: 0.00, avg: 0.00
  Unique values: 1, First 5: [0.0, 0.0, 0.0, 0.0, 0.0]
```

**原因：** 这是**正常现象**！

**解释：**
- Survival Score 只在节点存活数变化时才有值
- 如果仿真期间没有节点死亡或复活，该分数就是0
- 这不影响其他维度的显示

---

### 情况4：字段名错误

**特征：**
- ❌ First record keys 缺少某些键
- ❌ 报错 `KeyError`

**示例：**
```
First record keys: ['time_step', 'total_score']  # 缺少其他维度
```

**原因：** `compute_network_feedback_score` 返回的 details 字典不完整。

**解决：** 检查调度器的 `compute_network_feedback_score` 方法。

---

### 情况5：数组引用相同

**特征：**
- ❌ 某些数组检查为 True
- ❌ 所有数组实际指向同一个列表

**示例：**
```
Data arrays are same object?
  balance == survival: True  # 问题！
  balance == efficiency: True  # 问题！
```

**原因：** 代码错误，数组赋值有问题。

**解决：** 检查列表推导式是否正确。

---

## 常见问题解答

### Q1: Survival Score 全为0，是bug吗？

**A**: 不是！这很正常。

- Survival Score 只反映**存活节点数的变化**
- 如果仿真期间：
  - 没有节点死亡（能量降到0）
  - 没有节点复活（能量从0变为正）
- 那么这个分数就始终为0

**验证方法：**
```python
# 在调试信息中查看
alive_change = post_alive - pre_alive
print(f"Alive change: {alive_change}")  # 应该是0
```

---

### Q2: 为什么 Energy Score 值很小？

**A**: 因为它的**权重只有10%**。

```python
energy_score = energy_change_ratio * 0.1 * 100
```

- Balance Score: × 0.4
- Survival Score: × 0.3
- Efficiency Score: × 0.2
- Energy Score: × 0.1 ← 最小

**这是设计如此**，能量水平变化对整体影响最小。

---

### Q3: 所有数据都相同怎么办？

**A**: 检查以下几点：

1. **确认能量传输开启**
   ```python
   enable_energy_sharing=True
   ```

2. **增加仿真时间**
   ```python
   duration=500  # 至少100步以上
   ```

3. **检查网络参数**
   - 节点数量够多（>10）
   - 初始能量有差异
   - 传输间隔合理

---

### Q4: 如何手动检查数据？

**A**: 查看保存的 CSV 文件：

```bash
# 找到最新的输出目录
cd data/20251109_xxxxxx/

# 查看 feedback_scores.csv
head -n 20 feedback_scores.csv
```

CSV 列：
- `time_step`: 时间步
- `total_score`: 总分
- `balance_score`: 均衡分数
- `survival_score`: 存活分数
- `efficiency_score`: 效率分数
- `energy_score`: 能量分数
- `impact`: 影响描述

---

## 预期正常输出示例

```
[Feedback Scores Debug Info]
Total records: 8

First record keys: ['time_step', 'total_score', 'balance_score', 'survival_score', 'efficiency_score', 'energy_score', 'impact']
First record: {'time_step': 60, 'total_score': -1.23, 'balance_score': 0.85, 'survival_score': 0.0, 'efficiency_score': -2.10, 'energy_score': -0.08, 'impact': '中性'}

Balance scores - min: -2.15, max: 3.42, avg: 0.85
  Unique values: 8, First 5: [0.85, 1.20, -0.45, 2.30, -2.15]
  ✓ 数据正常：有8个不同的值

Survival scores - min: 0.00, max: 0.00, avg: 0.00
  Unique values: 1, First 5: [0.0, 0.0, 0.0, 0.0, 0.0]
  ✓ 正常：节点存活数未变化

Efficiency scores - min: -8.50, max: 2.30, avg: -2.10
  Unique values: 8, First 5: [-2.10, -8.50, 1.20, 2.30, -3.45]
  ✓ 数据正常：有8个不同的值

Energy scores - min: -0.35, max: 0.12, avg: -0.08
  Unique values: 7, First 5: [-0.08, -0.12, 0.05, 0.12, -0.35]
  ✓ 数据正常：有7个不同的值（值较小是正常的）

Data arrays are same object?
  balance == survival: False  ✓
  balance == efficiency: False  ✓
  balance == energy: False  ✓
  ✓ 所有数组都是独立的

  Plotted Balance Score (40%) with color #2E86AB
  Plotted Survival Score (30%) with color #A23B72
  Plotted Efficiency Score (20%) with color #F18F01
  Plotted Energy Level Score (10%) with color #C73E1D
  ✓ 所有维度都已绘制
```

---

## 修改的文件

- **`src/core/simulation_stats.py`**
  - `plot_feedback_scores()` 方法
  - 添加详细数据诊断输出

---

## 下一步

1. **运行仿真**查看调试输出
2. **根据输出**判断问题类型
3. **参考本文档**找到对应的解决方案

---

## 更新日期

2025-01-09

---

## 总结

通过详细的数据诊断，可以准确定位问题是在：
- ❌ 数据记录环节
- ❌ 数据提取环节  
- ✅ 数据可视化环节（颜色设置）← 最可能

运行一次，查看调试输出，问题就清楚了！

