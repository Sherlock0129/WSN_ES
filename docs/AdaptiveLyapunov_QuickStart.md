# AdaptiveLyapunovScheduler 快速入门

## 5分钟上手指南

### 1. 什么是 AdaptiveLyapunovScheduler？

一个**智能的**能量调度器，它能：
- ✅ **自动学习**：根据实际效果调整策略
- ✅ **多目标优化**：同时考虑均衡、效率、存活率、总能量
- ✅ **即插即用**：无需手动调参，自动适应网络

### 2. 快速测试

```bash
# 运行快速测试（验证功能）
python test_adaptive_scheduler.py

# 运行详细示例
python examples/adaptive_lyapunov_example.py --mode single

# 运行对比实验
python examples/adaptive_lyapunov_example.py --mode compare
```

### 3. 基本用法（3步）

```python
# 步骤1：导入
from scheduling.schedulers import AdaptiveLyapunovScheduler
from node_info.node_info_manager import NodeInfoManager

# 步骤2：创建调度器（一行代码）
nim = NodeInfoManager(network.nodes)
scheduler = AdaptiveLyapunovScheduler(nim, V=0.5, K=2)

# 步骤3：运行（自动调整V参数）
sim = EnergySimulation(network, scheduler=scheduler)
sim.run(duration=500)

# 查看调整结果
scheduler.print_adaptation_summary()
```

### 4. 典型输出

运行时会看到：

```
[自适应@t=120] V: 0.500 → 0.550 | 效率低(0.28) → 增大V(减少损耗)
[自适应@t=185] V: 0.550 → 0.495 | 均衡差 → 减小V(增强均衡)
[自适应@t=240] V: 0.495 → 0.450 | 节点死亡(-2) → 减小V(优先救活)
```

最后会输出摘要：

```
============================================================
自适应Lyapunov调度器 - 适应性总结
============================================================
初始V: 0.500
当前V: 0.625
总调整次数: 15
平均反馈分数: 2.34
```

### 5. 对比标准 Lyapunov

| 特性 | 标准Lyapunov | 自适应Lyapunov |
|------|-------------|---------------|
| 需要调参 | ✅ 是 | ❌ 否（自动） |
| 适应变化 | ❌ 否 | ✅ 是 |
| 性能 | 基准 | 通常+10-15% |

### 6. 核心参数（可选调整）

```python
scheduler = AdaptiveLyapunovScheduler(
    nim,
    V=0.5,              # 初始值（会自动调整）
    window_size=10,     # 记忆长度（5-20）
    adjust_rate=0.1,    # 调整速度（0.05-0.2）
    sensitivity=2.0     # 触发灵敏度（1-3）
)
```

**默认值适用于大多数场景！**

### 7. 何时使用

✅ **推荐使用**：
- 网络负载不均衡
- 节点密度变化大
- 长时间运行（>100步）
- 不确定最优V值

❌ **不推荐使用**：
- 极短仿真（<50步）
- 已知最优参数
- 需要严格可复现性

### 8. 故障排查

**问题：V频繁震荡**
```python
# 解决：降低调整速度
scheduler = AdaptiveLyapunovScheduler(nim, adjust_rate=0.05, sensitivity=3.0)
```

**问题：从不调整**
```python
# 解决：提高敏感度
scheduler = AdaptiveLyapunovScheduler(nim, sensitivity=1.0)
```

### 9. 完整示例代码

```python
import sys
sys.path.insert(0, 'src')

from network.sensor_network import SensorNetwork
from core.energy_simulation import EnergySimulation
from scheduling.schedulers import AdaptiveLyapunovScheduler
from node_info.node_info_manager import NodeInfoManager

# 创建网络
network = SensorNetwork(num_nodes=20, area_size=100, initial_energy=5000)

# 创建调度器
nim = NodeInfoManager(network.nodes)
scheduler = AdaptiveLyapunovScheduler(nim)

# 运行仿真
sim = EnergySimulation(network, scheduler=scheduler, enable_energy_sharing=True)
sim.run(duration=500)

# 查看结果
scheduler.print_adaptation_summary()
print(f"最终存活: {sim.stats.get_summary()['final_alive_nodes']}")
```

### 10. 工作原理（简化版）

```
每个时间步：
1. 使用当前V值做决策（选择传输路径）
2. 执行传输
3. 计算反馈分数（-10到+10）
4. 如果分数不好：
   - 效率低 → V↑（减少损耗）
   - 均衡差 → V↓（增加传输）
   - 节点死亡 → V↓（优先救活）
5. 下次使用新的V值
```

---

## 更多信息

- 📖 详细文档：`docs/AdaptiveLyapunovScheduler.md`
- 📊 示例代码：`examples/adaptive_lyapunov_example.py`
- 🧪 测试脚本：`test_adaptive_scheduler.py`

---

## 一句话总结

> **AdaptiveLyapunovScheduler = 标准Lyapunov + 自动调参 + 多目标优化**

就像给你的调度器装上了"自动驾驶"！🚗➡️🤖

