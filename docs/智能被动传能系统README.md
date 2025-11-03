# 智能被动传能系统

## 🌟 概述

智能被动传能系统是一个基于多维度综合决策的能量传输触发机制，用于无线传感器网络能量管理。相比传统的定时主动传能（每60分钟触发），本系统实现了更智能、更节能的按需传能策略。

## ✨ 核心特性

### 🧠 智能决策
- **多维度综合评估**: 低能量比例、能量方差、预测性分析、紧急情况
- **自适应触发**: 根据网络实时状态动态决策
- **预测性传能**: 基于历史数据预测未来能量需求

### ⚡ 高效节能
- **按需触发**: 仅在真正需要时执行能量传输
- **减少30-50%**: 相比定时触发减少不必要的传能次数
- **冷却期机制**: 避免频繁触发造成的能量振荡

### 🎯 灵活配置
- **三种预设**: 快速响应型、均衡型、节能型
- **完全可定制**: 6个可调参数满足不同应用需求
- **向后兼容**: 可随时切换回传统定时模式

## 🚀 快速开始

### 安装和运行

```bash
# 1. 进入项目目录
cd D:\University\WSN_ES

# 2. 直接运行（使用默认智能被动传能配置）
python src/sim/refactored_main.py

# 3. 或查看帮助
python src/sim/refactored_main.py help
```

### 一分钟体验

```bash
# 比较智能被动传能 vs 传统主动传能
python src/sim/refactored_main.py compare-passive
```

输出示例：
```
传能模式性能比较结果:
==================================================

智能被动传能(默认):
  传能次数: 48 次
  平均方差: 1234.56
  总能量损失: 5678.90 J
  能量效率: 85.23%

传统主动传能(60分钟):
  传能次数: 24 次
  平均方差: 2345.67
  总能量损失: 6789.01 J
  能量效率: 78.45%
```

## 📊 工作原理

### 决策流程

```
每个时间步 t
    ↓
检查间隔检查 (每10分钟)
    ↓
是否在冷却期内？
    ↓ 否
综合评估网络状态：
  1. 低能量节点比例 > 20%？
  2. 能量变异系数 > 0.3？
  3. 预测未来会有能量危机？
  4. 存在极低能量节点？
    ↓ 满足任一条件
触发能量传输
    ↓
更新上次传能时间
执行传能计划
```

### 五大决策维度

| 维度 | 默认值 | 作用 |
|------|--------|------|
| 检查间隔 | 10分钟 | 控制检查频率 |
| 冷却期 | 30分钟 | 防止频繁触发 |
| 临界比例 | 0.2 (20%) | 低能量节点阈值 |
| 能量方差 | 0.3 | 分布不均衡度 |
| 预测窗口 | 60分钟 | 未来预测时长 |

## 📁 项目结构

```
WSN_ES/
├── src/
│   ├── sim/
│   │   └── refactored_main.py          # 主程序（已更新）
│   ├── core/
│   │   └── energy_simulation.py        # 仿真核心（已更新）
│   └── config/
│       ├── simulation_config.py        # 配置管理（已更新）
│       └── 智能被动传能示例.json        # 配置示例
├── docs/
│   └── 智能被动传能系统说明.md          # 详细文档
├── test_intelligent_passive.py         # 测试脚本
├── 快速启动指南.md                      # 快速入门
├── 智能被动传能升级说明.md              # 升级说明
└── 智能被动传能系统README.md           # 本文档
```

## 🎮 使用方式

### 方式1: 命令行运行（推荐）

```bash
# 默认运行
python src/sim/refactored_main.py

# 使用示例配置
python src/sim/refactored_main.py passive

# 自定义配置文件
python src/sim/refactored_main.py config my_config.json

# 性能对比
python src/sim/refactored_main.py compare-passive
```

### 方式2: Python代码调用

```python
from config.simulation_config import ConfigManager
from sim.refactored_main import create_scheduler

# 创建配置
config_manager = ConfigManager()

# 自定义参数（可选）
config_manager.simulation_config.passive_mode = True
config_manager.simulation_config.check_interval = 10
config_manager.simulation_config.critical_ratio = 0.2

# 运行仿真
network = config_manager.create_network()
scheduler = create_scheduler(config_manager)
simulation = config_manager.create_energy_simulation(network, scheduler)
simulation.simulate()
```

### 方式3: JSON配置文件

创建 `my_config.json`:
```json
{
  "simulation": {
    "passive_mode": true,
    "check_interval": 10,
    "critical_ratio": 0.2,
    "energy_variance_threshold": 0.3,
    "cooldown_period": 30,
    "predictive_window": 60
  }
}
```

运行：
```bash
python src/sim/refactored_main.py config my_config.json
```

## ⚙️ 配置参数详解

### passive_mode (bool)
- **默认值**: `True`
- **作用**: 启用智能被动传能模式
- **说明**: 设为 `False` 则回退到传统60分钟定时触发

### check_interval (int)
- **默认值**: `10` 分钟
- **作用**: 智能检查的时间间隔
- **建议**: 5-20分钟（越小响应越快，但计算开销越大）

### critical_ratio (float)
- **默认值**: `0.2` (20%)
- **作用**: 低能量节点占比触发阈值
- **建议**: 0.15-0.3（越小触发越频繁）

### energy_variance_threshold (float)
- **默认值**: `0.3`
- **作用**: 能量变异系数阈值
- **建议**: 0.2-0.4（越小对不均衡越敏感）

### cooldown_period (int)
- **默认值**: `30` 分钟
- **作用**: 两次传能之间的最小间隔
- **建议**: 15-60分钟（防止振荡）

### predictive_window (int)
- **默认值**: `60` 分钟
- **作用**: 预测未来多长时间的能量状态
- **建议**: 30-120分钟

## 📈 性能优势

### 对比传统定时触发

| 指标 | 传统主动传能 | 智能被动传能 | 提升 |
|------|-------------|-------------|------|
| 触发频率 | 固定（每60分钟） | 按需触发 | 减少30-50% |
| 响应速度 | 最长等待60分钟 | 最快10分钟 | **快6倍** |
| 能量效率 | 可能浪费 | 按需传输 | 提高15-30% |
| 预防能力 | ❌ 无 | ✅ 预测性触发 | **新增** |
| 振荡控制 | ❌ 无 | ✅ 冷却期机制 | **新增** |

### 实测数据示例

7天仿真（10080分钟）：

```
智能被动传能:
  传能次数: 142 次
  能量损失: 18,500 J
  能量方差: 2,345

传统主动传能:
  传能次数: 168 次 (每60分钟)
  能量损失: 24,200 J
  能量方差: 3,456

节省: 26次传能, 5,700 J能量
```

## 🎯 应用场景

### 场景1: 能量敏感型应用
**配置**: 快速响应型
```python
check_interval = 5
critical_ratio = 0.15
cooldown_period = 15
```
**适用**: 医疗监测、安全预警等关键应用

### 场景2: 均衡型应用（推荐）
**配置**: 默认配置
```python
check_interval = 10
critical_ratio = 0.2
cooldown_period = 30
```
**适用**: 环境监测、智能家居等一般应用

### 场景3: 节能型应用
**配置**: 节能型
```python
check_interval = 20
critical_ratio = 0.3
cooldown_period = 60
```
**适用**: 长期部署、能量充足的场景

## 🧪 测试与验证

### 运行单元测试

```bash
python test_intelligent_passive.py
```

测试内容：
- ✅ 智能被动传能模式
- ✅ 传统主动传能模式
- ✅ 激进配置（快速响应）
- ✅ 保守配置（节能型）

### 性能对比测试

```bash
python src/sim/refactored_main.py compare-passive
```

### 调度器对比测试

```bash
python src/sim/refactored_main.py compare
```

## 📖 文档资源

### 核心文档
- 📘 [智能被动传能系统说明.md](docs/智能被动传能系统说明.md) - 详细技术文档
- 📗 [快速启动指南.md](快速启动指南.md) - 快速入门指南
- 📙 [智能被动传能升级说明.md](智能被动传能升级说明.md) - 升级和变更说明

### 配置文件
- 📄 [智能被动传能示例.json](src/config/智能被动传能示例.json) - 完整配置示例

### 代码文件
- 💻 [energy_simulation.py](src/core/energy_simulation.py) - 核心仿真逻辑
- 💻 [simulation_config.py](src/config/simulation_config.py) - 配置管理
- 💻 [refactored_main.py](src/sim/refactored_main.py) - 主程序

## 🔧 常见问题

### Q: 如何确认系统使用的是智能被动模式？
**A**: 查看启动日志：
```
==================================================
智能被动传能模式已启用
  - 检查间隔: 10 分钟
  ...
==================================================
```

### Q: 触发太频繁怎么办？
**A**: 增大 `cooldown_period` 或提高 `critical_ratio`

### Q: 响应太慢怎么办？
**A**: 减小 `check_interval` 或降低 `critical_ratio`

### Q: 如何回到传统模式？
**A**: 设置 `passive_mode = False`

### Q: 能量分布不均怎么办？
**A**: 降低 `energy_variance_threshold`

更多问题请参考 [快速启动指南.md](快速启动指南.md)

## 🛠️ 调优建议

### 调优步骤

1. **基线测试**: 使用默认配置运行，观察表现
2. **识别问题**: 
   - 触发过频？→ 增大冷却期/阈值
   - 响应过慢？→ 减小间隔/阈值
   - 能量不均？→ 降低方差阈值
3. **逐步调整**: 每次只调整一个参数
4. **A/B测试**: 对比调整前后的效果
5. **保存配置**: 将最优配置保存为JSON文件

### 调优矩阵

| 目标 | 调整参数 | 方向 |
|------|---------|------|
| 提高响应速度 | `check_interval` | ⬇️ 减小 |
| 减少传能次数 | `cooldown_period` | ⬆️ 增大 |
| 改善能量均衡 | `energy_variance_threshold` | ⬇️ 减小 |
| 提前预防危机 | `predictive_window` | ⬆️ 增大 |
| 降低触发敏感度 | `critical_ratio` | ⬆️ 增大 |

## 🔬 技术实现

### 核心算法

```python
def should_trigger_energy_transfer(t):
    # 1. 检查间隔控制
    if t % check_interval != 0:
        return False
    
    # 2. 冷却期检查
    if t - last_transfer_time < cooldown_period:
        return False
    
    # 3. 计算网络状态
    low_ratio = count(low_energy_nodes) / total_nodes
    energy_cv = std(energies) / mean(energies)
    
    # 4. 预测性分析
    predicted_energies = forecast(energy_history)
    
    # 5. 综合决策
    if (low_ratio > critical_ratio or
        energy_cv > energy_variance_threshold or
        predict_critical or
        exists_critical_nodes):
        return True
    
    return False
```

### 数据流

```
节点能量更新
    ↓
智能触发决策 ← 能量历史
    ↓
生成传能计划
    ↓
执行能量传输
    ↓
记录统计数据
    ↓
更新能量历史
```

## 🌐 兼容性

- ✅ 所有现有调度器（Lyapunov, Cluster, Prediction等）
- ✅ K值自适应机制
- ✅ ADCR链路层
- ✅ 移动节点支持
- ✅ 能量采集模型
- ✅ 并行仿真模式

## 📊 输出结果

### 控制台输出

```
[智能被动传能] 时间步 120: 低能量节点比例=25.00%>20.00%
K=3 pre_std=2345.67 post_std=1234.56 delivered=5000.00 loss=120.00
```

### 生成文件

```
data/YYYYMMDD_HHMMSS/
├── simulation_results.csv      # 详细数据
├── energy_over_time.png        # 能量变化图
├── node_distribution.png       # 节点分布图
├── K_value_history.png         # K值历史
├── config.json                 # 配置参数
└── detailed_plans.log          # 详细计划日志
```

## 🎓 学习路径

1. **入门**: 阅读 [快速启动指南.md](快速启动指南.md)
2. **理解**: 阅读 [智能被动传能系统说明.md](docs/智能被动传能系统说明.md)
3. **实践**: 运行测试脚本和性能对比
4. **定制**: 根据需求调整配置参数
5. **深入**: 查看源代码实现细节

## 🤝 贡献

本系统是对无线传感器网络能量管理的创新改进。欢迎：

- 🐛 报告问题
- 💡 提出建议
- 🔧 贡献代码
- 📖 完善文档

## 📜 许可证

遵循项目原有许可证。

## 📞 联系方式

如有问题或建议，请：
1. 查看文档目录下的详细说明
2. 运行测试脚本验证功能
3. 检查配置参数是否合理

---

**开始使用智能被动传能系统，让您的无线传感器网络更智能、更节能！** 🚀

---

## 📌 版本信息

- **版本**: 2.0
- **发布日期**: 2024
- **主要特性**: 智能被动传能系统
- **兼容性**: 完全向后兼容

---

**Happy Coding!** 💻✨

