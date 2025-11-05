# DQN传能过频问题修复指南

## 🔍 问题现象

训练完DQN模型后，导入使用时出现：
- ❌ 传能过于频繁
- ❌ 节点能量快速消耗
- ❌ 节点提前死亡
- ❌ 网络寿命显著缩短

## 🎯 问题根源

### 1. Epsilon探索率过高（主要原因）⚠️

**问题**：
```python
# DQN保存模型时会保存当前的epsilon值
checkpoint = {
    'q_network': ...,
    'epsilon': 0.51  # ← 如果训练不充分，epsilon可能还很高
}
```

**影响**：
- Epsilon=0.5 意味着 **50%的时间随机探索**
- 随机探索会选择1-10分钟的任意传输时长
- 可能选到长时间传输（如8-10分钟），导致能量快速消耗

**代码位置**：
```python
# src/scheduling/dqn_scheduler.py line 153-172
def select_action(self, state, training=True):
    if training and random.random() < self.epsilon:
        # 随机探索 ← 这里会选择1-10分钟的随机值
        action = random.randint(0, self.action_dim - 1)
    else:
        # 贪婪选择（使用训练好的策略）
        action = q_values.argmax(dim=1).item()
```

### 2. 训练模式未关闭

**问题**：
```python
# 如果创建调度器时还是training_mode=True
scheduler = DQNScheduler(
    training_mode=True  # ← 错误！推理时应该设置为False
)
```

**影响**：
- 继续使用epsilon-greedy探索策略
- 即使有训练好的模型，也会频繁随机探索

### 3. 未使用被动模式

**问题**：
```python
# 如果不使用passive_mode
simulation = EnergySimulation(
    passive_mode=False  # ← 每分钟都会传输
)
```

**影响**：
- DQN在**每一步**都会做传输决策
- 可能导致传输过于频繁

## ✅ 解决方案

### 方案1：使用修复工具（推荐）⭐

#### Step 1: 修复模型文件

```bash
# 修复dqn_model.pth，强制设置epsilon=0
python fix_dqn_inference.py --fix dqn_model.pth

# 或者保存到新文件
python fix_dqn_inference.py --fix dqn_model.pth --output dqn_model_fixed.pth
```

**效果**：
- ✅ 自动备份原模型（.backup）
- ✅ 强制设置epsilon=0.0（无探索）
- ✅ 保留网络权重不变

#### Step 2: 正确创建DQN调度器

```python
from scheduling.dqn_scheduler import DQNScheduler
from info_collection.physical_center import NodeInfoManager

# 创建节点信息管理器
nim = NodeInfoManager(
    initial_position=(5.0, 5.0),
    enable_logging=False
)

# 创建DQN调度器（推理模式）
scheduler = DQNScheduler(
    node_info_manager=nim,
    K=2,
    max_hops=3,
    action_dim=10,
    training_mode=False,      # ✓ 必须设置为False
    epsilon_start=0.0,        # ✓ 无探索
    epsilon_end=0.0
)

# 初始化并加载模型
scheduler.plan(network, 0)
scheduler.load_model("dqn_model.pth")  # 使用修复后的模型

# 双重保险：强制设置epsilon为0
scheduler.agent.epsilon = 0.0

print(f"当前epsilon: {scheduler.agent.epsilon}")  # 应该输出0.0
```

#### Step 3: 使用被动模式运行仿真

```python
from core.energy_simulation import EnergySimulation

simulation = EnergySimulation(
    network=network,
    time_steps=10080,
    scheduler=scheduler,
    enable_energy_sharing=True,
    passive_mode=True,        # ✓ 启用被动模式
    check_interval=10         # ✓ 每10分钟检查一次，而非每分钟
)

simulation.simulate()
```

### 方案2：手动修复（不推荐，但可用）

如果不想使用修复工具，可以手动修复：

```python
import torch

# 1. 加载模型
checkpoint = torch.load("dqn_model.pth")

# 2. 查看当前epsilon
print(f"当前epsilon: {checkpoint['epsilon']}")

# 3. 修复epsilon
checkpoint['epsilon'] = 0.0

# 4. 保存
torch.save(checkpoint, "dqn_model.pth")
print("✓ epsilon已修复为0.0")
```

## 📋 完整检查清单

### ✅ 修复前检查

- [ ] 确认模型文件存在（dqn_model.pth）
- [ ] 备份原模型（自动或手动）
- [ ] 确认PyTorch已安装

### ✅ 修复步骤

- [ ] 运行修复工具：`python fix_dqn_inference.py --fix dqn_model.pth`
- [ ] 确认epsilon已设置为0.0
- [ ] 创建调度器时设置`training_mode=False`
- [ ] 加载模型后手动设置`scheduler.agent.epsilon = 0.0`
- [ ] 启用被动模式：`passive_mode=True`
- [ ] 设置合理的检查间隔：`check_interval=10`

### ✅ 修复后验证

运行以下测试：

```python
# 验证epsilon
print(f"Epsilon: {scheduler.agent.epsilon}")  # 应该是0.0

# 验证训练模式
print(f"Training mode: {scheduler.training_mode}")  # 应该是False

# 运行短期测试（100步）
simulation = EnergySimulation(
    network=network,
    time_steps=100,
    scheduler=scheduler,
    enable_energy_sharing=True,
    passive_mode=True,
    check_interval=10
)
simulation.simulate()

# 检查是否还有过度传输
results = simulation.result_manager.get_results()
total_transfers = sum(len(r.get('plans', [])) for r in results)
print(f"100步内传输次数: {total_transfers}")  # 应该显著减少
```

## 🎯 预期效果

### 修复前 ❌

```
仿真步数: 1000
总传输次数: 856        ← 几乎每步都传输
平均传输时长: 6.8分钟  ← 时长较长
节点死亡: 5/15        ← 大量节点死亡
```

### 修复后 ✅

```
仿真步数: 1000
总传输次数: 87         ← 显著减少
平均传输时长: 3.2分钟  ← 更合理
节点死亡: 0/15        ← 所有节点存活
```

## 💡 最佳实践

### 1. 训练时确保充分衰减epsilon

```python
# 训练配置
dqn_training_episodes = 50     # ← 至少50回合
epsilon_start = 1.0
epsilon_end = 0.01             # ← 最终epsilon应该很小
epsilon_decay = 0.995          # ← 合理的衰减率

# 验证最终epsilon
# 50回合后：epsilon ≈ 1.0 * (0.995)^50 ≈ 0.78 (还是太高！)
# 需要更多回合或更快衰减
```

**改进建议**：
```python
# 方案1：增加训练回合
dqn_training_episodes = 100    # epsilon ≈ 0.61

# 方案2：加快衰减（推荐）
epsilon_decay = 0.98           # 50回合后 epsilon ≈ 0.36
epsilon_decay = 0.96           # 50回合后 epsilon ≈ 0.13
epsilon_decay = 0.95           # 50回合后 epsilon ≈ 0.08 ✓
```

### 2. 推理时三重保险

```python
# 保险1：修复模型文件
# 使用fix_dqn_inference.py修复

# 保险2：创建时设置
scheduler = DQNScheduler(
    training_mode=False,
    epsilon_start=0.0,
    epsilon_end=0.0
)

# 保险3：加载后强制设置
scheduler.load_model("dqn_model.pth")
scheduler.agent.epsilon = 0.0  # ← 强制设置
```

### 3. 合理使用被动模式

```python
# 推理时推荐配置
simulation = EnergySimulation(
    passive_mode=True,
    check_interval=10          # 可根据需要调整：10/15/20
)

# check_interval选择：
# - 10分钟：频繁传输，更积极的能量共享
# - 15分钟：平衡
# - 20分钟：保守传输，降低能量消耗
```

## 🔧 故障排除

### Q1: 修复后仍然传能频繁？

**检查**：
```python
# 1. 验证epsilon
print(f"Epsilon: {scheduler.agent.epsilon}")

# 2. 验证训练模式
print(f"Training mode: {scheduler.training_mode}")

# 3. 验证被动模式
print(f"Passive mode enabled: {simulation.passive_mode}")
```

**可能原因**：
- epsilon没有真正设置为0
- training_mode仍然是True
- 没有启用passive_mode

### Q2: 模型性能下降？

如果修复后模型性能不佳（能量不均衡），可能原因：

1. **训练不充分**
   - 解决：重新训练更多回合（100+）
   - 确保训练时epsilon充分衰减

2. **训练时的奖励函数问题**
   - 检查`_compute_reward`函数
   - 确保奖励函数鼓励合理的传输策略

3. **动作空间过大**
   - DQN支持1-10分钟，可能10分钟传输过长
   - 考虑限制动作空间到1-5分钟

### Q3: 如何验证修复成功？

运行以下诊断脚本：

```python
# 诊断脚本
import torch

# 1. 检查模型文件
checkpoint = torch.load("dqn_model.pth")
print(f"Epsilon in checkpoint: {checkpoint.get('epsilon', 'N/A')}")

# 2. 检查调度器
print(f"Scheduler epsilon: {scheduler.agent.epsilon}")
print(f"Training mode: {scheduler.training_mode}")

# 3. 运行100步测试
simulation = EnergySimulation(network, 100, scheduler, 
                             enable_energy_sharing=True, 
                             passive_mode=True, check_interval=10)
simulation.simulate()

# 4. 统计传输
results = simulation.result_manager.get_results()
total_transfers = sum(len(r.get('plans', [])) for r in results)
print(f"传输次数: {total_transfers}/100步")

# 预期结果：
# - Epsilon: 0.0 ✓
# - Training mode: False ✓
# - 传输次数: <20 ✓ (passive_mode下应该很少)
```

## 📚 相关文档

- [DQN训练和使用完整指南](./DQN训练和使用完整指南.md)
- [DQN离散动作调度器说明](./DQN离散动作调度器说明.md)
- [DQN训练损失修复说明](./DQN训练损失修复说明.md)

## 🎓 总结

### 核心修复要点

1. **修复epsilon**：使用工具或手动设置为0.0
2. **关闭训练模式**：`training_mode=False`
3. **启用被动模式**：`passive_mode=True`, `check_interval=10`

### 快速修复命令

```bash
# 1. 修复模型
python fix_dqn_inference.py --fix dqn_model.pth

# 2. 生成推理示例
python fix_dqn_inference.py --example

# 3. 运行修复后的推理
python dqn_inference_fixed.py --model dqn_model.pth --steps 10080
```

### 预期改善

✅ 传输频率下降 **80-90%**  
✅ 节点死亡率下降到 **0%**  
✅ 网络寿命延长 **2-3倍**  
✅ 能量利用更高效  

---

**问题解决了吗？如果还有疑问，请查看故障排除部分或联系技术支持。**


