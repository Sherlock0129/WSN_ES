# ADCR 算法改进日志

## [2025-10-27] 增加簇内通信能耗结算

### 改进摘要
在 ADCR 链路层算法中增加了簇内成员向簇头发送数据的能耗结算功能，使能耗模型更加完整和真实。

### 修改文件
- `src/acdr/adcr_link_layer.py` - 修改 `_settle_comm_energy()` 方法

### 关键变更

#### 1. 增强能耗结算方法 `_settle_comm_energy()`

**原有功能：** 
- 仅结算簇头到虚拟中心的路径能耗

**新增功能：**
- ✅ 第一阶段：簇内通信能耗结算（成员 → 簇头）
- ✅ 第二阶段：簇间路径能耗结算（簇头 → 锚点）
- ✅ 第三阶段：虚拟跳能耗结算（锚点 → 虚拟中心）

#### 2. 新增簇内通信处理逻辑

```python
# 对每个簇的每个非簇头成员
for member_id in member_ids:
    if member_id == ch_id:  # 跳过簇头自己
        continue
    
    member_node = id2node[member_id]
    
    # 成员向簇头发送基础数据（使用base_data_size）
    # 使用 _energy_consume_one_hop 方法扣除双向能量
    Eu, Ev = self._energy_consume_one_hop(member_node, ch_node, transfer_WET=False)
    
    # 记录通信
    self.last_comms.append({
        "type": "intra_cluster",
        "hop": (member_id, ch_id),
        "E_member": Eu,
        "E_ch": Ev,
        "distance": member_node.distance_to(ch_node)
    })
```

#### 3. 增强通信记录

所有通信记录现在包含 `type` 字段，便于区分不同类型的通信：
- `"intra_cluster"` - 簇内通信
- `"inter_cluster"` - 簇间路径通信
- `"virtual_hop"` - 虚拟跳通信

#### 4. 增强调试输出

新增详细的阶段性输出：
```
[ADCR-DEBUG] Phase 1: Settling intra-cluster communication energy
[ADCR-DEBUG] Intra-cluster communication: 27 hops, total energy: 180.50J
[ADCR-DEBUG] Phase 2: Settling cluster head to virtual center path energy
[ADCR-DEBUG] Inter-cluster paths: 8 hops, total energy: 95.30J
[ADCR-DEBUG] Virtual hops: 5 hops, total energy: 45.20J
[ADCR-DEBUG] Total energy consumption: 321.00J
```

### 影响分析

#### ✅ 正面影响

1. **更真实的能耗模型**
   - 完整模拟分簇协议的所有通信开销
   - 符合实际WSN系统的能耗特征

2. **更准确的簇头负载评估**
   - 簇头需要接收所有成员的数据
   - 簇头能量消耗明显高于普通成员

3. **自然的簇头轮换机制**
   - 簇头能量消耗大 → 下轮被选为簇头概率降低
   - 实现负载均衡和能量均衡

4. **更准确的网络寿命评估**
   - 真实反映网络能耗
   - 避免高估网络寿命

#### ⚠️ 注意事项

1. **总能耗增加**
   - 增加簇内通信能耗后，总能耗会增加
   - 这是符合实际的，不是bug

2. **网络寿命可能缩短**
   - 更真实的能耗模型可能导致评估的网络寿命缩短
   - 这反映了真实场景，而非性能下降

3. **不影响当前轮决策**
   - 能耗结算在分簇决策之后执行
   - 不会影响当前轮的簇头选择和成簇

4. **影响下一轮决策**
   - 能量消耗会影响下一周期的簇头选择
   - 这是合理且必要的行为

### 配置参数

相关配置参数（`src/config/simulation_config.py`）：

```python
@dataclass
class ADCRConfig:
    consume_energy: bool = True          # 是否启用能耗结算
    base_data_size: int = 1000000        # 基础数据量（bits）
    aggregation_ratio: float = 1.0       # 聚合比例
    enable_dynamic_data_size: bool = True # 动态数据量
```

### 测试验证

**测试脚本：** `test_adcr_intra_cluster_energy.py`

**测试内容：**
1. ✅ 验证簇内通信能耗是否正确扣除
2. ✅ 验证簇头能量消耗是否大于普通成员
3. ✅ 验证能耗结算不影响当前轮的分簇决策
4. ✅ 验证能耗统计输出的完整性

**运行测试：**
```bash
python test_adcr_intra_cluster_energy.py
```

### 文档

创建的文档：
1. `ADCR_ENERGY_SETTLEMENT_IMPROVEMENT.md` - 详细改进说明
2. `docs/ADCR_Energy_Settlement_Diagram.md` - 能耗结算流程图
3. `test_adcr_intra_cluster_energy.py` - 测试脚本
4. `CHANGELOG_ADCR.md` - 本文档

### 向后兼容性

✅ **完全向后兼容**

- 可通过 `consume_energy=False` 禁用能耗结算
- 不会影响现有代码的正常运行
- 仅增强功能，不改变原有接口

### 使用示例

```python
from config.simulation_config import ConfigManager

# 创建配置
config = ConfigManager()

# 启用ADCR链路层
config.simulation_config.enable_adcr_link_layer = True

# 配置ADCR参数
config.adcr_config.consume_energy = True         # 启用能耗结算
config.adcr_config.base_data_size = 1000000      # 1Mb基础数据
config.adcr_config.aggregation_ratio = 1.0       # 完全聚合
config.adcr_config.round_period = 1440           # 每天重聚类

# 创建网络和ADCR
network = config.create_network()
adcr = config.create_adcr_link_layer(network)
network.adcr_link = adcr

# 运行仿真
simulation = config.create_energy_simulation(network)
simulation.simulate()
```

### 未来改进方向

1. **能量感知的簇大小控制**
   - 根据簇头剩余能量动态调整簇大小
   - 避免簇头过载

2. **自适应数据聚合**
   - 根据网络状态调整聚合比例
   - 平衡能耗和数据质量

3. **多目标优化**
   - 同时考虑能耗、延迟、可靠性
   - 实现更优的簇头选择策略

4. **异构网络支持**
   - 支持不同类型的节点
   - 支持不同的能耗模型

### 相关Issue/需求

- 需求：完善ADCR算法的能耗模型
- 问题：簇内成员向簇头发送数据时未扣除能量
- 解决：增加簇内通信能耗结算

### 贡献者

- 实现：AI Assistant
- 日期：2025-10-27

---

**注意：** 本改进基于ADCR（Adaptive Duty Cycle Routing）算法的理论模型，旨在提供更真实和完整的无线传感器网络仿真。

