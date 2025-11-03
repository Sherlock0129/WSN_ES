# ADCR算法能耗结算改进说明

## 改进概述

在ADCR链路层算法中增加了**簇内成员向簇头发送数据的能耗结算**，使能耗模型更加完整和真实。

## 修改内容

### 文件：`src/acdr/adcr_link_layer.py`

修改了 `_settle_comm_energy()` 方法，将能耗结算分为三个阶段：

#### 第一阶段：簇内通信能耗结算（新增）
- **通信方向**：成员节点 → 簇头
- **数据量**：基础数据量（base_data_size）
- **扣能方式**：双向扣能（成员节点和簇头都扣除能量）
- **处理逻辑**：
  ```python
  # 对每个簇的每个非簇头成员
  for member_node in cluster_members:
      if member_node != cluster_head:
          # 成员向簇头发送数据，双向扣能
          Eu, Ev = self._energy_consume_one_hop(member_node, ch_node, transfer_WET=False)
  ```

#### 第二阶段：簇间路径通信能耗结算（原有）
- **通信方向**：簇头 → 锚点（多跳路径）
- **数据量**：聚合数据量（aggregated_data_size）
- **扣能方式**：逐跳双向扣能

#### 第三阶段：虚拟跳能耗结算（原有）
- **通信方向**：锚点 → 虚拟中心
- **数据量**：聚合数据量（aggregated_data_size）
- **扣能方式**：仅发送端扣能（虚拟中心不消耗能量）

## 执行时序分析

### 同一轮内：不影响当前分簇决策 ✓

```
时间 t (满足重聚类条件):
  1. 计算虚拟中心
  2. 估计最优簇数 K*
  3. 选择簇头（基于当前能量）         ← 使用未扣能的能量状态
  4. 执行成簇（基于当前能量）         ← 使用未扣能的能量状态
  5. 规划上报路径
  6. 结算通信能耗                     ← 扣除能量
     6.1 簇内通信能耗（新增）
     6.2 簇间路径能耗
     6.3 虚拟跳能耗
```

**结论**：能耗结算在分簇决策之后执行，因此不影响当前轮的决策。

### 下一轮：会影响下一周期的分簇决策 ✓

能量消耗后，节点的 `current_energy` 会降低，影响下一个 `round_period` 时的：
- 簇头选择概率：`p_i = p_star * (n.current_energy / (1.0 + meanE))`
- 成簇成本函数：`cost = distance_weight * d + energy_weight * (1.0 / E_CH)`

**这是合理且必要的行为**，符合实际系统中能量消耗影响后续决策的特性。

## 能耗影响分析

### 新增能耗的影响

| 项目 | 影响 |
|------|------|
| **成员节点能量** | ⬇️ 降低（发送数据） |
| **簇头能量** | ⬇️ 降低更多（接收所有成员数据） |
| **下轮簇头选择** | 📊 簇头能量消耗大，下轮被选为簇头的概率降低 |
| **能量均衡** | ✅ 改善（更真实反映簇头负载） |
| **网络寿命** | 📉 可能略微降低（总能耗增加），但更符合实际 |

### 能耗计算公式

成员向簇头发送数据的能耗使用标准通信模型：

```
E_tx = E_elec × B + ε_amp × B × d^τ
E_rx = E_elec × B
E_com = (E_tx + E_rx) / 2 + E_sen
```

其中：
- `E_elec`：电子学能耗系数
- `ε_amp`：功放损耗系数
- `B`：数据量（bits）
- `d`：传输距离
- `τ`：路径损耗指数
- `E_sen`：传感能耗

## 调试输出

启用ADCR后，能耗结算阶段会输出详细信息：

```
[ADCR-DEBUG] _settle_comm_energy() called, consume_energy=True
[ADCR-DEBUG] Phase 1: Settling intra-cluster communication energy
[ADCR-DEBUG] Intra-cluster communication: 25 hops, total energy: 150.50J
[ADCR-DEBUG] Phase 2: Settling cluster head to virtual center path energy
[ADCR-DEBUG] Processing 5 upstream paths
[ADCR-DEBUG] Inter-cluster paths: 8 hops, total energy: 95.30J
[ADCR-DEBUG] Virtual hops: 5 hops, total energy: 45.20J
[ADCR-DEBUG] Total energy consumption: 291.00J
[ADCR-DEBUG] Total communication records: 38
```

## 配置参数

相关配置参数位于 `src/config/simulation_config.py`：

```python
@dataclass
class ADCRConfig:
    # 是否对通信过程执行能耗结算
    consume_energy: bool = True
    
    # 基础数据大小（bits），簇内成员发送的数据量
    base_data_size: int = 1000000
    
    # 信息聚合比例，簇头上报时的数据压缩率
    aggregation_ratio: float = 1.0
    
    # 是否启用基于簇大小的动态数据量
    enable_dynamic_data_size: bool = True
```

## 数据记录

所有通信记录保存在 `self.last_comms` 列表中，包含三种类型：

### 1. 簇内通信（新增）
```python
{
    "type": "intra_cluster",
    "hop": (member_id, ch_id),
    "E_member": Eu,      # 成员节点消耗的能量
    "E_ch": Ev,          # 簇头接收消耗的能量
    "distance": d        # 传输距离
}
```

### 2. 簇间路径通信
```python
{
    "type": "inter_cluster",
    "hop": (u.node_id, v.node_id),
    "E_tx": Eu,          # 发送端能量
    "E_rx": Ev           # 接收端能量
}
```

### 3. 虚拟跳通信
```python
{
    "type": "virtual_hop",
    "hop": (last_real.node_id, "VIRTUAL"),
    "E_tx_only": E_com,  # 仅发送端能量
    "data_size": aggregated_data_size,
    "cluster_size": cluster_size
}
```

## 启用方式

在仿真配置中启用ADCR链路层：

```python
# simulation_config.py
@dataclass
class SimulationConfig:
    enable_adcr_link_layer: bool = True  # 启用ADCR链路层
```

确保能耗结算开关打开：

```python
# simulation_config.py
@dataclass
class ADCRConfig:
    consume_energy: bool = True  # 启用能耗结算
```

## 验证建议

建议通过以下方式验证改进效果：

1. **对比实验**：
   - 对比启用/禁用簇内通信能耗的仿真结果
   - 观察簇头能量下降速度的变化
   - 分析能量均衡性指标（标准差、方差等）

2. **能耗统计**：
   - 查看调试输出中的能耗分布
   - 确认簇内通信能耗占总能耗的比例是否合理

3. **网络寿命**：
   - 观察首个节点死亡时间
   - 比较网络整体存活时间

## 理论依据

该改进基于ADCR（Adaptive Duty Cycle Routing）算法的理论模型，在实际的无线传感器网络中：

1. **簇内成员必须向簇头发送采集的数据**，这会消耗能量
2. **簇头接收数据也需要能量**（接收机功耗）
3. **簇头负载更重**，因为要接收所有成员的数据并进行聚合
4. **能量消耗会影响下一轮的簇头选择**，形成轮换机制

忽略簇内通信能耗会导致：
- 低估簇头的能量消耗
- 高估网络寿命
- 无法准确评估能量均衡策略的效果

## 注意事项

1. **能耗增加**：增加簇内通信能耗后，节点整体能量消耗会增加，这是符合实际的。
2. **簇头负载**：簇头消耗的能量会明显高于普通成员，这也符合分簇协议的特点。
3. **轮换机制**：由于簇头能量消耗大，下一轮被选为簇头的概率降低，有助于实现簇头轮换。
4. **网络寿命**：虽然总能耗增加可能导致网络寿命缩短，但这是真实场景下的表现。

## 作者与时间

- **改进时间**：2025年10月27日
- **修改文件**：`src/acdr/adcr_link_layer.py`
- **改进类型**：功能增强 - 完善能耗模型

