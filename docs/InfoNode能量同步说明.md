# InfoNode能量同步机制说明

## 🔍 问题发现

在引入`InfoNode`架构后，模拟过程中发现能量传输无法正常工作。

### 问题根源

**症状**：调度器无法生成有效的能量传输计划

**原因**：
1. 模拟循环中，真实节点（`SensorNode`）的能量会不断变化：
   - 能量采集（太阳能）
   - 能量传输（donor消耗，receiver获得）
   - 能量衰减

2. 但是`NodeInfoManager`中的`InfoNode`能量从未更新

3. 调度器每次使用的都是**初始能量信息**，导致：
   - 低能量节点看起来能量充足
   - 高能量节点看起来能量不足
   - 无法正确规划能量传输

---

## ✅ 解决方案

### 核心原理

在每次调度**之前**，必须将真实节点的当前能量同步到`InfoNode`！

### 实现位置

**文件**：`src/core/energy_simulation.py`

**位置**：在`scheduler.plan()`之前添加同步代码

```python
# 检查是否启用能量传输
if self.enable_energy_sharing:
    # ★ 优先使用外部调度器（若提供）
    if self.scheduler is not None:
        # 【关键】更新NodeInfoManager中的节点能量信息
        # 在每次调度之前，必须同步真实节点的当前能量到InfoNode
        if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
            self.scheduler.nim.batch_update_node_info(
                nodes=self.network.nodes,
                current_time=t
            )
        
        # 调度
        result = self.scheduler.plan(self.network, t)
        ...
```

---

## 🔄 同步流程

### 完整流程（正确的分布式架构）

```
每个时间步 t：
  1. 更新节点能量（采集 + 衰减）
     → network.update_network_energy(t)
  
  2. ADCR链路层处理（如果启用）
     → network.adcr_link.step(t)
  
  3. 判断是否触发能量传输
     → passive_manager.should_trigger_transfer(t, network)
  
  4. 如果触发传输：
     a. 【关键】节点上报信息到物理中心
        → nim.batch_update_node_info(nodes, current_time=t)
        
        数据流向：
        真实节点 → [模拟上报] → 节点信息表（L1/L2/L3） → InfoNode
        
        说明：
        - batch_update_node_info 模拟节点主动上报信息
        - 信息存储到节点信息表（latest_info）
        - InfoNode从节点信息表同步更新
        - 物理中心不直接访问真实节点（避免上帝视角）
     
     b. 调度器使用InfoNode规划
        → scheduler.plan(network, t)
        → 调度器只能访问InfoNode，无法访问真实节点
     
     c. 执行能量传输
        → network.execute_energy_transfer(plans, current_time=t)
  
  5. 记录能量状态
```

---

## 📊 同步机制详解（正确的分布式架构）

### 架构原则

**关键理念**：物理中心不能直接访问真实节点（避免"上帝视角"）

**数据流向**：
```
真实节点（Ground Truth）
    ↓ [模拟上报通信]
节点信息表（latest_info）
    ↓ [自动同步]
InfoNode（调度器可见）
    ↓ [只读访问]
调度器/路由算法
```

---

### 1. `batch_update_node_info(nodes, current_time)`

**位置**：`src/acdr/physical_center.py`

**功能**：模拟节点上报信息到物理中心

```python
def batch_update_node_info(self, nodes: List[SensorNode], current_time: int,
                           cluster_mapping: Dict[int, int] = None,
                           data_sizes: Dict[int, int] = None):
    """
    批量更新多个节点的信息（模拟节点上报信息到物理中心）
    
    这个方法模拟真实的信息上报过程：
    1. 节点将当前状态信息发送给物理中心
    2. 物理中心将信息存储到节点信息表（L1/L2/L3）
    3. InfoNode基于节点信息表同步更新（而非直接读取真实节点）
    
    注意：这里读取node.current_energy是模拟"节点上报当前能量"的过程，
    而不是物理中心直接访问节点（上帝视角）。上报后，InfoNode只能
    从节点信息表获取数据，保持了分布式系统的真实性。
    """
    for node in nodes:
        # 模拟节点上报信息到物理中心
        self.update_node_info(
            node_id=node.node_id,
            energy=node.current_energy,  # 节点上报的能量值
            freshness=current_time,
            arrival_time=current_time,
            position=tuple(node.position),
            is_solar=node.has_solar,
            ...
        )
```

**关键点**：
- ✅ 模拟节点主动上报信息（不是中心读取）
- ✅ 信息存储到节点信息表
- ✅ InfoNode从表中同步（不直接访问节点）

---

### 2. `update_node_info(...)`

**位置**：`src/acdr/physical_center.py`

**功能**：更新节点信息表，并同步到InfoNode

```python
def update_node_info(self, node_id: int, energy: float, freshness: int, 
                    arrival_time: int, position: Tuple[float, float] = None,
                    ...):
    """
    更新节点信息到三级缓存表
    """
    # L1: 更新最新状态表
    self.latest_info[node_id] = info
    
    # L2: 添加到近期历史
    self.recent_history.append(history_record)
    
    # L3: 添加到归档缓冲区
    self.archive_buffer.append({...})
    
    # 【关键】从节点信息表同步更新InfoNode（如果已创建）
    # 注意：这里使用的是参数energy和position，这些数据来自节点上报的信息
    # InfoNode不直接访问真实节点，只从节点信息表获取数据
    # 数据流向：真实节点 → 上报 → 节点信息表 → InfoNode
    if node_id in self.info_nodes:
        info_node = self.info_nodes[node_id]
        info_node.current_energy = energy  # 从上报的能量值更新
        if position is not None:
            info_node.position = list(position)  # 从上报的位置更新
```

**关键点**：
- ✅ 参数`energy`来自节点上报，不是直接读取节点
- ✅ InfoNode从参数更新，参数来自节点信息表
- ✅ 保持了数据的单向流动：节点 → 表 → InfoNode

---

### 3. `sync_info_nodes_from_table()`（新增）

**位置**：`src/acdr/physical_center.py`

**功能**：显式地从节点信息表同步到InfoNode

```python
def sync_info_nodes_from_table(self):
    """
    从节点信息表同步数据到InfoNode
    
    这个方法展示了正确的数据流向：
    节点信息表（latest_info）→ InfoNode
    
    InfoNode不直接访问真实节点，只从节点信息表获取数据。
    这保持了分布式系统的真实性，避免了"上帝视角"。
    """
    for node_id, info in self.latest_info.items():
        if node_id in self.info_nodes:
            info_node = self.info_nodes[node_id]
            # 从节点信息表读取数据
            info_node.current_energy = info['energy']
            if info['position'] is not None:
                info_node.position = list(info['position'])
```

**关键点**：
- ✅ 数据源是节点信息表（`latest_info`），不是真实节点
- ✅ 展示了纯粹的"表 → InfoNode"数据流
- ✅ 通常不需要调用（`update_node_info`已自动同步）

---

## 🎯 关键点（分布式架构）

### 1. **避免"上帝视角"（最重要）**

**错误做法**（上帝视角）：
```python
# ❌ 物理中心直接访问真实节点
energy = node.current_energy  # 物理中心不应该能直接读取！
```

**正确做法**（分布式架构）：
```python
# ✅ 节点主动上报信息
batch_update_node_info(nodes, current_time)  # 模拟节点上报
# ✅ 调度器从信息表读取
info_nodes = nim.get_info_nodes()  # 从表中获取
```

**数据流向**：
```
真实节点 → [上报通信] → 节点信息表 → InfoNode → 调度器
        ✅ 模拟        ✅ 存储      ✅ 同步   ✅ 只读
```

### 2. **InfoNode是常驻对象**
- `InfoNode`实例在初始化时创建一次
- 存储在`NodeInfoManager.info_nodes`字典中
- 后续只更新其属性，不重新创建对象
- 数据来源是节点信息表，不是真实节点

### 3. **单一真实来源**
- **节点信息表**（`latest_info`）是唯一的数据源
- InfoNode从表同步，不直接访问真实节点
- 调度器访问InfoNode，不直接访问真实节点
- 保持了信息的单向流动和数据一致性

### 4. **时机至关重要**
- ❌ **错误**：调度后更新 → 调度器用的是旧数据
- ✅ **正确**：调度前上报并更新 → 调度器用的是最新数据

---

## 📈 测试验证

### 测试场景

1. **初始化**：创建节点和`InfoNode`，验证能量一致
2. **模拟能量变化**：手动修改节点能量（模拟传能）
3. **验证不一致**：确认此时`InfoNode`未同步
4. **执行同步**：调用`batch_update_node_info()`
5. **验证一致**：确认同步后能量一致
6. **验证常驻性**：确认`InfoNode`对象引用相同

### 测试结果

```
初始状态:
  节点0: 真实能量=1000.0J, InfoNode能量=1000.0J ✅
  
能量变化后（未同步）:
  节点0: 真实能量=900.0J, InfoNode能量=1000.0J (差值: -100.0J) ✅

同步后:
  节点0: 真实能量=900.0J, InfoNode能量=900.0J (差值: 0.0J) ✅

InfoNode实例是常驻的（引用相同） ✅
```

---

## 🎉 总结

### 架构设计（正确的分布式系统）

**核心原则**：避免"上帝视角"，保持分布式系统的真实性

**数据流向**：
```
真实节点（Ground Truth）
    ↓ [节点主动上报] - batch_update_node_info()
节点信息表（L1: latest_info, L2: history, L3: archive）
    ↓ [自动同步] - update_node_info()
InfoNode（常驻对象）
    ↓ [只读访问] - get_info_nodes()
调度器/路由算法
```

### 关键修复

#### 1. **在模拟循环中添加信息上报**

**位置**：`src/core/energy_simulation.py`

```python
# 【关键】节点上报信息到物理中心
if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
    self.scheduler.nim.batch_update_node_info(
        nodes=self.network.nodes,
        current_time=t
    )
```

#### 2. **明确数据流向和架构原则**

**位置**：`src/acdr/physical_center.py`

- `batch_update_node_info()`: 模拟节点上报
- `update_node_info()`: 存储到表并同步到InfoNode
- `sync_info_nodes_from_table()`: 从表同步到InfoNode（新增）

### 效果

- ✅ 调度器使用**基于节点信息表**的最新信息
- ✅ **避免了上帝视角**，符合分布式系统设计
- ✅ InfoNode从**节点信息表**更新，不直接访问真实节点
- ✅ 能量传输计划正确生成
- ✅ 系统正常工作

---

**日期**：2025-10-28  
**状态**：✅ 已修复并验证  
**架构**：✅ 正确的分布式系统设计

