# EETOR路由算法逻辑说明

**文件**: `src/routing/energy_transfer_routing.py`  
**算法名称**: EETOR (Energy-Efficient Transfer Opportunistic Routing)  
**设计目标**: 专门为无线能量传输设计的路由算法

---

## 📋 整体架构

### **核心组件**
```
EETOR算法
├── 1. 能量传输效率模型
├── 2. 实际通信能耗模型
├── 3. 节点能量状态感知
├── 4. 邻居构建（能量传输优化）
├── 5. 期望代价计算
├── 6. 前缀选择（能量感知）
└── 7. 路径查找（主接口）
```

---

## 🔧 核心逻辑详解

### **1. 能量传输效率模型**

#### **1.1 `calculate_energy_transfer_efficiency()`**
```python
计算单跳能量传输效率

输入: 节点间距离 d
输出: 传输效率 η ∈ [0, 1]

公式:
  if d <= 1.0:
      η = η₀ + (1.0 - η₀) × (1.0 - d)  # 线性插值
  else:
      η = η₀ / d^γ                        # 逆幂律衰减

参数:
  - η₀ = 0.6  (1米处的参考效率)
  - γ = 2.0   (衰减因子)
```

**特点**:
- 与 `SensorNode.energy_transfer_efficiency()` **完全一致**
- 距离≤1m时线性插值，>1m时逆幂律衰减

#### **1.2 `calculate_path_efficiency()`**
```python
计算多跳路径的总效率

公式: η_path = η₁ × η₂ × ... × ηₙ

特点:
- 累积乘积（体现多跳能量损耗）
- 与调度器的 _path_eta() 逻辑一致
```

---

### **2. 实际通信能耗模型**

#### **`calculate_communication_energy()`**
```python
计算节点间通信的实际能耗

公式:
  E_tx = E_elec × B + ε_amp × B × d^τ   # 发送能耗
  E_rx = E_elec × B                       # 接收能耗
  E_com = (E_tx + E_rx) / 2 + E_sen     # 通信总能耗
  
  if transfer_WET:
      E_com += E_char                     # 加上能量传输开销

特点:
- 与 SensorNode.energy_consumption() **完全一致**
- 包含固定部分（E_elec × B）和距离相关部分（ε_amp × B × d^τ）
```

**与EEOR的区别**:
- ❌ **EEOR**: 简化功率模型 `w = base × d^τ`
- ✅ **EETOR**: 实际通信能耗模型，包含所有组件

---

### **3. 节点能量状态感知**

#### **`get_energy_state_penalty()`**
```python
根据节点能量状态计算代价惩罚系数

能量惩罚规则:
  - 能量比例 < 20%:  惩罚系数 = 1.5  (代价增加50%)
  - 能量比例 20-50%: 惩罚系数 = 1.2  (代价增加20%)
  - 能量比例 > 50%:  惩罚系数 = 1.0  (无惩罚)

太阳能奖励:
  - 有太阳能的节点: 惩罚系数 × 0.9  (降低10%)

返回: 惩罚系数 (>1表示增加代价，<1表示降低代价)
```

**设计思想**:
- 低能量节点应避免作为中继（容易耗尽）
- 有太阳能的节点更适合作为中继（未来能量充足）

---

### **4. 邻居构建（能量传输优化）**

#### **`build_neighbors_energy_transfer()`**
```python
为能量传输优化的邻居发现

特点:
1. 移除硬性距离限制（使用大范围，默认10米）
2. 只考虑传输效率≥阈值的链路（默认1%）
3. 排除物理中心节点（ID=0）

返回:
  - neighbor_map: {node_id: [(neighbor_id, distance, efficiency), ...]}
  - node_dict: {node_id: node_object}
```

**与EEOR的区别**:
- ❌ **EEOR**: 固定范围 √3米，或自适应到约6个邻居
- ✅ **EETOR**: 大范围10米，基于效率阈值筛选

---

### **5. 期望代价计算（核心算法）**

#### **`expected_cost_for_energy_transfer()`**
```python
针对能量传输的期望代价计算

代价定义:
  C_u = (通信能耗 / 路径效率) + 后续转发期望代价

详细步骤:

1. 计算路径效率（累积乘积）
   η_path = ∏ η(u, v_i)  for all v_i in Fwd_ids

2. 计算实际通信能耗
   E_com = calculate_communication_energy(u, max_receiver)
   if energy_state_aware:
       E_com *= get_energy_state_penalty(u)

3. 计算后续转发期望代价（考虑效率损耗）
   β = Σ (success_ratio_i × C_v_i)
   其中:
     success_ratio_i = cumulative_efficiency × η_uv_i
     cumulative_efficiency *= (1 - η_uv_i)  # 剩余能量比例

4. 归一化到有效接收能量
   C_h = E_com / η_path      # 通信能耗归一化
   C_f = β / η_path          # 转发代价归一化
   C_u = C_h + C_f

返回: (期望代价, 路径效率)
```

**关键特点**:
- ✅ 使用**路径效率**（累积乘积），而非误码率
- ✅ 使用**实际通信能耗**，而非简化功率
- ✅ 考虑**节点能量状态**
- ✅ 考虑**多跳累积损耗**

---

### **6. 前缀选择（能量感知）**

#### **`select_forwarder_prefix_energy_aware()`**
```python
能量感知的前缀选择算法

算法流程:

1. 为每个邻居计算综合评分
   score = η × (1 / max(C_v, 1.0))
   
   其中:
     - η: 链路传输效率
     - C_v: 节点v到目标的代价（考虑能量状态惩罚）

2. 按评分降序排序

3. 贪心扩展前缀
   - 依次添加邻居到转发前缀
   - 每次计算期望代价
   - 如果代价不再下降，停止（贪心策略）

返回: (最优代价, 最优转发前缀, 路径效率)
```

**设计思想**:
- 优先选择效率高且代价低的节点
- 贪心策略保证局部最优

---

### **7. 全网期望代价计算**

#### **`compute_energy_transfer_costs()`**
```python
计算网络中所有节点到目标节点的期望代价

算法: 迭代松弛（类似Bellman-Ford）

初始化:
  C[target] = 0.0
  C[其他节点] = ∞
  FWD[所有节点] = []

迭代过程:
  for iteration in range(max_iter):
      for each node u (except target):
          neighbors = get_neighbors(u)
          new_cost, new_fwd = select_forwarder_prefix_energy_aware(...)
          if new_cost < C[u]:
              C[u] = new_cost
              FWD[u] = new_fwd
              updated = True
      
      if not updated:
          break  # 收敛，提前终止

返回: (代价字典C, 转发前缀字典FWD)
```

**特点**:
- 使用迭代松弛算法
- 最多迭代20次
- 提前终止优化

---

### **8. 路径查找（主接口）**

#### **`find_energy_transfer_path()`**
```python
从源节点到目标节点查找能量传输路径

步骤:

1. 确保源节点和目标节点在nodes列表中
   - 如果不在，自动添加

2. 计算所有节点到目标的期望代价
   C, FWD = compute_energy_transfer_costs(nodes, target_id, ...)

3. 从源节点开始，沿转发前缀查找路径
   path = [source_node]
   cur = source_node.node_id
   
   while cur != dest_node.node_id and hops < max_hops:
       fwd = FWD.get(cur, [])
       if not fwd:
           break
       
       nxt = fwd[0]  # 选择转发前缀中代价最小的节点
       
       # 避免环路
       if nxt in path:
           break
       
       path.append(get_node(nxt))
       cur = nxt
       hops += 1

4. 检查是否成功到达目标
   if path[-1].node_id == dest_node.node_id:
       return path
   else:
       return None
```

#### **`eetor_find_path_adaptive()`（兼容接口）**
```python
与EEOR兼容的接口

函数签名:
  eetor_find_path_adaptive(nodes, source_node, dest_node, 
                          max_hops=5, target_neighbors=6)

内部实现:
  1. 根据网络密度自适应调整通信范围
  2. 调用 find_energy_transfer_path()

特点:
  - 接口与 eeor_find_path_adaptive() 完全一致
  - 可直接替换EEOR使用
```

---

## 🔄 完整工作流程示例

### **场景**: 从节点A到节点D查找能量传输路径

```
步骤1: 邻居构建
  - 构建所有节点的邻居图（效率≥1%的链路）
  - 排除物理中心节点

步骤2: 计算期望代价（目标=D）
  - 初始化: C[D] = 0, C[其他] = ∞
  - 迭代松弛，计算所有节点到D的代价
  - 得到转发前缀: FWD[A] = [B, C]

步骤3: 路径查找（A→D）
  - 从A开始: path = [A]
  - 查找FWD[A][0] = B: path = [A, B]
  - 查找FWD[B][0] = C: path = [A, B, C]
  - 查找FWD[C][0] = D: path = [A, B, C, D]
  - 成功到达D，返回路径

步骤4: 路径效率计算
  η_path = η(A→B) × η(B→C) × η(C→D)
```

---

## 📊 与EEOR的关键区别

| 维度 | EEOR（数据传输） | EETOR（能量传输） |
|------|----------------|------------------|
| **链路模型** | 误码率 `e = k × d^γ` | 传输效率 `η = η₀ / d^γ` |
| **功率模型** | 简化 `w = base × d^τ` | 实际 `E = E_elec×B + ε×B×d^τ` |
| **路径效率** | 多路径可靠性（至少一个成功） | 累积乘积（逐跳衰减） |
| **能量状态** | 不考虑 | ✅ 考虑（低能量惩罚） |
| **太阳能** | 不考虑 | ✅ 考虑（降低代价） |
| **邻居范围** | 固定√3米或自适应 | 大范围10米+效率阈值 |

---

## ✅ 核心优势

1. **模型匹配**: 使用项目实际的能量传输效率模型和通信能耗模型
2. **路径优化**: 考虑多跳累积损耗，避免选择效率过低的路径
3. **能量感知**: 避免低能量节点作为中继，延长网络寿命
4. **太阳能感知**: 优先使用有太阳能的节点
5. **兼容接口**: 可直接替代EEOR，无需修改调度器代码

---

## 🚀 使用方式

```python
# 在 schedulers.py 中替换导入
from routing.energy_transfer_routing import eetor_find_path_adaptive

# 使用方式与EEOR完全相同
path = eetor_find_path_adaptive(nodes, source_node, dest_node, 
                                max_hops=5, target_neighbors=6)
```

---

## 📝 注意事项

1. **节点列表**: 应包含源节点和目标节点
2. **物理中心**: 自动排除物理中心节点（ID=0）
3. **路径效率**: 多跳路径效率为累积乘积，可能很低
4. **能量状态**: 默认启用能量状态感知（可关闭）
5. **兼容性**: 与InfoNode和SensorNode都兼容

---

**最后更新**: 2024-10-29  
**算法版本**: 1.0

