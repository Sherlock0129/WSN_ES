# EEOR路由算法详细逻辑说明

**文件路径**: `src/routing/EEOR.py`  
**算法名称**: Energy-Efficient Opportunistic Routing (EEOR)  
**实现版本**: 自适应邻居发现版本 + 基础版本

---

## 📋 **算法概述**

EEOR是一种基于**机会路由（Opportunistic Routing）**的能量高效路由算法。与传统的确定性路由不同，EEOR通过计算**期望代价（Expected Cost）**来选择最优的转发节点集合（前缀），而不是只选择单个下一跳节点。

### **核心思想**
1. **多路径机会传输**：源节点广播数据包，多个候选转发节点可能收到
2. **优先级转发**：按距离目标的代价排序，代价最小的节点优先转发
3. **能量感知**：考虑链路误码率、传输功率、转发代价等因素

---

## 🏗️ **算法架构**

```
┌─────────────────────────────────────────────────────────┐
│                  EEOR算法流程                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  步骤1: 构建邻居图                                       │
│  - _build_neighbors() 或 _build_neighbors_adaptive()    │
│  - 确定每个节点的邻居节点集合                             │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  步骤2: 计算期望代价                                     │
│  - eeor_compute_costs() 或 eeor_compute_costs_adaptive()│
│  - 从目标节点开始，迭代计算所有节点到目标的期望代价       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  步骤3: 前缀选择                                         │
│  - _select_forwarder_prefix()                           │
│  - 为每个节点选择最优的转发节点前缀                       │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│  步骤4: 路径查找                                         │
│  - eeor_find_path() 或 eeor_find_path_adaptive()       │
│  - 从源节点开始，沿着最优转发路径到达目标                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 **核心组件详解**

### **1. 邻居构建（Neighbor Discovery）**

#### **1.1 基础版本：`_build_neighbors()`**

```python
def _build_neighbors(nodes):
    """构建固定通信范围的邻居图"""
    R = math.sqrt(3)  # 固定通信范围：√3 米
    # 对于每个节点，找到所有距离 ≤ R 的邻居节点
```

**逻辑**：
- 固定通信范围：`R = √3` 米
- 遍历所有节点对，计算距离
- 如果 `distance ≤ R`，则建立邻居关系
- 返回：`neighbor_map`（节点ID → [(邻居ID, 距离), ...]）

---

#### **1.2 自适应版本：`_build_neighbors_adaptive()`** ⭐

```python
def _build_neighbors_adaptive(nodes, target_neighbors=6):
    """自适应邻居发现：动态调整通信范围，使每个节点有目标数量的邻居"""
```

**关键特性**：
- ✅ **排除物理中心节点**：ID=0 完全不参与WET，不作为邻居
- ✅ **自适应通信范围**：根据网络密度动态调整 `R`

**算法流程**：
```
对于每个节点 u：
  1. 计算到所有其他节点（排除物理中心）的距离
  2. 按距离排序
  3. 确定通信范围 R：
     - 如果邻居数 ≥ target_neighbors：
       R = 到第 target_neighbors 个邻居的距离 × 1.1
     - 否则：
       R = 最大距离（或默认值 √3）
  4. 建立邻居关系：所有距离 ≤ R 的节点
```

**优势**：
- 🎯 **自适应网络密度**：稀疏网络自动扩大范围，密集网络缩小范围
- 🎯 **控制邻居数量**：保持每个节点有约 `target_neighbors` 个邻居
- 🎯 **避免孤立节点**：确保网络连通性

---

### **2. 链路模型（Link Model）**

#### **2.1 链路误码率：`_link_error_prob()`**

```python
def _link_error_prob(d, k=0.05, gamma=2.0):
    """计算链路误码率"""
    return min(0.95, k * (d ** gamma))
```

**公式**：`e(u,v) = min(0.95, k * d^γ)`

**参数**：
- `k = 0.05`：误码率系数
- `gamma = 2.0`：路径损耗指数
- `d`：节点间距离

**特性**：
- 距离越远，误码率越高
- 上限为 0.95，避免链路完全不可用

---

#### **2.2 最小发射功率：`_min_tx_power()`**

```python
def _min_tx_power(d, base=1.0, tau=2.0):
    """计算最小发射功率需求"""
    return base * (d ** tau)
```

**公式**：`w(u) = base * d^τ`

**参数**：
- `base = 1.0`：基础功率
- `tau = 2.0`：功率衰减指数
- `d`：到最远转发节点的距离

**用途**：用于计算发射能耗 `C_h = w(u) / ρ`

---

### **3. 期望代价计算：`_expected_cost_given_fwd()`** ⭐⭐⭐

这是EEOR算法的**核心函数**，计算给定转发节点集合 `Fwd` 时的期望代价。

#### **3.1 输入参数**
- `u_id`：当前节点ID
- `Fwd_ids`：候选转发节点ID列表（按代价升序排列）
- `C`：所有节点到目标的当前期望代价字典
- `neighbor_map`：邻居图
- `node_dict`：节点字典

#### **3.2 计算流程**

```
步骤1: 计算链路误码率乘积 α
───────────────────────────────
α = ∏(e(u, v))  for v in Fwd
   = e(u, v1) × e(u, v2) × ... × e(u, vk)

含义：所有转发节点都失败的概率

步骤2: 计算至少一个节点收到的概率 ρ
────────────────────────────────────
ρ = 1 - α

含义：至少有一个转发节点成功接收的概率

步骤3: 计算发射功率 w(u)
─────────────────────────
w(u) = min_tx_power(max_d)
      = base * (max_d^τ)

其中 max_d = max{distance(u, v) for v in Fwd}

含义：覆盖最远转发节点所需的最小功率

步骤4: 计算发射能耗 C_h
───────────────────────
C_h = w(u) / ρ

含义：期望发射能耗（论文式(2)）
      = 发射到"至少一人收到"的期望发射能耗

步骤5: 计算后续转发代价 β
───────────────────────────
按优先级顺序（C 从小到大）计算：
β = Σ [prefix_fail × (1 - e_uv) × C_v]

其中：
  - prefix_fail：前面所有节点都失败的概率
  - (1 - e_uv)：节点 v 成功接收的概率
  - C_v：节点 v 到目标的期望代价

步骤6: 计算后续转发期望代价 C_f
─────────────────────────────────
C_f = β / ρ

含义：后续转发代价（论文式(4)）

步骤7: 计算总期望代价 C_u
──────────────────────────
C_u = C_h + C_f
    = (w(u) / ρ) + (β / ρ)
    = (w(u) + β) / ρ
```

#### **3.3 数学公式总结**

```
给定转发集合 Fwd = {v1, v2, ..., vk}（按 C 升序）

α = ∏ e(u, vi)              (所有转发节点都失败的概率)
ρ = 1 - α                   (至少一个转发节点成功的概率)
w(u) = base * max_d^τ       (到最远转发节点的最小功率)
C_h = w(u) / ρ              (期望发射能耗)

β = Σ [prefix_fail_i × (1 - e_uv_i) × C_vi]
    其中 prefix_fail_i = ∏ e(u, vj) for j < i
C_f = β / ρ                 (期望转发代价)

C_u = C_h + C_f = (w(u) + β) / ρ  (总期望代价)
```

---

### **4. 前缀选择：`_select_forwarder_prefix()`** ⭐⭐

为节点 `u` 选择最优的转发节点前缀（子集）。

#### **4.1 算法流程**

```
输入：
  - u_id：当前节点ID
  - neighbors_ids：所有邻居节点ID列表
  - C：所有节点到目标的期望代价字典
  - neighbor_map：邻居图

算法（贪心策略）：
─────────────────
1. 按期望代价 C 升序排序邻居节点
   ordered = sorted(neighbors_ids, key=lambda vid: C.get(vid, inf))

2. 初始化
   best_cost = inf
   best_fwd = []
   trial_fwd = []

3. 贪心扩展前缀
   for vid in ordered:
       trial_fwd.append(vid)
       cost, _, _ = _expected_cost_given_fwd(u_id, trial_fwd, C, neighbor_map, None)
       
       if cost < best_cost:
           best_cost = cost
           best_fwd = list(trial_fwd)  # 保存当前最优前缀
       else:
           break  # 成本不再下降，停止（Theorem 2/3）

4. 返回最优前缀
   return best_cost, best_fwd
```

#### **4.2 理论依据**

- **Theorem 2/3**：前缀按代价升序排列时，期望代价函数具有**单调性**
- **Algorithm 1**：贪心策略可以在 `O(n)` 时间内找到最优前缀
- **停止条件**：当前缀添加新节点后，总代价不再下降，说明已找到最优

---

### **5. 全网代价计算：`eeor_compute_costs()`** ⭐⭐

计算网络中所有节点到目标节点的期望代价。

#### **5.1 基础版本：`eeor_compute_costs()`**

```python
def eeor_compute_costs(nodes, target_node_id, max_iter=20):
    """使用固定通信范围的EEOR代价计算"""
```

**算法流程**：
```
1. 构建邻居图
   neighbor_map, node_dict = _build_neighbors(nodes)

2. 初始化代价字典
   C = {vid: inf for vid in V}        # 所有节点初始代价为无穷大
   FWD = {vid: [] for vid in V}        # 转发节点前缀字典
   C[target_node_id] = 0.0             # 目标节点代价为0

3. 迭代松弛（最多 max_iter 次）
   for iteration in range(max_iter):
       updated = False
       
       for u_id in V:  # 遍历所有节点
           if u_id == target_node_id:
               FWD[u_id] = []
               continue
           
           # 获取邻居节点
           neigh_ids = [nid for (nid, _) in neighbor_map[u_id]]
           if not neigh_ids:
               continue
           
           # 计算新的期望代价和转发前缀
           new_cost, new_fwd = _select_forwarder_prefix(u_id, neigh_ids, C, neighbor_map)
           
           # 如果代价降低，更新
           if new_cost < C[u_id] - 1e-12:  # 容差避免浮点误差
               C[u_id] = new_cost
               FWD[u_id] = new_fwd
               updated = True
       
       # 如果没有更新，提前终止
       if not updated:
           break

4. 返回结果
   return C, FWD
```

**时间复杂度**：`O(max_iter × |V| × |neighbors| × log|neighbors|)`

---

#### **5.2 自适应版本：`eeor_compute_costs_adaptive()`** ⭐

```python
def eeor_compute_costs_adaptive(nodes, target_node_id, max_iter=20, target_neighbors=6):
    """使用自适应邻居发现的EEOR代价计算"""
```

**与基础版本的区别**：
- 使用 `_build_neighbors_adaptive()` 构建邻居图
- 其他逻辑完全相同

**优势**：
- 自适应网络密度
- 更好的网络连通性
- 避免孤立节点

---

### **6. 路径查找：`eeor_find_path()`** ⭐

从源节点到目标节点，沿着最优转发路径查找实际传输路径。

#### **6.1 基础版本：`eeor_find_path()`**

```python
def eeor_find_path(nodes, source_node, dest_node, max_hops=5):
    """从源节点到目标节点查找路径"""
```

**算法流程**：
```
1. 计算所有节点到目标的期望代价和转发前缀
   C, FWD = eeor_compute_costs(nodes, dest_node.node_id)

2. 初始化路径
   path = [source_node]
   cur = source_node.node_id
   hops = 0

3. 沿着最优转发路径前进
   while cur != dest_node.node_id and hops < max_hops:
       # 获取当前节点的最优转发前缀
       fwd = FWD.get(cur, [])
       if not fwd:
           break  # 无法继续转发
       
       # 选择转发前缀中代价最小的节点（第一个）
       nxt = fwd[0]  # 已按 C 升序排列
       
       # 避免环路
       if nxt == cur or any(n.node_id == nxt for n in path):
           break
       
       # 添加到路径
       path.append(next(n for n in nodes if n.node_id == nxt))
       cur = nxt
       hops += 1

4. 检查是否成功到达目标
   if path[-1].node_id != dest_node.node_id:
       return None  # 路径查找失败
   return path
```

---

#### **6.2 自适应版本：`eeor_find_path_adaptive()`** ⭐

```python
def eeor_find_path_adaptive(nodes, source_node, dest_node, max_hops=5, target_neighbors=6):
    """使用自适应邻居发现的EEOR路径查找"""
```

**与基础版本的区别**：
- 使用 `eeor_compute_costs_adaptive()` 计算代价
- 物理中心节点已在邻居构建阶段被排除

**特性**：
- ✅ **自动处理单跳和多跳**：如果源和目标直接可达，返回单跳路径；否则返回多跳路径
- ✅ **能量感知**：选择能量高效的路径
- ✅ **环路检测**：防止路径中出现环路

---

## 📊 **算法执行示例**

### **示例场景**

假设网络中有以下节点：
```
Node 1 (源) → ... → Node 5 (目标)
```

**执行流程**：

```
步骤1: 构建邻居图
───────────────────
Node 1 的邻居: [Node 2, Node 3]
Node 2 的邻居: [Node 1, Node 4]
Node 3 的邻居: [Node 1, Node 4]
Node 4 的邻居: [Node 2, Node 3, Node 5]
Node 5 的邻居: [Node 4]  (目标节点)

步骤2: 计算期望代价（迭代过程）
───────────────────────────────
迭代0:
  C[5] = 0.0  (目标节点)
  
迭代1:
  C[4] = 计算期望代价（考虑转发到 Node 5）
  FWD[4] = [5]  (转发前缀)
  
迭代2:
  C[2] = 计算期望代价（考虑转发到 Node 4）
  C[3] = 计算期望代价（考虑转发到 Node 4）
  FWD[2] = [4]
  FWD[3] = [4]
  
迭代3:
  C[1] = 计算期望代价（考虑转发到 [2, 3]）
  FWD[1] = [2, 3]  (可能有多个候选)

步骤3: 路径查找
────────────────
从 Node 1 开始：
  FWD[1] = [2, 3]
  选择 Node 2 (代价最小)
  path = [1, 2]
  
从 Node 2 开始：
  FWD[2] = [4]
  选择 Node 4
  path = [1, 2, 4]
  
从 Node 4 开始：
  FWD[4] = [5]
  选择 Node 5
  path = [1, 2, 4, 5]  ✅ 到达目标
```

---

## 🎯 **算法特点与优势**

### **优势**

1. **能量高效**
   - 考虑链路误码率和传输功率
   - 选择期望能耗最小的路径

2. **鲁棒性强**
   - 多路径机会传输，提高可靠性
   - 链路失败时有备用路径

3. **自适应**
   - 自适应邻居发现，适应网络密度
   - 动态调整通信范围

4. **物理中心节点隔离**
   - 自动排除物理中心节点（ID=0）
   - 物理中心不参与能量传输路由

---

### **局限性**

1. **计算复杂度**
   - 每次路径查找需要重新计算全网代价
   - 时间复杂度：`O(max_iter × |V|²)`

2. **固定误码率模型**
   - 使用简化的误码率模型 `e = k × d^γ`
   - 未考虑实际信道衰落、干扰等因素

3. **固定功率模型**
   - 使用简化的功率模型 `w = base × d^τ`
   - 未考虑实际功率控制算法

---

## 📝 **代码调用链**

```
用户调用
  ↓
eeor_find_path_adaptive(nodes, source, dest, max_hops, target_neighbors)
  ↓
eeor_compute_costs_adaptive(nodes, target_id, max_iter, target_neighbors)
  ↓
_build_neighbors_adaptive(nodes, target_neighbors)
  ├─→ 构建邻居图
  └─→ 排除物理中心节点
  ↓
迭代计算期望代价：
  for each node u:
    _select_forwarder_prefix(u, neighbors, C, neighbor_map)
      ↓
      _expected_cost_given_fwd(u, Fwd, C, neighbor_map, node_dict)
        ├─→ 计算 α, ρ
        ├─→ 计算 w(u)
        ├─→ 计算 C_h = w(u) / ρ
        ├─→ 计算 β
        └─→ 计算 C_f = β / ρ
        └─→ 返回 C_u = C_h + C_f
  ↓
返回 C（代价字典）和 FWD（转发前缀字典）
  ↓
沿着 FWD 路径从源节点到目标节点
  ↓
返回路径列表 [source, relay1, ..., dest]
```

---

## ⚙️ **参数说明**

### **算法参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_iter` | 20 | 代价计算的迭代次数上限 |
| `target_neighbors` | 6 | 自适应邻居发现的目标邻居数 |
| `max_hops` | 5 | 路径查找的最大跳数 |

### **链路模型参数**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `k` | 0.05 | 链路误码率系数 |
| `gamma` | 2.0 | 误码率的路径损耗指数 |
| `base` | 1.0 | 最小发射功率的基础值 |
| `tau` | 2.0 | 功率衰减指数 |

---

## 🔍 **实现细节**

### **物理中心节点排除机制**

```python
# 在 _build_neighbors_adaptive() 中
for ni in nodes:
    # 排除物理中心节点（ID=0完全不参与WET）
    if hasattr(ni, 'is_physical_center') and ni.is_physical_center:
        continue  # 跳过作为源的节点
    
    for nj in nodes:
        # 排除物理中心节点作为邻居
        if hasattr(nj, 'is_physical_center') and nj.is_physical_center:
            continue  # 跳过作为邻居的节点
```

**效果**：
- 物理中心节点不会出现在任何路径中
- 路径只包含普通节点
- 符合物理中心不参与WET的设计

---

### **环路检测机制**

```python
# 在路径查找中
if nxt == cur or any(n.node_id == nxt for n in path):
    break  # 避免环路
```

**检测条件**：
1. 下一跳节点是当前节点（自环）
2. 下一跳节点已在路径中（环路）

---

### **浮点数容差处理**

```python
# 在代价更新中
if new_cost < C[u_id] - 1e-12:  # 容差避免浮点误差
    C[u_id] = new_cost
    FWD[u_id] = new_fwd
```

**原因**：避免浮点数比较误差导致算法不收敛

---

## 📈 **性能分析**

### **时间复杂度**

- **邻居构建**：`O(|V|²)` 或 `O(|V|² log|V|)`（自适应版本需要排序）
- **代价计算**：`O(max_iter × |V| × |neighbors| × log|neighbors|)`
  - 最坏情况：`O(max_iter × |V|³)`
- **路径查找**：`O(max_hops)`

**总体复杂度**：`O(max_iter × |V|³)`（最坏情况）

---

### **空间复杂度**

- **邻居图**：`O(|V| × avg_degree)`
- **代价字典**：`O(|V|)`
- **转发前缀字典**：`O(|V| × avg_fwd_size)`

**总体复杂度**：`O(|V|²)`（平均情况）

---

## 🔄 **与机会路由（Opportunistic Routing）的对比**

| 特性 | EEOR | 机会路由（Dijkstra） |
|------|------|---------------------|
| **路由策略** | 基于期望代价的多路径 | 基于距离的最短路径 |
| **能量感知** | ✅ 是 | ❌ 否 |
| **链路误码率** | ✅ 考虑 | ❌ 不考虑 |
| **转发机制** | 多节点候选转发 | 单节点确定性转发 |
| **计算复杂度** | 较高（O(|V|³)） | 较低（O(|V|²)） |
| **适用场景** | 能量受限网络 | 距离优先网络 |

---

## ✅ **总结**

EEOR是一个**能量高效的机会路由算法**，通过计算期望代价来选择最优的转发节点集合。其主要特点：

1. ✅ **多路径机会传输**：提高可靠性
2. ✅ **能量感知**：考虑链路误码率和传输功率
3. ✅ **自适应邻居发现**：适应网络密度变化
4. ✅ **物理中心节点隔离**：符合系统设计
5. ✅ **贪心前缀选择**：高效找到最优转发集合

目前系统中，**所有调度器都使用EEOR的自适应版本**进行能量传输路径规划，确保了路径选择的能量高效性。

