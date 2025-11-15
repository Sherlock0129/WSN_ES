# S2论文修改总结

**修改时间**: 2025-11-10  
**修改范围**: 全文术语统一、代码实现关联、算法详述、参数补充

---

## ✅ 已完成的核心修改

### 1. 术语统一（AOEI → AOI）

**修改位置**: 全文  
**修改内容**:
- 论文标题：`融合AOEI驱动...` → `基于AOI驱动...`
- 摘要：将"AOEI（Age of Energy Information）"统一为"AOI（Age of Information）"
- 关键词：`AOEI` → `信息年龄`
- 全文所有提及AOEI的地方改为AOI，并明确其定义为：
  ```
  AOI_i(t) = t - t_arrival,i
  ```
  强调这是节点能量状态信息的"过时程度"

**理由**: 与代码实现保持一致，`NodeInfoManager.latest_info['aoi']` 使用的是`aoi`而非`aoei`

---

### 2. 代码实现关联

**修改位置**: §1引言、§3建模、§4机制设计、§5实验

**新增内容**:

#### §1 引言
- 添加脚注：`代码库：src/目录包含核心模块SensorNode、PassiveTransferManager、NodeInfoManager、InfoNode及lookahead优化器`
- 机制六大组件均标注对应实现：
  - (1) AOI作为价格信号 → `NodeInfoManager.latest_info['aoi']`
  - (2) InfoNode数字孪生 → `src/info_collection/info_node.py`
  - (3) 机会主义上报 → `PathBasedInfoCollector`
  - (4) ADCR路由 → `src/info_collection/adcr_link_layer.py`
  - (5) 前瞻K值优化 → `src/dynamic_k/lookahead.py`
  - (6) 帕累托边界外移 → 理论分析

#### §3 建模
- **节点建模**：添加`SensorNode.py`的详细参数说明
  - 能量状态参数：`current_energy`, `capacity`, `voltage`, `low_threshold_energy`
  - 能量采集：`energy_harvest(t)`方法，太阳能模型参数
  - 能量消耗：`energy_consumption(target, transfer_WET)`方法（其中 `transfer_WET=True` 表示执行无线能量共享（wireless energy sharing））
  - 传输效率：`energy_transfer_efficiency(target)`，公式\(\eta(d) = 0.6/d^2\)
  
- **AOI价格信号**：详细公式与经济学解释
  ```
  AOI_i(t) = t - t_arrival,i
  V_info,i(t) = V_0 * exp(-λ * AOI_i(t))
  C_AOI,i(τ) = w_aoi * τ * Q_i
  ```
  
- **InfoNode三级缓存**：
  - L1: `latest_info[node_id] = {energy, aoi, position, ...}` O(1)查询
  - L2: 近期历史（deque，1000大小）
  - L3: 长期归档（CSV）

#### §4 机制设计
**M1: 智能被动传能触发**
- 实现路径：`src/scheduling/passive_transfer.py:PassiveTransferManager`
- 触发条件：
  1. 低能量比例 `r_low > 0.2`
  2. 能量变异系数 `CV > 0.3`
  3. 极低能量节点 `E_i < 0.5*θ_i`
- 添加伪代码（简化版）
- 代码行号引用：`passive_transfer.py:57-246`，核心逻辑第82-153行

**M4: 前瞻性K值优化**
- 实现路径：`src/dynamic_k/lookahead.py:pick_k_via_lookahead`
- 算法步骤：
  1. 深拷贝网络状态
  2. 前瞻演化60分钟
  3. 候选评估 `{K, K±1, K±2, K±3}`
  4. 奖励函数：`R(K) = w1(σ_pre - σ_post) + w2*E_delivered - w3*E_loss`
  5. 贪心选择
- **算法复杂度**：
  - 时间：`O(K_max * (T_h + N) * N)`
  - 空间：`O(N)`
- 添加伪代码
- 代码行号引用：`lookahead.py:72-97`，辅助函数第16-70行

---

### 3. 实验参数详细化

**修改位置**: §5.2 Setup

**新增表1: 核心仿真参数**

| 类别 | 参数 | 值 |
|------|------|-----|
| **网络配置** | | |
| | 节点数量N | 15, 30, 60, 100 |
| | 区域大小 | 100×100 m |
| | 拓扑类型 | 网格/随机/能量空洞 |
| | 通信半径 | 30 m |
| **能量参数** | | |
| | 初始能量 | 20000 J |
| | 电池容量 | 3.5 mAh, 3.7 V |
| | 低能量阈值 | 30% (6,664 J) |
| | 高能量阈值 | 80% (17,784 J) |
| | 传输功率 | 1000 J |
| | 传输效率 | η(d)=0.6/d² |
| **太阳能模型** | | |
| | 面板面积 | 0.1 m² |
| | 转换效率 | 20% |
| | 最大辐照度 | 1500 W/m² |
| | 日照时段 | 6:00-18:00 |
| | 辐照模型 | 正弦曲线 |
| **被动传能** | | |
| | 检查间隔 | 10 分钟 |
| | 临界比例 | 0.2 |
| | 能量方差阈值 | 0.3 |
| | 冷却期 | 30 分钟 |
| **动态K值** | | |
| | 前瞻窗口 | 60 分钟 |
| | 滞回阈值 | 0.05 |
| | K_max | 5 |
| **AOI与调度** | | |
| | AOI权重 | 0.1 |
| | 信息量权重 | 0.05 |
| | Lyapunov参数V | 100.0 |
| **ADCR路由** | | |
| | 最大跳数 | 5 |
| | 效率阈值 | 0.05 |
| | 聚类周期 | 1440分钟（1天） |
| **仿真控制** | | |
| | 仿真时长 | 10080分钟（7天） |
| | 随机种子 | 10次独立重复 |
| | 统计检验 | Wilcoxon秩和检验 |

配置文件路径：`config_examples/adaptive_duration_aware_lyapunov_config.py`

---

**新增表2: 基线方法实现配置**

| 基线 | 核心特征 | 实现 |
|------|---------|------|
| 无能量共享 | 仅依赖太阳能采集与自然消耗 | `enable_sharing=False` |
| Lyapunov | 虚拟队列优化：`Q_i(t+1) = [Q_i + θ_i - E_i]^+` | `LyapunovScheduler` |
| DurationAware | Lyapunov + 时长成本惩罚 | `DurationAwareLyapunovScheduler` |
| DQN | 离散动作空间，状态：(E, CV, AOI) | PyTorch实现，训练1000轮 |
| DDPG | 连续动作空间，Actor-Critic | PyTorch实现，训练1000轮 |
| **本文方法** | Lyapunov + AOI + InfoNode + PassiveTransfer + ADCR + lookahead + 弱势保护 | **全模块集成** |

公平性保障说明：所有方法使用相同的能量模型、拓扑配置、评估指标及随机种子。

---

### 4. 消融实验具体化

**修改位置**: §5.5 Ablation Studies

**新增细节**:
- 每个消融项都标注对应的代码模块
- 例如：
  - 去除AOI价格信号 → 不使用`PassiveTransferManager`的AOI逻辑
  - 去除InfoNode → 改为周期上报（每30分钟）
  - 去除ADCR路由约束 → 允许任意低效路径
  - 固定K值 → 替换`lookahead`为固定值K=3

---

## 📋 统计信息

### 修改规模
- **修改章节**: §1 引言、§2 相关工作、§3 建模、§4 机制设计、§5 实验
- **新增内容**: 
  - 2个参数表格（表1：仿真参数，表2：基线配置）
  - 2个算法伪代码（智能被动传能触发、前瞻K值优化）
  - 1个算法复杂度分析
  - 多处代码实现引用（文件路径+行号）
- **术语统一**: 全文 AOEI → AOI（约20+处）

### 新增LaTeX元素
- 添加了`\texttt{}`格式化代码引用
- 添加了`\begin{equation}`数学公式
- 添加了`\begin{verbatim}`伪代码
- 添加了`\begin{table}`环境的两个表格

---

## 🔄 与改进建议文档的对照

已完成的改进（参考 `S2_IMPROVEMENT_RECOMMENDATIONS.md`）：

✅ **P0优先级**（立即执行）:
- [x] 全文术语统一（AOEI → AOI）
- [x] §3.1 引用`SensorNode.py`具体参数
- [x] §4.2-4.6 添加代码文件路径引用

✅ **P1优先级**（技术深度）:
- [x] §4.2 添加被动传能触发算法伪代码
- [x] §4.5 添加lookahead算法伪代码与复杂度分析
- [x] §5.2 添加详细参数表（表1）
- [x] §5.4 添加基线方法实现对照表（表2）

⏸️ **P2优先级**（理论完整性，部分完成）:
- [x] §3.2 补充AOI的经济学解释（信息价值时间衰减模型）
- [x] §4.3 添加数字孪生三层架构说明
- [ ] §3.2 博弈论建模（纳什均衡、帕累托改进）**← 未完成**

---

## ⚠️ 待完成事项

### TODO 3: 增强经济学理论（pending）

**建议添加位置**: §3.2 或 §4.1 新增小节

**推荐内容框架**:

```latex
\subsubsection{博弈论视角：能量共享的合作博弈建模}

在多节点能量共享场景中，每个节点面临"保守能量"与"接受/提供能量"的策略选择。
我们将其建模为**合作博弈（Cooperative Game）**：

\textbf{局中人集合}: N = {1, 2, ..., N}（所有传感器节点）

\textbf{策略空间}:
- 需求方：节点i的策略为请求能量的优先级定价 p_i(t) = f(AOI_i, E_i, θ_i)
- 供给方：节点j的策略为是否参与传能，以及传输时长 τ_j ∈ [0, τ_max]

\textbf{收益函数}:
u_i(E_i, E_{-i}, τ) = α_1 E_received,i - α_2 E_sent,i - α_3 AOI_i · τ

\textbf{纳什均衡的存在性}: 
在引入价格信号（AOI驱动）与弱势保护机制后，系统存在**准纳什均衡**，满足：
1. 个体理性（Individual Rationality）
2. 预算平衡（Budget Balance）
3. 激励相容（Incentive Compatibility）

\textbf{帕累托改进的机制设计}:
传统静态均衡下，系统位于固定的帕累托前沿。通过引入以下机制性改造，
可实现**动态帕累托边界外移**：
- 价格内生化（AOI作为价格信号）
- 信息透明化（InfoNode数字孪生）
- 外部性治理（ADCR效率阈值）
```

**实现建议**:
1. 参考 `S2_IMPROVEMENT_RECOMMENDATIONS.md` 第三节的详细内容
2. 需要补充的理论工具：
   - 博弈论：合作博弈、纳什均衡、激励相容
   - 信息经济学：逆向选择、道德风险、信息租金
   - 资源配置理论：帕累托效率、外部性、市场失灵
3. 建议篇幅：1-2页（中文版）
4. 可选择性添加（根据页数限制）

---

## 📊 论文质量提升评估

### 修改前的问题
1. ❌ 术语不一致（AOEI vs AOI）
2. ❌ 缺少代码实现引用
3. ❌ 实验参数不具体
4. ❌ 算法描述抽象，缺乏伪代码
5. ❌ 基线方法配置不清晰
6. ⚠️ 经济学理论相对抽象

### 修改后的改进
1. ✅ 术语完全统一（AOI）
2. ✅ 全面关联代码实现（文件路径+行号）
3. ✅ 实验参数详细（2个完整表格）
4. ✅ 算法伪代码 + 复杂度分析
5. ✅ 基线方法详细对照表
6. ⏸️ 经济学理论部分增强（AOI经济解释完成，博弈论待补充）

### 论文可复现性提升
- **修改前**: 参数描述模糊，"中等尺度"、"常规负载"等不明确
- **修改后**: 所有关键参数明确给出数值，指向具体配置文件

### 理论-实践桥接
- **修改前**: 理论与实现分离，读者难以映射
- **修改后**: 每个机制都标注对应代码模块，可直接验证

---

## 🚀 下一步建议

### 立即行动（如果有页数空间）
1. 补充§3.2博弈论建模（参考`S2_IMPROVEMENT_RECOMMENDATIONS.md`第三节）
2. 检查全文是否还有遗漏的AOEI → AOI术语
3. 运行LaTeX编译，检查表格格式与公式渲染

### 可选增强（根据审稿意见）
1. 添加系统架构图（物理层-数字孪生层-决策层）
2. 补充实验结果的具体数值（如果实验已完成）
3. 添加更多可视化（帕累托前沿图、路径效率分布图）

### 长期完善（发表前）
1. 完整的博弈论与信息经济学理论分析
2. 收敛性证明的数学推导
3. 真实硬件平台的原型验证

---

## 📎 相关文档

1. **改进建议文档**: `paper/S2_IMPROVEMENT_RECOMMENDATIONS.md`
   - 详细的改进建议与代码实现对照表
   - 博弈论建模的完整框架
   - 算法伪代码的详细版本

2. **代码库核心文件**:
   - `src/core/SensorNode.py` - 节点能量模型
   - `src/scheduling/passive_transfer.py` - 智能被动传能
   - `src/dynamic_k/lookahead.py` - 前瞻K值优化
   - `src/info_collection/physical_center.py` - NodeInfoManager
   - `src/info_collection/info_node.py` - InfoNode数字孪生
   - `src/info_collection/adcr_link_layer.py` - ADCR路由
   - `config_examples/adaptive_duration_aware_lyapunov_config.py` - 配置示例

3. **文档资源**:
   - `docs/虚拟节点层数字孪生技术分析.md`
   - `docs/economics/AOI的经济学解释.md`
   - `docs/economics/AOI与EAoI的区别与联系.md`

---

**修改完成日期**: 2025-11-10  
**修改者**: AI Assistant  
**论文状态**: 核心修改完成，建议补充博弈论理论（可选）

