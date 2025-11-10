# S2论文改进建议与代码实现对照

**生成时间**: 2025-11-10  
**目标**: 将S2论文与WSN_ES代码实现紧密关联，强化经济学理论基础

---

## 一、核心术语调整

### 1.1 AOI vs AOEI 的概念统一

**当前论文**: 使用 **AOEI (Age of Energy Information, 能量信息年龄)** 作为核心概念

**代码实现**: 主要使用 **AOI (Age of Information, 信息年龄)**

**问题分析**:
- 代码库中的核心指标是 `AOI = current_time - arrival_time`
- `NodeInfoManager` 维护的是 `'aoi'` 字段，而非 `'aoei'`
- 调度器（如 `DurationAwareLyapunovScheduler`）使用 `aoi_cost` 和 `aoi_penalty`

**建议方案**:

**选项A（推荐）**: 在论文中统一使用 **AOI**，强调这是"能量-信息协同决策"中的信息年龄
```latex
% 修改前
AOEI（Age of Energy Information，能量信息年龄）

% 修改后
AOI（Age of Information，信息年龄）作为能量共享决策的价格信号
```

**选项B**: 保留AOEI但明确定义其与AOI的关系
```latex
本文定义能量-信息协同场景下的AOEI（Age of Energy Information），
其核心度量即为信息年龄AOI，用于刻画节点能量状态信息的时效性...
```

---

## 二、代码实现关联表

### 2.1 核心模块与论文章节对应

| 论文章节 | 核心概念 | 代码实现 | 文件路径 |
|---------|---------|---------|---------|
| §3.2 经济学解释与AOEI | AOI价格信号 | `NodeInfoManager.latest_info['aoi']` | `src/info_collection/physical_center.py:84` |
| §4.2 M1: AOEI定价与触发 | 触发决策 | `PassiveTransferManager.should_trigger_transfer()` | `src/scheduling/passive_transfer.py:57` |
| §4.3 M2: InfoNode + 机会主义上报 | 数字孪生 | `InfoNode` + `NodeInfoManager` | `src/info_collection/info_node.py:13` <br> `src/info_collection/physical_center.py:36` |
| §4.4 M3: EETOR路由 | 能量传输路由 | `ADCRLinkLayerVirtual` (ADCR协议) | `src/info_collection/adcr_link_layer.py:21` |
| §4.5 M4: 自适应时长 | 动态K值调整 | `pick_k_via_lookahead()` | `src/dynamic_k/lookahead.py:72` |
| §4.6 M5: 弱势保护 | 低能量节点保护 | `PassiveTransferManager.critical_ratio` | `src/scheduling/passive_transfer.py:25` |
| §3.1 节点建模 | 能量模型 | `SensorNode` | `src/core/SensorNode.py:6` |

### 2.2 关键算法与代码对应

#### 算法1: 智能被动传能触发决策

**论文描述** (§4.2):
> 以 AOEI 为核心价格信号，构造价格函数 \(p_i(t)=f(A_i(t), \text{稀缺度}, \text{能量占比})\)

**代码实现**:
```python
# src/scheduling/passive_transfer.py: lines 57-153
def should_trigger_transfer(self, t: int, network) -> Tuple[bool, Optional[str]]:
    """
    综合决策因素：
    1. 低能量节点比例 > critical_ratio (默认0.2)
    2. 能量变异系数CV > energy_variance_threshold (默认0.3)
    3. 极低能量节点存在（< 阈值50%）
    """
    # 获取节点能量状态（通过InfoNode - 数字孪生层）
    if self.node_info_manager is not None:
        all_info = self.node_info_manager.get_all_nodes_info()
        energies = np.array([info['energy'] for info in regular_nodes_info.values()])
        
    # 决策条件
    low_energy_ratio = np.sum(energies < thresholds) / total_nodes
    energy_cv = np.std(energies) / np.mean(energies)  # 变异系数
    
    if low_energy_ratio > self.critical_ratio:
        should_trigger = True
        reasons.append(f"低能量节点比例={low_energy_ratio:.2%}")
```

**建议论文修改**:
```latex
% §4.2 增加伪代码
\begin{algorithmic}
\STATE \textbf{输入:} 当前时间 $t$, 节点集合 $\mathcal{N}$, InfoNode信息表
\STATE \textbf{输出:} 是否触发传能 $\tau \in \{0, 1\}$
\IF{$t \bmod T_{\text{check}} \neq 0$ \OR $(t - t_{\text{last}}) < T_{\text{cool}}$}
    \RETURN False  \COMMENT{检查间隔与冷却期}
\ENDIF
\STATE 从InfoNode获取能量状态: $E = \{e_i | i \in \mathcal{N}\}$
\STATE 计算低能量比例: $r_{\text{low}} = |\{i: e_i < \theta_i\}| / |\mathcal{N}|$
\STATE 计算能量变异系数: $CV = \sigma(E) / \mu(E)$
\IF{$r_{\text{low}} > r_{\text{crit}}$ \OR $CV > CV_{\text{th}}$ \OR $\exists i: e_i < 0.5\theta_i$}
    \RETURN True
\ENDIF
\RETURN False
\end{algorithmic}

% 引用代码实现
\textit{实现细节参见}: \texttt{src/scheduling/passive\_transfer.py:57-246}
```

#### 算法2: 前瞻性动态K值优化

**论文描述** (§4.5):
> 基于前瞻优化的K值调整

**代码实现**:
```python
# src/dynamic_k/lookahead.py: lines 72-97
def pick_k_via_lookahead(network, scheduler, t, current_K, direction, improve, 
                         hysteresis, K_max, horizon_minutes=60, reward_fn):
    """
    前瞻策略:
    1. 深拷贝当前网络状态
    2. 模拟未来 horizon_minutes 分钟的能量演化
    3. 对候选K值 {K, K±1, K±2, K±3} 评估传能后的网络状态
    4. 选择使 reward_fn(stats) 最大的K值
    
    reward_fn评分指标:
    - 能量标准差变化: post_std - pre_std
    - 有效传递能量: delivered_total
    - 能量损耗: total_loss
    """
    # 基于improve与滞回判定搜索方向
    if improve > hysteresis:
        candidates = [current_K, current_K + d, current_K + 2*d, current_K + 3*d]
    # ... 候选集生成
    
    best_k, best_reward = current_K, -1e18
    for k in candidates:
        r = _eval_one_candidate(network, scheduler, k, t, horizon_minutes, reward_fn)
        if r > best_reward:
            best_reward = r
            best_k = k
    return best_k, best_reward
```

**建议论文添加**:
```latex
% §4.5 添加算法复杂度分析
\textbf{算法复杂度}:
\begin{itemize}
    \item \textbf{时间复杂度}: $O(|C_K| \times (T_h \times E_{\text{sim}} + P_{\text{plan}} + E_{\text{exec}}))$
    \begin{itemize}
        \item $|C_K|$: 候选K值数量（通常5个）
        \item $T_h$: 前瞻时间窗口（默认60分钟）
        \item $E_{\text{sim}}$: 单步能量演化复杂度 $O(N)$
        \item $P_{\text{plan}}$: 规划复杂度（依赖于调度器，Lyapunov为 $O(N^2)$）
        \item $E_{\text{exec}}$: 执行复杂度 $O(K \times N)$
    \end{itemize}
    \item \textbf{空间复杂度}: $O(N)$（深拷贝网络状态）
\end{itemize}

\textit{实现细节}: \texttt{src/dynamic\_k/lookahead.py:72-97}
```

#### 算法3: 数字孪生状态同步

**论文描述** (§4.3):
> 以 InfoNode 为数字孪生账户，维护当前/历史/预测的多层状态

**代码实现**:
```python
# src/info_collection/physical_center.py: lines 36-96
class NodeInfoManager:
    """
    三级缓存架构:
    L1: latest_info (Dict) - 最新状态，O(1)查询
    L2: recent_history (deque) - 近期历史，FIFO
    L3: archive_buffer (CSV) - 长期归档，持久化
    """
    def __init__(self, ...):
        self.latest_info: Dict[int, Dict] = {}  # L1缓存
        # latest_info[node_id] = {
        #   'energy': float,           # 能量（可能是估算值）
        #   'aoi': int,                # Age of Information（信息年龄）
        #   'record_time': int,        # 信息记录时间
        #   'arrival_time': int,       # 到达物理中心时间
        #   'position': (x, y),        # 节点位置
        #   'is_solar': bool,          # 是否有太阳能
        #   'is_estimated': bool,      # 是否为估算值
        # }

# src/info_collection/info_node.py: lines 13-66
class InfoNode:
    """
    轻量级节点代理，与SensorNode接口一致
    所有计算公式（距离、效率、能耗）与SensorNode完全相同
    """
    def distance_to(self, other): ...
    def energy_transfer_efficiency(self, target_node): ...
    def energy_consumption(self, target_node, transfer_WET): ...
```

**建议论文添加**:
```latex
% §4.3 添加数字孪生架构图与说明
\textbf{数字孪生三层架构}:

物理层（Physical Layer）与数字孪生层（Digital Twin Layer）通过信息上报机制实时同步：
\begin{itemize}
    \item \textbf{L1-最新状态表}: 维护每个节点的当前状态（能量、位置、AOI等），支持$O(1)$查询。
    \item \textbf{L2-近期历史}: 固定大小的FIFO队列，用于趋势分析和异常检测。
    \item \textbf{L3-长期归档}: CSV批量写入，支持离线分析与模型训练。
\end{itemize}

\textbf{InfoNode的关键特性}:
\begin{enumerate}
    \item \textbf{接口一致性}: 与SensorNode的\texttt{distance\_to()}、\texttt{energy\_transfer\_efficiency()}等方法完全一致，确保计算结果可复现。
    \item \textbf{避免上帝视角}: 调度器仅能通过InfoNode访问信息，不直接访问SensorNode的物理状态。
    \item \textbf{能量估算}: 对未及时上报的节点，基于历史消耗率与太阳能模型进行能量估算，并标记\texttt{is\_estimated=True}。
\end{enumerate}

\textit{实现}: \texttt{src/info\_collection/physical\_center.py:36-96}, \\
\texttt{src/info\_collection/info\_node.py:13-66}
```

---

## 三、经济学理论框架增强

### 3.1 补充博弈论视角

**当前不足**: 论文强调"经济学机制"，但缺少具体的博弈论建模

**建议添加** (§3.2或§4.1):

```latex
\subsection{博弈论视角：能量共享的合作博弈建模}

在多节点能量共享场景中，每个节点面临"保守能量"与"接受/提供能量"的策略选择。我们将其建模为\textbf{合作博弈（Cooperative Game）}：

\textbf{局中人集合}: $\mathcal{N} = \{1, 2, \ldots, N\}$（所有传感器节点）

\textbf{策略空间}:
\begin{itemize}
    \item \textbf{需求方}: 节点$i$的策略为请求能量的优先级定价 $p_i(t) = f(\text{AOI}_i, E_i, \theta_i)$
    \item \textbf{供给方}: 节点$j$的策略为是否参与传能，以及传输时长/额度 $\tau_j \in [0, \tau_{\max}]$
\end{itemize}

\textbf{收益函数}:
\begin{equation}
u_i(E_i, E_{-i}, \tau) = \alpha_1 E_{\text{received},i} - \alpha_2 E_{\text{sent},i} - \alpha_3 \text{AOI}_i \cdot \tau
\end{equation}
其中:
\begin{itemize}
    \item $E_{\text{received},i}$: 节点$i$接收的净能量
    \item $E_{\text{sent},i}$: 节点$i$作为供能方发送的能量（含损耗）
    \item $\text{AOI}_i \cdot \tau$: 传能时长导致的信息年龄增长成本
\end{itemize}

\textbf{纳什均衡的存在性}:

在引入价格信号（AOI驱动）与弱势保护机制后，系统存在\textbf{准纳什均衡（Quasi-Nash Equilibrium）}，满足：
\begin{enumerate}
    \item \textbf{个体理性}（Individual Rationality）: 每个节点的收益不低于不参与合作的收益（保底收益）。
    \item \textbf{预算平衡}（Budget Balance）: 系统总能量守恒，$\sum E_{\text{sent}} = \sum E_{\text{received}} + E_{\text{loss}}$。
    \item \textbf{激励相容}（Incentive Compatibility）: 节点如实报告能量状态（通过InfoNode机制实现）是其最优策略。
\end{enumerate}

\textbf{帕累托改进的机制设计}:

传统静态均衡下，系统位于固定的帕累托前沿。通过引入以下机制性改造，可实现\textbf{动态帕累托边界外移}：
\begin{itemize}
    \item \textbf{价格内生化}（AOI作为价格信号）：将外生的"谁需要能量"转化为内生的市场定价。
    \item \textbf{信息透明化}（InfoNode数字孪生）：降低信息不对称，减少逆向选择与道德风险。
    \item \textbf{外部性治理}（EETOR效率阈值）：抑制低效多跳路径的负外部性。
\end{itemize}

该机制设计的核心命题为：
\begin{theorem}[动态帕累托边界外移]
在给定物理资源约束下，若同时满足：
\begin{enumerate}
    \item 价格信号的单调性与预算一致性（命题1）
    \item 外部性抑制的有效性（命题2）
    \item 信息透明度增益（InfoNode的时效性与准确性）
\end{enumerate}
则系统在$(效率, 公平)$二维空间的可达解集$\mathcal{F}_{\text{dynamic}}$严格包含基线解集$\mathcal{F}_{\text{static}}$，即存在$(\eta, \phi) \in \mathcal{F}_{\text{dynamic}}$使得$\eta > \eta_{\text{base}}$且$\phi > \phi_{\text{base}}$对$\forall (\eta_{\text{base}}, \phi_{\text{base}}) \in \mathcal{F}_{\text{static}}$成立。
\end{theorem}
```

### 3.2 信息经济学：价值时间衰减模型

**建议添加** (§3.2):

```latex
\subsubsection{信息价值的时间衰减：经济学建模}

借鉴金融学的\textbf{资产折现理论}和供应链管理的\textbf{商品保质期模型}，我们将节点能量状态信息视为一种"信息资产"，其价值随时间衰减。

\textbf{线性衰减模型}（当前实现）:
\begin{equation}
V_{\text{info}}(t) = V_0 \cdot \left(1 - \frac{\text{AOI}(t)}{\text{AOI}_{\max}}\right)
\end{equation}

\textbf{指数衰减模型}（理论扩展）:
\begin{equation}
V_{\text{info}}(t) = V_0 \cdot e^{-\lambda \cdot \text{AOI}(t)}
\end{equation}
其中$\lambda$为衰减率（类似金融学中的折现率）。

\textbf{经济学解释}:
\begin{itemize}
    \item \textbf{$V_0$（基础价值）}: 信息在新鲜状态下对决策的贡献（类似商品原价）。
    \item \textbf{AOI（资产年龄）}: 信息到达物理中心后的时间，$\text{AOI} = t - t_{\text{arrival}}$。
    \item \textbf{$\lambda$（衰减率）}: 信息失效速度，取决于环境动态性（能量采集波动、负载变化）。
\end{itemize}

\textbf{决策成本函数}:

在Lyapunov优化框架中，AOI以惩罚项的形式内生化：
\begin{equation}
\text{AOI\_penalty} = w_{\text{aoi}} \cdot \Delta \text{AOI} \cdot Q_i
\end{equation}
其中:
\begin{itemize}
    \item $\Delta \text{AOI} = \tau$（传输时长导致的AOI增量）
    \item $Q_i$: 节点$i$的能量虚拟队列长度（需求紧迫度）
    \item $w_{\text{aoi}}$: AOI权重系数（可调参数）
\end{itemize}

\textit{代码实现}: \texttt{src/scheduling/schedulers.py:DurationAwareLyapunovScheduler}
```

---

## 四、实验部分具体化

### 4.1 参数设置与代码配置对照

**当前不足**: 论文§5.2描述了场景与配置，但缺少具体参数值

**建议添加** (§5.2):

```latex
\subsection{Setup：场景与配置（详细参数）}

\textbf{表1: 核心仿真参数}

\begin{table}[h]
\centering
\caption{仿真参数设置（参考\texttt{src/config/}）}
\begin{tabular}{lll}
\hline
\textbf{参数类别} & \textbf{参数名} & \textbf{取值} \\
\hline
\multirow{4}{*}{网络配置} 
  & 节点数量 $N$ & 15, 30, 60, 100 \\
  & 区域大小 & $100 \times 100$ m \\
  & 拓扑类型 & 规则网格/随机/能量空洞 \\
  & 通信半径 & 30 m \\
\hline
\multirow{6}{*}{能量参数} 
  & 初始能量 & 20000 J \\
  & 电池容量 & 3.5 mAh, 3.7 V \\
  & 低能量阈值 & 30\% \\
  & 高能量阈值 & 80\% \\
  & 传输效率模型 & $\eta(d) = 0.6 / d^2$ \\
  & 传输功率 & 1000 J \\
\hline
\multirow{4}{*}{太阳能模型} 
  & 面板面积 & 0.1 $\text{m}^2$ \\
  & 转换效率 & 20\% \\
  & 最大辐照度 $G_{\max}$ & 1500 W/m$^2$ \\
  & 日照周期 & 6:00-18:00 (正弦模型) \\
\hline
\multirow{3}{*}{被动传能} 
  & 检查间隔 $T_{\text{check}}$ & 10 分钟 \\
  & 临界比例 $r_{\text{crit}}$ & 0.2 \\
  & 能量方差阈值 $CV_{\text{th}}$ & 0.3 \\
  & 冷却期 $T_{\text{cool}}$ & 30 分钟 \\
\hline
\multirow{3}{*}{动态K值} 
  & 前瞻时间窗 $T_h$ & 60 分钟 \\
  & 滞回阈值 $h$ & 0.05 \\
  & $K_{\max}$ & 5 \\
\hline
\multirow{2}{*}{AOI参数} 
  & AOI权重 $w_{\text{aoi}}$ & 0.1 \\
  & 信息量权重 $w_{\text{info}}$ & 0.05 \\
\hline
\multirow{2}{*}{仿真控制} 
  & 仿真时长 & 10080 分钟 (7天) \\
  & 随机种子 & 10次独立重复 \\
\hline
\end{tabular}
\end{table}

\textit{配置文件}: \texttt{config\_examples/adaptive\_duration\_aware\_lyapunov\_config.py}
```

### 4.2 对比基线的具体实现

**建议添加** (§5.4):

```latex
\subsection{Baselines：对照方法的实现细节}

\begin{table}[h]
\centering
\caption{基线方法实现配置}
\begin{tabular}{lp{6cm}l}
\hline
\textbf{基线方法} & \textbf{核心特征} & \textbf{实现文件} \\
\hline
无能量共享 
  & 仅依赖太阳能采集与自然消耗 
  & \texttt{enable\_energy\_sharing=False} \\
\hline
Lyapunov 
  & 目标：最小化能量方差 $V(Q)$ 
  & \texttt{src/scheduling/} \\
  & 虚拟队列：$Q_i(t+1) = [Q_i(t) + \theta_i - E_i(t)]^+$ 
  & \texttt{schedulers.py:} \\
  & 传能决策：$\max \sum Q_j \Delta E_j$ 
  & \texttt{LyapunovScheduler} \\
\hline
DurationAware 
  & Lyapunov + 时长成本惩罚 
  & \texttt{schedulers.py:} \\
  & 成本：$C = Q_j E_j - V \cdot \text{loss} - w_{\tau} \tau$ 
  & \texttt{DurationAware-} \\
  &  & \texttt{LyapunovScheduler} \\
\hline
DQN 
  & 离散动作空间：$a \in \{0, 1, 2, ..., K_{\max}\}$ 
  & PyTorch实现 \\
  & 状态：$(E, \text{CV}, \text{AOI})$ 
  & 训练轮次：1000 \\
  & 奖励：$r = -\text{CV} + 0.1 \cdot E_{\text{eff}}$ 
  & $\epsilon$-greedy: 0.1 \\
\hline
DDPG 
  & 连续动作空间：$a \in [0, \tau_{\max}]$ 
  & PyTorch实现 \\
  & Actor-Critic架构 
  & 训练轮次：1000 \\
  & 奖励：同DQN 
  & Ornstein-Uhlenbeck噪声 \\
\hline
\textbf{本文方法} 
  & Lyapunov + AOI + InfoNode + PassiveTransfer 
  & \texttt{全模块集成} \\
  & + ADCR路由 + 动态K + 弱势保护 
  &  \\
\hline
\end{tabular}
\end{table}

\textbf{公平性说明}: 所有方法使用相同的：
\begin{itemize}
    \item 能量模型（传输效率、消耗、采集）
    \item 拓扑配置（节点位置、初始能量）
    \item 评估指标（寿命、CV、效率、AOI）
    \item 随机种子（10次重复）
\end{itemize}
```

---

## 五、具体修改行动清单

### 5.1 立即执行的修改

| 优先级 | 修改位置 | 修改内容 | 预期效果 |
|-------|---------|---------|---------|
| **P0** | 全文 | 将"AOEI"替换为"AOI"（或明确定义AOEI=AOI） | 与代码实现一致 |
| **P0** | §3.1 | 在节点建模中引用`SensorNode.py`的具体参数 | 增强可复现性 |
| **P0** | §4.2-4.6 | 每个机制M1-M6添加代码文件路径引用 | 理论-实践桥接 |
| **P1** | §4.2 | 添加被动传能触发算法伪代码 | 算法可读性 |
| **P1** | §4.5 | 添加lookahead算法伪代码与复杂度分析 | 技术深度 |
| **P1** | §3.2 | 补充博弈论建模与纳什均衡分析 | 经济学理论深度 |
| **P1** | §5.2 | 添加详细参数表（表1） | 实验可复现性 |
| **P1** | §5.4 | 添加基线方法实现对照表（表2） | 对比公平性说明 |
| **P2** | §3.2 | 补充信息价值时间衰减模型的经济学解释 | 理论完整性 |
| **P2** | §4.3 | 添加数字孪生三层架构图 | 可视化辅助理解 |

### 5.2 推荐的章节重组

**当前结构**:
- §3 Modeling
- §4 Problems and Mechanism Design (E-T-M框架)
- §5 Experiments

**建议调整**:
```
§3 System Model and Problem Formulation
  §3.1 Network and Node Model (引用SensorNode.py)
  §3.2 Economic Interpretation: AOI as Price Signal (新增博弈论)
  §3.3 Objective Function and Constraints

§4 Proposed Mechanism: Design and Implementation
  §4.1 Overview: "Price-Account-Rule-Governance" Framework
  §4.2 Mechanism M1: AOI-driven Triggering (对应PassiveTransferManager)
  §4.3 Mechanism M2: Digital Twin with InfoNode (对应NodeInfoManager)
  §4.4 Mechanism M3: Energy-Efficient Routing (对应ADCR)
  §4.5 Mechanism M4: Adaptive Duration via Lookahead (对应lookahead.py)
  §4.6 Mechanism M5: Fairness Protection (对应critical_ratio等)
  §4.7 Algorithm Complexity and Convergence Analysis (新增)

§5 Experiments and Evaluation
  §5.1 Experimental Protocol
  §5.2 Setup and Configuration (新增详细参数表)
  §5.3 Baseline Methods (新增实现对照表)
  §5.4 Results: Overall Performance
  §5.5 Pareto Frontier Analysis
  §5.6 Ablation Studies
  §5.7 Robustness Analysis
```

---

## 六、可选的高级扩展

### 6.1 EAoI（Energy Age of Information）的引入

如果希望保留"能量信息年龄"概念并与AOI区分，可以定义：

```latex
\textbf{定义（EAoI）}: 能量信息年龄（Energy Age of Information）是考虑节点能量状态与信息年龄的复合指标：
\begin{equation}
\text{EAoI}_i(t) = \text{AOI}_i(t) \cdot \left(1 + \beta \cdot \frac{\theta_i - E_i(t)}{\theta_i}\right)
\end{equation}
其中：
\begin{itemize}
    \item $\text{AOI}_i(t)$: 节点$i$的信息年龄
    \item $E_i(t)$: 节点$i$的当前能量
    \item $\theta_i$: 节点$i$的能量阈值
    \item $\beta$: 能量敏感系数（权重参数）
\end{itemize}

\textbf{经济学解释}: EAoI刻画了"低能量节点的信息更需要及时更新"的紧迫性，类似于金融学中的"压力测试下的资产估值"。
```

**代码扩展点**: 在`NodeInfoManager`中添加`'eaoi'`字段：
```python
# src/info_collection/physical_center.py
def calculate_eaoi(self, node_id):
    info = self.latest_info[node_id]
    aoi = info['aoi']
    energy = info['energy']
    threshold = self.thresholds.get(node_id, 6000.0)  # 默认阈值
    beta = 1.0  # 能量敏感系数
    
    energy_urgency = max(0, (threshold - energy) / threshold)
    eaoi = aoi * (1 + beta * energy_urgency)
    return eaoi
```

### 6.2 ADCR协议的详细说明

论文中提到"EETOR"但代码中实际使用"ADCR"（Adaptive Dynamic Clustering Routing），建议统一：

```latex
\subsection{M3: ADCR路由与能量传输路径治理}

\textbf{ADCR（Adaptive Dynamic Clustering Routing）}是面向能量传输的专用路由协议，核心特征：

\begin{enumerate}
    \item \textbf{自适应簇数估计}: 基于节点密度与邻居统计启发式估计最优簇数$K^*$。
    \item \textbf{能量感知簇头选择}: 概率$P_{\text{CH},i} = P_{\min} + (P_{\max} - P_{\min}) \cdot \frac{E_i - E_{\min}}{E_{\max} - E_{\min}}$。
    \item \textbf{虚拟中心路径规划}: 以网络几何中心作为虚拟汇聚点（无限能量），簇头通过多跳路径上报。
    \item \textbf{效率阈值与跳数限制}: 
    \begin{itemize}
        \item 路径效率$\eta_{\text{path}} = \prod_{h=1}^{H} \eta(d_h) > \eta_{\text{th}}$
        \item 最大跳数$H \leq H_{\max}$ (默认5)
    \end{itemize}
\end{enumerate}

\textit{实现}: \texttt{src/info\_collection/adcr\_link\_layer.py:21-840}
```

---

## 七、总结与下一步行动

### 7.1 当前论文的优势
- ✅ 经济学框架清晰（价格信号-账户-规则-治理）
- ✅ 问题分解合理（E-T-M结构）
- ✅ 评估维度全面（效率-公平-时效-寿命）

### 7.2 待改进的关键点
- ⚠️ 术语不一致（AOEI vs AOI）
- ⚠️ 缺少代码实现引用
- ⚠️ 经济学理论抽象（缺乏博弈论等具体工具）
- ⚠️ 实验参数不具体

### 7.3 下一步行动（建议优先级）

**第一阶段**（核心修正，2-3小时）:
1. 全文术语统一（AOEI→AOI或明确定义）
2. 在§4各节添加代码文件路径引用
3. 添加§5.2详细参数表

**第二阶段**（理论增强，3-4小时）:
4. §3.2补充博弈论建模
5. §4各节添加算法伪代码
6. §4.7新增复杂度分析

**第三阶段**（实验细化，2-3小时）:
7. §5.4添加基线实现对照表
8. 补充实验结果的统计显著性分析
9. 添加更多可视化（Pareto前沿、路径分布图等）

**总预计时间**: 7-10小时

---

## 八、参考资源

### 代码实现核心文件
```
src/
├── core/SensorNode.py                      # 节点能量模型
├── scheduling/
│   ├── passive_transfer.py                 # 被动传能管理器
│   └── schedulers.py                       # Lyapunov调度器
├── dynamic_k/lookahead.py                  # 前瞻K值优化
├── info_collection/
│   ├── physical_center.py                  # NodeInfoManager
│   ├── info_node.py                        # InfoNode数字孪生
│   └── adcr_link_layer.py                  # ADCR路由
└── config/
    └── adaptive_duration_aware_lyapunov_config.py  # 配置示例
```

### 文档参考
```
docs/
├── 虚拟节点层数字孪生技术分析.md
├── economics/
│   ├── AOI的经济学解释.md
│   ├── AOI与EAoI的区别与联系.md
│   └── AOI驱动的能量共享-经济一体化框架分析.md
└── 子课题_融合AOI与数字孪生的能量共享经济机制研究.md
```

---

**文档维护者**: AI Assistant  
**最后更新**: 2025-11-10

