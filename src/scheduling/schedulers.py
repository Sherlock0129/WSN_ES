# schedulers.py
# 可插拔的主流能量调度方法（兼容 Python 2.7/3.x：不使用变量注解/不使用dataclass）

from __future__ import print_function
import math
import random
import numpy as np

from routing.energy_transfer_routing import eetor_find_path_adaptive


# ------------------ 通用基类 ------------------
class BaseScheduler(object):
    def __init__(self, node_info_manager, K=2, max_hops=5):
        """
        初始化调度器
        
        :param node_info_manager: 节点信息管理器（NodeInfoManager实例）
        :param K: 最大捐能者数量
        :param max_hops: 最大跳数
        """
        self.nim = node_info_manager
        self.K = K
        self.max_hops = max_hops

    def get_name(self):
        """获取调度器名称"""
        return self.__class__.__name__
    
    def _filter_regular_nodes(self, nodes):
        """
        过滤出普通节点（排除物理中心节点）
        
        物理中心节点不参与能量传输（WET），包括：
        - 不能作为donor（捐能者）
        - 不能作为receiver（接收者）
        - 不能作为relay（中继节点，由EETOR处理）
        
        :param nodes: 节点列表（可以是InfoNode或SensorNode）
        :return: 过滤后的普通节点列表
        """
        return [n for n in nodes if not n.is_physical_center]
    
    def _filter_unlocked_nodes(self, nodes, current_time: int):
        """
        过滤出未锁定的节点（排除正在传输中的节点）
        
        正在传输中的节点（锁定状态）不能参与新的能量传输，包括：
        - 不能作为donor（捐能者）
        - 不能作为receiver（接收者）
        - 不能作为relay（中继节点）
        
        :param nodes: 节点列表（InfoNode或SensorNode）
        :param current_time: 当前时间（分钟）
        :return: 过滤后的未锁定节点列表
        """
        unlocked = []
        for node in nodes:
            if not self.nim.is_node_locked(node.node_id, current_time):
                unlocked.append(node)
        return unlocked

    def plan(self, network, t):
        """返回路由/传能计划列表：[{receiver, donor, path, distance, (可选)energy_sent}, ...]"""
        raise NotImplementedError

    def compute_network_feedback_score(self, pre_state, post_state, stats):
        """
        计算整体网络反馈分数，评估本次调度对网络的影响
        
        评分维度：
        1. 能量均衡性改善（标准差变化）：权重 0.4
        2. 网络存活率变化（死亡节点比例）：权重 0.3
        3. 能量传输效率（delivered/sent）：权重 0.2
        4. 整体能量水平变化：权重 0.1
        
        :param pre_state: 调度前的网络状态字典 {
            'energies': 节点能量数组,
            'alive_nodes': 存活节点数,
            'total_energy': 总能量,
            'std': 能量标准差
        }
        :param post_state: 调度后的网络状态字典（同上）
        :param stats: 调度统计信息（来自compute_step_stats）
        :return: (总分, 详细评分字典)
            - 总分：正值表示正向影响，负值表示负向影响，0表示无影响
            - 详细评分：包含各维度的具体分数和说明
        """
        feedback_score = 0.0
        details = {}
        
        # 1. 能量均衡性改善（标准差变化）- 权重 0.4
        # 标准差减少是好的（正分），标准差增加是坏的（负分）
        pre_std = pre_state.get('std', 0.0)
        post_std = post_state.get('std', 0.0)
        std_change = pre_std - post_std  # 正值表示标准差减少（改善）
        
        # 归一化：相对于调度前标准差的变化百分比
        if pre_std > 0:
            std_change_ratio = std_change / pre_std
        else:
            std_change_ratio = 0.0
        
        balance_score = std_change_ratio * 0.4 * 100  # 乘以100使分数更直观
        feedback_score += balance_score
        details['balance_score'] = balance_score
        details['std_change'] = std_change
        details['std_change_ratio'] = std_change_ratio
        
        # 2. 网络存活率变化 - 权重 0.3
        # 存活节点增加是好的，减少是坏的
        pre_alive = pre_state.get('alive_nodes', 0)
        post_alive = post_state.get('alive_nodes', 0)
        alive_change = post_alive - pre_alive
        
        total_nodes = len(pre_state.get('energies', []))
        if total_nodes > 0:
            alive_change_ratio = float(alive_change) / total_nodes
        else:
            alive_change_ratio = 0.0
        
        survival_score = alive_change_ratio * 0.3 * 100
        feedback_score += survival_score
        details['survival_score'] = survival_score
        details['alive_change'] = alive_change
        details['alive_change_ratio'] = alive_change_ratio
        
        # 3. 能量传输效率 - 权重 0.2
        # 传输效率越高越好
        delivered = stats.get('delivered_total', 0.0)
        sent = delivered + stats.get('total_loss', 0.0)
        
        if sent > 0:
            efficiency = delivered / sent
        else:
            efficiency = 0.0
        
        # 效率分数：效率范围[0, 1]，映射到[-20, +20]
        # 效率高于50%得正分，低于50%得负分
        efficiency_score = (efficiency - 0.5) * 0.2 * 100
        feedback_score += efficiency_score
        details['efficiency_score'] = efficiency_score
        details['efficiency'] = efficiency
        
        # 4. 整体能量水平变化 - 权重 0.1
        # 总能量增加是好的（考虑采集），减少是坏的
        pre_total = pre_state.get('total_energy', 0.0)
        post_total = post_state.get('total_energy', 0.0)
        energy_change = post_total - pre_total
        
        if pre_total > 0:
            energy_change_ratio = energy_change / pre_total
        else:
            energy_change_ratio = 0.0
        
        energy_score = energy_change_ratio * 0.1 * 100
        feedback_score += energy_score
        details['energy_score'] = energy_score
        details['energy_change'] = energy_change
        details['energy_change_ratio'] = energy_change_ratio
        
        # 综合评价
        if feedback_score > 5:
            impact = "正相关（显著改善）"
        elif feedback_score > 1:
            impact = "正相关（轻微改善）"
        elif feedback_score > -1:
            impact = "中性（影响很小）"
        elif feedback_score > -5:
            impact = "负相关（轻微恶化）"
        else:
            impact = "负相关（显著恶化）"
        
        details['total_score'] = feedback_score
        details['impact'] = impact
        
        return feedback_score, details

    def post_step(self, network, t, feedback):
        """在一步传能后（拿到 stats）做自更新，可选"""
        pass


# ------------------ Baseline：与你 run_routing 类似的启发式 ------------------
class BaselineHeuristic(BaseScheduler):
    def plan(self, network, t):
        # 从信息表获取InfoNode（常驻实例，自动同步更新）
        info_nodes = self.nim.get_info_nodes()
        
        # 构建ID到真实节点的映射（用于最后转换）
        id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        plans = []
        used = set()

        avgE = float(np.mean([n.current_energy for n in nodes]))
        lows = sorted([n for n in nodes if n.current_energy < avgE],
                      key=lambda x: (avgE - x.current_energy), reverse=True)

        for r in lows:
            cand = [n for n in nodes if (n is not r) and (n.current_energy > avgE) and (n not in used)]
            if not cand:
                continue
            cand.sort(key=lambda d: (-d.current_energy, r.distance_to(d)))
            quota = self.K
            for d in cand:
                if quota <= 0:
                    break
                dist = r.distance_to(d)
                # 使用自适应路径查找（自动处理单跳和多跳）
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops, 
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                
                # 转换回真实节点对象（保持返回格式兼容）
                receiver = id2node[r.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({"receiver": receiver, "donor": donor, "path": real_path, "distance": dist})
                used.add(d)
                quota -= 1
        return plans


# ------------------ Lyapunov：漂移+惩罚（无需预测） ------------------
class LyapunovScheduler(BaseScheduler):
    def __init__(self, node_info_manager, V=0.5, K=2, max_hops=5):
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        self.V = float(V)

    def _path_eta(self, path):
        eta = 1.0
        for i in range(len(path) - 1):
            eta *= path[i].energy_transfer_efficiency(path[i + 1])
        if eta < 1e-6:
            eta = 1e-6
        if eta > 1.0:
            eta = 1.0
        return eta

    def plan(self, network, t):
        # 从信息表创建InfoNode
        info_nodes = self.nim.get_info_nodes()
        id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        E = np.array([n.current_energy for n in nodes], dtype=float)
        E_bar = float(E.mean())
        Q = dict((n, max(0.0, E_bar - n.current_energy)) for n in nodes)

        receivers = sorted([n for n in nodes if Q[n] > 0], key=lambda x: Q[x], reverse=True)
        donors = [n for n in nodes if n.current_energy > E_bar]
        used = set()
        plans = []
        all_candidates = []  # 收集所有候选信息

        for r in receivers:
            cand = []
            for d in donors:
                if d in used or d is r:
                    continue
                dist = r.distance_to(d)
                # 使用自适应路径查找（自动处理单跳和多跳）
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops, 
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                eta = self._path_eta(path)
                # 效率低于10%的传输直接放弃
                if eta < 0.1:
                    continue
                    
                E_char = getattr(d, "E_char", 300.0)
                delivered = E_char * eta
                loss = max(0.0, E_char - delivered)
                Q_normalized = Q[r] / E_bar
                score = eta * (Q_normalized + self.V) - self.V
                # score = Q[r] * delivered - self.V * loss
                cand.append((score, d, r, path, dist, delivered, loss))
            if not cand:
                continue
            cand.sort(key=lambda x: x[0], reverse=True)
            all_candidates.extend(cand)  # 收集所有候选信息
            quota = self.K
            for sc, d, rr, path, dist, delivered, loss in cand:
                if quota <= 0:
                    break
                
                # 转换回真实节点对象
                receiver = id2node[rr.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({
                    "receiver": receiver, 
                    "donor": donor, 
                    "path": real_path, 
                    "distance": dist,
                    "delivered": delivered,
                    "loss": loss
                })
                used.add(d)
                quota -= 1
        return plans, all_candidates  # 返回plans和所有候选信息


# ------------------ AdaptiveLyapunov：基于反馈自适应调整参数的Lyapunov ------------------
class AdaptiveLyapunovScheduler(LyapunovScheduler):
    """
    自适应参数调整的Lyapunov调度器
    
    核心创新：
    1. 继承标准Lyapunov调度器的决策逻辑
    2. 根据反馈分数动态调整V参数（能量-损耗权衡）
    3. 多维度自适应：响应均衡性、效率、存活率等多个指标
    4. 带记忆的平滑调整：避免参数震荡
    
    自适应策略：
    - 效率低 → 增大V（更重视减少损耗，选择更近的路径）
    - 均衡性差 → 减小V（更重视能量均衡，增加传输量）
    - 存活率下降 → 减小V（优先救活节点）
    """
    
    def __init__(self, node_info_manager, V=0.5, K=2, max_hops=5, 
                 window_size=10, V_min=0.1, V_max=2.0, 
                 adjust_rate=0.1, sensitivity=2.0):
        """
        初始化自适应Lyapunov调度器
        
        :param node_info_manager: 节点信息管理器
        :param V: 初始V参数（能量-损耗权衡）
        :param K: 最大捐能者数量
        :param max_hops: 最大跳数
        :param window_size: 反馈窗口大小（记忆最近N次）
        :param V_min: V的最小值（不能太小，避免过度传输）
        :param V_max: V的最大值（不能太大，避免过度保守）
        :param adjust_rate: 参数调整速率（0-1，越大调整越快）
        :param sensitivity: 反馈敏感度（触发调整的阈值）
        """
        LyapunovScheduler.__init__(self, node_info_manager, V, K, max_hops)
        
        # 自适应参数
        self.V_initial = float(V)
        self.V_min = float(V_min)
        self.V_max = float(V_max)
        self.adjust_rate = float(adjust_rate)
        self.sensitivity = float(sensitivity)
        
        # 反馈历史记录
        self.window_size = int(window_size)
        self.recent_feedbacks = []  # 最近N次的总分
        self.recent_details = []    # 最近N次的详细信息
        
        # 统计信息
        self.total_adjustments = 0  # 总调整次数
        self.adjustment_history = []  # 调整历史：[(time, old_V, new_V, reason)]
        
        # 性能指标
        self.best_feedback_score = float('-inf')
        self.worst_feedback_score = float('inf')
        
    def post_step(self, network, t, feedback):
        """
        接收反馈并自适应调整参数
        
        :param network: 网络对象
        :param t: 当前时间步
        :param feedback: 反馈信息（来自compute_network_feedback_score）
        """
        if feedback is None or not isinstance(feedback, dict):
            return
        
        # 提取反馈分数和详细信息
        feedback_score = feedback.get('total_score', 0.0)
        details = feedback.get('details', {}) if 'details' in feedback else feedback
        
        # 记录到历史
        self.recent_feedbacks.append(feedback_score)
        self.recent_details.append(details)
        
        # 维持窗口大小
        if len(self.recent_feedbacks) > self.window_size:
            self.recent_feedbacks.pop(0)
            self.recent_details.pop(0)
        
        # 更新最佳/最差记录
        if feedback_score > self.best_feedback_score:
            self.best_feedback_score = feedback_score
        if feedback_score < self.worst_feedback_score:
            self.worst_feedback_score = feedback_score
        
        # 只有积累足够历史后才开始调整
        if len(self.recent_feedbacks) < min(5, self.window_size):
            return
        
        # 计算平均反馈和趋势
        avg_feedback = float(np.mean(self.recent_feedbacks))
        recent_trend = float(np.mean(self.recent_feedbacks[-3:])) if len(self.recent_feedbacks) >= 3 else avg_feedback
        
        # 提取关键指标
        balance_score = details.get('balance_score', 0.0)
        efficiency_score = details.get('efficiency_score', 0.0)
        survival_score = details.get('survival_score', 0.0)
        efficiency = details.get('efficiency', 0.5)
        
        # 决策：是否需要调整V
        old_V = self.V
        adjustment_reason = None
        
        # 策略1：持续负反馈 → 需要调整
        if avg_feedback < -self.sensitivity:
            # 诊断问题所在
            if efficiency_score < -2.0:
                # 问题：传输效率太低，损耗过大
                # 解决：增大V，更重视损耗，倾向选择更近的路径
                self.V = min(self.V_max, self.V * (1.0 + self.adjust_rate))
                adjustment_reason = f"效率低({efficiency:.2f}) → 增大V(减少损耗)"
                
            elif balance_score < -2.0:
                # 问题：能量分布不均衡
                # 解决：减小V，更重视均衡，增加给低能量节点的传输
                self.V = max(self.V_min, self.V * (1.0 - self.adjust_rate))
                adjustment_reason = f"均衡差(std变化{details.get('std_change', 0):.2f}) → 减小V(增强均衡)"
                
            elif survival_score < -1.0:
                # 问题：节点死亡
                # 解决：减小V，优先救活节点
                self.V = max(self.V_min, self.V * (1.0 - self.adjust_rate * 1.5))
                adjustment_reason = f"节点死亡({details.get('alive_change', 0)}) → 减小V(优先救活)"
        
        # 策略2：趋势恶化 → 预防性调整
        elif recent_trend < avg_feedback - 1.0:
            # 最近趋势明显下降
            if efficiency < 0.3:
                # 效率过低且趋势恶化
                self.V = min(self.V_max, self.V * (1.0 + self.adjust_rate * 0.5))
                adjustment_reason = "趋势恶化+效率低 → 轻微增大V"
        
        # 策略3：持续正反馈但可以优化 → 微调
        elif avg_feedback > self.sensitivity:
            # 表现良好，但检查是否可以进一步优化
            if efficiency > 0.7 and balance_score > 0:
                # 效率很高且均衡性好，可以稍微减小V增加吞吐量
                if self.V > self.V_initial * 0.8:
                    self.V = max(self.V_min, self.V * (1.0 - self.adjust_rate * 0.3))
                    adjustment_reason = "表现优秀 → 轻微减小V(增加吞吐)"
        
        # 策略4：重置机制 → 如果长期表现不佳，重置到初始值
        if len(self.recent_feedbacks) >= self.window_size:
            all_negative = all(score < 0 for score in self.recent_feedbacks[-5:])
            if all_negative and abs(self.V - self.V_initial) > 0.5:
                self.V = self.V_initial
                adjustment_reason = "长期不佳 → 重置V到初始值"
        
        # 记录调整
        if adjustment_reason is not None:
            self.total_adjustments += 1
            self.adjustment_history.append((t, old_V, self.V, adjustment_reason))
            print(f"[自适应@t={t}] V: {old_V:.3f} → {self.V:.3f} | {adjustment_reason}")
            print(f"           反馈: 总分={feedback_score:.2f}, 均衡={balance_score:.2f}, "
                  f"效率={efficiency_score:.2f}(η={efficiency:.2f}), 存活={survival_score:.2f}")
    
    def get_adaptation_stats(self):
        """
        获取自适应统计信息
        
        :return: 统计信息字典
        """
        return {
            'current_V': self.V,
            'initial_V': self.V_initial,
            'V_min': self.V_min,
            'V_max': self.V_max,
            'total_adjustments': self.total_adjustments,
            'adjustment_history': self.adjustment_history,
            'recent_feedbacks': list(self.recent_feedbacks),
            'avg_feedback': float(np.mean(self.recent_feedbacks)) if self.recent_feedbacks else 0.0,
            'best_feedback': self.best_feedback_score,
            'worst_feedback': self.worst_feedback_score
        }
    
    def print_adaptation_summary(self):
        """打印自适应调整摘要"""
        stats = self.get_adaptation_stats()
        print("\n" + "="*60)
        print("自适应Lyapunov调度器 - 适应性总结")
        print("="*60)
        print(f"初始V: {stats['initial_V']:.3f}")
        print(f"当前V: {stats['current_V']:.3f}")
        print(f"V范围: [{stats['V_min']:.3f}, {stats['V_max']:.3f}]")
        print(f"总调整次数: {stats['total_adjustments']}")
        print(f"平均反馈分数: {stats['avg_feedback']:.2f}")
        print(f"最佳反馈分数: {stats['best_feedback']:.2f}")
        print(f"最差反馈分数: {stats['worst_feedback']:.2f}")
        
        if stats['adjustment_history']:
            print(f"\n最近5次调整:")
            for t, old_v, new_v, reason in stats['adjustment_history'][-5:]:
                print(f"  t={t}: {old_v:.3f}→{new_v:.3f} | {reason}")
        print("="*60 + "\n")


# ------------------ Cluster：LEACH-lite 轮换簇首 + 簇内均衡 ------------------
class ClusterScheduler(BaseScheduler):
    def __init__(self, node_info_manager, round_period=360, K=2, max_hops=5, p_ch=0.05):
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        self.round_period = int(round_period)
        self.p_ch = float(p_ch)
        self.cluster_of = {}  # node_id -> ch_id
        self.ch_set = set()

    def _recluster(self, nodes, t):
        self.ch_set.clear()
        meanE = float(np.mean([x.current_energy for x in nodes]))
        for n in nodes:
            prob = self.p_ch * (n.current_energy / (1.0 + meanE))
            if random.random() < prob:
                self.ch_set.add(n.node_id)
        if not self.ch_set:
            top = max(nodes, key=lambda x: x.current_energy)
            self.ch_set.add(top.node_id)

        self.cluster_of.clear()
        id2node = dict((n.node_id, n) for n in nodes)
        ch_nodes = [id2node[i] for i in self.ch_set]
        for n in nodes:
            best = min(ch_nodes, key=lambda ch: n.distance_to(ch))
            self.cluster_of[n.node_id] = best.node_id

    def plan(self, network, t):
        # 从信息表创建InfoNode
        info_nodes = self.nim.get_info_nodes()
        real_id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        if (t % self.round_period == 0) or (not self.cluster_of):
            self._recluster(nodes, t)

        id2node = dict((n.node_id, n) for n in nodes)
        clusters = {}
        for n in nodes:
            ch_id = self.cluster_of[n.node_id]
            clusters.setdefault(ch_id, []).append(n)

        plans = []
        for ch_id, members in clusters.items():
            ch = id2node[ch_id]
            avgE = float(np.mean([m.current_energy for m in members]))
            lows = sorted([m for m in members if m.current_energy < avgE],
                          key=lambda x: avgE - x.current_energy, reverse=True)
            highs = [m for m in members if m.current_energy > avgE]

            if ch in highs:
                # CH 给簇内低能成员
                for r in lows:
                    if r is ch:
                        continue
                    dist = r.distance_to(ch)
                    # 使用自适应路径查找（自动处理单跳和多跳）
                    path = eetor_find_path_adaptive(nodes, ch, r, max_hops=self.max_hops,
                                                     node_info_manager=self.nim)
                    if path is None:
                        continue
                    
                    # 转换回真实节点对象
                    receiver = real_id2node[r.node_id]
                    donor = real_id2node[ch.node_id]
                    real_path = [real_id2node[n.node_id] for n in path]
                    
                    plans.append({"receiver": receiver, "donor": donor, "path": real_path, "distance": dist})
            else:
                # 先给 CH 充能
                highs_sorted = sorted(highs, key=lambda x: (-x.current_energy, ch.distance_to(x)))
                quota = self.K
                for d in highs_sorted:
                    if quota <= 0:
                        break
                    dist = ch.distance_to(d)
                    # 使用自适应路径查找（自动处理单跳和多跳）
                    path = eetor_find_path_adaptive(nodes, d, ch, max_hops=self.max_hops,
                                                     node_info_manager=self.nim)
                    if path is None:
                        continue
                    
                    # 转换回真实节点对象
                    receiver = real_id2node[ch.node_id]
                    donor = real_id2node[d.node_id]
                    real_path = [real_id2node[n.node_id] for n in path]
                    
                    plans.append({"receiver": receiver, "donor": donor, "path": real_path, "distance": dist})
                    quota -= 1
        return plans


# ------------------ Prediction：EWMA 的"未来富余量"排序 ------------------
class PredictionScheduler(BaseScheduler):
    def __init__(self, node_info_manager, alpha=0.6, horizon_min=60, K=2, max_hops=5):
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        self.alpha = float(alpha)
        self.horizon = int(horizon_min)
        self.prev_harv = {}  # node_id -> 上一步的估计

    def _predict_harvest(self, node, t):
        obs = node.energy_harvest(t)  # InfoNode有简化的energy_harvest方法
        last = self.prev_harv.get(node.node_id, obs)
        est = self.alpha * obs + (1.0 - self.alpha) * last
        self.prev_harv[node.node_id] = est
        # 近似未来一小时汇总（你的 energy_harvest 单位为"此次步瞬时能量"，这里做一个简化）
        return est * (self.horizon / 60.0)

    def plan(self, network, t):
        # 从信息表创建InfoNode
        info_nodes = self.nim.get_info_nodes()
        id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        avgE = float(np.mean([n.current_energy for n in nodes]))
        pred_surplus = dict((n, (n.current_energy + self._predict_harvest(n, t))) for n in nodes)

        receivers = sorted([n for n in nodes if n.current_energy < avgE],
                           key=lambda n: (avgE - n.current_energy), reverse=True)
        plans = []
        used = set()

        for r in receivers:
            cand = [n for n in nodes if (n is not r) and (pred_surplus[n] > avgE) and (n not in used)]
            if not cand:
                continue
            cand.sort(key=lambda d: (-pred_surplus[d], r.distance_to(d)))
            quota = self.K
            for d in cand:
                if quota <= 0:
                    break
                dist = r.distance_to(d)
                # 使用自适应路径查找（自动处理单跳和多跳）
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops, 
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                
                # 转换回真实节点对象
                receiver = id2node[r.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({"receiver": receiver, "donor": donor, "path": real_path, "distance": dist})
                used.add(d)
                quota -= 1
        return plans


# ------------------ PowerControl：EAST-like 的功率/下发量收缩 ------------------
class AdaptiveDurationLyapunovScheduler(BaseScheduler):
    """
    自适应传输时长的Lyapunov调度器（简化版）
    
    核心思想：
    - 基于Lyapunov漂移加惩罚框架
    - 对每个donor-receiver对，尝试不同传输时长（1-5分钟）
    - 选择使Lyapunov函数减少最多的时长
    - 纯粹的能量优化，不考虑AoI和信息量
    
    得分函数：delivered × Q[r] - V × loss
    """
    def __init__(self, node_info_manager, V=0.5, K=2, max_hops=5,
                 min_duration=1, max_duration=5):
        """
        :param node_info_manager: 节点信息管理器
        :param V: Lyapunov控制参数（能量-损耗权衡）
        :param K: 每个receiver最多接受的donor数量
        :param max_hops: 最大跳数
        :param min_duration: 最小传输时长（分钟）
        :param max_duration: 最大传输时长（分钟）
        """
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        self.V = float(V)
        self.min_duration = min_duration
        self.max_duration = max_duration
    
    def _path_eta(self, path):
        """计算路径总效率"""
        eta = 1.0
        for i in range(len(path) - 1):
            eta *= path[i].energy_transfer_efficiency(path[i + 1])
        return max(1e-6, min(1.0, eta))
    
    def _compute_duration_score(self, donor, receiver, path, eta, E_bar, Q_r, duration):
        """
        计算特定传输时长的Lyapunov得分（考虑信息价值）
        
        得分 = delivered × Q[r] - V × loss + 信息价值奖励
        
        :param duration: 传输时长（分钟）
        :return: (score, energy_delivered, energy_loss)
        """
        E_char = getattr(donor, "E_char", 300.0)
        
        # 能量计算
        energy_sent_total = duration * E_char
        energy_delivered = energy_sent_total * eta
        energy_loss = energy_sent_total - energy_delivered
        
        # Lyapunov得分计算
        Q_normalized = Q_r / E_bar if E_bar > 0 else 0
        
        # 基础得分 = 能量收益 - 能量损耗惩罚
        base_score = energy_delivered * Q_normalized - self.V * energy_loss
        
        # 信息价值奖励（如果可用）
        info_bonus = 0.0
        if hasattr(self, 'path_collector') and self.path_collector and hasattr(self, 'current_time'):
            receiver_info = self.nim.get_node_info(receiver.node_id)
            if receiver_info:
                is_reported = receiver_info.get('info_is_reported', True)
                if not is_reported:
                    # 使用信息价值判断是否有未上报信息
                    info_value = self.path_collector.calculate_info_value(receiver_info, self.current_time)
                    if info_value > 0:
                        # 信息价值奖励：信息价值越大，奖励越大（归一化）
                        max_info_volume = 1000000  # 默认最大值
                        normalized_value = min(info_value / max_info_volume, 1.0)
                        info_bonus = normalized_value * 0.1 * Q_normalized  # 信息价值奖励系数
        
        score = base_score + info_bonus
        
        return score, energy_delivered, energy_loss
    
    def plan(self, network, t):
        """
        规划能量传输，自适应选择传输时长
        
        对每个donor-receiver对，尝试不同的传输时长（1-5分钟），
        选择Lyapunov得分最高的时长
        """
        # 保存当前时间，用于信息价值计算
        self.current_time = t
        
        # 从信息表创建InfoNode
        info_nodes = self.nim.get_info_nodes()
        id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        E = np.array([n.current_energy for n in nodes], dtype=float)
        E_bar = float(E.mean())
        Q = dict((n, max(0.0, E_bar - n.current_energy)) for n in nodes)
        
        receivers = sorted([n for n in nodes if Q[n] > 0], key=lambda x: Q[x], reverse=True)
        donors = [n for n in nodes if n.current_energy > E_bar]
        used = set()
        plans = []
        all_candidates = []
        
        for r in receivers:
            cand = []
            for d in donors:
                if d in used or d is r:
                    continue
                
                dist = r.distance_to(d)
                
                # 使用自适应路径查找
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops,
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                
                eta = self._path_eta(path)
                
                # 效率低于10%的传输直接放弃
                if eta < 0.1:
                    continue
                
                # 尝试不同的传输时长，选择最优的
                best_duration = self.min_duration
                best_score = float('-inf')
                best_metrics = None
                
                for duration in range(self.min_duration, self.max_duration + 1):
                    score, delivered, loss = \
                        self._compute_duration_score(d, r, path, eta, E_bar, Q[r], duration)
                    
                    if score > best_score:
                        best_score = score
                        best_duration = duration
                        best_metrics = (delivered, loss)
                
                if best_metrics:
                    delivered, loss = best_metrics
                    cand.append((best_score, d, r, path, dist, delivered, loss, best_duration))
            
            if not cand:
                continue
            
            cand.sort(key=lambda x: x[0], reverse=True)
            all_candidates.extend(cand)
            quota = self.K
            
            for sc, d, rr, path, dist, delivered, loss, duration in cand:
                if quota <= 0:
                    break
                
                # 转换回真实节点对象
                receiver = id2node[rr.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({
                    "receiver": receiver,
                    "donor": donor,
                    "path": real_path,
                    "distance": dist,
                    "delivered": delivered,
                    "loss": loss,
                    "duration": duration,  # 传输时长（分钟）
                    "score": sc  # Lyapunov得分
                })
                used.add(d)
                quota -= 1
        
        return plans, all_candidates


class DurationAwareLyapunovScheduler(BaseScheduler):
    """
    传输时长感知的Lyapunov调度器
    
    核心创新：将传输时长作为新的优化维度，综合考虑：
    1. 能量传输量：duration × E_char
    2. AoI变化：传输期间AoI增长
    3. 信息量累积：传输期间信息继续采集
    4. 多目标权衡：能量、AoI、信息量的综合优化
    """
    def __init__(self, node_info_manager, V=0.5, K=2, max_hops=5,
                 min_duration=1, max_duration=5,
                 w_aoi=0.1, w_info=0.05, info_collection_rate=10000.0):
        """
        :param node_info_manager: 节点信息管理器
        :param V: Lyapunov控制参数（能量-损耗权衡）
        :param K: 每个receiver最多接受的donor数量
        :param max_hops: 最大跳数
        :param min_duration: 最小传输时长（分钟）
        :param max_duration: 最大传输时长（分钟）
        :param w_aoi: AoI惩罚权重
        :param w_info: 信息量奖励权重
        :param info_collection_rate: 信息采集速率（bits/分钟）
        """
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        self.V = float(V)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.w_aoi = w_aoi
        self.w_info = w_info
        self.info_collection_rate = info_collection_rate
    
    def _path_eta(self, path):
        """计算路径总效率"""
        eta = 1.0
        for i in range(len(path) - 1):
            eta *= path[i].energy_transfer_efficiency(path[i + 1])
        return max(1e-6, min(1.0, eta))
    
    def _compute_duration_score(self, donor, receiver, path, eta, E_bar, Q_r, duration):
        """
        计算特定传输时长的综合得分
        
        综合考虑：
        1. 能量收益：receiver获得的能量 × Q权重
        2. 能量损耗：传输损耗惩罚
        3. AoI惩罚：传输时间越长，AoI增长越多
        4. 信息量奖励：传输时间越长，累积信息量越多（可搭便车）
        
        :return: (score, energy_delivered, energy_loss, aoi_cost, info_gain)
        """
        E_char = getattr(donor, "E_char", 300.0)
        
        # 1. 能量计算
        energy_sent_total = duration * E_char
        energy_delivered = energy_sent_total * eta
        energy_loss = energy_sent_total - energy_delivered
        
        # 2. AoI代价：传输期间所有未上报节点的AoI都会增长
        # 简化模型：只考虑receiver的AoI增长
        aoi_cost = duration  # 传输持续时间即为AoI增长
        
        # 3. 信息量收益：传输期间，receiver可能累积更多信息
        # 如果receiver有传感任务，持续采集信息，传输时长越长，累积越多
        info_gain = duration * self.info_collection_rate  # bits
        
        # 4. 综合得分计算（基于Lyapunov漂移加惩罚框架）
        Q_normalized = Q_r / E_bar if E_bar > 0 else 0
        
        # 能量收益项（考虑队列backlog）
        energy_benefit_score = energy_delivered * Q_normalized
        
        # 能量损耗惩罚项
        energy_loss_penalty = self.V * energy_loss
        
        # AoI惩罚项（传输时间越长，其他节点AoI增长越多）
        aoi_penalty = self.w_aoi * aoi_cost * Q_normalized
        
        # 信息量奖励项（传输时间越长，可能携带更多信息，减少后续上报）
        # 检查receiver是否有未上报信息（使用信息价值而非信息量）
        receiver_info = self.nim.get_node_info(receiver.node_id)
        has_info = False
        info_value = 0.0
        if receiver_info:
            is_reported = receiver_info.get('info_is_reported', True)
            if not is_reported:
                # 优先使用信息价值（如果path_collector可用）
                if hasattr(self, 'path_collector') and self.path_collector:
                    info_value = self.path_collector.calculate_info_value(receiver_info, self.current_time)
                    has_info = (info_value > 0)
                else:
                    # 后备方案：使用信息量（如果没有path_collector）
                    info_volume = receiver_info.get('info_volume', 0)
                    has_info = (info_volume > 0)
        
        # 信息奖励策略：
        # - 如果有未上报信息：全额奖励（鼓励搭便车）
        # - 否则：仍给予部分奖励（鼓励长传输，为未来信息传输铺路）
        if has_info:
            info_bonus = self.w_info * info_gain  # 全额奖励
        else:
            info_bonus = self.w_info * info_gain * 0.5  # 半额奖励（即使没有当前信息）
        
        # 总得分 = 能量收益 - 能量损耗 - AoI惩罚 + 信息奖励
        total_score = energy_benefit_score - energy_loss_penalty - aoi_penalty + info_bonus
        
        return total_score, energy_delivered, energy_loss, aoi_cost, info_gain
    
    def plan(self, network, t):
        """
        规划能量传输，优化传输时长
        
        对每个donor-receiver对，尝试不同的传输时长（1-5分钟），
        选择综合得分最高的时长
        """
        # 保存当前时间，用于信息价值计算
        self.current_time = t
        
        # 从信息表创建InfoNode
        info_nodes = self.nim.get_info_nodes()
        id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        # DurationAwareLyapunovScheduler需要过滤锁定节点（传输时长优化特性）
        nodes = self._filter_unlocked_nodes(nodes, t)
        E = np.array([n.current_energy for n in nodes], dtype=float)
        E_bar = float(E.mean())
        Q = dict((n, max(0.0, E_bar - n.current_energy)) for n in nodes)
        
        receivers = sorted([n for n in nodes if Q[n] > 0], key=lambda x: Q[x], reverse=True)
        donors = [n for n in nodes if n.current_energy > E_bar]
        used = set()
        plans = []
        all_candidates = []
        
        for r in receivers:
            cand = []
            for d in donors:
                if d in used or d is r:
                    continue
                
                dist = r.distance_to(d)
                
                # 使用自适应路径查找
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops,
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                
                eta = self._path_eta(path)
                
                # 效率低于10%的传输直接放弃
                if eta < 0.1:
                    continue
                
                # 尝试不同的传输时长，选择最优的
                best_duration = self.min_duration
                best_score = float('-inf')
                best_metrics = None
                
                for duration in range(self.min_duration, self.max_duration + 1):
                    score, delivered, loss, aoi_cost, info_gain = \
                        self._compute_duration_score(d, r, path, eta, E_bar, Q[r], duration)
                    
                    if score > best_score:
                        best_score = score
                        best_duration = duration
                        best_metrics = (delivered, loss, aoi_cost, info_gain)
                
                if best_metrics:
                    delivered, loss, aoi_cost, info_gain = best_metrics
                    cand.append((best_score, d, r, path, dist, delivered, loss, 
                                best_duration, aoi_cost, info_gain))
            
            if not cand:
                continue
            
            cand.sort(key=lambda x: x[0], reverse=True)
            all_candidates.extend(cand)
            quota = self.K
            
            for sc, d, rr, path, dist, delivered, loss, duration, aoi_cost, info_gain in cand:
                if quota <= 0:
                    break
                
                # 转换回真实节点对象
                receiver = id2node[rr.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({
                    "receiver": receiver,
                    "donor": donor,
                    "path": real_path,
                    "distance": dist,
                    "delivered": delivered,
                    "loss": loss,
                    "duration": duration,  # 新增：传输时长（分钟）
                    "aoi_cost": aoi_cost,  # 新增：AoI代价
                    "info_gain": info_gain,  # 新增：信息量收益
                    "score": sc  # 新增：综合得分
                })
                used.add(d)
                quota -= 1
        
        return plans, all_candidates


class PowerControlScheduler(BaseScheduler):
    def __init__(self, node_info_manager, target_eta=0.25, K=2, max_hops=5):
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        self.target_eta = float(target_eta)

    def _path_eta(self, path):
        eta = 1.0
        for i in range(len(path) - 1):
            eta *= path[i].energy_transfer_efficiency(path[i + 1])
        if eta < 1e-6:
            eta = 1e-6
        if eta > 1.0:
            eta = 1.0
        return eta

    def plan(self, network, t):
        # 从信息表创建InfoNode
        info_nodes = self.nim.get_info_nodes()
        id2node = {n.node_id: n for n in network.nodes}
        
        # 排除物理中心节点
        nodes = self._filter_regular_nodes(info_nodes)
        avgE = float(np.mean([n.current_energy for n in nodes]))
        receivers = sorted([n for n in nodes if n.current_energy < avgE],
                           key=lambda n: (avgE - n.current_energy), reverse=True)
        plans = []
        used = set()

        for r in receivers:
            cand = [n for n in nodes if (n is not r) and (n.current_energy > avgE) and (n not in used)]
            cand.sort(key=lambda d: (-d.current_energy, r.distance_to(d)))
            quota = self.K
            for d in cand:
                if quota <= 0:
                    break
                dist = r.distance_to(d)
                # 使用自适应路径查找（自动处理单跳和多跳）
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops, 
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                eta = self._path_eta(path)
                E_char = getattr(d, "E_char", 300.0)
                scale = min(1.0, self.target_eta / eta)  # 低效率→更小发送量
                energy_sent = max(0.0, E_char * scale)
                
                # 转换回真实节点对象
                receiver = id2node[r.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({
                    "receiver": receiver, "donor": donor, "path": real_path, "distance": dist,
                    "energy_sent": energy_sent
                })
                used.add(d)
                quota -= 1
        return plans

    @staticmethod
    def execute_plans(network, plans):
        """按计划中的 energy_sent 执行（避免改网络默认执行器）"""
        for plan in plans:
            d = plan["donor"]; r = plan["receiver"]; path = plan["path"]; dist = plan["distance"]
            energy_sent = float(plan.get("energy_sent", getattr(d, "E_char", 300.0)))

            # 根据路径长度判断单跳或多跳（自适应路径查找已确定最优路径）
            if len(path) == 2:
                # 单跳直接传输
                eta = d.energy_transfer_efficiency(r)
                energy_received = energy_sent * eta
                d.current_energy -= d.energy_consumption(r, transfer_WET=True)
                r.current_energy += energy_received
                d.transferred_history.append(energy_sent)
                r.received_history.append(energy_received)
            else:
                # 多跳传输
                energy_left = energy_sent
                d.transferred_history.append(energy_sent)
                for i in range(len(path) - 1):
                    s = path[i]; rr = path[i + 1]
                    eta = s.energy_transfer_efficiency(rr)
                    delivered = energy_left * eta
                    transfer_WET = (i == 0)
                    s.current_energy -= s.energy_consumption(rr, transfer_WET=transfer_WET)
                    rr.current_energy += delivered
                    rr.received_history.append(delivered)
                    energy_left = delivered
