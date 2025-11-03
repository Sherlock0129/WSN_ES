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

    def plan(self, network, t):
        """返回路由/传能计划列表：[{receiver, donor, path, distance, (可选)energy_sent}, ...]"""
        raise NotImplementedError

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
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops)
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
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops)
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
                    path = eetor_find_path_adaptive(nodes, ch, r, max_hops=self.max_hops)
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
                    path = eetor_find_path_adaptive(nodes, d, ch, max_hops=self.max_hops)
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
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops)
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
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops)
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
