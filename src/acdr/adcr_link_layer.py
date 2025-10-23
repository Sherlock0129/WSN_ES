# src/adcr_link_layer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import random
import os
from collections import defaultdict
from utils.output_manager import OutputManager

try:
    from routing.opportunistic_routing import opportunistic_routing
except Exception:
    opportunistic_routing = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


class ADCRLinkLayerVirtual(object):
    """
    - 估计最优簇数 K*（近邻统计启发式）
    - 能量感知 + 空间抑制 选择簇头
    - 成簇与一次性细化
    - 以网络几何中心为“虚拟link中心（无限能量）”，为每个簇头规划一条上报路径
    - 按通信能耗模型为真实节点扣除能量；虚拟中心不扣
    - 提供 Plotly 可视化：节点/簇/路径/虚拟中心
    """
    def __init__(self, network,
                 round_period: int = 1440,
                 r_neighbor: float = 1.732,
                 r_min_ch: float = 1.0,
                 c_k: float = 1.2,
                 plan_paths: bool = True,
                 consume_energy: bool = True,
                 max_hops: int = 5,
                 output_dir: str = "adcr",
                 # 簇头选择参数
                 max_probability: float = 0.9,
                 min_probability: float = 0.05,
                 # 聚类成本函数参数
                 distance_weight: float = 1.0,
                 energy_weight: float = 0.2,
                 # 通信能耗参数
                 tx_rx_ratio: float = 0.5,
                 sensor_energy: float = 0.1,
                 # 信息聚合参数
                 base_data_size: int = 1000000,
                 aggregation_ratio: float = 1.0,
                 enable_dynamic_data_size: bool = True,
                 # 直接传输优化参数
                 enable_direct_transmission_optimization: bool = True,
                 direct_transmission_threshold: float = 0.1,
                 # 可视化参数
                 image_width: int = 900,
                 image_height: int = 700,
                 image_scale: int = 3,
                 node_marker_size: int = 7,
                 ch_marker_size: int = 10,
                 vc_marker_size: int = 12,
                 line_width: float = 1.0,
                 path_line_width: float = 2.0):
        self.net = network
        self.round_period = int(round_period)
        self.r_neighbor = float(r_neighbor)
        self.r_min_ch = float(r_min_ch)
        self.c_k = float(c_k)
        self.plan_paths = bool(plan_paths)
        self.consume_energy = bool(consume_energy)
        self.max_hops = int(max_hops)
        self.output_dir = output_dir
        
        # 簇头选择参数
        self.max_probability = float(max_probability)
        self.min_probability = float(min_probability)
        
        # 聚类成本函数参数
        self.distance_weight = float(distance_weight)
        self.energy_weight = float(energy_weight)
        
        # 通信能耗参数
        self.tx_rx_ratio = float(tx_rx_ratio)
        self.sensor_energy = float(sensor_energy)
        
        # 信息聚合参数
        self.base_data_size = int(base_data_size)
        self.aggregation_ratio = float(aggregation_ratio)
        self.enable_dynamic_data_size = bool(enable_dynamic_data_size)
        
        # 直接传输优化参数
        self.enable_direct_transmission_optimization = bool(enable_direct_transmission_optimization)
        self.direct_transmission_threshold = float(direct_transmission_threshold)
        
        # 可视化参数
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.image_scale = int(image_scale)
        self.node_marker_size = int(node_marker_size)
        self.ch_marker_size = int(ch_marker_size)
        self.vc_marker_size = int(vc_marker_size)
        self.line_width = float(line_width)
        self.path_line_width = float(path_line_width)
        # 运行态
        self.last_round_t = None
        self.virtual_center = (0.0, 0.0)   # (cx, cy)
        self.cluster_of = {}               # node_id -> ch_id
        self.ch_set = set()
        self.cluster_stats = {}            # ch_id -> summary
        self.upstream_paths = {}           # ch_id -> [SensorNode,...] 真实节点路径（不含虚拟中心“节点”）

        # 每轮通信统计
        self.last_comms = []               # list of dicts: {hop:(u->v), E_tx, E_rx, etc.}

    # ---------------- 工具：几何中心/距离 ----------------

    def _geo_center(self):
        xs, ys = zip(*[n.position for n in self.net.nodes]) if self.net.nodes else ([0],[0])
        return (sum(xs)/float(len(xs)), sum(ys)/float(len(ys)))

    @staticmethod
    def _dist_xy(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx*dx + dy*dy) ** 0.5

    # ---------------- 估计 K* / 选簇头 / 成簇细化 ----------------

    def _estimate_K_star(self):
        nodes = self.net.nodes
        N = len(nodes)
        if N <= 1:
            return 1
        # 近邻度方差
        deg = []
        for i, ni in enumerate(nodes):
            cnt = 0
            for j, nj in enumerate(nodes):
                if i == j:
                    continue
                if ni.distance_to(nj) <= self.r_neighbor:
                    cnt += 1
            deg.append(cnt)
        # 方差越大，降低 K*
        import numpy as np
        # 计算deg均值
        mean_k = float(np.mean(np.array(deg, dtype=float))) if deg else 0.0
        var_k = float(np.var(np.array(deg, dtype=float))) if len(deg) > 1 else 0.0
        # 标准差
        std_k = float(np.std(np.array(deg, dtype=float))) if len(deg) > 1 else 0.0
        K_star = max(1, round(round(self.c_k * (N ** 0.5) / (1.0 + std_k + 1e-9))))
        return K_star

    def _select_ch(self, K_star):
        nodes = self.net.nodes
        import numpy as np
        meanE = float(np.mean([n.current_energy for n in nodes])) if nodes else 0.0
        p_star = float(K_star) / max(1.0, float(len(nodes)))

        cand = sorted(nodes, key=lambda x: x.current_energy, reverse=True)
        chosen = []
        for n in cand:
            p_i = p_star * (n.current_energy / (1.0 + meanE))
            if random.random() > min(self.max_probability, max(self.min_probability, p_i)):
                continue
            ok = True
            for ch in chosen:
                if n.distance_to(ch) < self.r_min_ch:
                    ok = False
                    break
            if ok:
                chosen.append(n)
            if len(chosen) >= K_star:
                break
        if not chosen:
            chosen = [max(nodes, key=lambda x: x.current_energy)]
        self.ch_set = set([n.node_id for n in chosen])
        return chosen

    def _clustering(self, ch_nodes):
        def cost(n, ch):
            d = n.distance_to(ch)
            E_CH = max(1.0, ch.current_energy)
            return self.distance_weight * d + self.energy_weight * (1.0 / E_CH)

        nodes = self.net.nodes
        id2node = {n.node_id: n for n in nodes}

        # 初分配
        cluster_of = {}
        for n in nodes:
            best = min(ch_nodes, key=lambda ch: cost(n, ch))
            cluster_of[n.node_id] = best.node_id

        # 细化：把过载簇的远端成员迁到次优簇
        from collections import defaultdict
        groups = defaultdict(list)
        for nid, chid in cluster_of.items():
            groups[chid].append(nid)

        import numpy as np
        sizes = [len(v) for v in groups.values()]
        if sizes:
            avg_sz = float(np.mean(sizes))
            std_sz = float(np.std(sizes))
            limit = avg_sz + std_sz
            for chid, members in list(groups.items()):
                if len(members) > limit:
                    ch = id2node[chid]
                    far = sorted(members, key=lambda nid: id2node[nid].distance_to(ch), reverse=True)
                    move_quota = int(len(members) - limit)
                    for nid in far[:max(0, move_quota)]:
                        n = id2node[nid]
                        alts = sorted(ch_nodes, key=lambda c_: cost(n, c_))
                        for c2 in alts:
                            if c2.node_id != chid:
                                cluster_of[nid] = c2.node_id
                                break

        self.cluster_of = cluster_of
        return cluster_of

    def _summarize_clusters(self):
        import numpy as np
        id2node = {n.node_id: n for n in self.net.nodes}
        groups = defaultdict(list)
        for nid, chid in self.cluster_of.items():
            groups[chid].append(nid)

        stats = {}
        for chid, members in groups.items():
            Es = np.array([id2node[nid].current_energy for nid in members], dtype=float)
            ch_node = id2node[chid]
            dists = [id2node[nid].distance_to(ch_node) for nid in members]
            r_mean = float(np.mean(dists)) if dists else 0.0
            r_max = float(np.max(dists)) if dists else 0.0
            stats[chid] = {
                "size": len(members),
                "E_mean": float(Es.mean()) if len(Es) else 0.0,
                "E_std": float(Es.std()) if len(Es) else 0.0,
                "E_ch": float(ch_node.current_energy),
                "r_mean": r_mean,
                "r_max": r_max,
                "members": list(members)
            }
        self.cluster_stats = stats
        return stats

    # ---------------- 信息聚合计算 ----------------
    
    def _calculate_cluster_data_size(self, ch_id):
        """
        计算簇头需要传输的聚合数据量
        
        :param ch_id: 簇头节点ID
        :return: 聚合后的数据量（bits）
        """
        if not self.enable_dynamic_data_size:
            # 如果禁用动态数据量，使用基础数据量
            return self.base_data_size
        
        # 计算该簇的成员数量（包括簇头自己）
        cluster_members = [nid for nid, cid in self.cluster_of.items() if cid == ch_id]
        cluster_size = len(cluster_members)
        
        # 计算聚合数据量：基础数据量 × 簇大小 × 聚合比例
        aggregated_data_size = int(self.base_data_size * cluster_size * self.aggregation_ratio)
        
        print(f"[ADCR-DEBUG] Cluster {ch_id}: {cluster_size} members, data size: {aggregated_data_size} bits")
        return aggregated_data_size

    def _calculate_energy_cost(self, sender, receiver, distance, data_size=None):
        """
        计算从sender到receiver的能耗
        
        :param sender: 发送节点
        :param receiver: 接收节点（可以是虚拟中心）
        :param distance: 传输距离
        :param data_size: 数据量（bits），如果为None则使用sender.B
        :return: 能耗值
        """
        if data_size is None:
            data_size = sender.B
        
        E_elec = sender.E_elec
        eps = sender.epsilon_amp
        tau = sender.tau
        
        # 计算发送和接收能耗
        E_tx = E_elec * data_size + eps * data_size * (distance ** tau)
        E_rx = E_elec * data_size
        
        # 使用与SensorNode.energy_consumption相同的公式
        E_com = (E_tx + E_rx) / 2 + sender.sensor_energy
        
        return E_com

    def _should_use_direct_transmission(self, ch, anchor):
        """
        判断是否应该直接传输到虚拟中心
        
        :param ch: 簇头节点
        :param anchor: 锚点节点
        :return: True表示直接传输，False表示通过锚点传输
        """
        if not self.enable_direct_transmission_optimization:
            return False
        
        # 计算距离
        ch_to_vc_dist = self._dist_xy(ch.position, self.virtual_center)
        ch_to_anchor_dist = ch.distance_to(anchor)
        anchor_to_vc_dist = self._dist_xy(anchor.position, self.virtual_center)
        
        # 计算该簇的聚合数据量
        aggregated_data_size = self._calculate_cluster_data_size(ch.node_id)
        
        # 计算能耗
        direct_energy = self._calculate_energy_cost(ch, self.virtual_center, ch_to_vc_dist, aggregated_data_size)
        via_anchor_energy = (self._calculate_energy_cost(ch, anchor, ch_to_anchor_dist, aggregated_data_size) + 
                            self._calculate_energy_cost(anchor, self.virtual_center, anchor_to_vc_dist, aggregated_data_size))
        
        # 判断是否应该直接传输
        should_direct = direct_energy <= via_anchor_energy * (1 + self.direct_transmission_threshold)
        
        print(f"[ADCR-DEBUG] CH {ch.node_id} transmission decision:")
        print(f"  Direct: {ch_to_vc_dist:.2f}m, {direct_energy:.2f}J")
        print(f"  Via anchor: {ch_to_anchor_dist:.2f}m + {anchor_to_vc_dist:.2f}m, {via_anchor_energy:.2f}J")
        print(f"  Decision: {'Direct' if should_direct else 'Via anchor'}")
        
        return should_direct

    # ---------------- 路径规划到"虚拟中心" + 通信能耗结算 ----------------

    def _plan_paths_to_virtual(self):
        """
        为每个簇头规划一条真实节点路径（CH -> … -> "靠近几何中心的真实节点"），
        最后一跳视为"到虚拟中心"的逻辑汇报（不需要接收方能量）。
        """
        print(f"[ADCR-DEBUG] _plan_paths_to_virtual: plan_paths={self.plan_paths}, opportunistic_routing={opportunistic_routing is not None}")
        if not self.plan_paths or opportunistic_routing is None:
            print(f"[ADCR-DEBUG] Path planning disabled or routing function unavailable")
            self.upstream_paths = {}
            return {}

        id2node = {n.node_id: n for n in self.net.nodes}
        # 选取“最靠近几何中心的真实节点”作为锚点（终点），终点到虚拟中心的“虚拟跳”不计接收能量
        cx, cy = self.virtual_center
        anchor = min(self.net.nodes, key=lambda n: self._dist_xy(n.position, (cx, cy)))

        paths = {}
        for ch_id in self.ch_set:
            ch = id2node[ch_id]
            if ch is anchor:
                paths[ch_id] = [ch]  # 只有虚拟跳
                continue
            
            # 判断是否应该直接传输到虚拟中心
            if self._should_use_direct_transmission(ch, anchor):
                paths[ch_id] = [ch]  # 直接传输，只有虚拟跳
                print(f"[ADCR-DEBUG] CH {ch_id}: Using direct transmission to virtual center")
            else:
                path = opportunistic_routing(self.net.nodes, ch, anchor, max_hops=self.max_hops, t=0)
                paths[ch_id] = path  # 通过锚点传输
                print(f"[ADCR-DEBUG] CH {ch_id}: Using path through anchor")
        
        self.upstream_paths = paths
        return paths

    def _energy_consume_one_hop(self, u, v, transfer_WET=False):
        """
        对真实节点的一跳通信扣能量：发送方/接收方各自调用能耗模型。
        注意：你的模型里 E_com= (E_tx+E_rx)/2 + E_sen [+E_char]，我们按照该实现扣除。:contentReference[oaicite:1]{index=1}
        """
        # 发送方→接收方：两端都要通信（有 ACK），各自按自身参数扣费
        Eu = u.energy_consumption(target_node=v, transfer_WET=transfer_WET)
        Ev = v.energy_consumption(target_node=u, transfer_WET=False)
        u.current_energy = max(0.0, u.current_energy - Eu)
        v.current_energy = max(0.0, v.current_energy - Ev)
        return Eu, Ev

    def _settle_comm_energy(self):
        """
        为所有簇头上报路径结算通信能耗。
        - 对每条真实节点路径：逐跳扣费 (u<->v)
        - 最后一跳到虚拟中心：只对"最后一个真实节点"按到虚拟中心的距离扣"单端发射+感知"的通信费用，
          这里按照你的 energy_consumption 形式近似：我们构造一个"等效距离"的 E_tx+E_rx/2 + E_sen，
          但没有真实接收方，因此只扣发送端一次。
        """
        print(f"[ADCR-DEBUG] _settle_comm_energy() called, consume_energy={self.consume_energy}")
        self.last_comms = []
        if not self.consume_energy:
            print("[ADCR-DEBUG] Energy consumption disabled, skipping")
            return

        if not self.upstream_paths:
            print("[ADCR-DEBUG] No upstream paths available, skipping")
            return
        
        print(f"[ADCR-DEBUG] Processing {len(self.upstream_paths)} upstream paths")

        cx, cy = self.virtual_center

        for ch_id, path in self.upstream_paths.items():
            if (path is None) or (len(path) == 0):
                continue

            # 真实节点逐跳
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                Eu, Ev = self._energy_consume_one_hop(u, v, transfer_WET=False)
                self.last_comms.append({"hop": (u.node_id, v.node_id), "E_tx": Eu, "E_rx": Ev})

            # 最后一跳到虚拟中心（只有发送端计费）
            last_real = path[-1]
            # 计算到虚拟中心的欧式距离
            d = self._dist_xy(last_real.position, (cx, cy))
            
            # 计算该簇的聚合数据量
            aggregated_data_size = self._calculate_cluster_data_size(ch_id)
            
            # 使用聚合数据量计算能耗
            E_elec = last_real.E_elec
            eps = last_real.epsilon_amp
            tau = last_real.tau
            # 使用聚合数据量B_aggregated替代固定的B
            E_tx_virtual = E_elec * aggregated_data_size + eps * aggregated_data_size * (d ** tau)
            E_rx_virtual = E_elec * aggregated_data_size
            E_com = self.tx_rx_ratio * (E_tx_virtual + E_rx_virtual) + self.sensor_energy
            last_real.current_energy = max(0.0, last_real.current_energy - E_com)
            self.last_comms.append({
                "hop": (last_real.node_id, "VIRTUAL"), 
                "E_tx_only": E_com,
                "data_size": aggregated_data_size,
                "cluster_size": len([nid for nid, cid in self.cluster_of.items() if cid == ch_id])
            })

    # ---------------- Plotly 可视化 ----------------

    def plot_clusters_and_paths(self, output_dir=None, title="ADCR clustering & info paths to virtual center"):
        """
        画：所有节点、簇头（强调）、簇内连线、簇头到锚点路径、虚拟中心位置。
        保存为 PNG（需要 kaleido）。
        """
        if go is None:
            print("[ADCR-Link-Virtual] Plotly not available.")
            return

        import os
        import numpy as np

        # 统一输出目录：优先函数入参，其次类属性 self.output_dir，最后退回 "data"
        if output_dir is None:
            output_dir = getattr(self, "output_dir", "data")
        OutputManager.ensure_dir_exists(output_dir)

        nodes = self.net.nodes
        id2node = {n.node_id: n for n in nodes}
        cx, cy = self.virtual_center

        # 基础散点：所有节点
        x_all = [n.position[0] for n in nodes]
        y_all = [n.position[1] for n in nodes]
        ids = [n.node_id for n in nodes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_all, y=y_all, mode='markers+text', name='Nodes',
            marker=dict(color='#888', size=self.node_marker_size),
            text=[str(i) for i in ids[:100]], textposition='bottom right',
            hovertemplate='Node %{text}<br>(%{x}, %{y})<extra></extra>'
        ))

        # 簇头
        ch_nodes = [id2node[i] for i in self.ch_set]
        fig.add_trace(go.Scatter(
            x=[n.position[0] for n in ch_nodes],
            y=[n.position[1] for n in ch_nodes],
            mode='markers', name='Cluster Heads',
            marker=dict(color='#d62728', size=self.ch_marker_size, symbol='diamond'),
            hovertemplate='CH %{x}, %{y}<extra></extra>'
        ))

        # 虚拟中心
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers', name='Virtual Center',
            marker=dict(color='#1f77b4', size=self.vc_marker_size, symbol='x'),
            hovertemplate='Virtual center<br>(%{x}, %{y})<extra></extra>'
        ))

        # 簇内连线（成员→CH）
        for nid, chid in self.cluster_of.items():
            if nid == chid:
                continue
            a = id2node[nid].position
            b = id2node[chid].position
            fig.add_trace(go.Scatter(
                x=[a[0], b[0]], y=[a[1], b[1]],
                mode='lines', line=dict(width=self.line_width, color='rgba(0,0,0,0.2)'),
                showlegend=False, hoverinfo='skip'
            ))

        # 路径：CH → 锚点（真实节点路径）
        for ch_id, path in self.upstream_paths.items():
            if not path or len(path) == 1:
                # 只有“虚拟跳”，画一条到虚拟中心
                p = id2node[ch_id].position
                fig.add_trace(go.Scatter(
                    x=[p[0], cx], y=[p[1], cy], mode='lines',
                    line=dict(width=self.path_line_width, color='rgba(31,119,180,0.6)'),
                    name=f'CH {ch_id} → VC', showlegend=False
                ))
                continue
            xs = [n.position[0] for n in path]
            ys = [n.position[1] for n in path]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines+markers',
                line=dict(width=self.path_line_width, color='rgba(31,119,180,0.6)'),
                marker=dict(size=6),
                name=f'Path CH {ch_id}', showlegend=False
            ))
            # 最后一段到虚拟中心
            last = path[-1].position
            fig.add_trace(go.Scatter(
                x=[last[0], cx], y=[last[1], cy], mode='lines',
                line=dict(width=self.path_line_width, dash='dot', color='rgba(31,119,180,0.6)'),
                showlegend=False
            ))

        fig.update_layout(
            title=title,
            xaxis_title="X (m)", yaxis_title="Y (m)",
            template='plotly_white',
            legend=dict(x=1.05, y=1, xanchor='left', yanchor='top'),
            margin=dict(r=150),
            xaxis=dict(scaleanchor='y', scaleratio=1)
        )

        # 使用与其他图相同的保存方式
        session_dir = OutputManager.get_session_dir(output_dir)
        save_path = OutputManager.get_file_path(session_dir, 'adcr_info_paths.png')
        try:
            # 使用与其他图相同的参数
            fig.write_image(save_path, width=800, height=600, scale=3)
            print(f"ADCR聚类和路径图已保存到: {save_path}")
        except Exception as e:
            print(f"[ADCR-Link-Virtual] save image failed: {e}")
            # 如果保存失败，尝试保存为HTML文件
            try:
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                print(f"ADCR聚类和路径图已保存为HTML: {html_path}")
            except Exception as e2:
                print(f"[ADCR-Link-Virtual] HTML save also failed: {e2}")

        fig.show()

    # ---------------- 主入口：Step 1.5 调用 ----------------

    def step(self, t):
        # 到期才重聚类 + 路径 + 结算
        if (self.last_round_t is not None) and (t - self.last_round_t < self.round_period):
            return
        if not self.net.nodes:
            print("[ADCR-DEBUG] Skipping - no nodes in network")
            return

        print("[ADCR-DEBUG] Starting ADCR clustering process...")

        # 1) 更新虚拟几何中心
        self.virtual_center = self._geo_center()
        print(f"[ADCR-DEBUG] Virtual center updated: {self.virtual_center}")

        # 2) 估计 K* 、选择簇头、成簇
        try:
            K_star = self._estimate_K_star()
            print(f"[ADCR-DEBUG] Estimated K* = {K_star}")
        except Exception as e:
            print(f"[ADCR-DEBUG] Error in _estimate_K_star: {e}")
            return
        
        try:
            ch_nodes = self._select_ch(K_star)
            print(f"[ADCR-DEBUG] Selected {len(ch_nodes)} cluster heads: {[n.node_id for n in ch_nodes]}")
        except Exception as e:
            print(f"[ADCR-DEBUG] Error in _select_ch: {e}")
            return
        
        try:
            self._clustering(ch_nodes)
            print(f"[ADCR-DEBUG] Clustering completed, {len(self.cluster_of)} nodes assigned to clusters")
        except Exception as e:
            print(f"[ADCR-DEBUG] Error in _clustering: {e}")
            return
        
        self._summarize_clusters()
        print(f"[ADCR-DEBUG] Cluster summary: {len(self.cluster_stats)} clusters")

        # 3) 规划 CH→锚点（真实节点）的上报路径；最后对虚拟中心做"虚拟跳"
        self._plan_paths_to_virtual()
        print(f"[ADCR-DEBUG] Path planning completed, {len(self.upstream_paths)} paths planned")
        
        # 详细输出路径信息
        for ch_id, path in self.upstream_paths.items():
            if path is None:
                print(f"[ADCR-DEBUG] CH {ch_id}: No path found")
            elif len(path) == 1:
                print(f"[ADCR-DEBUG] CH {ch_id}: Direct to virtual center (path length: 1)")
            else:
                print(f"[ADCR-DEBUG] CH {ch_id}: Path length {len(path)}, nodes: {[n.node_id for n in path]}")

        # 4) 结算通信能耗（逐跳 + 虚拟跳只扣发送端）
        self._settle_comm_energy()
        print(f"[ADCR-DEBUG] Energy settlement completed, {len(self.last_comms)} communication hops processed")

        print("[ADCR-Link-Virtual] t={} | VC=({:.3f},{:.3f}) | K*={} | CHs={}".format(
            t, self.virtual_center[0], self.virtual_center[1], K_star, sorted(list(self.ch_set))
        ))
        self.last_round_t = t
        print(f"[ADCR-DEBUG] ADCR step completed, last_round_t updated to {self.last_round_t}")
