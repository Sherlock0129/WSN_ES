# src/adcr_link_layer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import random
import os
from collections import defaultdict
from utils.output_manager import OutputManager
from acdr.physical_center import VirtualCenter

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
    ADCR (Adaptive Dynamic Collaborative Routing) 链路层协议
    
    核心功能：
    - 估计最优簇数 K*（近邻统计启发式）
    - 能量感知 + 空间抑制 选择簇头
    - 成簇与一次性细化
    - 簇头向物理中心节点（ID=0）报告信息
    - 按通信能耗模型为真实节点扣除能量
    - 提供 Plotly 可视化：节点/簇/路径/物理中心
    
    架构更新（物理中心节点）：
    - 不再使用"虚拟中心"和"锚点"概念
    - 所有通信都是真实节点之间的通信
    - 物理中心节点（ID=0）是固定的特殊节点，位于几何中心
    - VirtualCenter 类保留，仅用于节点信息表管理（未来将重构为 PhysicalCenterHelper）
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
                 # 自动画图参数
                 auto_plot: bool = False,
                 plot_filename_template: str = "adcr_day{day}_t{t}.png",
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
        
        # 自动画图参数
        self.auto_plot = bool(auto_plot)
        self.plot_filename_template = str(plot_filename_template)
        
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
        self.last_round_t = 0  # 从0开始，避免在t=0时执行ADCR
        
        # 物理中心信息管理器（位置固定）
        # 归档文件路径将在第一次使用时由 OutputManager 设置
        # 获取物理中心节点位置（如果启用）
        physical_center = self.net.get_physical_center() if hasattr(self.net, 'get_physical_center') else None
        if physical_center:
            initial_pos = tuple(physical_center.position)
            print(f"[ADCR] 使用物理中心节点位置: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
        else:
            # 如果没有物理中心，使用几何中心
            if self.net.nodes:
                initial_pos = (
                    sum(n.position[0] for n in self.net.nodes) / len(self.net.nodes),
                    sum(n.position[1] for n in self.net.nodes) / len(self.net.nodes)
                )
                print(f"[ADCR] 使用网络几何中心: ({initial_pos[0]:.3f}, {initial_pos[1]:.3f})")
            else:
                initial_pos = (0.0, 0.0)
                print(f"[ADCR] 使用默认中心位置: (0.0, 0.0)")
        
        self.vc = VirtualCenter(
            initial_position=initial_pos,
            enable_logging=True,
            history_size=1000,  # 保留最近1000条历史记录
            archive_path=None   # 稍后通过 set_archive_path 设置
        )
        
        self.cluster_of = {}               # node_id -> ch_id
        self.ch_set = set()
        self.cluster_stats = {}            # ch_id -> summary
        self.upstream_paths = {}           # ch_id -> [SensorNode,...] 真实节点路径（不含虚拟中心"节点"）

        # 每轮通信统计
        self.last_comms = []               # list of dicts: {hop:(u->v), E_tx, E_rx, etc.}
        
        # 初始化虚拟中心节点信息表（填充初始值）
        self._initialize_virtual_center_info()
    
    def _initialize_virtual_center_info(self):
        """
        初始化虚拟中心节点信息表
        
        在ADCR创建时立即填充所有节点的初始状态信息
        """
        if self.net.nodes:
            print(f"[ADCR] 初始化虚拟中心节点信息表...")
            self.vc.initialize_node_info(
                nodes=self.net.nodes,
                initial_time=0  # 初始时间为0
            )
            print(f"[ADCR] 虚拟中心节点信息表初始化完成，已记录 {len(self.vc.latest_info)} 个节点")
    
    def set_archive_path(self, session_dir: str):
        """
        设置虚拟中心的归档路径
        
        :param session_dir: 会话目录路径
        """
        import os
        archive_path = os.path.join(session_dir, "virtual_center_node_info.csv")
        self.vc.archive_path = archive_path
        print(f"[ADCR] 虚拟中心归档路径已设置: {archive_path}")

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

    # 注意：部分计算逻辑在 VirtualCenter 类中（将来会重构为 PhysicalCenterHelper）

    # ---------------- 路径规划到物理中心 + 通信能耗结算 ----------------

    def _plan_paths_to_physical_center(self):
        """
        为每个簇头规划一条真实节点路径：CH -> … -> 物理中心节点（ID=0）
        
        所有路径都是真实节点之间的通信，不再有"虚拟跳"的概念
        """
        if not self.plan_paths or opportunistic_routing is None:
            print(f"[ADCR-DEBUG] Path planning disabled or routing function unavailable")
            self.upstream_paths = {}
            return {}
        
        # 获取物理中心节点
        physical_center = self.net.get_physical_center()
        if physical_center is None:
            print(f"[ADCR-DEBUG] No physical center found, skipping path planning")
            self.upstream_paths = {}
            return {}
        
        # 获取簇头节点列表
        id2node = {n.node_id: n for n in self.net.nodes}
        cluster_heads = [id2node[ch_id] for ch_id in self.ch_set if ch_id in id2node]
        
        # 为每个簇头规划到物理中心的路径
        self.upstream_paths = {}
        for ch in cluster_heads:
            if ch.node_id == physical_center.node_id:
                # 簇头就是物理中心（不应该发生，因为调度器已排除）
                self.upstream_paths[ch.node_id] = [ch]
                continue
            
            # 使用机会路由规划路径
            path = opportunistic_routing(
                nodes=self.net.nodes,
                source_node=ch,
                dest_node=physical_center,
                max_hops=self.max_hops
            )
            
            if path is None or len(path) == 0:
                print(f"[ADCR-DEBUG] No path found from CH {ch.node_id} to physical center")
                path = [ch, physical_center]  # 回退：直接通信
            
            self.upstream_paths[ch.node_id] = path
        
        return self.upstream_paths

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
        
        所有通信都是真实节点之间的通信，包括：
        1. 簇内通信：成员 → 簇头
        2. 簇间通信：簇头 → … → 物理中心节点（ID=0）
        
        所有跳都使用相同的能耗模型，不再有"虚拟跳"概念
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

        for ch_id, path in self.upstream_paths.items():
            if (path is None) or (len(path) == 0):
                continue

            # 阶段1：簇内通信（成员 → 簇头）
            cluster_members = [nid for nid, cid in self.cluster_of.items() if cid == ch_id]
            id2node = {n.node_id: n for n in self.net.nodes}
            
            if ch_id in id2node:
                ch_node = id2node[ch_id]
                for member_id in cluster_members:
                    if member_id != ch_id and member_id in id2node:
                        member = id2node[member_id]
                        # 成员向簇头通信
                        Eu_member, Ev_ch = self._energy_consume_one_hop(member, ch_node, transfer_WET=False)
                        self.last_comms.append({
                            "hop": (member_id, ch_id),
                            "E_tx": Eu_member,
                            "E_rx": Ev_ch,
                            "type": "intra_cluster"
                        })

            # 阶段2：簇间通信（簇头 → 物理中心节点）
            # 所有跳都是真实节点之间的通信，包括最后一跳到物理中心
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                Eu, Ev = self._energy_consume_one_hop(u, v, transfer_WET=False)
                
                # 标记是否是最后一跳（到物理中心）
                is_final_hop = (i == len(path) - 2)
                hop_type = "to_physical_center" if is_final_hop else "inter_cluster"
                
                self.last_comms.append({
                    "hop": (u.node_id, v.node_id),
                    "E_tx": Eu,
                    "E_rx": Ev,
                    "type": hop_type,
                    "cluster_id": ch_id
                })
    
    def _update_virtual_center_info(self, t: int):
        """
        更新虚拟中心的节点信息表
        
        在ADCR通信结算后调用，将所有节点的最新状态信息上报到虚拟中心
        
        :param t: 当前时间步
        """
        if not self.consume_energy:
            # 如果不进行能量结算，也不更新节点信息
            return
        
        # 计算每个簇的聚合数据量
        data_sizes = {}
        for ch_id in self.ch_set:
            data_sizes[ch_id] = self._calculate_cluster_data_size(ch_id)
        
        # 批量更新所有节点信息到虚拟中心
        self.vc.batch_update_node_info(
            nodes=self.net.nodes,
            current_time=t,
            cluster_mapping=self.cluster_of,
            data_sizes=data_sizes
        )
        
        print(f"[ADCR] 虚拟中心节点信息表已更新，当前记录: {len(self.vc.latest_info)} 个节点")

    # ---------------- Plotly 可视化 ----------------

    def plot_clusters_and_paths(self, output_dir=None, title="ADCR clustering & info paths to physical center", filename=None):
        """
        画：所有节点、簇头（强调）、簇内连线、簇头到物理中心路径、物理中心节点（ID=0）。
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
        
        # 获取物理中心节点位置
        physical_center = self.net.get_physical_center()
        if physical_center is not None:
            cx, cy = physical_center.position[0], physical_center.position[1]
        else:
            # 回退：使用几何中心
            cx, cy = self.vc.get_position()

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

        # 物理中心节点（ID=0）
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers', name='Physical Center (ID=0)',
            marker=dict(color='#1f77b4', size=self.vc_marker_size, symbol='star'),
            hovertemplate='Physical Center (ID=0)<br>(%{x}, %{y})<extra></extra>'
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

        # 路径：CH → 物理中心节点（ID=0）
        for ch_id, path in self.upstream_paths.items():
            if not path or len(path) == 1:
                # 直接连接到物理中心
                p = id2node[ch_id].position
                fig.add_trace(go.Scatter(
                    x=[p[0], cx], y=[p[1], cy], mode='lines',
                    line=dict(width=self.path_line_width, color='rgba(31,119,180,0.6)'),
                    name=f'CH {ch_id} → PC', showlegend=False
                ))
                continue
            
            # 绘制完整路径（包括到物理中心的最后一跳）
            xs = [n.position[0] for n in path]
            ys = [n.position[1] for n in path]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines+markers',
                line=dict(width=self.path_line_width, color='rgba(31,119,180,0.6)'),
                marker=dict(size=6),
                name=f'Path CH {ch_id} → PC', showlegend=False
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
        if filename is None:
            filename = 'adcr_info_paths.png'
        save_path = OutputManager.get_file_path(session_dir, filename)
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
        # 从第一个周期开始执行（跳过t=0）
        if t - self.last_round_t < self.round_period:
            return
        if not self.net.nodes:
            print("[ADCR-DEBUG] Skipping - no nodes in network")
            return

        print("[ADCR-DEBUG] Starting ADCR clustering process...")

        # 1) 获取物理中心节点（位置固定不变）
        physical_center = self.net.get_physical_center()
        if physical_center:
            print(f"[ADCR-DEBUG] Physical center (ID=0) at: ({physical_center.position[0]:.3f}, {physical_center.position[1]:.3f})")

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

        # 3) 规划 CH→物理中心节点（ID=0）的上报路径
        self._plan_paths_to_physical_center()
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
        
        # 4.5) 更新虚拟中心节点信息表
        self._update_virtual_center_info(t)

        physical_center = self.net.get_physical_center()
        if physical_center:
            pc_pos = (physical_center.position[0], physical_center.position[1])
            print("[ADCR-Link] t={} | PC(ID=0)=({:.3f},{:.3f}) | K*={} | CHs={}".format(
                t, pc_pos[0], pc_pos[1], K_star, sorted(list(self.ch_set))
            ))
        else:
            print("[ADCR-Link] t={} | K*={} | CHs={}".format(t, K_star, sorted(list(self.ch_set))))
        
        # 5) 自动画图（如果启用）
        if self.auto_plot:
            self._auto_plot_clusters_and_paths(t)
        
        self.last_round_t = t
        print(f"[ADCR-DEBUG] ADCR step completed, last_round_t updated to {self.last_round_t}")

    def _auto_plot_clusters_and_paths(self, t):
        """自动画图方法"""
        try:
            # 计算天数
            day = t // 1440
            
            # 生成文件名
            filename = self.plot_filename_template.format(day=day, t=t)
            
            # 创建ADCR专用子目录
            adcr_output_dir = os.path.join(self.output_dir, "adcr_plots")
            OutputManager.ensure_dir_exists(adcr_output_dir)
            
            # 调用画图方法
            self.plot_clusters_and_paths(output_dir=adcr_output_dir, filename=filename)
            print(f"[ADCR-Auto-Plot] 自动画图已保存: {filename}")
            
        except Exception as e:
            print(f"[ADCR-Auto-Plot] 自动画图失败: {e}")
