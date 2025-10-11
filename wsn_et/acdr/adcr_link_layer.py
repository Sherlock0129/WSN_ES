# src/adcr_link_layer.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import math
import random
import os
from collections import defaultdict

try:
    from wsn_et.routing.opportunistic_routing import opportunistic_routing
except Exception:
    opportunistic_routing = None

try:
    import plotly.graph_objects as go
except Exception:
    go = None


class ADCRLinkLayerVirtual(object):
    def __init__(self, network,
                 round_period=1440,
                 r_neighbor=math.sqrt(3.0),
                 r_min_ch=1.0,
                 c_k=1.2,
                 plan_paths=True,
                 consume_energy=True,
                 max_hops=5,
                 output_dir="data"):
        self.net = network
        self.round_period = int(round_period)
        self.r_neighbor = float(r_neighbor)
        self.r_min_ch = float(r_min_ch)
        self.c_k = float(c_k)
        self.plan_paths = bool(plan_paths)
        self.consume_energy = bool(consume_energy)
        self.max_hops = int(max_hops)

        self.output_dir = output_dir
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception:
            pass

        self.last_round_t = None
        self.virtual_center = (0.0, 0.0)
        self.cluster_of = {}
        self.ch_set = set()
        self.cluster_stats = {}
        self.upstream_paths = {}
        self.last_comms = []


class ADCRLinkLayerVirtual(object):
    """
    - 估计最优簇数 dynamic_k*（近邻统计启发式）
    - 能量感知 + 空间抑制 选择簇头
    - 成簇与一次性细化
    - 以网络几何中心为“虚拟link中心（无限能量）”，为每个簇头规划一条上报路径
    - 按通信能耗模型为真实节点扣除能量；虚拟中心不耗
    - 提供 Plotly 可视化：节点/簇路径/虚拟中心
    """
    def __init__(self, network,
                 round_period=1440,
                 r_neighbor=math.sqrt(3.0),
                 r_min_ch=1.0,
                 c_k=1.2,
                 plan_paths=True,
                 consume_energy=True,
                 max_hops=5,
                 output_dir="adcr"):
        self.net = network
        self.round_period = int(round_period)
        self.r_neighbor = float(r_neighbor)
        self.r_min_ch = float(r_min_ch)
        self.c_k = float(c_k)
        self.plan_paths = bool(plan_paths)
        self.consume_energy = bool(consume_energy)
        self.max_hops = int(max_hops)
        self.output_dir = output_dir
        self.last_round_t = None
        self.virtual_center = (0.0, 0.0)
        self.cluster_of = {}
        self.ch_set = set()
        self.cluster_stats = {}
        self.upstream_paths = {}
        self.last_comms = []

    def _geo_center(self):
        xs, ys = zip(*[n.position for n in self.net.nodes]) if self.net.nodes else ([0],[0])
        return (sum(xs)/float(len(xs)), sum(ys)/float(len(ys)))

    @staticmethod
    def _dist_xy(a, b):
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return (dx*dx + dy*dy) ** 0.5

    def _estimate_K_star(self):
        nodes = self.net.nodes
        N = len(nodes)
        if N <= 1:
            return 1
        deg = []
        for i, ni in enumerate(nodes):
            cnt = 0
            for j, nj in enumerate(nodes):
                if i == j:
                    continue
                if ni.distance_to(nj) <= self.r_neighbor:
                    cnt += 1
            deg.append(cnt)
        import numpy as np
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
            if random.random() > min(0.9, max(0.05, p_i)):
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
            return 1.0 * d + 0.2 * (1.0 / E_CH)

        nodes = self.net.nodes
        id2node = {n.node_id: n for n in nodes}

        cluster_of = {}
        for n in nodes:
            best = min(ch_nodes, key=lambda ch: cost(n, ch))
            cluster_of[n.node_id] = best.node_id

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

    def _plan_paths_to_virtual(self):
        if not self.plan_paths or opportunistic_routing is None:
            self.upstream_paths = {}
            return {}

        id2node = {n.node_id: n for n in self.net.nodes}
        cx, cy = self.virtual_center
        anchor = min(self.net.nodes, key=lambda n: self._dist_xy(n.position, (cx, cy)))

        paths = {}
        for ch_id in self.ch_set:
            ch = id2node[ch_id]
            if ch is anchor:
                paths[ch_id] = [ch]
                continue
            path = opportunistic_routing(self.net.nodes, ch, anchor, max_hops=self.max_hops, t=0)
            paths[ch_id] = path
        self.upstream_paths = paths
        return paths

    def _energy_consume_one_hop(self, u, v, transfer_WET=False):
        Eu = u.energy_consumption(target_node=v, transfer_WET=transfer_WET)
        Ev = v.energy_consumption(target_node=u, transfer_WET=False)
        u.current_energy = max(0.0, u.current_energy - Eu)
        v.current_energy = max(0.0, v.current_energy - Ev)
        return Eu, Ev

    def _settle_comm_energy(self):
        self.last_comms = []
        if not self.consume_energy:
            return
        if not self.upstream_paths:
            return

        cx, cy = self.virtual_center

        for ch_id, path in self.upstream_paths.items():
            if (path is None) or (len(path) == 0):
                continue

            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                Eu, Ev = self._energy_consume_one_hop(u, v, transfer_WET=False)
                self.last_comms.append({"hop": (u.node_id, v.node_id), "E_tx": Eu, "E_rx": Ev})

            last_real = path[-1]
            d = self._dist_xy(last_real.position, (cx, cy))
            B = last_real.B
            E_elec = last_real.E_elec
            eps = last_real.epsilon_amp
            tau = last_real.tau
            E_tx_virtual = E_elec * B + eps * B * (d ** tau)
            E_rx_virtual = E_elec * B
            E_com = 0.5 * (E_tx_virtual + E_rx_virtual) + 0.1
            last_real.current_energy = max(0.0, last_real.current_energy - E_com)
            self.last_comms.append({"hop": (last_real.node_id, "VIRTUAL"), "E_tx_only": E_com})

    def plot_clusters_and_paths(self, output_dir=None, title="ADCR clustering & info paths to virtual center"):
        if go is None:
            print("[ADCR-Link-Virtual] Plotly not available.")
            return

        import os
        import numpy as np

        if output_dir is None:
            output_dir = getattr(self, "output_dir", "data")
        os.makedirs(output_dir, exist_ok=True)

        nodes = self.net.nodes
        id2node = {n.node_id: n for n in nodes}
        cx, cy = self.virtual_center

        x_all = [n.position[0] for n in nodes]
        y_all = [n.position[1] for n in nodes]
        ids = [n.node_id for n in nodes]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_all, y=y_all, mode='markers+text', name='Nodes',
            marker=dict(color='#888', size=7),
            text=[str(i) for i in ids[:100]], textposition='bottom right',
            hovertemplate='Node %{text}<br>(%{x}, %{y})<extra></extra>'
        ))

        ch_nodes = [id2node[i] for i in self.ch_set]
        fig.add_trace(go.Scatter(
            x=[n.position[0] for n in ch_nodes],
            y=[n.position[1] for n in ch_nodes],
            mode='markers', name='Cluster Heads',
            marker=dict(color='#d62728', size=10, symbol='diamond'),
            hovertemplate='CH %{x}, %{y}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=[cx], y=[cy], mode='markers', name='Virtual Center',
            marker=dict(color='#1f77b4', size=12, symbol='x'),
            hovertemplate='Virtual center<br>(%{x}, %{y})<extra></extra>'
        ))

        for nid, chid in self.cluster_of.items():
            if nid == chid:
                continue
            a = id2node[nid].position
            b = id2node[chid].position
            fig.add_trace(go.Scatter(
                x=[a[0], b[0]], y=[a[1], b[1]],
                mode='lines', line=dict(width=1, color='rgba(0,0,0,0.2)'),
                showlegend=False, hoverinfo='skip'
            ))

        for ch_id, path in self.upstream_paths.items():
            if not path or len(path) == 1:
                p = id2node[ch_id].position
                fig.add_trace(go.Scatter(
                    x=[p[0], cx], y=[p[1], cy], mode='lines',
                    line=dict(width=2, color='rgba(31,119,180,0.6)'),
                    name=f'CH {ch_id} -> VC', showlegend=False
                ))
                continue
            xs = [n.position[0] for n in path]
            ys = [n.position[1] for n in path]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode='lines+markers',
                line=dict(width=2, color='rgba(31,119,180,0.6)'),
                marker=dict(size=6),
                name=f'Path CH {ch_id}', showlegend=False
            ))
            last = path[-1].position
            fig.add_trace(go.Scatter(
                x=[last[0], cx], y=[last[1], cy], mode='lines',
                line=dict(width=2, dash='dot', color='rgba(31,119,180,0.6)'),
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

        out_png = os.path.join(output_dir, 'adcr_info_paths.png')
        try:
            fig.write_image(out_png, width=900, height=700, scale=3)
        except Exception as e:
            print("[ADCR-Link-Virtual] save image failed:", e)

        fig.show()

    def step(self, t):
        if (self.last_round_t is not None) and (t - self.last_round_t < self.round_period):
            return
        if not self.net.nodes:
            return

        self.virtual_center = self._geo_center()
        K_star = self._estimate_K_star()
        ch_nodes = self._select_ch(K_star)
        self._clustering(ch_nodes)
        self._summarize_clusters()
        self._plan_paths_to_virtual()
        self._settle_comm_energy()

        print("[ADCR-Link-Virtual] t={} | VC=({:.3f},{:.3f}) | dynamic_k*={} | CHs={}".format(
            t, self.virtual_center[0], self.virtual_center[1], K_star, sorted(list(self.ch_set))
        ))
        self.last_round_t = t

