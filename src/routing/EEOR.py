# --- EEOR minimal reproducible implementation ---

import math
import heapq
from collections import defaultdict

# ===== 1) 基础：邻接、e(u,v)、w(u,v) =====

def _neighbor_range():
    return math.sqrt(3)

def _build_neighbors(nodes):
    nmap = {n.node_id: [] for n in nodes}
    ndict = {n.node_id: n for n in nodes}
    R = _neighbor_range()
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i == j: continue
            d = ni.distance_to(nj)
            if d <= R:
                nmap[ni.node_id].append((nj.node_id, d))
    return nmap, ndict

def _build_neighbors_adaptive(nodes, target_neighbors=6):
    """
    自适应邻居发现：动态调整通信范围，使每个节点有目标数量的邻居
    
    注意：物理中心节点（ID=0）完全不参与能量传输，在邻居构建时直接排除
    """
    nmap = {n.node_id: [] for n in nodes}
    ndict = {n.node_id: n for n in nodes}
    
    for ni in nodes:
        # 排除物理中心节点（ID=0完全不参与WET）
        if hasattr(ni, 'is_physical_center') and ni.is_physical_center:
            continue
        
        # 1. 计算到所有其他节点的距离
        distances = []
        for nj in nodes:
            if ni != nj:
                # 排除物理中心节点作为邻居
                if hasattr(nj, 'is_physical_center') and nj.is_physical_center:
                    continue
                d = ni.distance_to(nj)
                distances.append((d, nj))
        
        distances.sort(key=lambda x: x[0])  # 按距离排序
        
        # 2. 确定通信范围
        if len(distances) >= target_neighbors:
            # 使用到第target_neighbors个邻居的距离作为通信范围
            R = distances[target_neighbors-1][0] * 1.1  # 稍微扩大
        else:
            # 节点太少，使用最大距离
            R = distances[-1][0] if distances else math.sqrt(3)
        
        # # 3. 限制通信范围上限，避免过度扩大
        # original_R = math.sqrt(3)
        # R = min(R, original_R * 1.5)  # 最多扩大50%
        
        # 4. 建立邻居关系
        for d, nj in distances:
            if d <= R:
                nmap[ni.node_id].append((nj.node_id, d))
    
    return nmap, ndict

def _link_error_prob(d, k=0.05, gamma=2.0):
    # 简化误码率模型：随距离升高而增大；上限设为0.95避免全断
    return min(0.95, k * (d ** gamma))

def _min_tx_power(d, base=1.0, tau=2.0):
    # 简化最小发射功率需求：与 d^tau 成正比
    return base * (d ** tau)

# ===== 2) 论文公式：给定 Fwd 计算 C_u =====
def _expected_cost_given_fwd(u_id, Fwd_ids, C, neighbor_map, node_dict):
    """
    C: dict[node_id] -> 当前对目标的期望代价 (已知/上轮)
    返回: Cu(Fwd) 以及用于中间过程的 (rho, alpha)
    """
    if not Fwd_ids:
        return float('inf'), 0.0, 1.0

    # 计算 alpha, rho
    prod = 1.0
    w_u = 0.0  # 固定功率: 用达到最远Fwd的最小功率 or 统一常数。这里取到最远邻居的功率
    max_d = 0.0

    for v_id in Fwd_ids:
        # 找 u->v 的距离与误码
        d = None
        for nid, dist in neighbor_map[u_id]:
            if nid == v_id:
                d = dist
                break
        if d is None:
            return float('inf'), 0.0, 1.0
        e_uv = _link_error_prob(d)
        prod *= e_uv
        if d > max_d: max_d = d

    alpha = prod
    rho = 1.0 - alpha
    if rho <= 1e-12:
        return float('inf'), rho, alpha

    w_u = _min_tx_power(max_d)  # 固定功率模型也可取常数，这里取到最远邻居的最小可达功率

    # 计算 β：按优先级（C 从小到大）次序
    # Fwd 已保证按 C 升序
    beta = 0.0
    prefix_fail = 1.0
    for i, v_id in enumerate(Fwd_ids):
        # e(u, v)
        d_uv = None
        for nid, dist in neighbor_map[u_id]:
            if nid == v_id:
                d_uv = dist
                break
        e_uv = _link_error_prob(d_uv)
        success = (1.0 - e_uv)
        beta += prefix_fail * success * C.get(v_id, float('inf'))
        prefix_fail *= e_uv

    C_h = w_u / rho           # 论文式(2)发射到“至少一人收到”的期望发射能耗
    C_f = beta / rho          # 论文式(4)后续转发代价（忽略协调项）
    return (C_h + C_f), rho, alpha

# ===== 3) 前缀选择（Algorithm 1）=====
def _select_forwarder_prefix(u_id, neighbors_ids, C, neighbor_map):
    # 邻居按 C 升序；若Cv为inf代表其还未知，放后面
    ordered = sorted(neighbors_ids, key=lambda vid: C.get(vid, float('inf')))
    best_cost = float('inf')
    best_fwd = []
    trial_fwd = []

    for vid in ordered:
        trial_fwd.append(vid)
        cost, _, _ = _expected_cost_given_fwd(u_id, trial_fwd, C, neighbor_map, None)
        if cost < best_cost:
            best_cost = cost
            best_fwd = list(trial_fwd)
        else:
            # 成本不再下降，停止（Theorem 2/3 + Algorithm 1）
            break
    return best_cost, best_fwd

# ===== 4) 全网期望代价计算（Algorithm 4 简化）=====
def eeor_compute_costs(nodes, target_node_id, max_iter=20):
    neighbor_map, node_dict = _build_neighbors(nodes)
    V = [n.node_id for n in nodes]

    C = {vid: float('inf') for vid in V}
    FWD = {vid: [] for vid in V}
    C[target_node_id] = 0.0

    for _ in range(max_iter):
        updated = False
        # 只用"能直达的邻居集合"参与，按论文思路反复松弛
        for u_id in V:
            if u_id == target_node_id:
                FWD[u_id] = []
                continue
            neigh_ids = [nid for (nid, _) in neighbor_map[u_id]]
            if not neigh_ids:
                continue
            new_cost, new_fwd = _select_forwarder_prefix(u_id, neigh_ids, C, neighbor_map)
            if new_cost < C[u_id] - 1e-12:
                C[u_id] = new_cost
                FWD[u_id] = new_fwd
                updated = True
        if not updated:
            break
    return C, FWD

def eeor_compute_costs_adaptive(nodes, target_node_id, max_iter=20, target_neighbors=6):
    """使用自适应邻居发现的EEOR代价计算"""
    neighbor_map, node_dict = _build_neighbors_adaptive(nodes, target_neighbors)
    V = [n.node_id for n in nodes]

    C = {vid: float('inf') for vid in V}
    FWD = {vid: [] for vid in V}
    C[target_node_id] = 0.0

    for _ in range(max_iter):
        updated = False
        # 只用"能直达的邻居集合"参与，按论文思路反复松弛
        for u_id in V:
            if u_id == target_node_id:
                FWD[u_id] = []
                continue
            neigh_ids = [nid for (nid, _) in neighbor_map[u_id]]
            if not neigh_ids:
                continue
            new_cost, new_fwd = _select_forwarder_prefix(u_id, neigh_ids, C, neighbor_map)
            if new_cost < C[u_id] - 1e-12:
                C[u_id] = new_cost
                FWD[u_id] = new_fwd
                updated = True
        if not updated:
            break
    return C, FWD

# ===== 5) 从源到目的导出一条路径（每步选前缀里 C 最小的）=====
def eeor_find_path(nodes, source_node, dest_node, max_hops=5):
    C, FWD = eeor_compute_costs(nodes, dest_node.node_id)

    path = [source_node]
    cur = source_node.node_id
    hops = 0

    while cur != dest_node.node_id and hops < max_hops:
        fwd = FWD.get(cur, [])
        if not fwd:
            break
        # 选其前缀中 C 最小者（已是升序，取第一个）
        nxt = fwd[0]
        # 避免环
        if nxt == cur or any(n.node_id == nxt for n in path):
            break
        path.append(next(n for n in nodes if n.node_id == nxt))
        cur = nxt
        hops += 1

    if path[-1].node_id != dest_node.node_id:
        return None
    return path

def eeor_find_path_adaptive(nodes, source_node, dest_node, max_hops=5, target_neighbors=6):
    """
    使用自适应邻居发现的EEOR路径查找
    
    注意：物理中心节点（ID=0）已在邻居构建阶段被排除，完全不参与能量传输
    """
    C, FWD = eeor_compute_costs_adaptive(nodes, dest_node.node_id, target_neighbors=target_neighbors)

    path = [source_node]
    cur = source_node.node_id
    hops = 0

    while cur != dest_node.node_id and hops < max_hops:
        fwd = FWD.get(cur, [])
        if not fwd:
            break
        
        # 选其前缀中 C 最小者（已是升序，取第一个）
        nxt = fwd[0]
        # 避免环
        if nxt == cur or any(n.node_id == nxt for n in path):
            break
        
        path.append(next(n for n in nodes if n.node_id == nxt))
        cur = nxt
        hops += 1

    if path[-1].node_id != dest_node.node_id:
        return None
    return path
