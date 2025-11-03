# src/routing/energy_transfer_routing.py
# -*- coding: utf-8 -*-
"""
专门为能量传输设计的路由算法

EETOR: Energy-Efficient Transfer Opportunistic Routing

核心特性：
1. 使用能量传输效率模型（而非误码率）
2. 使用实际通信能耗模型（而非简化功率）
3. 考虑路径总效率（累积乘积，而非多路径可靠性）
4. 集成节点能量状态感知
5. 移除硬性邻居范围限制（或使用更大的范围）
6. 优化目标：最小化总损耗 + 最大化最终接收能量
"""

import math
from typing import List, Dict, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.SensorNode import SensorNode
    from scheduling.info_node import InfoNode
    from config.simulation_config import EETORConfig

# 统一的节点类型（可以是SensorNode或InfoNode）
NodeType = type('Node', (), {})

# 全局EETOR配置（可以从ConfigManager获取）
_global_eetor_config = None


def set_eetor_config(config):
    """设置全局EETOR配置对象"""
    global _global_eetor_config
    _global_eetor_config = config


def get_eetor_config():
    """获取全局EETOR配置对象（如果没有设置则使用默认值）"""
    if _global_eetor_config is None:
        # 尝试从ConfigManager获取
        try:
            from config.simulation_config import get_config
            config_manager = get_config()
            return config_manager.eetor_config
        except:
            # 如果无法获取，创建默认配置
            from config.simulation_config import EETORConfig
            return EETORConfig()
    return _global_eetor_config


# ==================== 能量传输效率模型 ====================

def calculate_energy_transfer_efficiency(distance: float, eta_0: float = None, gamma: float = None) -> float:
    """
    计算能量传输效率（与SensorNode.energy_transfer_efficiency完全相同）
    
    :param distance: 节点间距离（米）
    :param eta_0: 1米处的参考效率（如果None则从配置获取，默认0.6）
    :param gamma: 衰减因子（如果None则从配置获取，默认2.0）
    :return: 传输效率 [0, 1]
    """
    # 从配置获取参数
    config = get_eetor_config()
    if eta_0 is None:
        eta_0 = config.eta_0
    if gamma is None:
        gamma = config.gamma
    
    if distance <= 1.0:
        # 距离≤1m时，效率为eta_0到1之间的线性插值
        efficiency = eta_0 + (1.0 - eta_0) * (1.0 - distance)
    else:
        # 距离>1m时，使用逆幂律衰减
        efficiency = eta_0 / (distance ** gamma)
    
    return min(1.0, max(0.0, efficiency))


def calculate_path_efficiency(path: List, config=None) -> float:
    """
    计算路径的总效率（累积乘积）
    
    :param path: 路径节点列表（节点对象，不是ID）
    :param config: EETOR配置对象，如果None则从全局配置获取
    :return: 路径总效率 [0, 1]
    """
    if len(path) < 2:
        return 1.0
    
    # 从配置获取参数
    if config is None:
        config = get_eetor_config()
    
    total_eta = 1.0
    for i in range(len(path) - 1):
        sender = path[i]
        receiver = path[i + 1]
        # 使用节点对象的distance_to方法
        distance = sender.distance_to(receiver)
        # 使用配置的eta_0和gamma
        eta = calculate_energy_transfer_efficiency(distance, eta_0=config.eta_0, gamma=config.gamma)
        total_eta *= eta
    
    return max(1e-6, min(1.0, total_eta))  # 限制在合理范围内


# ==================== 实际通信能耗模型 ====================

def calculate_communication_energy(sender, receiver, 
                                   energy_elec: float = None,
                                   epsilon_amp: float = None,
                                   bit_rate: float = None,
                                   path_loss_exponent: float = None,
                                   sensor_energy: float = None,
                                   transfer_WET: bool = False,
                                   energy_char: float = None) -> float:
    """
    计算实际通信能耗（与SensorNode.energy_consumption完全相同）
    
    优先使用节点自身的参数，如果没有则使用默认值
    
    :param sender: 发送节点（SensorNode或InfoNode）
    :param receiver: 接收节点（SensorNode或InfoNode）
    :param energy_elec: 电子能耗（J/bit），如果None则从节点获取
    :param epsilon_amp: 功放能耗系数（J/bit/m²），如果None则从节点获取
    :param bit_rate: 比特率（bits），如果None则从节点获取
    :param path_loss_exponent: 路径损耗指数，如果None则从节点获取
    :param sensor_energy: 传感器能耗（J），如果None则从节点获取
    :param transfer_WET: 是否包含能量传输开销
    :param energy_char: 能量传输特征能量（J），如果None则从节点获取
    :return: 通信总能耗（J）
    """
    # 优先使用节点自身的参数
    B = bit_rate if bit_rate is not None else getattr(sender, 'bit_rate', getattr(sender, 'B', 1000000.0))
    E_elec = energy_elec if energy_elec is not None else getattr(sender, 'energy_elec', 1e-4)
    eps_amp = epsilon_amp if epsilon_amp is not None else getattr(sender, 'epsilon_amp', 1e-5)
    tau = path_loss_exponent if path_loss_exponent is not None else getattr(sender, 'path_loss_exponent', getattr(sender, 'tau', 2.0))
    E_sen = sensor_energy if sensor_energy is not None else getattr(sender, 'sensor_energy', 0.1)
    E_char = energy_char if energy_char is not None else getattr(sender, 'energy_char', 1000.0)
    
    d = sender.distance_to(receiver)
    
    # 发送能耗（与SensorNode.energy_consumption公式相同）
    E_tx = E_elec * B + eps_amp * B * (d ** tau)
    
    # 接收能耗（假设双向确认通信）
    E_rx = E_elec * B
    
    # 通信总能耗
    E_com = (E_tx + E_rx) / 2 + E_sen
    
    # 如果包含能量传输，加上能量传输开销
    if transfer_WET:
        E_com += E_char
    
    return E_com


# ==================== 节点能量状态感知 ====================

def get_energy_state_penalty(node, capacity: float = None, config=None) -> float:
    """
    根据节点能量状态计算代价惩罚系数
    
    低能量节点应该避免作为中继，增加其代价
    有太阳能的节点降低代价（未来能量充足）
    
    :param node: 节点对象（SensorNode或InfoNode）
    :param capacity: 电池容量（J），如果为None则从节点获取
    :param config: EETOR配置对象，如果None则从全局配置获取
    :return: 惩罚系数（>1表示增加代价，<1表示降低代价）
    """
    # 获取配置
    if config is None:
        config = get_eetor_config()
    
    # 如果未启用能量状态感知，返回无惩罚
    if not config.enable_energy_state_aware:
        return 1.0
    
    # 获取当前能量
    current_energy = getattr(node, 'current_energy', 0.0)
    
    # 获取电池容量
    if capacity is None:
        # 尝试从节点属性获取
        if hasattr(node, 'capacity') and hasattr(node, 'V'):
            capacity = node.capacity * node.V * 3600  # mAh → J
        elif hasattr(node, 'capacity') and hasattr(node, 'voltage'):
            capacity = node.capacity * node.voltage * 3600  # mAh → J
        else:
            # 默认值
            capacity = 3.5 * 3.7 * 3600  # 默认电池容量（3.5 mAh × 3.7 V × 3600）
    
    energy_ratio = current_energy / capacity if capacity > 0 else 0.0
    
    # 能量惩罚：低能量节点增加代价（从配置获取）
    if energy_ratio < config.low_energy_threshold:
        penalty = config.low_energy_penalty
    elif energy_ratio < config.medium_energy_threshold:
        penalty = config.medium_energy_penalty
    else:
        penalty = 1.0  # 高能量节点无惩罚
    
    # 太阳能奖励：有太阳能的节点降低代价（从配置获取）
    has_solar = getattr(node, 'has_solar', False) or getattr(node, 'is_solar', False)
    if has_solar:
        penalty *= config.solar_bonus
    
    return penalty


# ==================== 邻居构建（能量传输优化）====================

def build_neighbors_energy_transfer(nodes: List, 
                                     max_range: float = None,
                                     min_efficiency: float = None,
                                     config=None) -> Tuple[Dict, Dict]:
    """
    为能量传输优化的邻居构建
    
    特性：
    1. 移除硬性距离限制（或使用很大的范围）
    2. 只考虑传输效率≥阈值的链路
    3. 排除物理中心节点（ID=0）
    
    :param nodes: 节点列表
    :param max_range: 最大通信范围（米），如果None则从配置获取
    :param min_efficiency: 最小传输效率阈值，如果None则从配置获取
    :param config: EETOR配置对象，如果None则从全局配置获取
    :return: (neighbor_map, node_dict)
    """
    # 从配置获取参数
    if config is None:
        config = get_eetor_config()
    if max_range is None:
        max_range = config.max_range
    if min_efficiency is None:
        min_efficiency = config.min_efficiency
    nmap = {n.node_id: [] for n in nodes}
    ndict = {n.node_id: n for n in nodes}
    
    for ni in nodes:
        # 排除物理中心节点（ID=0完全不参与WET）
        if hasattr(ni, 'is_physical_center') and ni.is_physical_center:
            continue
        
        for nj in nodes:
            if ni == nj:
                continue
            
            # 排除物理中心节点作为邻居
            if hasattr(nj, 'is_physical_center') and nj.is_physical_center:
                continue
            
            d = ni.distance_to(nj)
            
            # 距离检查（可选，但设置较大的范围）
            if d > max_range:
                continue
            
            # 效率检查：只考虑效率≥阈值的链路（使用配置的eta_0和gamma）
            eta = calculate_energy_transfer_efficiency(d, eta_0=config.eta_0, gamma=config.gamma)
            if eta < min_efficiency:
                continue
            
            # 建立邻居关系
            nmap[ni.node_id].append((nj.node_id, d, eta))
    
    return nmap, ndict


# ==================== 期望代价计算（能量传输优化）====================

def expected_cost_for_energy_transfer(u_id: int, 
                                      Fwd_ids: List[int],
                                      C: Dict[int, float],
                                      neighbor_map: Dict[int, List[Tuple]],
                                      node_dict: Dict[int, NodeType],
                                      energy_state_aware: bool = True,
                                      config=None) -> Tuple[float, float]:
    """
    针对能量传输的期望代价计算
    
    代价定义：
    C_u = (通信能耗 / 路径效率) + 后续转发期望代价
    
    其中：
    - 通信能耗：使用实际模型 E_com = (E_tx + E_rx)/2 + E_sen [+ E_char]
    - 路径效率：η_path = ∏η(di)（累积乘积，而非多路径可靠性）
    - 后续转发代价：考虑效率损耗的转发期望代价
    
    :param u_id: 当前节点ID
    :param Fwd_ids: 候选转发节点ID列表（按代价升序排列）
    :param C: 所有节点到目标的当前期望代价字典
    :param neighbor_map: 邻居图 {(node_id, distance, efficiency), ...}
    :param node_dict: 节点字典
    :param energy_state_aware: 是否考虑节点能量状态
    :return: (期望代价, 路径效率)
    """
    if not Fwd_ids:
        return float('inf'), 0.0
    
    # 获取当前节点
    u = node_dict.get(u_id)
    if u is None:
        return float('inf'), 0.0
    
    # 1. 计算路径效率（累积乘积）
    path_efficiency = 1.0
    max_d = 0.0
    
    for v_id in Fwd_ids:
        # 查找距离和效率
        d = None
        eta_uv = None
        for nid, dist, eta in neighbor_map.get(u_id, []):
            if nid == v_id:
                d = dist
                eta_uv = eta
                break
        
        if d is None or eta_uv is None:
            return float('inf'), 0.0
        
        path_efficiency *= eta_uv  # 累积效率损耗
        if d > max_d:
            max_d = d
    
    # 路径效率不能为0
    if path_efficiency < 1e-6:
        return float('inf'), path_efficiency
    
    # 2. 计算实际通信能耗（到最远转发节点）
    # 选择最远的转发节点作为通信目标（需要覆盖所有转发节点）
    max_receiver_id = None
    max_d_found = 0.0
    for v_id in Fwd_ids:
        for nid, dist, eta in neighbor_map.get(u_id, []):
            if nid == v_id and dist > max_d_found:
                max_receiver_id = v_id
                max_d_found = dist
                break
    
    if max_receiver_id is None:
        return float('inf'), path_efficiency
    
    max_receiver = node_dict.get(max_receiver_id)
    if max_receiver is None:
        return float('inf'), path_efficiency
    
    # 计算实际通信能耗（使用节点自身参数）
    E_com = calculate_communication_energy(
        sender=u,
        receiver=max_receiver,
        transfer_WET=False  # 这里只计算通信能耗，不包括WET开销
    )
    
    # 能量状态惩罚（如果启用）
    if energy_state_aware:
        penalty = get_energy_state_penalty(u, config=config)
        E_com *= penalty
    
    # 3. 计算后续转发期望代价（考虑效率损耗）
    beta = 0.0
    cumulative_efficiency = 1.0  # 累积未接收的能量比例
    
    for v_id in Fwd_ids:
        # 获取链路效率
        eta_uv = None
        for nid, dist, eta in neighbor_map.get(u_id, []):
            if nid == v_id:
                eta_uv = eta
                break
        
        if eta_uv is None:
            continue
        
        # 节点v成功接收的能量比例 × 节点v到目标的代价
        # 注意：这里考虑的是"通过节点v转发的能量比例"
        success_ratio = cumulative_efficiency * eta_uv
        C_v = C.get(v_id, float('inf'))
        
        # 如果节点v有能量状态，调整其代价
        if energy_state_aware and v_id in node_dict:
            v_penalty = get_energy_state_penalty(node_dict[v_id], config=config)
            C_v *= v_penalty
        
        beta += success_ratio * C_v
        
        # 更新累积未接收的能量比例（剩余能量）
        cumulative_efficiency *= (1.0 - eta_uv)
    
    # 4. 计算期望代价
    # C_h：通信能耗（归一化到有效接收能量）
    C_h = E_com / path_efficiency
    
    # C_f：后续转发期望代价（归一化到有效接收能量）
    C_f = beta / path_efficiency
    
    total_cost = C_h + C_f
    
    return total_cost, path_efficiency


# ==================== 前缀选择（能量感知）====================

def select_forwarder_prefix_energy_aware(u_id: int,
                                         neighbors_ids: List[int],
                                         C: Dict[int, float],
                                         neighbor_map: Dict[int, List[Tuple]],
                                         node_dict: Dict[int, NodeType],
                                         energy_state_aware: bool = True,
                                         config=None,
                                         node_info_manager=None) -> Tuple[float, List[int], float]:
    """
    能量感知的前缀选择：考虑路径总效率和节点能量状态
    
    算法：
    1. 按综合评分排序邻居（效率 × 代价倒数 × 信息奖励）
    2. 贪心扩展前缀，选择使总代价最小的前缀
    
    :param u_id: 当前节点ID
    :param neighbors_ids: 所有邻居节点ID列表
    :param C: 所有节点到目标的期望代价字典
    :param neighbor_map: 邻居图
    :param node_dict: 节点字典
    :param energy_state_aware: 是否考虑节点能量状态
    :param config: EETOR配置对象
    :param node_info_manager: NodeInfoManager实例（用于信息感知路由）
    :return: (最优代价, 最优转发前缀, 路径效率)
    """
    if not neighbors_ids:
        return float('inf'), [], 0.0
    
    # 1. 为每个邻居计算综合评分（用于初始排序）
    candidates = []
    for v_id in neighbors_ids:
        # 获取距离和效率
        d = None
        eta = None
        for nid, dist, eff in neighbor_map.get(u_id, []):
            if nid == v_id:
                d = dist
                eta = eff
                break
        
        if d is None or eta is None:
            continue
        
        # 获取代价（考虑能量状态）
        C_v = C.get(v_id, float('inf'))
        if energy_state_aware and v_id in node_dict:
            v_penalty = get_energy_state_penalty(node_dict[v_id], config=config)
            C_v *= v_penalty
        
        # 信息感知路由：考虑节点的信息量（info_volume）
        info_bonus = 1.0  # 默认无奖励
        if config and config.enable_info_aware_routing and node_info_manager:
            node_info = node_info_manager.get_node_info(v_id)
            if node_info:
                info_volume = node_info.get('info_volume', 0)
                is_reported = node_info.get('info_is_reported', True)
                
                # 如果节点有未上报的信息量，给予奖励（降低代价）
                if not is_reported and info_volume > 0:
                    # 信息奖励：信息量越大，奖励越大
                    # info_reward_factor 控制奖励强度（0~1）
                    # 归一化信息量（假设最大信息量为 base_data_size × max_hops）
                    max_info_volume = 1000000  # 默认最大值（可从配置获取）
                    normalized_volume = min(info_volume / max_info_volume, 1.0)
                    info_bonus = 1.0 - (config.info_reward_factor * normalized_volume)
                    # bonus < 1 表示降低代价（优先选择）
                    C_v *= info_bonus
        
        # 综合评分：效率 × (1 / 代价)
        # 效率高的节点优先，代价低的节点优先
        # 如果有信息奖励，代价降低，评分提高
        score = eta * (1.0 / max(C_v, 1.0))
        candidates.append((score, v_id, eta, C_v))
    
    if not candidates:
        return float('inf'), [], 0.0
    
    # 按评分降序排序
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 2. 贪心选择最优前缀
    best_cost = float('inf')
    best_fwd = []
    best_path_efficiency = 0.0
    trial_fwd = []
    
    for score, v_id, eta, C_v in candidates:
        trial_fwd.append(v_id)
        
        # 计算新的期望代价
        cost, path_eff = expected_cost_for_energy_transfer(
            u_id=u_id,
            Fwd_ids=trial_fwd,
            C=C,
            neighbor_map=neighbor_map,
            node_dict=node_dict,
            energy_state_aware=energy_state_aware,
            config=config
        )
        
        if cost < best_cost:
            best_cost = cost
            best_path_efficiency = path_eff
            best_fwd = list(trial_fwd)
        else:
            # 代价不再下降，停止（类似EEOR的贪心策略）
            break
    
    return best_cost, best_fwd, best_path_efficiency


# ==================== 全网期望代价计算 =====================

def compute_energy_transfer_costs(nodes: List,
                                   target_node_id: int,
                                   max_iter: int = None,
                                   max_range: float = None,
                                   min_efficiency: float = None,
                                   energy_state_aware: bool = None,
                                   config=None,
                                   node_info_manager=None) -> Tuple[Dict[int, float], Dict[int, List[int]]]:
    """
    计算网络中所有节点到目标节点的能量传输期望代价
    
    :param nodes: 节点列表（应包含目标节点）
    :param target_node_id: 目标节点ID
    :param max_iter: 最大迭代次数，如果None则从配置获取
    :param max_range: 最大通信范围（米），如果None则从配置获取
    :param min_efficiency: 最小传输效率阈值，如果None则从配置获取
    :param energy_state_aware: 是否考虑节点能量状态，如果None则从配置获取
    :param config: EETOR配置对象，如果None则从全局配置获取
    :return: (代价字典C, 转发前缀字典FWD)
    """
    # 从配置获取参数
    if config is None:
        config = get_eetor_config()
    if max_iter is None:
        max_iter = config.max_iter
    if max_range is None:
        max_range = config.max_range
    if min_efficiency is None:
        min_efficiency = config.min_efficiency
    if energy_state_aware is None:
        energy_state_aware = config.enable_energy_state_aware
    
    # 构建邻居图
    neighbor_map, node_dict = build_neighbors_energy_transfer(
        nodes=nodes,
        max_range=max_range,
        min_efficiency=min_efficiency,
        config=config
    )
    
    V = [n.node_id for n in nodes]
    
    # 初始化
    C = {vid: float('inf') for vid in V}
    FWD = {vid: [] for vid in V}
    
    # 检查目标节点是否在nodes中
    if target_node_id not in node_dict:
        # 如果目标节点不在nodes中，返回空结果
        return C, FWD
    
    C[target_node_id] = 0.0  # 目标节点代价为0
    
    # 迭代松弛
    for iteration in range(max_iter):
        updated = False
        
        for u_id in V:
            if u_id == target_node_id:
                FWD[u_id] = []
                continue
            
            # 获取邻居节点
            neighbors_ids = [nid for nid, _, _ in neighbor_map.get(u_id, [])]
            if not neighbors_ids:
                continue
            
            # 选择最优转发前缀
            new_cost, new_fwd, path_eff = select_forwarder_prefix_energy_aware(
                u_id=u_id,
                neighbors_ids=neighbors_ids,
                C=C,
                neighbor_map=neighbor_map,
                node_dict=node_dict,
                energy_state_aware=energy_state_aware,
                config=config,
                node_info_manager=node_info_manager
            )
            
            # 如果代价降低，更新
            if new_cost < C[u_id] - 1e-12:
                C[u_id] = new_cost
                FWD[u_id] = new_fwd
                updated = True
        
        # 如果没有更新，提前终止
        if not updated:
            break
    
    return C, FWD


# ==================== 路径查找 =====================

def find_energy_transfer_path(nodes: List,
                              source_node,
                              dest_node,
                              max_hops: int = 5,
                              max_range: float = None,
                              min_efficiency: float = None,
                              energy_state_aware: bool = None,
                              config=None,
                              node_info_manager=None) -> Optional[List]:
    """
    从源节点到目标节点查找能量传输路径
    
    :param nodes: 节点列表（应包含源节点和目标节点）
    :param source_node: 源节点
    :param dest_node: 目标节点
    :param max_hops: 最大跳数
    :param max_range: 最大通信范围（米）
    :param min_efficiency: 最小传输效率阈值
    :param energy_state_aware: 是否考虑节点能量状态
    :return: 路径节点列表，如果失败返回None
    """
    # 检查源节点和目标节点是否在nodes中
    # 如果不在，尝试添加到nodes（但不修改原列表）
    nodes_to_use = list(nodes)
    source_in_nodes = any(n.node_id == source_node.node_id for n in nodes_to_use)
    dest_in_nodes = any(n.node_id == dest_node.node_id for n in nodes_to_use)
    
    if not source_in_nodes:
        nodes_to_use.append(source_node)
    if not dest_in_nodes and dest_node.node_id != source_node.node_id:
        nodes_to_use.append(dest_node)
    
    # 从配置获取参数
    if config is None:
        config = get_eetor_config()
    if max_range is None:
        max_range = config.max_range
    if min_efficiency is None:
        min_efficiency = config.min_efficiency
    if energy_state_aware is None:
        energy_state_aware = config.enable_energy_state_aware
    
    # 计算所有节点到目标的期望代价和转发前缀
    C, FWD = compute_energy_transfer_costs(
        nodes=nodes_to_use,
        target_node_id=dest_node.node_id,
        max_range=max_range,
        min_efficiency=min_efficiency,
        energy_state_aware=energy_state_aware,
        config=config,
        node_info_manager=node_info_manager
    )
    
    # 构建节点字典（用于查找）
    id2node = {n.node_id: n for n in nodes_to_use}
    
    # 从源节点开始查找路径
    path = [source_node]
    cur = source_node.node_id
    hops = 0
    
    while cur != dest_node.node_id and hops < max_hops:
        fwd = FWD.get(cur, [])
        if not fwd:
            break  # 无法继续转发
        
        # 选择转发前缀中代价最小的节点（第一个）
        nxt = fwd[0]
        
        # 避免环路
        if nxt == cur or any(n.node_id == nxt for n in path):
            break
        
        # 获取下一个节点
        if nxt not in id2node:
            # 如果不在id2node中，尝试从原始nodes中查找（兼容性）
            next_node = next((n for n in nodes if n.node_id == nxt), None)
            if next_node is None:
                break
        else:
            next_node = id2node[nxt]
        
        path.append(next_node)
        cur = nxt
        hops += 1
    
    # 检查是否成功到达目标
    if path[-1].node_id != dest_node.node_id:
        return None
    
    return path


# ==================== 自适应版本（兼容现有接口）====================

def find_energy_transfer_path_adaptive(nodes: List,
                                       source_node,
                                       dest_node,
                                       max_hops: int = 5,
                                       target_neighbors: int = None,
                                       energy_state_aware: bool = None,
                                       config=None,
                                       node_info_manager=None) -> Optional[List]:
    """
    自适应版本的能量传输路径查找（兼容现有调度器接口）
    
    注意：target_neighbors参数保留以兼容接口，但实际使用更大的通信范围
    
    :param nodes: 节点列表
    :param source_node: 源节点
    :param dest_node: 目标节点
    :param max_hops: 最大跳数
    :param target_neighbors: 目标邻居数（用于兼容，实际不使用）
    :param energy_state_aware: 是否考虑节点能量状态
    :return: 路径节点列表，如果失败返回None
    """
    # 从配置获取参数
    if config is None:
        config = get_eetor_config()
    if target_neighbors is None:
        target_neighbors = config.target_neighbors
    if energy_state_aware is None:
        energy_state_aware = config.enable_energy_state_aware
    
    # 根据target_neighbors动态调整通信范围（但保持较大范围）
    # 如果网络密度高，可以适当缩小范围以提高效率
    # 如果网络密度低，使用更大的范围
    avg_neighbors = estimate_average_neighbors(nodes)
    if avg_neighbors > config.dense_network_threshold:
        max_range = config.dense_network_range  # 密集网络，可以缩小范围
    else:
        max_range = config.sparse_network_range  # 稀疏网络，使用大范围
    
    return find_energy_transfer_path(
        nodes=nodes,
        source_node=source_node,
        dest_node=dest_node,
        max_hops=max_hops,
        max_range=max_range,
        min_efficiency=config.min_efficiency,
        energy_state_aware=energy_state_aware,
        config=config,
        node_info_manager=node_info_manager
    )


def estimate_average_neighbors(nodes: List, sample_range: float = 3.0) -> float:
    """
    估算网络的平均邻居数（用于自适应调整）
    
    :param nodes: 节点列表
    :param sample_range: 采样范围（米）
    :return: 平均邻居数
    """
    total_neighbors = 0
    count = 0
    
    for ni in nodes:
        if hasattr(ni, 'is_physical_center') and ni.is_physical_center:
            continue
        
        neighbors = 0
        for nj in nodes:
            if ni == nj:
                continue
            if hasattr(nj, 'is_physical_center') and nj.is_physical_center:
                continue
            
            d = ni.distance_to(nj)
            if d <= sample_range:
                neighbors += 1
        
        total_neighbors += neighbors
        count += 1
    
    return total_neighbors / count if count > 0 else 0.0


# ==================== 兼容接口（与EEOR一致）====================

def eetor_find_path_adaptive(nodes: List,
                              source_node,
                              dest_node,
                              max_hops: int = 5,
                              target_neighbors: int = 6,
                              node_info_manager=None) -> Optional[List]:
    """
    EETOR自适应路径查找接口（兼容EEOR接口）
    
    这是专门为能量传输设计的路由算法，直接替代 eeor_find_path_adaptive
    
    使用方式：
        from routing.energy_transfer_routing import eetor_find_path_adaptive
        # 替换原来的：
        # from routing.EEOR import eeor_find_path_adaptive
        
        path = eetor_find_path_adaptive(nodes, source_node, dest_node, max_hops=5, target_neighbors=6)
    
    :param nodes: 节点列表
    :param source_node: 源节点
    :param dest_node: 目标节点
    :param max_hops: 最大跳数
    :param target_neighbors: 目标邻居数（用于兼容，实际使用更大的通信范围）
    :return: 路径节点列表，如果失败返回None
    """
    return find_energy_transfer_path_adaptive(
        nodes=nodes,
        source_node=source_node,
        dest_node=dest_node,
        max_hops=max_hops,
        target_neighbors=target_neighbors,
        energy_state_aware=True,  # 默认启用能量状态感知
        node_info_manager=node_info_manager
    )


# ==================== 辅助函数 =====================

def get_node_distance(u_id: int, v_id: int, neighbor_map: Dict[int, List[Tuple]]) -> Optional[float]:
    """从邻居图中获取节点间距离"""
    for nid, dist, eta in neighbor_map.get(u_id, []):
        if nid == v_id:
            return dist
    return None

