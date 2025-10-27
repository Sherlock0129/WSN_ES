# src/acdr/virtual_center.py
# -*- coding: utf-8 -*-
"""
虚拟中心（Virtual Center）模块

职责：
1. 管理虚拟中心位置（网络几何中心）
2. 提供锚点选择功能（最靠近虚拟中心的真实节点）
3. 规划簇头到虚拟中心的路径
4. 计算到虚拟中心的通信能耗
5. 支持可视化

设计原则：
- 单一职责：只负责虚拟中心相关逻辑
- 开放封闭：易于扩展（如支持多个虚拟中心、移动虚拟中心等）
- 依赖倒置：通过接口与其他模块交互
"""

from __future__ import annotations
import math
import csv
import os
from collections import deque
from typing import List, Tuple, Dict, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from core.SensorNode import SensorNode


class VirtualCenter:
    """
    虚拟中心类 - 表示WSN中的逻辑汇聚点（Sink）
    
    特性：
    - 假设拥有无限能量
    - 位置动态更新（基于网络节点分布）
    - 不对应真实节点，但通过锚点接收数据
    """
    
    def __init__(self, 
                 initial_position: Tuple[float, float] = (0.0, 0.0),
                 update_strategy: str = "geometric_center",
                 enable_logging: bool = True,
                 history_size: int = 1000,
                 archive_path: Optional[str] = None):
        """
        初始化虚拟中心
        
        :param initial_position: 初始位置 (x, y)
        :param update_strategy: 位置更新策略
            - "geometric_center": 几何中心（所有节点位置的算术平均）
            - "energy_weighted_center": 能量加权中心（未来扩展）
            - "fixed": 固定位置（未来扩展）
        :param enable_logging: 是否启用日志输出
        :param history_size: 近期历史缓存大小
        :param archive_path: 归档文件路径（如果为None，使用默认路径）
        """
        self.position: Tuple[float, float] = initial_position
        self.update_strategy = update_strategy
        self.enable_logging = enable_logging
        
        # 运行时状态
        self.anchor_node: Optional[SensorNode] = None  # 当前锚点节点
        self.last_update_time: int = -1  # 最后更新时间
        
        # ========== 节点信息表（三级缓存） ==========
        # L1: 最新状态表（字典，O(1)查询）
        # 结构: {node_id: {
        #   'energy': float,           # 节点能量
        #   'freshness': int,          # 信息新鲜度（采集时间）
        #   'arrival_time': int,       # 到达虚拟中心的时间
        #   'position': (x, y),        # 节点位置
        #   'is_solar': bool,          # 是否有太阳能
        #   'cluster_id': int,         # 所属簇ID
        #   'data_size': int           # 数据包大小
        # }}
        self.latest_info: Dict[int, Dict] = {}
        
        # L2: 近期历史（deque，固定大小）
        # 每个元素: (timestamp, node_id, energy, freshness, ...)
        self.recent_history: deque = deque(maxlen=history_size)
        
        # L3: 长期归档配置
        self.archive_path = archive_path
        self.archive_buffer = []  # 缓冲区，积累一定数量后批量写入
        self.archive_batch_size = 100  # 批量写入大小
        self.archive_initialized = False  # 归档文件是否已初始化
        
        self._log(f"[VirtualCenter] 初始化虚拟中心，位置: {self.position}, 策略: {update_strategy}")
        self._log(f"[VirtualCenter] 节点信息表已启用，历史缓存: {history_size} 条")
    
    def _log(self, message: str):
        """内部日志方法"""
        if self.enable_logging:
            print(message)
    
    # ==================== 位置管理 ====================
    
    def update_position(self, nodes: List[SensorNode], current_time: int = None) -> Tuple[float, float]:
        """
        更新虚拟中心位置
        
        :param nodes: 网络中的所有节点
        :param current_time: 当前时间步（可选，用于记录）
        :return: 更新后的位置 (x, y)
        """
        if not nodes:
            self._log("[VirtualCenter] 警告：没有节点，保持当前位置")
            return self.position
        
        if self.update_strategy == "geometric_center":
            self.position = self._calculate_geometric_center(nodes)
        elif self.update_strategy == "energy_weighted_center":
            self.position = self._calculate_energy_weighted_center(nodes)
        elif self.update_strategy == "fixed":
            pass  # 保持不变
        else:
            self._log(f"[VirtualCenter] 警告：未知策略 '{self.update_strategy}'，使用几何中心")
            self.position = self._calculate_geometric_center(nodes)
        
        if current_time is not None:
            self.last_update_time = current_time
        
        self._log(f"[VirtualCenter] 位置更新为: ({self.position[0]:.3f}, {self.position[1]:.3f})")
        return self.position
    
    def _calculate_geometric_center(self, nodes: List[SensorNode]) -> Tuple[float, float]:
        """计算几何中心（所有节点位置的算术平均）"""
        xs = [n.position[0] for n in nodes]
        ys = [n.position[1] for n in nodes]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def _calculate_energy_weighted_center(self, nodes: List[SensorNode]) -> Tuple[float, float]:
        """
        计算能量加权中心（能量越高的节点权重越大）
        
        未来扩展：可以让虚拟中心偏向能量充足的区域，减少整体通信开销
        """
        total_energy = sum(n.current_energy for n in nodes)
        if total_energy == 0:
            return self._calculate_geometric_center(nodes)
        
        weighted_x = sum(n.position[0] * n.current_energy for n in nodes) / total_energy
        weighted_y = sum(n.position[1] * n.current_energy for n in nodes) / total_energy
        return (weighted_x, weighted_y)
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前虚拟中心位置"""
        return self.position
    
    def set_position(self, x: float, y: float):
        """手动设置虚拟中心位置（用于固定位置模式）"""
        self.position = (x, y)
        self._log(f"[VirtualCenter] 手动设置位置: ({x:.3f}, {y:.3f})")
    
    # ==================== 距离计算 ====================
    
    def distance_to(self, position: Tuple[float, float]) -> float:
        """
        计算指定位置到虚拟中心的欧式距离
        
        :param position: 目标位置 (x, y)
        :return: 距离值
        """
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def distance_to_node(self, node: SensorNode) -> float:
        """计算节点到虚拟中心的距离"""
        return self.distance_to(node.position)
    
    # ==================== 锚点管理 ====================
    
    def find_anchor(self, nodes: List[SensorNode], 
                   min_energy_threshold: float = 0.0,
                   exclude_dead_nodes: bool = True) -> Optional[SensorNode]:
        """
        选择最靠近虚拟中心的真实节点作为锚点（Anchor）
        
        锚点是虚拟中心在物理网络中的"代表"，负责接收并转发到虚拟中心的数据
        
        :param nodes: 候选节点列表
        :param min_energy_threshold: 最小能量阈值（能量过低的节点不能作为锚点）
        :param exclude_dead_nodes: 是否排除死亡节点
        :return: 锚点节点，如果没有合适节点则返回 None
        """
        if not nodes:
            self._log("[VirtualCenter] 警告：没有可用节点作为锚点")
            return None
        
        # 过滤候选节点
        candidates = nodes
        if exclude_dead_nodes:
            candidates = [n for n in candidates if n.current_energy > 0]
        if min_energy_threshold > 0:
            candidates = [n for n in candidates if n.current_energy >= min_energy_threshold]
        
        if not candidates:
            self._log("[VirtualCenter] 警告：没有满足条件的锚点候选节点")
            return None
        
        # 选择最近的节点
        self.anchor_node = min(candidates, key=lambda n: self.distance_to_node(n))
        self._log(f"[VirtualCenter] 锚点选择: Node {self.anchor_node.node_id}, "
                 f"距离虚拟中心 {self.distance_to_node(self.anchor_node):.3f}m, "
                 f"能量 {self.anchor_node.current_energy:.1f}J")
        
        return self.anchor_node
    
    def get_anchor(self) -> Optional[SensorNode]:
        """获取当前锚点节点"""
        return self.anchor_node
    
    # ==================== 路径规划 ====================
    
    def plan_paths_from_cluster_heads(
        self,
        cluster_heads: List[SensorNode],
        all_nodes: List[SensorNode],
        routing_function: Optional[Callable] = None,
        max_hops: int = 5,
        enable_direct_transmission: bool = True,
        direct_transmission_threshold: float = 0.1
    ) -> Dict[int, List[SensorNode]]:
        """
        为所有簇头规划到虚拟中心的上行路径
        
        策略：
        1. 选择锚点（最靠近虚拟中心的节点）
        2. 为每个簇头规划路径：CH → ... → Anchor → VirtualCenter
        3. 支持直接传输优化（簇头直接到虚拟中心）
        
        :param cluster_heads: 簇头节点列表
        :param all_nodes: 所有网络节点
        :param routing_function: 路由函数 func(nodes, src, dst, max_hops, t) -> path
        :param max_hops: 最大跳数限制
        :param enable_direct_transmission: 是否启用直接传输优化
        :param direct_transmission_threshold: 直接传输距离阈值（相对于网络范围）
        :return: 路径字典 {ch_id: [ch, node1, ..., anchor]}
        """
        if not cluster_heads:
            self._log("[VirtualCenter] 没有簇头，跳过路径规划")
            return {}
        
        # 选择锚点
        anchor = self.find_anchor(all_nodes)
        if anchor is None:
            self._log("[VirtualCenter] 警告：无法找到锚点，路径规划失败")
            return {}
        
        paths = {}
        id2node = {n.node_id: n for n in all_nodes}
        
        for ch in cluster_heads:
            ch_id = ch.node_id
            
            # 情况1：簇头本身就是锚点
            if ch is anchor:
                paths[ch_id] = [ch]
                self._log(f"[VirtualCenter] CH {ch_id} 即为锚点，只有虚拟跳")
                continue
            
            # 情况2：启用直接传输优化
            if enable_direct_transmission:
                should_direct = self._should_use_direct_transmission(
                    ch, anchor, all_nodes, direct_transmission_threshold
                )
                if should_direct:
                    paths[ch_id] = [ch]
                    self._log(f"[VirtualCenter] CH {ch_id} 使用直接传输到虚拟中心")
                    continue
            
            # 情况3：多跳路径规划
            if routing_function is None:
                self._log(f"[VirtualCenter] 警告：没有提供路由函数，CH {ch_id} 使用直接传输")
                paths[ch_id] = [ch]
            else:
                path = routing_function(all_nodes, ch, anchor, max_hops=max_hops, t=0)
                if path and len(path) >= 2:
                    paths[ch_id] = path
                    self._log(f"[VirtualCenter] CH {ch_id} 多跳路径: {' → '.join([str(n.node_id) for n in path])} → VC")
                else:
                    # 路径规划失败，使用直接传输
                    paths[ch_id] = [ch]
                    self._log(f"[VirtualCenter] 警告：CH {ch_id} 路径规划失败，使用直接传输")
        
        return paths
    
    def _should_use_direct_transmission(
        self,
        ch: SensorNode,
        anchor: SensorNode,
        all_nodes: List[SensorNode],
        threshold: float
    ) -> bool:
        """
        判断簇头是否应该直接传输到虚拟中心
        
        策略：如果簇头到锚点的距离小于网络直径的阈值百分比，则直接传输
        
        :param ch: 簇头节点
        :param anchor: 锚点节点
        :param all_nodes: 所有节点（用于计算网络直径）
        :param threshold: 距离阈值（相对于网络直径的比例）
        :return: True 表示应该直接传输
        """
        # 计算网络直径（所有节点两两距离的最大值）
        max_dist = 0.0
        for i, n1 in enumerate(all_nodes):
            for n2 in all_nodes[i+1:]:
                dist = n1.distance_to(n2)
                if dist > max_dist:
                    max_dist = dist
        
        # 计算簇头到锚点的距离
        ch_to_anchor_dist = ch.distance_to(anchor)
        
        # 判断是否低于阈值
        should_direct = ch_to_anchor_dist <= (max_dist * threshold)
        
        if should_direct:
            self._log(f"[VirtualCenter] CH {ch.node_id}: 距锚点 {ch_to_anchor_dist:.2f}m "
                     f"< 网络直径 {max_dist:.2f}m × {threshold} = {max_dist * threshold:.2f}m, 使用直接传输")
        
        return should_direct
    
    # ==================== 能耗计算 ====================
    
    def calculate_virtual_hop_energy(
        self,
        sender: SensorNode,
        data_size: int,
        tx_rx_ratio: float = 0.5,
        sensor_energy: float = 0.1
    ) -> float:
        """
        计算从真实节点到虚拟中心的"虚拟跳"能耗
        
        特点：
        - 只扣发送端能量（虚拟中心不消耗能量）
        - 使用聚合数据量
        - 基于到虚拟中心的距离
        
        :param sender: 发送节点（通常是锚点或簇头）
        :param data_size: 数据大小（bits）
        :param tx_rx_ratio: 发送/接收能耗分配比例
        :param sensor_energy: 传感器能耗（J）
        :return: 能耗值（J）
        """
        # 计算到虚拟中心的距离
        distance = self.distance_to_node(sender)
        
        # 使用发送节点的能耗模型参数
        E_elec = sender.E_elec
        epsilon_amp = sender.epsilon_amp
        tau = sender.tau
        
        # 发射能耗
        E_tx = E_elec * data_size + epsilon_amp * data_size * (distance ** tau)
        
        # 接收能耗（虚拟的，用于计算总开销）
        E_rx = E_elec * data_size
        
        # 总通信能耗（按比例分配 + 传感器能耗）
        E_total = tx_rx_ratio * (E_tx + E_rx) + sensor_energy
        
        self._log(f"[VirtualCenter] 虚拟跳能耗: Node {sender.node_id} → VC, "
                 f"距离={distance:.2f}m, 数据={data_size}bits, 能耗={E_total:.4f}J")
        
        return E_total
    
    def settle_virtual_hop_energy(
        self,
        sender: SensorNode,
        data_size: int,
        tx_rx_ratio: float = 0.5,
        sensor_energy: float = 0.1
    ) -> float:
        """
        结算虚拟跳能耗（直接扣除发送节点的能量）
        
        :return: 消耗的能量值
        """
        energy_cost = self.calculate_virtual_hop_energy(sender, data_size, tx_rx_ratio, sensor_energy)
        sender.current_energy = max(0.0, sender.current_energy - energy_cost)
        
        self._log(f"[VirtualCenter] Node {sender.node_id} 能量扣除 {energy_cost:.2f}J, "
                 f"剩余 {sender.current_energy:.1f}J")
        
        return energy_cost
    
    # ==================== 节点信息表管理 ====================
    
    def update_node_info(self, node_id: int, energy: float, freshness: int, 
                        arrival_time: int, position: Tuple[float, float] = None,
                        is_solar: bool = None, cluster_id: int = None, 
                        data_size: int = None):
        """
        更新节点信息到虚拟中心的三级缓存表
        
        :param node_id: 节点ID
        :param energy: 节点能量（J）
        :param freshness: 信息新鲜度（信息采集时刻）
        :param arrival_time: 到达虚拟中心的时间
        :param position: 节点位置 (x, y)
        :param is_solar: 是否有太阳能
        :param cluster_id: 所属簇ID
        :param data_size: 数据包大小（bits）
        """
        # 构造信息记录
        info = {
            'energy': energy,
            'freshness': freshness,
            'arrival_time': arrival_time,
            'position': position,
            'is_solar': is_solar,
            'cluster_id': cluster_id,
            'data_size': data_size,
            'age': arrival_time - freshness  # 信息年龄（延迟）
        }
        
        # L1: 更新最新状态表
        self.latest_info[node_id] = info
        
        # L2: 添加到近期历史
        history_record = (arrival_time, node_id, energy, freshness, position, 
                         is_solar, cluster_id, data_size)
        self.recent_history.append(history_record)
        
        # L3: 添加到归档缓冲区
        self.archive_buffer.append({
            'arrival_time': arrival_time,
            'node_id': node_id,
            'energy': energy,
            'freshness': freshness,
            'age': info['age'],
            'position_x': position[0] if position else None,
            'position_y': position[1] if position else None,
            'is_solar': is_solar,
            'cluster_id': cluster_id,
            'data_size': data_size
        })
        
        # 批量写入归档
        if len(self.archive_buffer) >= self.archive_batch_size:
            self._flush_archive()
        
        self._log(f"[VirtualCenter] 更新节点信息: Node {node_id}, "
                 f"能量={energy:.1f}J, 新鲜度={freshness}, 到达时间={arrival_time}, "
                 f"信息年龄={info['age']}分钟")
    
    def batch_update_node_info(self, nodes: List[SensorNode], current_time: int,
                               cluster_mapping: Dict[int, int] = None,
                               data_sizes: Dict[int, int] = None):
        """
        批量更新多个节点的信息（通常在ADCR聚类后调用）
        
        :param nodes: 节点列表
        :param current_time: 当前时间
        :param cluster_mapping: 簇映射 {node_id: cluster_id}
        :param data_sizes: 数据大小映射 {node_id: data_size}
        """
        for node in nodes:
            cluster_id = cluster_mapping.get(node.node_id) if cluster_mapping else None
            data_size = data_sizes.get(node.node_id) if data_sizes else None
            
            self.update_node_info(
                node_id=node.node_id,
                energy=node.current_energy,
                freshness=current_time,  # 假设信息是当前时刻采集的
                arrival_time=current_time,  # 批量更新时，采集和到达同时发生
                position=tuple(node.position),
                is_solar=node.has_solar,
                cluster_id=cluster_id,
                data_size=data_size
            )
        
        self._log(f"[VirtualCenter] 批量更新 {len(nodes)} 个节点信息")
    
    def initialize_node_info(self, nodes: List[SensorNode], initial_time: int = 0):
        """
        初始化节点信息表（在网络创建后立即调用）
        
        填充所有节点的初始状态信息到虚拟中心
        
        :param nodes: 节点列表
        :param initial_time: 初始时间（默认为0）
        """
        self._log(f"[VirtualCenter] 开始初始化节点信息表，节点数: {len(nodes)}")
        
        for node in nodes:
            self.update_node_info(
                node_id=node.node_id,
                energy=node.initial_energy,  # 使用初始能量
                freshness=initial_time,
                arrival_time=initial_time,
                position=tuple(node.position),
                is_solar=node.has_solar,
                cluster_id=None,  # 初始时没有簇分配
                data_size=None    # 初始时没有数据传输
            )
        
        self._log(f"[VirtualCenter] 节点信息表初始化完成，记录数: {len(self.latest_info)}")
        
        # 输出统计信息
        stats = self.get_statistics()
        self._log(f"[VirtualCenter] 初始统计 - 平均能量: {stats['avg_energy']:.1f}J, "
                 f"太阳能节点: {stats['solar_nodes']}/{stats['total_nodes']}")
    
    def get_node_info(self, node_id: int) -> Optional[Dict]:
        """
        获取节点的最新信息（L1查询）
        
        :param node_id: 节点ID
        :return: 节点信息字典，如果不存在返回None
        """
        return self.latest_info.get(node_id)
    
    def get_all_nodes_info(self) -> Dict[int, Dict]:
        """
        获取所有节点的最新信息
        
        :return: {node_id: info_dict}
        """
        return self.latest_info.copy()
    
    def get_stale_nodes(self, current_time: int, staleness_threshold: int = 30) -> List[int]:
        """
        获取信息过期的节点列表
        
        :param current_time: 当前时间
        :param staleness_threshold: 过期阈值（分钟）
        :return: 过期节点ID列表
        """
        stale_nodes = []
        for node_id, info in self.latest_info.items():
            age = current_time - info['freshness']
            if age > staleness_threshold:
                stale_nodes.append(node_id)
        
        return stale_nodes
    
    def get_low_energy_nodes(self, energy_threshold: float) -> List[Tuple[int, float]]:
        """
        获取低能量节点列表
        
        :param energy_threshold: 能量阈值
        :return: [(node_id, energy), ...] 按能量升序排序
        """
        low_energy = [(nid, info['energy']) 
                      for nid, info in self.latest_info.items() 
                      if info['energy'] < energy_threshold]
        return sorted(low_energy, key=lambda x: x[1])
    
    def get_recent_history(self, limit: int = 100) -> List:
        """
        获取近期历史记录（L2查询）
        
        :param limit: 返回的记录数量
        :return: 历史记录列表
        """
        if limit >= len(self.recent_history):
            return list(self.recent_history)
        else:
            return list(self.recent_history)[-limit:]
    
    def _flush_archive(self):
        """
        将缓冲区中的数据批量写入归档文件（L3持久化）
        """
        if not self.archive_buffer:
            return
        
        if self.archive_path is None:
            self._log("[VirtualCenter] 警告：未设置归档路径，跳过归档")
            self.archive_buffer.clear()
            return
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.archive_path), exist_ok=True)
        
        # 初始化CSV文件（写入表头）
        if not self.archive_initialized:
            with open(self.archive_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'arrival_time', 'node_id', 'energy', 'freshness', 'age',
                    'position_x', 'position_y', 'is_solar', 'cluster_id', 'data_size'
                ])
                writer.writeheader()
            self.archive_initialized = True
            self._log(f"[VirtualCenter] 归档文件已初始化: {self.archive_path}")
        
        # 追加写入数据
        with open(self.archive_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'arrival_time', 'node_id', 'energy', 'freshness', 'age',
                'position_x', 'position_y', 'is_solar', 'cluster_id', 'data_size'
            ])
            writer.writerows(self.archive_buffer)
        
        self._log(f"[VirtualCenter] 归档 {len(self.archive_buffer)} 条记录到 {self.archive_path}")
        self.archive_buffer.clear()
    
    def force_flush_archive(self):
        """
        强制刷新归档缓冲区（在模拟结束时调用）
        """
        if self.archive_buffer:
            self._flush_archive()
            self._log("[VirtualCenter] 强制刷新归档完成")
    
    def get_statistics(self) -> Dict:
        """
        获取节点信息表的统计信息
        
        :return: 统计信息字典
        """
        if not self.latest_info:
            return {
                'total_nodes': 0,
                'avg_energy': 0,
                'min_energy': 0,
                'max_energy': 0,
                'avg_age': 0,
                'max_age': 0,
                'solar_nodes': 0,
                'non_solar_nodes': 0
            }
        
        energies = [info['energy'] for info in self.latest_info.values()]
        ages = [info['age'] for info in self.latest_info.values()]
        solar_count = sum(1 for info in self.latest_info.values() 
                         if info.get('is_solar', False))
        
        return {
            'total_nodes': len(self.latest_info),
            'avg_energy': sum(energies) / len(energies),
            'min_energy': min(energies),
            'max_energy': max(energies),
            'avg_age': sum(ages) / len(ages) if ages else 0,
            'max_age': max(ages) if ages else 0,
            'solar_nodes': solar_count,
            'non_solar_nodes': len(self.latest_info) - solar_count,
            'history_records': len(self.recent_history),
            'archive_buffer_size': len(self.archive_buffer)
        }
    
    # ==================== 可视化支持 ====================
    
    def get_visualization_data(self) -> Dict:
        """
        获取用于可视化的数据
        
        :return: 包含位置、锚点等信息的字典
        """
        return {
            'position': self.position,
            'x': self.position[0],
            'y': self.position[1],
            'anchor_id': self.anchor_node.node_id if self.anchor_node else None,
            'last_update_time': self.last_update_time,
            'update_strategy': self.update_strategy,
            'node_count': len(self.latest_info),
            'statistics': self.get_statistics()
        }
    
    # ==================== 扩展接口 ====================
    
    def reset(self):
        """重置虚拟中心状态"""
        self.anchor_node = None
        self.last_update_time = -1
        
        # 清空节点信息表
        self.latest_info.clear()
        self.recent_history.clear()
        
        # 强制刷新归档
        self.force_flush_archive()
        
        self._log("[VirtualCenter] 状态已重置，节点信息表已清空")
    
    def __repr__(self):
        return f"VirtualCenter(pos={self.position}, anchor={self.anchor_node.node_id if self.anchor_node else None})"
    
    def __str__(self):
        return f"虚拟中心 @ ({self.position[0]:.2f}, {self.position[1]:.2f})"


# ==================== 工厂方法 ====================

def create_virtual_center(strategy: str = "geometric_center", **kwargs) -> VirtualCenter:
    """
    工厂方法：创建虚拟中心实例
    
    :param strategy: 位置更新策略
    :param kwargs: 其他初始化参数
    :return: VirtualCenter 实例
    """
    return VirtualCenter(update_strategy=strategy, **kwargs)

