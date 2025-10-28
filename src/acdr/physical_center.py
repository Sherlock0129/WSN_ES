# src/acdr/physical_center.py
# -*- coding: utf-8 -*-
"""
物理中心节点信息管理模块

职责：
1. 管理物理中心节点位置（网络几何中心）
2. 维护全网节点信息表（三级缓存：内存 → 历史 → 归档）
3. 提供节点信息查询和统计功能
4. 支持可视化数据导出

说明：
本模块用于管理物理中心节点（ID=0）收集到的网络节点信息。
物理中心节点是一个真实的SensorNode，位于网络几何中心，
负责汇总和存储所有节点的状态信息（能量、位置、新鲜度等）。

设计原则：
- 单一职责：只负责信息表管理和位置计算
- 高性能：三级缓存架构（L1:字典 L2:deque L3:CSV）
- 可扩展：支持不同的位置更新策略
"""

from __future__ import annotations
import math
import csv
import os
from collections import deque
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.SensorNode import SensorNode


class VirtualCenter:
    """
    物理中心节点信息管理类
    
    功能：
    - 计算和维护物理中心节点的位置（通常为网络几何中心）
    - 管理全网节点信息表（能量、新鲜度、位置等）
    - 提供三级缓存架构的高效信息存储和查询
    
    注意：
    本类不持有物理中心的SensorNode实例，仅管理信息表和位置计算。
    物理中心节点实体由Network类管理（nodes列表中ID=0的节点）。
    """
    
    def __init__(self, 
                 initial_position: Tuple[float, float] = (0.0, 0.0),
                 update_strategy: str = "geometric_center",
                 enable_logging: bool = True,
                 history_size: int = 1000,
                 archive_path: Optional[str] = None):
        """
        初始化物理中心信息管理器
        
        :param initial_position: 初始位置 (x, y)
        :param update_strategy: 位置更新策略
            - "geometric_center": 几何中心（所有节点位置的算术平均）
            - "energy_weighted_center": 能量加权中心（未来扩展）
            - "fixed": 固定位置（未来扩展）
        :param enable_logging: 是否启用日志输出
        :param history_size: 近期历史缓存大小
        :param archive_path: 归档文件路径（如果为None，需后续设置）
        """
        self.position: Tuple[float, float] = initial_position
        self.update_strategy = update_strategy
        self.enable_logging = enable_logging
        self.last_update_time: int = -1  # 最后更新时间
        
        # ========== 节点信息表（三级缓存） ==========
        # L1: 最新状态表（字典，O(1)查询）
        # 结构: {node_id: {
        #   'energy': float,           # 节点能量
        #   'freshness': int,          # 信息新鲜度（采集时间）
        #   'arrival_time': int,       # 到达物理中心的时间
        #   'position': (x, y),        # 节点位置
        #   'is_solar': bool,          # 是否有太阳能
        #   'cluster_id': int,         # 所属簇ID
        #   'data_size': int,          # 数据包大小
        #   'age': int                 # 信息年龄（arrival_time - freshness）
        # }}
        self.latest_info: Dict[int, Dict] = {}
        
        # L2: 近期历史（deque，固定大小，FIFO）
        # 每个元素: (timestamp, node_id, energy, freshness, position, 
        #           is_solar, cluster_id, data_size)
        self.recent_history: deque = deque(maxlen=history_size)
        
        # L3: 长期归档（CSV文件）
        self.archive_path = archive_path
        self.archive_buffer = []  # 写入缓冲区
        self.archive_batch_size = 100  # 批量写入大小
        self.archive_initialized = False  # 归档文件是否已初始化
        
        self._log(f"[PhysicalCenter] 初始化信息管理器，位置: {self.position}, 策略: {update_strategy}")
        self._log(f"[PhysicalCenter] 节点信息表已启用，历史缓存: {history_size} 条")
    
    def _log(self, message: str):
        """内部日志方法"""
        if self.enable_logging:
            print(message)
    
    # ==================== 位置管理 ====================
    
    def update_position(self, nodes: List[SensorNode], current_time: int = None) -> Tuple[float, float]:
        """
        更新物理中心位置
        
        注意：此方法只计算位置，不更新实际的物理中心节点。
        实际的物理中心节点位置由Network类的update_physical_center_position()方法更新。
        
        :param nodes: 网络中的所有节点（通常应传入普通节点，不包括物理中心）
        :param current_time: 当前时间步（可选，用于记录）
        :return: 更新后的位置 (x, y)
        """
        if not nodes:
            self._log("[PhysicalCenter] 警告：没有节点，保持当前位置")
            return self.position
        
        if self.update_strategy == "geometric_center":
            self.position = self._calculate_geometric_center(nodes)
        elif self.update_strategy == "energy_weighted_center":
            self.position = self._calculate_energy_weighted_center(nodes)
        elif self.update_strategy == "fixed":
            pass  # 保持不变
        else:
            self._log(f"[PhysicalCenter] 警告：未知策略 '{self.update_strategy}'，使用几何中心")
            self.position = self._calculate_geometric_center(nodes)
        
        if current_time is not None:
            self.last_update_time = current_time
        
        self._log(f"[PhysicalCenter] 位置更新为: ({self.position[0]:.3f}, {self.position[1]:.3f})")
        return self.position
    
    def _calculate_geometric_center(self, nodes: List[SensorNode]) -> Tuple[float, float]:
        """计算几何中心（所有节点位置的算术平均）"""
        xs = [n.position[0] for n in nodes]
        ys = [n.position[1] for n in nodes]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    
    def _calculate_energy_weighted_center(self, nodes: List[SensorNode]) -> Tuple[float, float]:
        """
        计算能量加权中心（能量越高的节点权重越大）
        
        未来扩展：可以让物理中心偏向能量充足的区域
        """
        total_energy = sum(n.current_energy for n in nodes)
        if total_energy == 0:
            return self._calculate_geometric_center(nodes)
        
        weighted_x = sum(n.position[0] * n.current_energy for n in nodes) / total_energy
        weighted_y = sum(n.position[1] * n.current_energy for n in nodes) / total_energy
        return (weighted_x, weighted_y)
    
    def get_position(self) -> Tuple[float, float]:
        """获取当前计算的物理中心位置"""
        return self.position
    
    def set_position(self, x: float, y: float):
        """手动设置位置（用于固定位置模式）"""
        self.position = (x, y)
        self._log(f"[PhysicalCenter] 手动设置位置: ({x:.3f}, {y:.3f})")
    
    # ==================== 距离计算 ====================
    
    def distance_to(self, position: Tuple[float, float]) -> float:
        """
        计算指定位置到物理中心的欧式距离
        
        :param position: 目标位置 (x, y)
        :return: 距离值
        """
        dx = position[0] - self.position[0]
        dy = position[1] - self.position[1]
        return math.sqrt(dx * dx + dy * dy)
    
    def distance_to_node(self, node: SensorNode) -> float:
        """计算节点到物理中心的距离"""
        return self.distance_to(node.position)
    
    # ==================== 节点信息表管理 ====================
    
    def update_node_info(self, node_id: int, energy: float, freshness: int, 
                        arrival_time: int, position: Tuple[float, float] = None,
                        is_solar: bool = None, cluster_id: int = None, 
                        data_size: int = None):
        """
        更新节点信息到物理中心的三级缓存表
        
        :param node_id: 节点ID
        :param energy: 节点能量（J）
        :param freshness: 信息新鲜度（信息采集时刻）
        :param arrival_time: 到达物理中心的时间
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
        
        self._log(f"[PhysicalCenter] 更新节点信息: Node {node_id}, "
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
        
        self._log(f"[PhysicalCenter] 批量更新 {len(nodes)} 个节点信息")
    
    def initialize_node_info(self, nodes: List[SensorNode], initial_time: int = 0):
        """
        初始化节点信息表（在网络创建后立即调用）
        
        填充所有节点的初始状态信息到物理中心
        
        :param nodes: 节点列表
        :param initial_time: 初始时间（默认为0）
        """
        self._log(f"[PhysicalCenter] 开始初始化节点信息表，节点数: {len(nodes)}")
        
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
        
        self._log(f"[PhysicalCenter] 节点信息表初始化完成，记录数: {len(self.latest_info)}")
        
        # 输出统计信息
        stats = self.get_statistics()
        self._log(f"[PhysicalCenter] 初始统计 - 平均能量: {stats['avg_energy']:.1f}J, "
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
            self._log("[PhysicalCenter] 警告：未设置归档路径，跳过归档")
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
            self._log(f"[PhysicalCenter] 归档文件已初始化: {self.archive_path}")
        
        # 追加写入数据
        with open(self.archive_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'arrival_time', 'node_id', 'energy', 'freshness', 'age',
                'position_x', 'position_y', 'is_solar', 'cluster_id', 'data_size'
            ])
            writer.writerows(self.archive_buffer)
        
        self._log(f"[PhysicalCenter] 归档 {len(self.archive_buffer)} 条记录到 {self.archive_path}")
        self.archive_buffer.clear()
    
    def force_flush_archive(self):
        """
        强制刷新归档缓冲区（在模拟结束时调用）
        """
        if self.archive_buffer:
            self._flush_archive()
            self._log("[PhysicalCenter] 强制刷新归档完成")
    
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
        
        :return: 包含位置、节点统计等信息的字典
        """
        return {
            'position': self.position,
            'x': self.position[0],
            'y': self.position[1],
            'last_update_time': self.last_update_time,
            'update_strategy': self.update_strategy,
            'node_count': len(self.latest_info),
            'statistics': self.get_statistics()
        }
    
    # ==================== 扩展接口 ====================
    
    def reset(self):
        """重置状态"""
        self.last_update_time = -1
        
        # 清空节点信息表
        self.latest_info.clear()
        self.recent_history.clear()
        
        # 强制刷新归档
        self.force_flush_archive()
        
        self._log("[PhysicalCenter] 状态已重置，节点信息表已清空")
    
    def __repr__(self):
        return f"PhysicalCenter(pos={self.position}, nodes={len(self.latest_info)})"
    
    def __str__(self):
        return f"物理中心 @ ({self.position[0]:.2f}, {self.position[1]:.2f})"


# ==================== 工厂方法 ====================

def create_virtual_center(strategy: str = "geometric_center", **kwargs) -> VirtualCenter:
    """
    工厂方法：创建物理中心信息管理器实例
    
    :param strategy: 位置更新策略
    :param kwargs: 其他初始化参数
    :return: VirtualCenter 实例
    """
    return VirtualCenter(update_strategy=strategy, **kwargs)
