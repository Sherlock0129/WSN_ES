# src/acdr/physical_center.py
# -*- coding: utf-8 -*-
"""
节点信息管理模块

职责：
1. 维护全网节点信息表（三级缓存：内存 → 历史 → 归档）
2. 提供节点信息查询和统计功能
3. 支持可视化数据导出
4. 管理物理中心的位置信息（固定不变）

说明：
本模块为物理中心节点（ID=0）提供信息表管理功能。
物理中心节点是一个真实的SensorNode（由Network类管理），
位于网络几何中心，负责汇总和存储所有节点的状态信息（能量、位置、新鲜度等）。

设计原则：
- 单一职责：只负责信息表管理，不持有节点实体
- 高性能：三级缓存架构（L1:字典 L2:deque L3:CSV）
- 轻量级：与物理中心节点实体（SensorNode）解耦
"""

from __future__ import annotations
import math
import csv
import os
from collections import deque
from typing import List, Tuple, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.SensorNode import SensorNode
    from scheduling.info_node import InfoNode


class NodeInfoManager:
    """
    节点信息管理类
    
    功能：
    - 管理物理中心节点的位置信息（固定不变）
    - 管理全网节点信息表（能量、新鲜度、位置等）
    - 提供三级缓存架构的高效信息存储和查询
    
    注意：
    本类不持有物理中心的SensorNode实例，仅管理信息表。
    物理中心节点实体由Network类管理（nodes列表中ID=0的节点）。
    """
    
    def __init__(self, 
                 initial_position: Tuple[float, float] = (0.0, 0.0),
                 enable_logging: bool = True,
                 history_size: int = 1000,
                 archive_path: Optional[str] = None):
        """
        初始化物理中心信息管理器
        
        :param initial_position: 初始位置 (x, y) - 固定不变
        :param enable_logging: 是否启用日志输出
        :param history_size: 近期历史缓存大小
        :param archive_path: 归档文件路径（如果为None，需后续设置）
        """
        self.position: Tuple[float, float] = initial_position
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
        
        # ========== InfoNode常驻缓存 ==========
        # 存储所有节点的InfoNode实例，key为node_id
        # 在initialize_node_info时创建，在update_node_info时更新
        self.info_nodes: Dict[int, 'InfoNode'] = {}
        
        # 能量传输参数缓存（从真实节点复制一次）
        self.energy_params: Dict[int, Dict] = {}
        
        self._log(f"[NodeInfoManager] 初始化信息管理器，固定位置: {self.position}")
        self._log(f"[NodeInfoManager] 节点信息表已启用，历史缓存: {history_size} 条")
    
    def _log(self, message: str):
        """内部日志方法"""
        if self.enable_logging:
            print(message)
    
    # ==================== 位置管理 ====================
    
    def get_position(self) -> Tuple[float, float]:
        """获取物理中心位置（固定不变）"""
        return self.position
    
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
        
        # 【关键】从节点信息表同步更新InfoNode（如果已创建）
        # 注意：这里使用的是参数energy和position，这些数据来自节点上报的信息
        # InfoNode不直接访问真实节点，只从节点信息表获取数据
        # 数据流向：真实节点 → 上报 → 节点信息表 → InfoNode
        if node_id in self.info_nodes:
            info_node = self.info_nodes[node_id]
            info_node.current_energy = energy  # 从上报的能量值更新
            if position is not None:
                info_node.position = list(position)  # 从上报的位置更新
        
        self._log(f"[NodeInfoManager] 更新节点信息: Node {node_id}, "
                 f"能量={energy:.1f}J, 新鲜度={freshness}, 到达时间={arrival_time}, "
                 f"信息年龄={info['age']}分钟")
    
    def batch_update_node_info(self, nodes: List[SensorNode], current_time: int,
                               cluster_mapping: Dict[int, int] = None,
                               data_sizes: Dict[int, int] = None):
        """
        批量更新多个节点的信息（模拟节点上报信息到物理中心）
        
        这个方法模拟真实的信息上报过程：
        1. 节点将当前状态信息发送给物理中心
        2. 物理中心将信息存储到节点信息表（L1/L2/L3）
        3. InfoNode基于节点信息表同步更新（而非直接读取真实节点）
        
        注意：这里读取node.current_energy是模拟"节点上报当前能量"的过程，
        而不是物理中心直接访问节点（上帝视角）。上报后，InfoNode只能
        从节点信息表获取数据，保持了分布式系统的真实性。
        
        :param nodes: 节点列表（用于模拟上报）
        :param current_time: 当前时间
        :param cluster_mapping: 簇映射 {node_id: cluster_id}
        :param data_sizes: 数据大小映射 {node_id: data_size}
        """
        for node in nodes:
            cluster_id = cluster_mapping.get(node.node_id) if cluster_mapping else None
            data_size = data_sizes.get(node.node_id) if data_sizes else None
            
            # 模拟节点上报信息到物理中心
            # 这里读取node.current_energy是模拟"节点发送自己的能量信息"
            self.update_node_info(
                node_id=node.node_id,
                energy=node.current_energy,  # 节点上报的能量值
                freshness=current_time,  # 信息采集时刻
                arrival_time=current_time,  # 信息到达时刻
                position=tuple(node.position),  # 节点上报的位置
                is_solar=node.has_solar,  # 节点属性
                cluster_id=cluster_id,
                data_size=data_size
            )
        
        self._log(f"[NodeInfoManager] 批量更新 {len(nodes)} 个节点信息")
    
    def sync_info_nodes_from_table(self):
        """
        从节点信息表同步数据到InfoNode
        
        这个方法展示了正确的数据流向：
        节点信息表（latest_info）→ InfoNode
        
        InfoNode不直接访问真实节点，只从节点信息表获取数据。
        这保持了分布式系统的真实性，避免了"上帝视角"。
        
        注意：update_node_info() 已经在更新表的同时自动同步InfoNode，
        所以这个方法通常不需要单独调用，除非需要显式地重新同步。
        """
        for node_id, info in self.latest_info.items():
            if node_id in self.info_nodes:
                info_node = self.info_nodes[node_id]
                # 从节点信息表读取数据
                info_node.current_energy = info['energy']
                if info['position'] is not None:
                    info_node.position = list(info['position'])
        
        self._log(f"[NodeInfoManager] 从节点信息表同步 {len(self.info_nodes)} 个InfoNode")
    
    def initialize_node_info(self, nodes: List[SensorNode], initial_time: int = 0):
        """
        初始化节点信息表（在网络创建后立即调用）
        
        填充所有节点的初始状态信息到物理中心，并创建InfoNode实例
        
        :param nodes: 节点列表
        :param initial_time: 初始时间（默认为0）
        """
        self._log(f"[NodeInfoManager] 开始初始化节点信息表，节点数: {len(nodes)}")
        
        # 先缓存所有节点的能量传输参数
        self._cache_energy_params(nodes)
        
        # 更新信息表并创建InfoNode
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
        
        # 创建所有InfoNode实例
        self._create_info_nodes()
        
        self._log(f"[NodeInfoManager] 节点信息表初始化完成，记录数: {len(self.latest_info)}")
        self._log(f"[NodeInfoManager] InfoNode实例创建完成，数量: {len(self.info_nodes)}")
        
        # 输出统计信息
        stats = self.get_statistics()
        self._log(f"[NodeInfoManager] 初始统计 - 平均能量: {stats['avg_energy']:.1f}J, "
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
            self._log("[NodeInfoManager] 警告：未设置归档路径，跳过归档")
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
            self._log(f"[NodeInfoManager] 归档文件已初始化: {self.archive_path}")
        
        # 追加写入数据
        with open(self.archive_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'arrival_time', 'node_id', 'energy', 'freshness', 'age',
                'position_x', 'position_y', 'is_solar', 'cluster_id', 'data_size'
            ])
            writer.writerows(self.archive_buffer)
        
        self._log(f"[NodeInfoManager] 归档 {len(self.archive_buffer)} 条记录到 {self.archive_path}")
        self.archive_buffer.clear()
    
    def force_flush_archive(self):
        """
        强制刷新归档缓冲区（在模拟结束时调用）
        """
        if self.archive_buffer:
            self._flush_archive()
            self._log("[NodeInfoManager] 强制刷新归档完成")
    
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
            'node_count': len(self.latest_info),
            'statistics': self.get_statistics()
        }
    
    # ==================== InfoNode管理 ====================
    
    def _cache_energy_params(self, nodes: List[SensorNode]):
        """
        缓存所有节点的能量传输参数（内部方法，在初始化时调用）
        
        :param nodes: 节点列表
        """
        for node in nodes:
            self.energy_params[node.node_id] = {
                'energy_char': getattr(node, 'energy_char', 300.0),
                'energy_elec': getattr(node, 'energy_elec', 1e-4),
                'epsilon_amp': getattr(node, 'epsilon_amp', 1e-5),
                'bit_rate': getattr(node, 'bit_rate', 1000000.0),
                'path_loss_exponent': getattr(node, 'path_loss_exponent', 2.0),
                'sensor_energy': getattr(node, 'sensor_energy', 0.1),
            }
        self._log(f"[NodeInfoManager] 缓存 {len(self.energy_params)} 个节点的能量传输参数")
    
    def _create_info_nodes(self):
        """
        创建所有InfoNode实例（内部方法，在初始化时调用）
        
        基于latest_info和energy_params创建InfoNode实例，
        存储在self.info_nodes字典中。
        """
        from scheduling.info_node import InfoNode
        
        for node_id, info in self.latest_info.items():
            # 获取该节点的能量传输参数
            params = self.energy_params.get(node_id, {})
            
            info_node = InfoNode(
                node_id=node_id,
                energy=info['energy'],
                position=info['position'],
                is_solar=info.get('is_solar', False),
                is_physical_center=(node_id == 0),  # 物理中心ID固定为0
                **params
            )
            self.info_nodes[node_id] = info_node
        
        self._log(f"[NodeInfoManager] 创建 {len(self.info_nodes)} 个InfoNode实例")
    
    def get_info_nodes(self) -> List['InfoNode']:
        """
        获取所有InfoNode实例的列表
        
        这些InfoNode实例会随着信息表的更新而自动同步更新。
        调度器和路由算法可以直接使用这些实例。
        
        :return: InfoNode列表
        """
        return list(self.info_nodes.values())
    
    # ==================== 扩展接口 ====================
    
    def reset(self):
        """重置状态"""
        self.last_update_time = -1
        
        # 清空节点信息表
        self.latest_info.clear()
        self.recent_history.clear()
        
        # 强制刷新归档
        self.force_flush_archive()
        
        self._log("[NodeInfoManager] 状态已重置，节点信息表已清空")
    
    def __repr__(self):
        return f"NodeInfoManager(pos={self.position}, nodes={len(self.latest_info)})"
    
    def __str__(self):
        return f"节点信息管理器 @ ({self.position[0]:.2f}, {self.position[1]:.2f})"


# ==================== 工厂方法 ====================

def create_node_info_manager(**kwargs) -> NodeInfoManager:
    """
    工厂方法：创建节点信息管理器实例
    
    :param kwargs: 初始化参数
    :return: NodeInfoManager 实例
    """
    return NodeInfoManager(**kwargs)


# ==================== 向后兼容 ====================

# 为了兼容旧代码，保留别名
VirtualCenter = NodeInfoManager
create_virtual_center = create_node_info_manager
