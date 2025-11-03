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
                 archive_path: Optional[str] = None,
                 # 能量估算参数
                 enable_energy_estimation: bool = True,
                 decay_rate_default: float = 5.0,
                 use_solar_model: bool = True):
        """
        初始化物理中心信息管理器
        
        :param initial_position: 初始位置 (x, y) - 固定不变
        :param enable_logging: 是否启用日志输出
        :param history_size: 近期历史缓存大小
        :param archive_path: 归档文件路径（如果为None，需后续设置）
        :param enable_energy_estimation: 是否启用能量估算（未上报节点）
        :param decay_rate_default: 默认能量衰减率（J/分钟）
        :param use_solar_model: 是否使用太阳能模型进行估算
        """
        self.position: Tuple[float, float] = initial_position
        self.enable_logging = enable_logging
        self.last_update_time: int = -1  # 最后更新时间
        
        # ========== 节点信息表（三级缓存） ==========
        # L1: 最新状态表（字典，O(1)查询）
        # 结构: {node_id: {
        #   'energy': float,           # 节点能量（可能是估算值）
        #   'record_time': int,        # 信息记录时间（采集时刻）
        #   'arrival_time': int,       # 到达物理中心的时间
        #   'position': (x, y),        # 节点位置
        #   'is_solar': bool,          # 是否有太阳能
        #   'cluster_id': int,         # 所属簇ID
        #   'data_size': int,          # 数据包大小
        #   'aoi': int,                # Age of Information（信息年龄，每分钟+1）
        #   'is_estimated': bool,      # 是否为估算值
        #   't': int,                  # 全局时间戳（统一时钟，从0开始每分钟+1）
        #   # 机会主义信息传递字段
        #   'info_volume': int,         # 累积的信息量（bits），0表示无信息量
        #   'info_waiting_since': int, # 开始等待的时间戳（分钟），-1表示未等待
        #   'info_is_reported': bool,  # 是否已上报（True表示已上报或立即上报模式）
        #   'info_source_nodes': List[int]  # 信息来源节点列表（可选，用于去重优化）
        # }}
        self.latest_info: Dict[int, Dict] = {}
        
        # L2: 近期历史（deque，固定大小，FIFO）
        # 每个元素: (timestamp, node_id, energy, record_time, position, 
        #           is_solar, cluster_id, data_size)
        self.recent_history: deque = deque(maxlen=history_size)
        
        # L3: 长期归档（CSV文件）
        self.archive_path = archive_path
        self.archive_buffer = []  # 写入缓冲区
        self.archive_batch_size = 100  # 批量写入大小
        self.archive_initialized = False  # 归档文件是否已初始化
        
        # ========== 能量估算参数 ==========
        self.enable_energy_estimation = enable_energy_estimation
        self.decay_rate_default = decay_rate_default
        self.use_solar_model = use_solar_model
        
        # ========== InfoNode常驻缓存 ==========
        # 存储所有节点的InfoNode实例，key为node_id
        # 在initialize_node_info时创建，在update_node_info时更新
        self.info_nodes: Dict[int, 'InfoNode'] = {}
        
        # 能量传输参数缓存（从真实节点复制一次）
        self.energy_params: Dict[int, Dict] = {}
        
        # ========== 信息传输能量消耗统计 ==========
        # 按节点统计信息传输的能量消耗
        # 结构: {node_id: {'adcr': float, 'path_collector': float, 'total': float}}
        self.info_transmission_energy: Dict[int, Dict[str, float]] = {}
        # 总体统计
        self.info_transmission_stats: Dict[str, float] = {
            'total_adcr_energy': 0.0,
            'total_path_collector_energy': 0.0,
            'total_energy': 0.0,
            'adcr_transmission_count': 0,  # ADCR传输次数（每次完整的传输，不是跳数）
            'path_collector_transmission_count': 0  # 路径收集器传输次数（每次路径收集）
        }
        
        self._log(f"[NodeInfoManager] 初始化信息管理器，固定位置: {self.position}")
        self._log(f"[NodeInfoManager] 节点信息表已启用，历史缓存: {history_size} 条")
        if self.enable_energy_estimation:
            self._log(f"[NodeInfoManager] 能量估算已启用，衰减率: {self.decay_rate_default} J/min")
    
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
    
    def update_node_info(self, node_id: int, energy: float, record_time: int, 
                        arrival_time: int, position: Tuple[float, float] = None,
                        is_solar: bool = None, cluster_id: int = None, 
                        data_size: int = None):
        """
        更新节点信息到物理中心的三级缓存表
        
        :param node_id: 节点ID
        :param energy: 节点能量（J）
        :param record_time: 信息记录时间（信息采集时刻）
        :param arrival_time: 到达物理中心的时间
        :param position: 节点位置 (x, y)
        :param is_solar: 是否有太阳能
        :param cluster_id: 所属簇ID
        :param data_size: 数据包大小（bits）
        """
        # 保存现有信息量字段（如果存在）
        existing_info_fields = {}
        if node_id in self.latest_info:
            for field in ['info_volume', 'info_waiting_since', 'info_is_reported', 'info_source_nodes']:
                if field in self.latest_info[node_id]:
                    existing_info_fields[field] = self.latest_info[node_id][field]
        
        # 构造信息记录
        info = {
            'energy': energy,
            'record_time': record_time,
            'arrival_time': arrival_time,
            'position': position,
            'is_solar': is_solar,
            'cluster_id': cluster_id,
            'data_size': data_size,
            'aoi': 0,  # Age of Information，刚到达时为0
            'is_estimated': False,  # 上报值不是估算值
            't': arrival_time  # 当前时间戳（信息到达时刻）
        }
        
        # 恢复信息量字段（如果存在）
        if existing_info_fields:
            info.update(existing_info_fields)
        else:
            # 初始化信息量字段（如果不存在）
            info['info_volume'] = 0
            info['info_waiting_since'] = -1
            info['info_is_reported'] = True
            info['info_source_nodes'] = []
        
        # L1: 更新最新状态表
        self.latest_info[node_id] = info
        
        # L2: 添加到近期历史
        history_record = (arrival_time, node_id, energy, record_time, position, 
                         is_solar, cluster_id, data_size)
        self.recent_history.append(history_record)
        
        # L3: 添加到归档缓冲区
        # 将info_source_nodes列表序列化为字符串（CSV不支持列表）
        info_source_nodes_str = ','.join(map(str, info.get('info_source_nodes', []))) if info.get('info_source_nodes') else ''
        
        self.archive_buffer.append({
            'arrival_time': arrival_time,
            'node_id': node_id,
            'energy': energy,
            'record_time': record_time,
            'aoi': info['aoi'],
            't': info['t'],
            'position_x': position[0] if position else None,
            'position_y': position[1] if position else None,
            'is_solar': is_solar,
            'cluster_id': cluster_id,
            'data_size': data_size,
            'info_volume': info.get('info_volume', 0),
            'info_waiting_since': info.get('info_waiting_since', -1),
            'info_is_reported': info.get('info_is_reported', True),
            'info_source_nodes': info_source_nodes_str
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
                record_time=current_time,  # 信息记录时刻
                arrival_time=current_time,  # 信息到达时刻
                position=tuple(node.position),  # 节点上报的位置
                is_solar=node.has_solar,  # 节点属性
                cluster_id=cluster_id,
                data_size=data_size
            )
        
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
    
    # ==================== 能量估算 ====================
    
    def _solar_irradiance(self, t: int, G_max: float) -> float:
        """
        计算太阳辐照度（与SensorNode.solar_irradiance完全相同）
        
        :param t: 时间（分钟）
        :param G_max: 最大辐照度 (W/m^2)
        :return: 太阳辐照度 (W/m^2)
        """
        t = t % 1440  # 标准化为每日周期
        if 360 <= t <= 1080:  # 日出到日落（6:00 AM - 6:00 PM）
            return G_max * math.sin(math.pi * (t - 360) / 720)
        return 0
    
    def _estimate_solar_harvest_single(self, t: int, params: dict) -> float:
        """
        估算单个时刻的太阳能采集（与SensorNode.energy_harvest完全相同）
        
        :param t: 时间（分钟）
        :param params: 节点的太阳能参数
        :return: 采集的能量（J）
        """
        if not params.get('enable_energy_harvesting', True):
            return 0
        
        G_t = self._solar_irradiance(t, params['G_max'])
        harvested_energy = (params['solar_efficiency'] * 
                           params['solar_area'] * 
                           G_t * 
                           params['env_correction_factor'])
        return harvested_energy
    
    def _estimate_energy(self, node_id: int, current_time: int) -> tuple:
        """
        估算节点当前能量（基于物理模型，与SensorNode逻辑完全一致）
        
        模型：E(t) = E(t0) + Σ harvest(t_i) - decay * Δt
        
        :param node_id: 节点ID
        :param current_time: 当前时间（分钟）
        :return: (estimated_energy, time_elapsed)
        """
        if node_id not in self.latest_info:
            return None, 0
        
        info = self.latest_info[node_id]
        last_report_time = info['arrival_time']
        last_report_energy = info['energy']
        time_elapsed = current_time - last_report_time
        
        # 如果是刚上报的（time_elapsed=0），直接返回上报值
        if time_elapsed == 0:
            return last_report_energy, 0
        
        # 获取节点参数
        if node_id in self.energy_params:
            params = self.energy_params[node_id]
            decay_rate = params.get('energy_decay_rate', self.decay_rate_default)
            capacity = params.get('capacity', 3.5)
            voltage = params.get('voltage', 3.7)
        else:
            params = {}
            decay_rate = self.decay_rate_default
            capacity = 3.5
            voltage = 3.7
        
        # 1. 计算自然衰减
        total_decay = decay_rate * time_elapsed
        
        # 2. 估算太阳能采集（与SensorNode.energy_harvest完全相同）
        total_harvest = 0.0
        if self.use_solar_model and info.get('is_solar', False) and params:
            # 对每一分钟求和，精确计算（与真实节点逻辑一致）
            for t in range(last_report_time, current_time):
                total_harvest += self._estimate_solar_harvest_single(t, params)
        
        # 3. 计算估算能量
        estimated = last_report_energy + total_harvest - total_decay
        estimated = max(0.0, estimated)  # 确保非负
        max_energy = capacity * voltage * 3600  # 电池容量上限（J）
        estimated = min(estimated, max_energy)  # 不超过容量
        
        return estimated, time_elapsed
    
    def estimate_all_nodes(self, current_time: int):
        """
        估算所有节点的当前能量并更新AoI到节点信息表
        
        工作流程：
        1. 遍历所有节点
        2. 对于未上报的节点（time_elapsed > 0），基于物理模型估算能量
        3. 更新节点信息表中的能量值、is_estimated字段和AoI（每分钟+1）
        4. 所有节点的 t 字段统一更新为 current_time（全局时钟）
        5. InfoNode会自动从信息表同步
        
        :param current_time: 当前时间（分钟）
        """
        if not self.enable_energy_estimation:
            return
        
        estimated_count = 0
        
        for node_id in self.latest_info.keys():
            estimated_energy, time_elapsed = self._estimate_energy(node_id, current_time)
            
            if estimated_energy is None:
                continue
            
            # 如果是估算值（time_elapsed > 0），更新信息表
            if time_elapsed > 0:
                # 更新节点信息表
                self.latest_info[node_id]['energy'] = estimated_energy
                self.latest_info[node_id]['is_estimated'] = True
                
                # 同步到InfoNode
                if node_id in self.info_nodes:
                    self.info_nodes[node_id].current_energy = estimated_energy
                
                estimated_count += 1
            
            # 【统一时钟】所有节点的 t 字段都更新为当前时刻
            # t 代表全局模拟时间，从0开始，每分钟+1
            self.latest_info[node_id]['t'] = current_time
            
            # 更新AoI：每分钟+1（无论是否估算）
            # AoI = current_time - arrival_time
            arrival_time = self.latest_info[node_id]['arrival_time']
            self.latest_info[node_id]['aoi'] = current_time - arrival_time
        
        if estimated_count > 0:
            self._log(f"[NodeInfoManager] 能量估算完成：{estimated_count} 个节点")
    
    def apply_energy_transfer_changes(self, plans: List[Dict], current_time: int):
        """
        根据能量传输计划，计算并更新节点信息表中的节点能量
        
        工作流程：
        1. 遍历所有传输计划（plans）
        2. 对于每条路径，计算每个节点的能量变化：
           - Donor: 消耗 = energy_consumption(receiver, transfer_WET=True)
           - 中继节点: 每跳的消耗和接收
           - Receiver: 获得 = energy_sent * 路径总效率
        3. 累计所有路径对每个节点的能量影响
        4. 更新节点信息表和InfoNode
        
        注意：
        - 中心节点规划了所有传输路径，知道每个节点的能量变化
        - 这是确定性的计算（不是估算），基于物理模型和已知路径
        
        :param plans: 能量传输计划列表 [{donor, receiver, path, distance}, ...]
        :param current_time: 当前时间步（分钟）
        """
        if not plans:
            return
        
        # 初始化能量变化字典：{node_id: energy_delta}
        energy_changes = {}
        
        # 1. 遍历所有传输计划，计算每个节点的能量变化
        for plan in plans:
            donor = plan.get("donor")
            receiver = plan.get("receiver")
            path = plan.get("path")
            
            if not donor or not receiver or not path:
                continue
            
            donor_id = donor.node_id
            receiver_id = receiver.node_id
            
            # 获取donor的InfoNode（用于计算能量消耗）
            if donor_id not in self.info_nodes:
                continue
            donor_info = self.info_nodes[donor_id]
            
            # 初始发送能量
            energy_sent = donor_info.E_char
            
            # 根据路径长度判断单跳或多跳
            if len(path) == 2:
                # 单跳直接传输
                # 获取receiver的InfoNode（用于计算距离和效率）
                if receiver_id not in self.info_nodes:
                    continue
                receiver_info = self.info_nodes[receiver_id]
                
                # 计算传输效率
                eta = donor_info.energy_transfer_efficiency(receiver_info)
                energy_received = energy_sent * eta
                
                # 计算donor消耗（使用InfoNode计算）
                # energy_consumption = E_elec * B + epsilon_amp * B * d^tau + sensor_energy
                # 如果 transfer_WET=True，还要加上 E_char
                B = donor_info.bit_rate
                d = donor_info.distance_to(receiver_info)
                E_tx = donor_info.energy_elec * B + donor_info.epsilon_amp * B * (d ** donor_info.path_loss_exponent)
                E_rx = donor_info.energy_elec * B
                E_com = (E_tx + E_rx) / 2  # 双向确认通信，平均
                E_sen = donor_info.sensor_energy
                energy_consumed = E_com + E_sen + donor_info.E_char  # 加上WET开销
                
                # 记录能量变化
                if donor_id not in energy_changes:
                    energy_changes[donor_id] = 0.0
                energy_changes[donor_id] -= energy_consumed
                
                if receiver_id not in energy_changes:
                    energy_changes[receiver_id] = 0.0
                energy_changes[receiver_id] += energy_received
                
            else:
                # 多跳传输：逐跳转发，每跳能量衰减
                energy_left = energy_sent
                
                for i in range(len(path) - 1):
                    sender_node = path[i]
                    receiver_i_node = path[i + 1]
                    sender_id = sender_node.node_id
                    receiver_i_id = receiver_i_node.node_id
                    
                    # 获取InfoNode（用于计算）
                    if sender_id not in self.info_nodes or receiver_i_id not in self.info_nodes:
                        continue
                    
                    sender_info = self.info_nodes[sender_id]
                    receiver_i_info = self.info_nodes[receiver_i_id]
                    
                    # 计算本跳的效率
                    eta = sender_info.energy_transfer_efficiency(receiver_i_info)
                    
                    # 计算本跳实际接收能量
                    energy_delivered = energy_left * eta
                    
                    # 计算发送方消耗（使用InfoNode计算）
                    transfer_WET = (i == 0)  # 仅第一跳计算WET模块消耗
                    B = sender_info.bit_rate
                    d = sender_info.distance_to(receiver_i_info)
                    E_tx = sender_info.energy_elec * B + sender_info.epsilon_amp * B * (d ** sender_info.path_loss_exponent)
                    E_rx = sender_info.energy_elec * B
                    E_com = (E_tx + E_rx) / 2  # 双向确认通信，平均
                    E_sen = sender_info.sensor_energy
                    energy_consumed = E_com + E_sen
                    if transfer_WET:
                        energy_consumed += sender_info.E_char  # WET开销
                    
                    # 记录能量变化
                    if sender_id not in energy_changes:
                        energy_changes[sender_id] = 0.0
                    energy_changes[sender_id] -= energy_consumed
                    
                    if receiver_i_id not in energy_changes:
                        energy_changes[receiver_i_id] = 0.0
                    energy_changes[receiver_i_id] += energy_delivered
                    
                    # 准备下一跳
                    energy_left = energy_delivered
        
        # 2. 更新节点信息表和InfoNode
        updated_count = 0
        for node_id, energy_delta in energy_changes.items():
            if abs(energy_delta) < 1e-9:  # 忽略极小的变化
                continue
            
            if node_id not in self.latest_info:
                continue
            
            # 获取当前能量
            current_energy = self.latest_info[node_id]['energy']
            
            # 计算新能量（不能小于0）
            new_energy = max(0.0, current_energy + energy_delta)
            
            # 更新节点信息表
            self.latest_info[node_id]['energy'] = new_energy
            self.latest_info[node_id]['is_estimated'] = False  # 基于传输计划的能量是确定的，不是估算
            self.latest_info[node_id]['arrival_time'] = current_time  # 更新到达时间
            self.latest_info[node_id]['record_time'] = current_time  # 更新记录时间
            self.latest_info[node_id]['aoi'] = 0  # 刚更新，AoI重置为0
            self.latest_info[node_id]['t'] = current_time  # 全局时间戳
            
            # 同步到InfoNode
            if node_id in self.info_nodes:
                self.info_nodes[node_id].current_energy = new_energy
            
            updated_count += 1
        
        if updated_count > 0:
            self._log(f"[NodeInfoManager] 能量传输更新完成：{updated_count} 个节点")
    
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
                record_time=initial_time,
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
                    'arrival_time', 'node_id', 'energy', 'record_time', 'aoi', 't',
                    'position_x', 'position_y', 'is_solar', 'cluster_id', 'data_size',
                    'info_volume', 'info_waiting_since', 'info_is_reported', 'info_source_nodes'
                ])
                writer.writeheader()
            self.archive_initialized = True
            self._log(f"[NodeInfoManager] 归档文件已初始化: {self.archive_path}")
        
        # 追加写入数据
        with open(self.archive_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'arrival_time', 'node_id', 'energy', 'record_time', 'aoi', 't',
                'position_x', 'position_y', 'is_solar', 'cluster_id', 'data_size',
                'info_volume', 'info_waiting_since', 'info_is_reported', 'info_source_nodes'
            ])
            writer.writerows(self.archive_buffer)
        
        # 去掉归档输出的日志（减少控制台输出）
        # self._log(f"[NodeInfoManager] 归档 {len(self.archive_buffer)} 条记录到 {self.archive_path}")
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
                'avg_aoi': 0,
                'max_aoi': 0,
                'solar_nodes': 0,
                'non_solar_nodes': 0
            }
        
        energies = [info['energy'] for info in self.latest_info.values()]
        aois = [info['aoi'] for info in self.latest_info.values()]
        solar_count = sum(1 for info in self.latest_info.values() 
                         if info.get('is_solar', False))
        
        return {
            'total_nodes': len(self.latest_info),
            'avg_energy': sum(energies) / len(energies),
            'min_energy': min(energies),
            'max_energy': max(energies),
            'avg_aoi': sum(aois) / len(aois) if aois else 0,
            'max_aoi': max(aois) if aois else 0,
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
        缓存所有节点的能量传输参数和太阳能参数（内部方法，在初始化时调用）
        
        :param nodes: 节点列表
        """
        for node in nodes:
            self.energy_params[node.node_id] = {
                # 能量传输参数
                'energy_char': getattr(node, 'energy_char', 300.0),
                'energy_elec': getattr(node, 'energy_elec', 1e-4),
                'epsilon_amp': getattr(node, 'epsilon_amp', 1e-5),
                'bit_rate': getattr(node, 'bit_rate', 1000000.0),
                'path_loss_exponent': getattr(node, 'path_loss_exponent', 2.0),
                'sensor_energy': getattr(node, 'sensor_energy', 0.1),
                # 能量衰减参数
                'energy_decay_rate': getattr(node, 'energy_decay_rate', self.decay_rate_default),
                # 电池参数
                'capacity': getattr(node, 'capacity', 3.5),
                'voltage': getattr(node, 'V', 3.7),
                # 太阳能参数
                'G_max': getattr(node, 'G_max', 1000.0),
                'solar_efficiency': getattr(node, 'solar_efficiency', 0.2),
                'solar_area': getattr(node, 'solar_area', 0.01),
                'env_correction_factor': getattr(node, 'env_correction_factor', 0.8),
                'enable_energy_harvesting': getattr(node, 'enable_energy_harvesting', True)
            }
        self._log(f"[NodeInfoManager] 缓存 {len(self.energy_params)} 个节点的能量传输和太阳能参数")
    
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
            
            # 只传递InfoNode需要的参数
            info_node_params = {
                'energy_char': params.get('energy_char', 300.0),
                'energy_elec': params.get('energy_elec', 1e-4),
                'epsilon_amp': params.get('epsilon_amp', 1e-5),
                'bit_rate': params.get('bit_rate', 1000000.0),
                'path_loss_exponent': params.get('path_loss_exponent', 2.0),
                'sensor_energy': params.get('sensor_energy', 0.1)
            }
            
            info_node = InfoNode(
                node_id=node_id,
                energy=info['energy'],
                position=info['position'],
                is_solar=info.get('is_solar', False),
                is_physical_center=(node_id == 0),  # 物理中心ID固定为0
                **info_node_params
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
    
    def check_timeout_and_force_report(self, current_time: int, max_wait_time: int = 10,
                                       path_collector=None, network=None) -> int:
        """
        检查等待中的节点是否超时，并强制上报（超时保护机制）
        
        :param current_time: 当前时间（分钟）
        :param max_wait_time: 最大等待时间（分钟）
        :param path_collector: PathBasedInfoCollector实例（用于上报信息）
        :param network: Network实例（用于查找节点）
        :return: 强制上报的节点数量
        """
        timeout_nodes = []
        
        # 1. 检查所有等待中的节点
        for node_id, node_info in self.latest_info.items():
            # 确保信息量字段存在（兼容旧数据）
            info_volume = node_info.get('info_volume', 0)
            is_reported = node_info.get('info_is_reported', True)
            waiting_since = node_info.get('info_waiting_since', -1)
            
            # 检查是否有未上报的信息量
            if not is_reported and info_volume > 0 and waiting_since >= 0:
                # 计算等待时间
                wait_time = current_time - waiting_since
                
                # 检查是否超时
                if wait_time >= max_wait_time:
                    timeout_nodes.append((node_id, node_info))
        
        # 2. 对超时节点强制上报
        if timeout_nodes and path_collector and network:
            for node_id, node_info in timeout_nodes:
                # 查找节点对象
                node = None
                if hasattr(network, 'get_node_by_id'):
                    node = network.get_node_by_id(node_id)
                elif hasattr(network, 'nodes'):
                    # 从nodes列表中查找
                    for n in network.nodes:
                        if n.node_id == node_id:
                            node = n
                            break
                if node is not None:
                    # 获取信息量
                    info_volume = node_info.get('info_volume', 0)
                    
                    # 获取等待时间（需要在循环内计算）
                    waiting_since = node_info.get('info_waiting_since', -1)
                    wait_time = current_time - waiting_since if waiting_since >= 0 else 0
                    
                    # 强制单独上报（通过PathBasedInfoCollector）
                    if hasattr(path_collector, '_report_info_to_center'):
                        path_collector._report_info_to_center(node, info_volume)
                    
                    # 在节点信息表中清零信息量状态
                    if node_id in self.latest_info:
                        self.latest_info[node_id]['info_is_reported'] = True
                        self.latest_info[node_id]['info_volume'] = 0
                        self.latest_info[node_id]['info_waiting_since'] = -1
                    
                    # 记录强制上报统计
                    if not hasattr(self, 'forced_reports_count'):
                        self.forced_reports_count = 0
                    self.forced_reports_count += 1
                    
                    self._log(f"[NodeInfoManager] 超时强制上报 - 节点 {node_id}, "
                             f"信息量: {info_volume} bits, 等待时间: {wait_time} 分钟")
        
        return len(timeout_nodes)
    
    def reset(self):
        """重置状态"""
        self.last_update_time = -1
        
        # 清空节点信息表
        self.latest_info.clear()
        self.recent_history.clear()
        
        # 清空信息传输能量消耗统计
        self.info_transmission_energy.clear()
        self.info_transmission_stats = {
            'total_adcr_energy': 0.0,
            'total_path_collector_energy': 0.0,
            'total_energy': 0.0,
            'adcr_transmission_count': 0,
            'path_collector_transmission_count': 0
        }
        
        # 强制刷新归档
        self.force_flush_archive()
        
        self._log("[NodeInfoManager] 状态已重置，节点信息表已清空")
    
    # ==================== 信息传输能量消耗统计 ====================
    
    def record_info_transmission_energy(self, node_id: int, energy: float, 
                                       transmission_type: str = "adcr"):
        """
        记录节点信息传输的能量消耗
        
        :param node_id: 节点ID
        :param energy: 消耗的能量（J）
        :param transmission_type: 传输类型 ("adcr" 或 "path_collector")
        
        注意：
        1. 此方法只记录能量消耗，不记录传输次数。
        2. 传输次数应该由调用方在完成一次完整传输后，调用 record_transmission_count() 单独记录。
        """
        if node_id not in self.info_transmission_energy:
            self.info_transmission_energy[node_id] = {
                'adcr': 0.0,
                'path_collector': 0.0,
                'total': 0.0
            }
        
        self.info_transmission_energy[node_id][transmission_type] += energy
        self.info_transmission_energy[node_id]['total'] += energy
        
        # 更新总体统计（能量累加）
        if transmission_type == "adcr":
            self.info_transmission_stats['total_adcr_energy'] += energy
        elif transmission_type == "path_collector":
            self.info_transmission_stats['total_path_collector_energy'] += energy
        
        self.info_transmission_stats['total_energy'] += energy
    
    def record_transmission_count(self, transmission_type: str = "adcr"):
        """
        记录一次完整的传输（用于统计传输次数）
        
        :param transmission_type: 传输类型 ("adcr" 或 "path_collector")
        
        注意：一次传输可能涉及多个节点和多次能量消耗记录，
        但传输次数应该只在完整传输完成时计数一次。
        """
        if transmission_type == "adcr":
            self.info_transmission_stats['adcr_transmission_count'] += 1
        elif transmission_type == "path_collector":
            self.info_transmission_stats['path_collector_transmission_count'] += 1
    
    def get_info_transmission_statistics(self) -> Dict:
        """
        获取信息传输能量消耗统计信息
        
        :return: 统计信息字典
        """
        if not self.info_transmission_energy:
            return {
                'total_nodes': 0,
                'total_energy': 0.0,
                'total_adcr_energy': 0.0,
                'total_path_collector_energy': 0.0,
                'avg_energy_per_node': 0.0,
                'max_node_energy': 0.0,
                'min_node_energy': 0.0,
                'node_breakdown': {},
                'adcr_transmission_count': 0,
                'path_collector_transmission_count': 0
            }
        
        total_energies = [stats['total'] for stats in self.info_transmission_energy.values()]
        
        # 按节点排序（按总能量降序）
        node_breakdown = {
            node_id: {
                'adcr': stats['adcr'],
                'path_collector': stats['path_collector'],
                'total': stats['total']
            }
            for node_id, stats in sorted(
                self.info_transmission_energy.items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )
        }
        
        return {
            'total_nodes': len(self.info_transmission_energy),
            'total_energy': self.info_transmission_stats['total_energy'],
            'total_adcr_energy': self.info_transmission_stats['total_adcr_energy'],
            'total_path_collector_energy': self.info_transmission_stats['total_path_collector_energy'],
            'avg_energy_per_node': sum(total_energies) / len(total_energies) if total_energies else 0.0,
            'max_node_energy': max(total_energies) if total_energies else 0.0,
            'min_node_energy': min(total_energies) if total_energies else 0.0,
            'node_breakdown': node_breakdown,
            'adcr_transmission_count': self.info_transmission_stats['adcr_transmission_count'],
            'path_collector_transmission_count': self.info_transmission_stats['path_collector_transmission_count']
        }
    
    def log_info_transmission_statistics(self):
        """
        打印信息传输能量消耗统计到日志
        """
        stats = self.get_info_transmission_statistics()
        
        self._log("\n" + "=" * 80)
        self._log("信息传输能量消耗统计")
        self._log("=" * 80)
        self._log(f"总体统计:")
        self._log(f"  - 总能量消耗: {stats['total_energy']:.2f} J")
        self._log(f"  - ADCR协议: {stats['total_adcr_energy']:.2f} J "
                 f"({stats['adcr_transmission_count']} 次完整传输)")
        self._log(f"  - 路径收集器: {stats['total_path_collector_energy']:.2f} J "
                 f"({stats['path_collector_transmission_count']} 次路径收集)")
        self._log(f"  - 参与节点数: {stats['total_nodes']} 个")
        self._log(f"  - 平均每节点: {stats['avg_energy_per_node']:.2f} J")
        self._log(f"  - 最高消耗节点: {stats['max_node_energy']:.2f} J")
        self._log(f"  - 最低消耗节点: {stats['min_node_energy']:.2f} J")
        
        if stats['total_nodes'] > 0:
            self._log(f"\n节点详细统计 (Top 10):")
            sorted_nodes = sorted(
                stats['node_breakdown'].items(),
                key=lambda x: x[1]['total'],
                reverse=True
            )[:10]
            
            self._log(f"  {'节点ID':<8} {'ADCR消耗(J)':<15} {'路径收集器消耗(J)':<20} {'总消耗(J)':<12}")
            self._log(f"  {'-' * 8} {'-' * 15} {'-' * 20} {'-' * 12}")
            for node_id, node_stats in sorted_nodes:
                self._log(f"  {node_id:<8} {node_stats['adcr']:<15.2f} "
                         f"{node_stats['path_collector']:<20.2f} {node_stats['total']:<12.2f}")
        
        self._log("=" * 80 + "\n")
    
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
