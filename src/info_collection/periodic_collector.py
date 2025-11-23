# src/info_collection/periodic_collector.py
# -*- coding: utf-8 -*-
"""
定期上报信息收集器

设计理念：
- 每个节点固定每隔一定时间（默认60分钟）向物理中心节点发送一次自己的信息
- 不依赖能量传输路径，独立运行
- 与路径收集器互斥（不能同时使用）

优势：
- 简单直接，易于理解
- 信息更新频率可控
- 适合作为基线对比实验

"""

from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.SensorNode import SensorNode
    from info_collection.physical_center import VirtualCenter


class PeriodicInfoCollector:
    """
    定期上报信息收集器
    
    工作流程：
    1. 每个节点维护自己的上次上报时间
    2. 每个时间步检查是否到达上报周期
    3. 如果到达，节点直接向物理中心节点发送自己的信息
    4. 更新虚拟中心的节点信息表
    """
    
    def __init__(self, 
                 virtual_center: VirtualCenter,
                 physical_center: SensorNode = None,
                 report_interval: int = 60,
                 base_data_size: int = 1000000,
                 enable_logging: bool = True):
        """
        初始化定期上报信息收集器
        
        :param virtual_center: 虚拟中心实例（用于节点信息表管理）
        :param physical_center: 物理中心节点（ID=0，信息上报目标）
        :param report_interval: 上报间隔（分钟），默认60分钟
        :param base_data_size: 基础数据包大小（bits）
        :param enable_logging: 是否启用日志
        """
        self.vc = virtual_center
        self.physical_center = physical_center
        self.report_interval = report_interval
        self.base_data_size = base_data_size
        self.enable_logging = enable_logging
        
        # 统计信息
        self.total_reports = 0  # 总上报次数
        self.total_energy_consumed = 0.0  # 总能量消耗
        
        # 每个节点的上次上报时间 {node_id: last_report_time}
        self.last_report_time: Dict[int, int] = {}
        
        self._log(f"[PeriodicCollector] 初始化完成 - 上报间隔: {report_interval} 分钟")
    
    def _log(self, message: str):
        """日志输出"""
        if self.enable_logging:
            print(message)
    
    def step(self, all_nodes: List[SensorNode], current_time: int):
        """
        每个时间步调用，检查并执行定期上报
        
        :param all_nodes: 所有网络节点
        :param current_time: 当前时间步（分钟）
        """
        if not self.physical_center:
            return
        
        # 遍历所有节点（排除物理中心节点）
        for node in all_nodes:
            # 跳过物理中心节点
            if node.node_id == self.physical_center.node_id:
                continue
            
            # 检查是否需要上报
            last_time = self.last_report_time.get(node.node_id, -1)
            
            # 如果是第一次（last_time == -1）或者已经过了上报间隔
            if last_time == -1 or (current_time - last_time) >= self.report_interval:
                # 执行上报
                self._report_node_info(node, current_time)
                # 更新上次上报时间
                self.last_report_time[node.node_id] = current_time
    
    def _report_node_info(self, node: SensorNode, current_time: int):
        """
        上报节点信息到物理中心
        
        :param node: 要上报的节点
        :param current_time: 当前时间步
        """
        # 1. 收集节点实时信息（上报前的能量）
        pre_report_energy = node.current_energy
        
        # 2. 计算并扣除能量消耗（节点到物理中心的通信）
        # 注意：这会修改 node.current_energy
        Eu, Ev, energy_cost = self._calculate_report_energy(node)
        self.total_energy_consumed += energy_cost
        
        # 3. 更新虚拟中心的节点信息表
        # 重要：使用上报后的能量（扣除上报消耗后的能量），这样信息表与真实节点能量一致
        self.vc.update_node_info(
            node_id=node.node_id,
            energy=node.current_energy,  # 使用上报后的能量（已扣除上报消耗）
            record_time=current_time,
            arrival_time=current_time,  # 立即到达
            position=tuple(node.position),
            is_solar=node.has_solar,
            cluster_id=None,  # 定期上报不使用簇
            data_size=None
        )
        
        # 4. 记录信息传输能量消耗
        if hasattr(self.vc, 'record_info_transmission_energy'):
            self.vc.record_info_transmission_energy(node.node_id, Eu, "periodic_collector")
            self.vc.record_info_transmission_energy(self.physical_center.node_id, Ev, "periodic_collector")
        
        # 5. 记录传输次数（每次上报算一次完整传输）
        if hasattr(self.vc, 'record_transmission_count'):
            self.vc.record_transmission_count("periodic_collector")
        
        # 6. 更新统计
        self.total_reports += 1
        
        self._log(f"[PeriodicCollector] 节点 {node.node_id} 上报信息 (时间: {current_time}, 能量消耗: {energy_cost:.2f}J)")
    
    def _calculate_report_energy(self, node: SensorNode) -> tuple:
        """
        计算节点上报到物理中心的能量消耗
        
        :param node: 上报节点
        :return: (发送方消耗Eu, 接收方消耗Ev, 总能量消耗)
        """
        if not self.physical_center:
            return (0.0, 0.0, 0.0)
        
        # 如果节点就是物理中心（不应该发生），返回0
        if node.node_id == self.physical_center.node_id:
            return (0.0, 0.0, 0.0)
        
        # 数据包大小固定
        data_size = self.base_data_size
        
        # 临时修改B参数
        original_B_node = node.B
        original_B_pc = self.physical_center.B
        
        node.B = data_size
        self.physical_center.B = data_size
        
        # 计算双向通信能耗
        Eu = node.energy_consumption(target_node=self.physical_center, transfer_WET=False)
        Ev = self.physical_center.energy_consumption(target_node=node, transfer_WET=False)
        
        # 恢复原始B值
        node.B = original_B_node
        self.physical_center.B = original_B_pc
        
        # 扣除能量
        node.current_energy = max(0.0, node.current_energy - Eu)
        self.physical_center.current_energy = max(0.0, self.physical_center.current_energy - Ev)
        
        total_energy = Eu + Ev
        return (Eu, Ev, total_energy)
    
    # ==================== 统计和监控 ====================
    
    def get_statistics(self) -> Dict:
        """
        获取收集器的统计信息
        
        :return: 统计信息字典
        """
        return {
            'total_reports': self.total_reports,
            'total_energy_consumed': self.total_energy_consumed,
            'avg_energy_per_report': self.total_energy_consumed / max(1, self.total_reports),
            'report_interval': self.report_interval
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("定期上报信息收集器统计")
        print("=" * 60)
        print(f"上报间隔: {stats['report_interval']} 分钟")
        print(f"总上报次数: {stats['total_reports']}")
        print(f"能量消耗:")
        print(f"  - 总计: {stats['total_energy_consumed']:.2f} J")
        print(f"  - 平均每次: {stats['avg_energy_per_report']:.2f} J")
        print("=" * 60 + "\n")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_reports = 0
        self.total_energy_consumed = 0.0
        self.last_report_time.clear()
        self._log("[PeriodicCollector] 统计信息已重置")

