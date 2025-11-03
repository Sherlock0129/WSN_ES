# src/info_collection/path_based_collector.py
# -*- coding: utf-8 -*-
"""
基于路径的机会主义信息收集器（物理中心架构）

设计理念：
- 利用能量传输路径收集节点信息（piggyback，搭载）
- 路径内节点：实时采集最新信息
- 路径外节点：不处理（不收集、不估算、不更新）
- Receiver（路径最后一个节点）作为信息汇聚点上报到物理中心节点（ID=0）

优势：
- 无需额外通信开销（信息搭载在传能路径上）
- 更新频率高（每次传能都更新）
- 信息新鲜度高（路径节点为实时采集）
- 能量消耗可配置（free/full模式）

架构更新（物理中心）：
- 不再有"虚拟跳"概念，所有通信都是真实节点之间的通信
- 信息上报目标从"虚拟中心"改为"物理中心节点（ID=0）"
- VirtualCenter 保留，仅用于节点信息表管理

"""

from __future__ import annotations
from typing import List, Dict, Set, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from core.SensorNode import SensorNode
    from acdr.physical_center import VirtualCenter


class PathBasedInfoCollector:
    """
    基于能量传输路径的机会主义信息收集器
    
    工作流程（以路径 a→b→c 为例）：
    1. 路径节点（a、b、c）各自收集信息，沿路径聚合到终点c
    2. 终点c将路径节点的聚合信息（固定1B）上报到物理中心节点（ID=0）
    3. 虚拟中心（节点信息表管理器）只更新路径节点（a、b、c）的信息
    4. 非路径节点不做任何处理（不收集、不估算、不更新）
    
    能量模型（物理中心架构）：
    - free模式：零能耗（信息完全搭载）
    - full模式：路径逐跳 + 上报跳（→物理中心节点）都消耗能量（与ADCR一致）
                所有跳都是真实节点之间的通信，不再有"虚拟跳"概念
    """
    
    def __init__(self, 
                 virtual_center: VirtualCenter,
                 physical_center: SensorNode = None,
                 energy_mode: str = "free",
                 base_data_size: int = 1000000,
                 enable_logging: bool = True,
                 # 估算参数
                 decay_rate: float = 5.0,
                 use_solar_model: bool = True,
                 # 优化选项
                 batch_update: bool = True,
                 # 数据包大小模式
                 enable_accumulative_data_size: bool = False,
                 # 机会主义信息传递参数
                 enable_opportunistic_info_forwarding: bool = True,
                 enable_delayed_reporting: bool = True,
                 max_wait_time: int = 10,
                 min_info_volume_threshold: int = 1):
        """
        初始化路径信息收集器
        
        :param virtual_center: 虚拟中心实例（用于节点信息表管理）
        :param physical_center: 物理中心节点（ID=0，信息上报目标）
        :param energy_mode: 能量消耗模式 ("free": 零能耗, "full": 完全真实)
        :param base_data_size: 基础数据大小（bits），每个节点的基础信息量（与ADCR一致）
        :param enable_logging: 是否启用日志
        :param decay_rate: 自然衰减率（J/分钟，用于估算）
        :param use_solar_model: 是否使用太阳能模型进行估算
        :param batch_update: 是否批量更新虚拟中心
        :param enable_accumulative_data_size: 是否启用累积数据包大小模式
                                           True: 数据包大小 = base_data_size × 经过的节点数
                                           False: 数据包大小 = base_data_size（固定）
        :param enable_opportunistic_info_forwarding: 是否启用机会主义信息传递
        :param enable_delayed_reporting: 是否启用延迟上报（False为立即上报）
        :param max_wait_time: 最大等待时间（分钟），超时强制上报
        :param min_info_volume_threshold: 最小信息量阈值（节点数），低于此值不等待
        """
        self.vc = virtual_center
        self.physical_center = physical_center
        self.energy_mode = energy_mode
        self.base_data_size = base_data_size
        self.enable_logging = enable_logging
        self.decay_rate = decay_rate
        self.use_solar_model = use_solar_model
        self.batch_update = batch_update
        self.enable_accumulative_data_size = enable_accumulative_data_size
        
        # 机会主义信息传递参数
        self.enable_opportunistic_info_forwarding = enable_opportunistic_info_forwarding
        self.enable_delayed_reporting = enable_delayed_reporting
        self.max_wait_time = max_wait_time
        self.min_info_volume_threshold = min_info_volume_threshold
        
        # 统计信息
        self.total_collections = 0
        self.total_real_info = 0
        self.total_estimated_info = 0
        self.total_energy_consumed = 0.0  # 总能量消耗
        
        data_size_mode = "累积模式" if enable_accumulative_data_size else "固定大小模式"
        self._log(f"[PathCollector] 初始化完成 - 能量模式={energy_mode}, "
                 f"数据包大小={base_data_size}bits ({data_size_mode}), "
                 f"衰减率={decay_rate}J/min, 太阳能模型={'启用' if use_solar_model else '禁用'}")
    
    def _log(self, message: str):
        """内部日志方法"""
        if self.enable_logging:
            print(message)
    
    # ==================== 核心方法 ====================
    
    def collect_and_report(self, path: List[SensorNode], all_nodes: List[SensorNode], 
                          current_time: int) -> Dict[str, int]:
        """
        从传能路径收集信息并上报到虚拟中心
        
        主要流程：
        1. 收集路径节点的实时信息
        2. 计算能量消耗（如果启用full模式）
        3. 更新虚拟中心（只更新路径节点）
        
        :param path: 能量传输路径 [donor, relay1, ..., receiver]
        :param all_nodes: 所有网络节点（保留参数以兼容接口，实际未使用）
        :param current_time: 当前时间步（分钟）
        :return: 统计信息 {'real': int, 'estimated': int}
        """
        if not path:
            self._log("[PathCollector] 警告：路径为空，跳过信息收集")
            return {'real': 0, 'estimated': 0}
        
        self.total_collections += 1
        
        # 1. 收集路径节点的实时信息
        path_info = self._collect_real_info(path, current_time)
        
        # 2. 能量消耗结算（如果启用）
        if self.energy_mode == "full":
            energy_cost = self._settle_energy_consumption(path)
            self.total_energy_consumed += energy_cost
        
        # 3. 更新虚拟中心（只更新路径节点）
        self._update_virtual_center(path_info, current_time)
        
        # 4. 机会主义信息传递：更新信息量状态（如果启用）
        if self.enable_opportunistic_info_forwarding:
            self._update_info_volume(path, current_time)
        
        # 5. 记录一次完整的路径收集传输次数（仅在full模式下）
        if self.energy_mode == "full" and hasattr(self.vc, 'record_transmission_count'):
            self.vc.record_transmission_count("path_collector")
        
        # 6. 更新统计
        path_node_count = len(path_info)
        self.total_real_info += path_node_count
        
        self._log(f"[PathCollector] 收集完成 - 路径节点: {path_node_count}, "
                 f"路径长度: {len(path)}, Receiver: Node {path[-1].node_id}")
        
        return {'real': path_node_count, 'estimated': 0}
    
    def _collect_real_info(self, path: List[SensorNode], current_time: int) -> Dict[int, Dict]:
        """
        收集路径上所有节点的实时信息
        
        :param path: 传能路径
        :param current_time: 当前时间
        :return: {node_id: info_dict}
        """
        info = {}
        for node in path:
            info[node.node_id] = {
                'energy': node.current_energy,
                'position': tuple(node.position),
                'is_solar': node.has_solar,
                'record_time': current_time,  # 实时采集，记录时间为当前时间
                'arrival_time': current_time,  # 立即到达
                'is_estimated': False
            }
        
        return info
    
    def _update_virtual_center(self, all_info: Dict[int, Dict], current_time: int):
        """
        批量更新虚拟中心的节点信息
        
        :param all_info: {node_id: info_dict}
        :param current_time: 当前时间
        """
        for node_id, info in all_info.items():
            # 更新虚拟中心
            self.vc.update_node_info(
                node_id=node_id,
                energy=info['energy'],
                record_time=info['record_time'],
                arrival_time=info['arrival_time'],
                position=info['position'],
                is_solar=info['is_solar'],
                cluster_id=None,  # PathBased不使用簇
                data_size=None    # 可选：记录路径长度等信息
            )
        
        self._log(f"[PathCollector] 虚拟中心更新: {len(all_info)} 个节点")
    
    def _update_info_volume(self, path: List[SensorNode], current_time: int):
        """
        更新路径完成后的信息量状态（机会主义信息传递）
        
        流程：
        1. 检查路径中是否有节点携带未上报信息（搭便车检查）
        2. 计算路径的信息量
        3. 更新终点节点的信息量（延迟上报或立即上报）
        4. 中间节点清零信息量
        
        :param path: 能量传输路径 [donor, relay1, ..., receiver]
        :param current_time: 当前时间步（分钟）
        """
        if not path:
            return
        
        receiver = path[-1]  # 终点节点
        
        # 1. 检查路径中是否有节点携带未上报信息（搭便车检查）
        nodes_with_info = []
        for node in path:
            node_info = self.vc.get_node_info(node.node_id)
            if node_info:
                info_volume = node_info.get('info_volume', 0)
                is_reported = node_info.get('info_is_reported', True)
                
                # 检查是否有未上报的信息量
                if not is_reported and info_volume > 0:
                    nodes_with_info.append((node, node_info))
        
        # 2. 计算整条路径的信息量
        if self.enable_accumulative_data_size:
            path_info_volume = self.base_data_size * len(path)
        else:
            path_info_volume = self.base_data_size
        
        # 3. 如果有节点携带信息，实现"搭便车"
        if nodes_with_info:
            # 计算路径上的总信息量（包括新路径信息和搭载的信息）
            total_info_volume = path_info_volume
            for node, node_info in nodes_with_info:
                total_info_volume += node_info.get('info_volume', 0)
            
            # 在路径终点上报总信息量（消耗能量）
            if self.energy_mode == "full" and self.physical_center:
                self._report_info_to_center(receiver, total_info_volume)
            
            # 标记搭载信息的节点已上报（在节点信息表中更新）
            for node, node_info in nodes_with_info:
                if node.node_id in self.vc.latest_info:
                    self.vc.latest_info[node.node_id]['info_is_reported'] = True
                    self.vc.latest_info[node.node_id]['info_volume'] = 0
                    self.vc.latest_info[node.node_id]['info_waiting_since'] = -1
            
            # 更新新路径终点的信息量（延迟上报模式）
            if self.enable_delayed_reporting:
                self._set_receiver_info_volume(receiver, path_info_volume, path, current_time)
            else:
                # 立即上报模式：信息量清零
                self._clear_receiver_info_volume(receiver)
            
            self._log(f"[PathCollector] 搭便车成功 - 搭载信息量: {total_info_volume - path_info_volume} bits, "
                     f"总信息量: {total_info_volume} bits")
        else:
            # 没有节点携带信息，正常处理新路径
            if self.enable_delayed_reporting:
                # 延迟上报模式：终点节点进入等待状态
                self._set_receiver_info_volume(receiver, path_info_volume, path, current_time)
            else:
                # 立即上报模式：立即上报，信息量清零
                if self.energy_mode == "full" and self.physical_center:
                    self._report_info_to_center(receiver, path_info_volume)
                self._clear_receiver_info_volume(receiver)
        
        # 4. 中间节点清零信息量（如果有的话）
        for i in range(len(path) - 1):
            intermediate_node = path[i]
            if intermediate_node.node_id in self.vc.latest_info:
                self._clear_node_info_volume(intermediate_node.node_id)
    
    def _set_receiver_info_volume(self, receiver: SensorNode, path_info_volume: int, 
                                   path: List[SensorNode], current_time: int):
        """
        设置终点节点的信息量状态（延迟上报模式）
        
        :param receiver: 终点节点
        :param path_info_volume: 路径信息量
        :param path: 路径
        :param current_time: 当前时间
        """
        # 确保终点节点在节点信息表中
        if receiver.node_id not in self.vc.latest_info:
            self.vc.update_node_info(
                node_id=receiver.node_id,
                energy=receiver.current_energy,
                record_time=current_time,
                arrival_time=current_time,
                position=tuple(receiver.position),
                is_solar=receiver.has_solar
            )
        
        # 确保信息量字段存在
        if 'info_volume' not in self.vc.latest_info[receiver.node_id]:
            self.vc.latest_info[receiver.node_id]['info_volume'] = 0
            self.vc.latest_info[receiver.node_id]['info_waiting_since'] = -1
            self.vc.latest_info[receiver.node_id]['info_is_reported'] = True
            self.vc.latest_info[receiver.node_id]['info_source_nodes'] = []
        
        # 检查信息量阈值
        current_volume = self.vc.latest_info[receiver.node_id]['info_volume']
        new_volume = current_volume + path_info_volume
        threshold_volume = self.base_data_size * self.min_info_volume_threshold
        
        if new_volume >= threshold_volume:
            # 达到阈值，进入等待状态
            self.vc.latest_info[receiver.node_id]['info_volume'] = new_volume
            self.vc.latest_info[receiver.node_id]['info_waiting_since'] = current_time
            self.vc.latest_info[receiver.node_id]['info_is_reported'] = False
            # 记录信息来源节点
            if 'info_source_nodes' not in self.vc.latest_info[receiver.node_id]:
                self.vc.latest_info[receiver.node_id]['info_source_nodes'] = []
            self.vc.latest_info[receiver.node_id]['info_source_nodes'].extend([n.node_id for n in path])
        else:
            # 未达到阈值，立即上报
            if self.energy_mode == "full" and self.physical_center:
                self._report_info_to_center(receiver, new_volume)
            self._clear_receiver_info_volume(receiver)
    
    def _clear_receiver_info_volume(self, receiver: SensorNode):
        """清空终点节点的信息量"""
        if receiver.node_id in self.vc.latest_info:
            self._clear_node_info_volume(receiver.node_id)
    
    def _clear_node_info_volume(self, node_id: int):
        """清空节点的信息量"""
        if node_id in self.vc.latest_info:
            if 'info_volume' not in self.vc.latest_info[node_id]:
                self.vc.latest_info[node_id]['info_volume'] = 0
                self.vc.latest_info[node_id]['info_waiting_since'] = -1
                self.vc.latest_info[node_id]['info_is_reported'] = True
            else:
                self.vc.latest_info[node_id]['info_volume'] = 0
                self.vc.latest_info[node_id]['info_waiting_since'] = -1
                self.vc.latest_info[node_id]['info_is_reported'] = True
    
    def _report_info_to_center(self, node: SensorNode, info_volume: int):
        """
        上报信息到物理中心（消耗能量）
        
        :param node: 上报节点
        :param info_volume: 信息量（bits）
        """
        if not self.physical_center:
            return
        
        # 使用实际的信息量计算能量消耗
        data_size = info_volume
        
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
        
        # 记录信息传输能量消耗
        if hasattr(self.vc, 'record_info_transmission_energy'):
            self.vc.record_info_transmission_energy(node.node_id, Eu, "path_collector")
            self.vc.record_info_transmission_energy(self.physical_center.node_id, Ev, "path_collector")
    
    # ==================== 统计和监控 ====================
    
    def get_statistics(self) -> Dict:
        """
        获取收集器的统计信息
        
        :return: 统计信息字典
        """
        total_info = self.total_real_info + self.total_estimated_info
        return {
            'total_collections': self.total_collections,
            'total_real_info': self.total_real_info,
            'total_estimated_info': self.total_estimated_info,
            'avg_real_per_collection': self.total_real_info / max(1, self.total_collections),
            'avg_estimated_per_collection': self.total_estimated_info / max(1, self.total_collections),
            'real_ratio': self.total_real_info / max(1, total_info),
            'estimated_ratio': self.total_estimated_info / max(1, total_info),
            'total_energy_consumed': self.total_energy_consumed,
            'avg_energy_per_collection': self.total_energy_consumed / max(1, self.total_collections)
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("路径信息收集器统计")
        print("=" * 60)
        print(f"能量模式: {self.energy_mode}")
        print(f"总收集次数: {stats['total_collections']}")
        print(f"收集的路径节点数: {stats['total_real_info']}")
        print(f"平均每次收集: {stats['avg_real_per_collection']:.1f} 个路径节点")
        if self.energy_mode == "full":
            print(f"能量消耗:")
            print(f"  - 总计: {stats['total_energy_consumed']:.2f} J")
            print(f"  - 平均每次: {stats['avg_energy_per_collection']:.2f} J")
        print("=" * 60 + "\n")
    
    def reset_statistics(self):
        """重置统计信息"""
        self.total_collections = 0
        self.total_real_info = 0
        self.total_estimated_info = 0
        self._log("[PathCollector] 统计信息已重置")
    
    # ==================== 高级特性（可选） ====================
    
    def evaluate_estimation_accuracy(self, all_nodes: List[SensorNode]) -> Dict:
        """
        评估估算准确度（用于调试和优化）
        
        对比虚拟中心的估算值和节点的实际值
        
        :param all_nodes: 所有节点
        :return: 准确度统计
        """
        errors = []
        for node in all_nodes:
            vc_info = self.vc.get_node_info(node.node_id)
            if vc_info and vc_info.get('is_estimated', False):
                estimated = vc_info['energy']
                actual = node.current_energy
                error = abs(estimated - actual)
                relative_error = error / max(1.0, actual)
                errors.append({
                    'node_id': node.node_id,
                    'estimated': estimated,
                    'actual': actual,
                    'error': error,
                    'relative_error': relative_error
                })
        
        if not errors:
            return {'count': 0}
        
        avg_error = sum(e['error'] for e in errors) / len(errors)
        avg_relative_error = sum(e['relative_error'] for e in errors) / len(errors)
        
        return {
            'count': len(errors),
            'avg_absolute_error': avg_error,
            'avg_relative_error': avg_relative_error,
            'max_error': max(e['error'] for e in errors),
            'errors': errors
        }
    
    # ==================== 能量消耗结算 ====================
    
    def _settle_energy_consumption(self, path: List[SensorNode]) -> float:
        """
        结算信息收集的能量消耗（仅在 energy_mode="full" 时调用）
        
        能量消耗包括：
        1. 路径逐跳信息传递：路径节点信息沿路径聚合传递，每跳消耗能量
        2. 上报跳：Receiver → 物理中心节点（ID=0）的真实通信能耗
        
        :param path: 传能路径 [donor, relay..., receiver]
        :return: 总能量消耗（J）
        """
        if len(path) < 1:
            return 0.0
        
        # 1. 计算路径逐跳能耗（信息沿路径传递）
        path_energy = self._calculate_path_hop_energy(path)
        
        # 2. 计算上报跳能耗（Receiver → 物理中心节点）
        receiver = path[-1]
        report_energy = self._calculate_report_hop_energy(receiver, path)
        
        total_energy = path_energy + report_energy
        
        self._log(f"[PathCollector] 能量消耗 - 路径逐跳={path_energy:.2f}J, "
                 f"上报跳(→PC)={report_energy:.2f}J, 总计={total_energy:.2f}J")
        
        return total_energy
    
    def _calculate_path_hop_energy(self, path: List[SensorNode]) -> float:
        """
        计算路径逐跳能耗（信息沿路径聚合传递）
        
        模型：
        - 路径节点（如a→b→c）各自收集信息，沿路径聚合
        - 如果启用累积模式：数据包大小 = base_data_size × 经过的节点数
        - 如果禁用累积模式：数据包大小 = base_data_size（固定大小）
        - 路径终点（Receiver）汇聚路径上所有节点的信息
        - 使用SensorNode.energy_consumption()方法（与ADCR一致）
        - 能耗 = [(E_tx + E_rx) / 2 + E_sen] × 2端
        
        :param path: 传能路径 [donor, relay..., receiver]
        :return: 路径总能量消耗（J）
        """
        if len(path) < 2:
            return 0.0
        
        total_energy = 0.0
        
        # 沿路径逐跳传输信息包
        for i in range(len(path) - 1):
            sender = path[i]
            receiver = path[i + 1]
            
            # 计算数据包大小
            if self.enable_accumulative_data_size:
                # 累积模式：数据包大小 = base_data_size × 经过的节点数（从起点到当前发送节点）
                # 例如：路径 a→b→c，第1跳(a→b)传输 base_data_size×1，第2跳(b→c)传输 base_data_size×2
                nodes_passed = i + 1  # 经过的节点数（包括发送节点）
                data_packet_size = self.base_data_size * nodes_passed
            else:
                # 固定大小模式：数据包大小固定为 base_data_size
                data_packet_size = self.base_data_size
            
            # 使用与ADCR相同的能量计算方法
            # 临时修改节点的B参数以传递数据大小
            original_B_sender = sender.B
            original_B_receiver = receiver.B
            
            sender.B = data_packet_size
            receiver.B = data_packet_size
            
            # 调用SensorNode的energy_consumption方法（与ADCR._energy_consume_one_hop一致）
            Eu = sender.energy_consumption(target_node=receiver, transfer_WET=False)
            Ev = receiver.energy_consumption(target_node=sender, transfer_WET=False)
            
            # 恢复原始B值
            sender.B = original_B_sender
            receiver.B = original_B_receiver
            
            # 扣除能量
            sender.current_energy = max(0.0, sender.current_energy - Eu)
            receiver.current_energy = max(0.0, receiver.current_energy - Ev)
            
            # 记录信息传输能量消耗（仅当energy_mode="full"时）
            if hasattr(self.vc, 'record_info_transmission_energy'):
                self.vc.record_info_transmission_energy(sender.node_id, Eu, "path_collector")
                self.vc.record_info_transmission_energy(receiver.node_id, Ev, "path_collector")
            
            total_energy += (Eu + Ev)
        
        return total_energy
    
    def _calculate_report_hop_energy(self, receiver: SensorNode, path: List[SensorNode] = None) -> float:
        """
        计算上报跳能耗（Receiver → 物理中心节点）
        
        这是真实节点之间的通信，不再有"虚拟跳"概念。
        使用与ADCR相同的能耗模型计算双向通信能量消耗。
        
        :param receiver: 路径终点节点（路径信息汇聚点）
        :param path: 传能路径（可选，用于计算累积信息量）
        :return: 上报跳能量消耗（J）
        
        注意：
        - 如果启用累积模式：数据包大小 = base_data_size × 路径节点数
        - 如果禁用累积模式：数据包大小 = base_data_size（固定大小）
        """
        # 如果没有物理中心节点，返回0
        if self.physical_center is None:
            self._log("[PathCollector] 警告：物理中心节点未设置，跳过上报跳能耗计算")
            return 0.0
        
        # 如果receiver就是物理中心（不应该发生），返回0
        if receiver.node_id == self.physical_center.node_id:
            return 0.0
        
        # 计算数据包大小
        if self.enable_accumulative_data_size:
            # 累积模式：数据包大小 = base_data_size × 路径节点数
            if path is not None:
                path_node_count = len(path)
            else:
                # 如果路径未提供，尝试从receiver获取（降级处理）
                path_node_count = 1
            data_size = self.base_data_size * path_node_count
        else:
            # 固定大小模式：数据包大小固定为 base_data_size
            data_size = self.base_data_size
        
        # 使用与ADCR._energy_consume_one_hop相同的能耗模型
        # 临时修改B参数
        original_B_receiver = receiver.B
        original_B_pc = self.physical_center.B
        
        receiver.B = data_size
        self.physical_center.B = data_size
        
        # 计算双向通信能耗
        Eu = receiver.energy_consumption(target_node=self.physical_center, transfer_WET=False)
        Ev = self.physical_center.energy_consumption(target_node=receiver, transfer_WET=False)
        
        # 恢复原始B值
        receiver.B = original_B_receiver
        self.physical_center.B = original_B_pc
        
        # 扣除能量
        receiver.current_energy = max(0.0, receiver.current_energy - Eu)
        self.physical_center.current_energy = max(0.0, self.physical_center.current_energy - Ev)
        
        # 记录信息传输能量消耗（Receiver上报到物理中心）
        # 记录发送方（Receiver）和接收方（物理中心）的能量消耗
        if hasattr(self.vc, 'record_info_transmission_energy'):
            self.vc.record_info_transmission_energy(receiver.node_id, Eu, "path_collector")
            self.vc.record_info_transmission_energy(self.physical_center.node_id, Ev, "path_collector")
        
        total_energy = Eu + Ev
        
        return total_energy


# ==================== 工厂方法 ====================

def create_path_based_collector(virtual_center, **kwargs):
    """
    工厂方法：创建路径信息收集器
    
    :param virtual_center: 虚拟中心实例
    :param kwargs: 其他配置参数
    :return: PathBasedInfoCollector实例
    """
    return PathBasedInfoCollector(virtual_center, **kwargs)

