# src/info_collection/path_based_collector.py
# -*- coding: utf-8 -*-
"""
基于路径的机会主义信息收集器

设计理念：
- 利用能量传输路径收集节点信息（piggyback，搭载）
- 路径内节点：实时采集最新信息
- 路径外节点：基于历史信息 + 自然损耗模型估算
- Receiver（路径最后一个节点）作为信息汇聚点上报虚拟中心

优势：
- 无需额外通信开销（信息搭载在传能路径上）
- 更新频率高（每次传能都更新）
- 信息新鲜度高（路径节点为实时采集）
- 能量消耗低（无专门的上报通信）

"""

from __future__ import annotations
from typing import List, Dict, Set, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from core.SensorNode import SensorNode
    from acdr.virtual_center import VirtualCenter


class PathBasedInfoCollector:
    """
    基于能量传输路径的信息收集器
    
    工作流程：
    1. 能量传输时，路径上的节点收集自身实时信息
    2. Receiver（路径终点）汇总路径上所有节点信息
    3. Receiver估算路径外节点的信息（基于历史+模型）
    4. Receiver将全部信息上报到虚拟中心
    """
    
    def __init__(self, 
                 virtual_center: VirtualCenter,
                 energy_mode: str = "free",
                 enable_logging: bool = True,
                 # 估算参数
                 decay_rate: float = 5.0,
                 use_solar_model: bool = True,
                 # 优化选项
                 batch_update: bool = True):
        """
        初始化路径信息收集器
        
        :param virtual_center: 虚拟中心实例
        :param energy_mode: 能量消耗模式 ("free": 零能耗, "full": 完全真实)
        :param enable_logging: 是否启用日志
        :param decay_rate: 自然衰减率（J/分钟，用于估算）
        :param use_solar_model: 是否使用太阳能模型进行估算
        :param batch_update: 是否批量更新虚拟中心
        """
        self.vc = virtual_center
        self.energy_mode = energy_mode
        self.enable_logging = enable_logging
        self.decay_rate = decay_rate
        self.use_solar_model = use_solar_model
        self.batch_update = batch_update
        
        # 统计信息
        self.total_collections = 0
        self.total_real_info = 0
        self.total_estimated_info = 0
        self.total_energy_consumed = 0.0  # 总能量消耗
        
        self._log(f"[PathCollector] 初始化完成 - 能量模式={energy_mode}, "
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
        2. 估算路径外节点的信息
        3. 批量更新虚拟中心
        
        :param path: 能量传输路径 [donor, relay1, ..., receiver]
        :param all_nodes: 所有网络节点
        :param current_time: 当前时间步（分钟）
        :return: 统计信息 {'real': int, 'estimated': int}
        """
        if not path:
            self._log("[PathCollector] 警告：路径为空，跳过信息收集")
            return {'real': 0, 'estimated': 0}
        
        self.total_collections += 1
        
        # 1. 收集路径节点的实时信息
        path_node_ids = {n.node_id for n in path}
        real_info = self._collect_real_info(path, current_time)
        
        # 2. 估算路径外节点的信息
        other_nodes = [n for n in all_nodes if n.node_id not in path_node_ids]
        estimated_info = self._estimate_other_nodes(other_nodes, current_time)
        
        # 3. 合并所有信息
        all_info = {**real_info, **estimated_info}
        
        # 4. 能量消耗结算（如果启用）
        if self.energy_mode == "full":
            energy_cost = self._settle_energy_consumption(path, all_info)
            self.total_energy_consumed += energy_cost
        
        # 5. 更新虚拟中心
        self._update_virtual_center(all_info, current_time)
        
        # 6. 更新统计
        real_count = len(real_info)
        est_count = len(estimated_info)
        self.total_real_info += real_count
        self.total_estimated_info += est_count
        
        self._log(f"[PathCollector] 收集完成 - 实时: {real_count}, 估算: {est_count}, "
                 f"路径长度: {len(path)}, Receiver: Node {path[-1].node_id}")
        
        return {'real': real_count, 'estimated': est_count}
    
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
                'freshness': current_time,  # 实时采集，新鲜度为当前时间
                'arrival_time': current_time,  # 立即到达
                'is_estimated': False,
                'confidence': 1.0  # 实时信息，100%置信度
            }
        
        return info
    
    def _estimate_other_nodes(self, other_nodes: List[SensorNode], 
                              current_time: int) -> Dict[int, Dict]:
        """
        估算路径外节点的信息
        
        估算策略：
        1. 从虚拟中心获取节点的历史信息
        2. 计算时间差
        3. 基于自然衰减和太阳能采集模型估算当前能量
        
        :param other_nodes: 路径外的节点列表
        :param current_time: 当前时间
        :return: {node_id: estimated_info_dict}
        """
        estimated = {}
        
        for node in other_nodes:
            # 从虚拟中心获取历史信息
            old_info = self.vc.get_node_info(node.node_id)
            
            if old_info is None:
                # 如果虚拟中心没有历史信息，使用节点当前实际值
                # 注意：这里获取的是节点对象的真实值，但标记为"估算"
                # 因为在实际应用中，虚拟中心无法直接访问路径外节点
                estimated_energy = node.current_energy
                freshness = current_time
                confidence = 0.5  # 中等置信度（无历史基准）
            else:
                # 基于历史信息估算
                time_elapsed = current_time - old_info['freshness']
                
                if time_elapsed == 0:
                    # 刚刚更新过，直接使用历史值
                    estimated_energy = old_info['energy']
                    confidence = 1.0
                else:
                    # 估算能量变化
                    estimated_energy, confidence = self._estimate_energy(
                        node=node,
                        old_energy=old_info['energy'],
                        time_elapsed=time_elapsed,
                        current_time=current_time
                    )
                
                freshness = old_info['freshness']  # 保持原有新鲜度
            
            estimated[node.node_id] = {
                'energy': estimated_energy,
                'position': tuple(node.position),
                'is_solar': node.has_solar,
                'freshness': freshness,
                'arrival_time': current_time,
                'is_estimated': True,
                'confidence': confidence
            }
        
        return estimated
    
    def _estimate_energy(self, node: SensorNode, old_energy: float, 
                        time_elapsed: int, current_time: int) -> tuple:
        """
        估算节点能量（基于物理模型）
        
        模型：E_new = E_old + 采集 - 损耗
        
        :param node: 节点对象（用于获取参数）
        :param old_energy: 历史能量值
        :param time_elapsed: 经过的时间（分钟）
        :param current_time: 当前时间
        :return: (estimated_energy, confidence)
        """
        # 1. 自然衰减（固定速率，从节点配置获取）
        decay_per_minute = getattr(node, 'energy_decay_rate', self.decay_rate)
        total_decay = decay_per_minute * time_elapsed
        
        # 2. 太阳能采集（如果启用且节点有太阳能）
        total_harvest = 0.0
        if self.use_solar_model and node.has_solar:
            # 使用节点的实际太阳能采集方法估算
            total_harvest = self._estimate_solar_harvest(node, time_elapsed, current_time)
        
        # 3. 计算估算能量
        estimated = old_energy + total_harvest - total_decay
        estimated = max(0.0, estimated)  # 确保非负
        estimated = min(estimated, node.capacity * node.V * 3600)  # 不超过容量
        
        # 4. 计算置信度（时间越长，置信度越低）
        # 置信度衰减模型：confidence = exp(-lambda * t)
        # 假设30分钟后置信度降至50%
        decay_factor = math.log(2) / 30.0  # ln(2)/30 ≈ 0.023
        confidence = math.exp(-decay_factor * time_elapsed)
        confidence = max(0.1, min(1.0, confidence))  # 限制在[0.1, 1.0]
        
        return estimated, confidence
    
    def _estimate_solar_harvest(self, node: SensorNode, time_elapsed: int, 
                                current_time: int) -> float:
        """
        估算太阳能采集量（简化模型）
        
        策略：
        - 计算time_elapsed期间的平均采集量
        - 考虑昼夜周期（简化为余弦函数）
        
        :param node: 节点对象
        :param time_elapsed: 时间跨度（分钟）
        :param current_time: 当前时间（分钟）
        :return: 估算的采集总量（J）
        """
        # 简化策略：使用当前时刻的采集率作为平均值
        # 注意：这是一个近似，更准确的方法需要积分
        avg_harvest = node.energy_harvest(current_time)
        total = avg_harvest * time_elapsed
        
        return total
    
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
                freshness=info['freshness'],
                arrival_time=info['arrival_time'],
                position=info['position'],
                is_solar=info['is_solar'],
                cluster_id=None,  # PathBased不使用簇
                data_size=None    # 可选：记录路径长度等信息
            )
        
        self._log(f"[PathCollector] 虚拟中心更新: {len(all_info)} 个节点")
    
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
        print(f"实时信息: {stats['total_real_info']} ({stats['real_ratio']*100:.1f}%)")
        print(f"估算信息: {stats['total_estimated_info']} ({stats['estimated_ratio']*100:.1f}%)")
        print(f"平均每次收集:")
        print(f"  - 实时: {stats['avg_real_per_collection']:.1f} 个节点")
        print(f"  - 估算: {stats['avg_estimated_per_collection']:.1f} 个节点")
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
    
    def _settle_energy_consumption(self, path: List[SensorNode], 
                                   collected_info: Dict) -> float:
        """
        结算信息收集的能量消耗（仅在 energy_mode="full" 时调用）
        
        能量消耗包括：
        1. 路径逐跳信息传递：信息沿路径累积传递，每跳消耗能量
        2. 虚拟跳上报：Receiver → 虚拟中心的上报能量
        
        :param path: 传能路径 [donor, relay..., receiver]
        :param collected_info: 收集的信息（用于计算数据量）
        :return: 总能量消耗（J）
        """
        if len(path) < 1:
            return 0.0
        
        # 1. 计算路径逐跳能耗（信息沿路径传递）
        path_energy = self._calculate_path_hop_energy(path)
        
        # 2. 计算虚拟跳能耗（Receiver → 虚拟中心）
        receiver = path[-1]
        virtual_energy = self._calculate_virtual_hop_energy(receiver, collected_info)
        
        total_energy = path_energy + virtual_energy
        
        self._log(f"[PathCollector] 能量消耗 - 路径逐跳={path_energy:.2f}J, "
                 f"虚拟跳={virtual_energy:.2f}J, 总计={total_energy:.2f}J")
        
        return total_energy
    
    def _calculate_path_hop_energy(self, path: List[SensorNode]) -> float:
        """
        计算路径逐跳能耗（信息沿路径传递）
        
        模型：
        - 每个节点收集自己的信息并转发前面节点的信息
        - Hop i: 传递前i+1个节点的信息
        - 能耗 = E_tx(sender) + E_rx(receiver)
        
        :param path: 传能路径
        :return: 路径总能量消耗（J）
        """
        if len(path) < 2:
            return 0.0
        
        total_energy = 0.0
        info_packet_size = 100  # bits/节点（node_id + energy + position + ...）
        
        # 沿路径逐跳累积信息并计算能耗
        for i in range(len(path) - 1):
            sender = path[i]
            receiver = path[i + 1]
            
            # 累积信息大小：前面i+1个节点的信息
            accumulated_size = (i + 1) * info_packet_size
            
            # 计算通信能耗（参考SensorNode.energy_consumption）
            # E_tx = E_elec * B + epsilon_amp * B * d^tau
            # E_rx = E_elec * B
            distance = sender.distance_to(receiver)
            E_tx = sender.E_elec * accumulated_size + \
                   sender.epsilon_amp * accumulated_size * (distance ** sender.tau)
            E_rx = receiver.E_elec * accumulated_size
            
            # 扣除能量
            sender.current_energy -= E_tx
            receiver.current_energy -= E_rx
            
            total_energy += (E_tx + E_rx)
        
        return total_energy
    
    def _calculate_virtual_hop_energy(self, receiver: SensorNode, 
                                      collected_info: Dict) -> float:
        """
        计算虚拟跳能耗（Receiver → 虚拟中心）
        
        使用虚拟中心的 settle_virtual_hop_energy 方法（与ADCR相同）
        
        :param receiver: 路径终点节点（信息汇聚点）
        :param collected_info: 收集的全部信息量
        :return: 虚拟跳能量消耗（J）
        """
        # 数据大小 = 所有节点信息数量 × 单节点信息包大小
        info_packet_size = 100  # bits/节点
        data_size = len(collected_info) * info_packet_size
        
        # 使用虚拟中心的能量结算方法（与ADCR相同）
        E_virtual = self.vc.settle_virtual_hop_energy(
            sender=receiver,
            data_size=data_size,
            tx_rx_ratio=0.5,  # 默认值，与ADCR一致
            sensor_energy=0.1  # 默认值，与ADCR一致
        )
        
        return E_virtual


# ==================== 工厂方法 ====================

def create_path_based_collector(virtual_center, **kwargs):
    """
    工厂方法：创建路径信息收集器
    
    :param virtual_center: 虚拟中心实例
    :param kwargs: 其他配置参数
    :return: PathBasedInfoCollector实例
    """
    return PathBasedInfoCollector(virtual_center, **kwargs)

