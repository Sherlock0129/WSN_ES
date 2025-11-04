# src/scheduling/info_node.py
# -*- coding: utf-8 -*-
"""
信息节点类 - 基于物理中心节点信息表的轻量级节点

用于调度器和路由算法，所有属性和方法与SensorNode保持一致，
确保计算结果完全相同。
"""

from __future__ import print_function


class InfoNode(object):
    """
    信息节点 - 基于节点信息表的轻量级节点代理
    
    特点：
    - 只包含调度和路由需要的属性和方法
    - 所有计算公式与SensorNode完全一致
    - 数据来源于物理中心的节点信息表
    """
    
    def __init__(self, node_id, energy, position, is_solar=False,
                 is_physical_center=False,
                 # 能量传输参数（从真实节点复制）
                 energy_char=300.0,
                 energy_elec=1e-4,
                 epsilon_amp=1e-5,
                 bit_rate=1000000.0,
                 path_loss_exponent=2.0,
                 sensor_energy=0.1):
        """
        初始化信息节点
        
        :param node_id: 节点ID
        :param energy: 当前能量（从信息表获取）
        :param position: 位置 [x, y] 或 (x, y)
        :param is_solar: 是否有太阳能
        :param is_physical_center: 是否是物理中心节点
        :param energy_char: 特征能量（单次传输能量）
        :param energy_elec: 电路能耗
        :param epsilon_amp: 功放能耗系数
        :param bit_rate: 比特率
        :param path_loss_exponent: 路径损耗指数
        :param sensor_energy: 传感器能耗
        """
        self.node_id = node_id
        self.current_energy = energy
        self.position = list(position) if position else [0.0, 0.0]
        self.has_solar = is_solar
        self.is_physical_center = is_physical_center
        
        # 能量传输参数
        self.E_char = energy_char  # 注意：调度器使用E_char访问
        self.energy_char = energy_char  # 兼容不同访问方式
        self.energy_elec = energy_elec
        self.epsilon_amp = epsilon_amp
        self.bit_rate = bit_rate
        self.path_loss_exponent = path_loss_exponent
        self.sensor_energy = sensor_energy
        
        # 标记为InfoNode（用于调试和特殊处理）
        self._is_info_node = True
    
    def distance_to(self, other):
        """
        计算到另一个节点的欧式距离
        
        与SensorNode.distance_to()完全相同
        
        :param other: 另一个节点（InfoNode或SensorNode）
        :return: 欧式距离
        """
        dx = self.position[0] - other.position[0]
        dy = self.position[1] - other.position[1]
        return (dx**2 + dy**2) ** 0.5
    
    def energy_transfer_efficiency(self, target_node):
        """
        计算无线能量传输效率
        
        ⚠️ 与SensorNode.energy_transfer_efficiency()完全相同的公式！
        
        :param target_node: 目标节点
        :return: 效率 (0~1)
        """
        d = self.distance_to(target_node)
        eta_0 = 0.6  # 1米处最大效率
        gamma = 2.0  # 衰减因子
        
        if d <= 1.0:
            # 距离≤1m时，效率为eta_0到1之间的线性插值
            efficiency = eta_0 + (1.0 - eta_0) * (1.0 - d)
        else:
            # 距离>1m时，使用指数衰减
            efficiency = eta_0 * (1.0 / (d ** gamma))
        
        return min(1.0, max(0.0, efficiency))
    
    def energy_harvest(self, t):
        """
        简化的能量采集估算
        
        注意：信息节点无法实时计算太阳能采集，
        使用简化策略：有太阳能的节点返回保守估计值
        
        :param t: 时间步（分钟）
        :return: 估算的采集能量（焦耳）
        """
        if not self.has_solar:
            return 0.0
        
        # 保守估计：假设平均太阳能采集率
        # 可以根据历史数据或时间调整
        # 这里使用固定的保守值
        average_harvest_rate = 0.5  # J/min（保守估计）
        return average_harvest_rate
    
    def __repr__(self):
        return "InfoNode(id={}, E={:.1f}J, pos=[{:.1f}, {:.1f}])".format(
            self.node_id, self.current_energy, 
            self.position[0], self.position[1]
        )
    
    def __str__(self):
        return "InfoNode{}".format(self.node_id)

