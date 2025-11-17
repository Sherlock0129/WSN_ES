"""
智能被动传能管理器
集中管理所有被动传能相关的逻辑和决策
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any, List


class PassiveTransferManager:
    """
    智能被动传能管理器
    
    负责综合决策是否应该触发能量传输，基于以下因素：
    1. 检查间隔控制
    2. 冷却期机制
    3. 低能量节点比例
    4. 能量分布方差
    5. 预测性分析
    """
    
    def __init__(self, 
                 passive_mode: bool = True,
                 check_interval: int = 10,
                 critical_ratio: float = 0.2,
                 energy_variance_threshold: float = 0.3,
                 cooldown_period: int = 30,
                 predictive_window: int = 60,
                 node_info_manager=None):
        """
        初始化被动传能管理器
        
        :param passive_mode: 是否启用被动模式（False则使用60分钟定时触发）
        :param check_interval: 检查间隔（分钟）
        :param critical_ratio: 低能量节点临界比例（0-1）
        :param energy_variance_threshold: 能量方差阈值
        :param cooldown_period: 冷却期（分钟）
        :param predictive_window: 预测窗口（分钟）
        :param node_info_manager: NodeInfoManager实例（可选，用于虚拟能量层）
        """
        self.passive_mode = passive_mode
        self.check_interval = check_interval
        self.critical_ratio = critical_ratio
        self.energy_variance_threshold = energy_variance_threshold
        self.cooldown_period = cooldown_period
        self.predictive_window = predictive_window
        self.node_info_manager = node_info_manager  # 虚拟能量层管理器
        
        # 状态变量
        self.last_transfer_time = -cooldown_period  # 上次传能时间
        self.energy_history = []  # 能量历史记录（用于预测）
        
        # 统计信息
        self.trigger_count = 0  # 触发次数
        self.trigger_reasons_stats = {}  # 触发原因统计
    
    def should_trigger_transfer(self, t: int, network) -> Tuple[bool, Optional[str]]:
        """
        智能综合决策：是否应该触发能量传输
        
        ⚠️ 优先使用虚拟能量层（NodeInfoManager）获取节点能量信息，
        避免直接访问SensorNode（"上帝视角"），保持分布式系统模型一致性。
        
        :param t: 当前时间步
        :param network: 网络对象
        :return: (是否触发, 触发原因)
        """
        # 如果不是被动模式，使用传统的定时触发（每60分钟）
        if not self.passive_mode:
            # 使用 check_interval 作为定时触发的间隔
            if self.check_interval <= 0:
                return False, None  # 间隔小于等于0时，禁用定时触发
            should_trigger = (t % self.check_interval == 0)
            return should_trigger, "定时触发" if should_trigger else None
        
        # 1. 检查间隔：只在指定间隔检查
        if t % self.check_interval != 0:
            return False, None
        
        # 2. 冷却期检查：避免过于频繁的传能
        if t - self.last_transfer_time < self.cooldown_period:
            return False, None
        
        # 3. 获取节点能量状态（直接从真实网络获取，仅用于触发决策）
        # 这一步是为了确保触发决策基于最新的能量状态，而不影响AOI的计算
        physical_center = network.get_physical_center() if hasattr(network, 'get_physical_center') else None
        physical_center_id = physical_center.node_id if physical_center else None

        regular_nodes = [node for node in network.nodes if node.node_id != physical_center_id]
        total_nodes = len(regular_nodes)
        
        if total_nodes == 0:
            return False, None
        
        # 直接从真实节点获取能量和阈值，仅用于本次决策
        energies = np.array([node.current_energy for node in regular_nodes])
        thresholds = np.array([node.low_threshold_energy for node in regular_nodes])

        # 3. 低能量节点比例检查(已禁用)
        low_energy_nodes = energies < thresholds
        # low_energy_ratio = np.sum(low_energy_nodes) / total_nodes
        low_energy_ratio = False

        # 4. 能量方差检查（归一化方差，考虑能量不均衡）
        if len(energies) > 0 and np.mean(energies) > 0:
            energy_cv = np.std(energies) / np.mean(energies)  # 变异系数
        else:
            energy_cv = 0
        
        # 5. 预测性检查：基于能量消耗速率（已禁用）
        # predict_critical = self._check_predictive_trigger(energies, thresholds, total_nodes)
        predict_critical = False  # 禁用预测性触发
        
        # 记录当前能量状态（用于未来预测）（已禁用）
        # self._update_energy_history(energies)
        
        # 综合决策逻辑
        should_trigger, reasons = self._make_decision(
            low_energy_ratio, energy_cv, predict_critical,
            energies, thresholds
        )
        
        if should_trigger:
            reason_str = " | ".join(reasons)
            self._record_trigger(reason_str)
            return True, reason_str
        
        return False, None
    
    def _check_predictive_trigger(self, energies: np.ndarray,
                                  thresholds: np.ndarray,
                                  total_nodes: int) -> bool:
        """
        预测性检查：基于能量消耗速率预测未来

        注意：energies 和 thresholds 数组已经排除了物理中心节点

        :param energies: 当前普通节点能量数组
        :param thresholds: 普通节点阈值数组
        :param total_nodes: 普通节点总数
        :return: 是否预测到危机
        """
        predict_critical = False
        
        if len(self.energy_history) >= 2:
            # 计算平均能量下降速率
            recent_energies = self.energy_history[-min(5, len(self.energy_history)):]
            
            if len(recent_energies) >= 2:
                energy_rates = []
                for i in range(len(recent_energies) - 1):
                    rate = (recent_energies[i] - recent_energies[i + 1]) / self.check_interval
                    energy_rates.append(rate)
                avg_rate = np.mean(energy_rates)
                
                # 预测未来predictive_window分钟后的能量
                predicted_energies = energies - avg_rate * self.predictive_window
                predicted_low = np.sum(predicted_energies < thresholds) / total_nodes
                
                if predicted_low > self.critical_ratio * 0.5:  # 预测性阈值更宽松
                    predict_critical = True
        
        return predict_critical
    
    def _update_energy_history(self, energies: np.ndarray):
        """
        更新能量历史记录

        注意：energies 数组已经排除了物理中心节点

        :param energies: 当前普通节点能量数组
        """
        self.energy_history.append(np.mean(energies))
        
        # 只保留最近20次记录
        if len(self.energy_history) > 20:
            self.energy_history.pop(0)
    
    def _make_decision(self, low_energy_ratio: float,
                      energy_cv: float,
                      predict_critical: bool,
                      energies: np.ndarray,
                      thresholds: np.ndarray) -> Tuple[bool, List[str]]:
        """
        综合决策逻辑

        注意：所有参数都只基于普通节点（已排除物理中心节点）

        :param low_energy_ratio: 普通节点中低能量节点比例
        :param energy_cv: 普通节点能量变异系数
        :param predict_critical: 是否预测到普通节点危机
        :param energies: 普通节点能量数组
        :param thresholds: 普通节点阈值数组
        :return: (是否触发, 原因列表)
        """
        reasons = []
        should_trigger = False
        
        # 条件1: 严重情况 - 低能量节点比例超过临界值（已禁用）
        # if low_energy_ratio > self.critical_ratio:
        #     should_trigger = True
        #     reasons.append(f"低能量节点比例={low_energy_ratio:.2%}>{self.critical_ratio:.2%}")
        
        # 条件2: 能量分布严重不均（仅保留此条件）
        if energy_cv > self.energy_variance_threshold:
            should_trigger = True
            reasons.append(f"能量变异系数={energy_cv:.3f}>{self.energy_variance_threshold:.3f}")
        
        # 条件3: 预测性触发（已禁用）
        # if predict_critical:
        #     should_trigger = True
        #     reasons.append(f"预测{self.predictive_window}分钟后将出现能量危机")
        
        # 条件4: 紧急情况 - 存在极低能量节点（低于阈值的50%）（已禁用）
        # critical_nodes = energies < (thresholds * 0.5)
        # if np.any(critical_nodes):
        #     should_trigger = True
        #     critical_count = np.sum(critical_nodes)
        #     reasons.append(f"存在{critical_count}个极低能量节点（<阈值50%）")
        
        return should_trigger, reasons
    
    def _record_trigger(self, reason: str):
        """
        记录触发事件
        
        :param reason: 触发原因
        """
        self.trigger_count += 1
        
        # 统计触发原因
        for r in reason.split(" | "):
            key = r.split("=")[0] if "=" in r else r
            self.trigger_reasons_stats[key] = self.trigger_reasons_stats.get(key, 0) + 1
    
    def update_last_transfer_time(self, t: int):
        """
        更新上次传能时间
        
        :param t: 当前时间步
        """
        self.last_transfer_time = t
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        :return: 统计信息字典
        """
        return {
            'trigger_count': self.trigger_count,
            'trigger_reasons': self.trigger_reasons_stats,
            'energy_history_length': len(self.energy_history),
            'mode': '智能被动传能' if self.passive_mode else '定时主动传能'
        }
    
    def reset(self):
        """重置管理器状态"""
        self.last_transfer_time = -self.cooldown_period
        self.energy_history = []
        self.trigger_count = 0
        self.trigger_reasons_stats = {}
    
    def get_config(self) -> Dict[str, Any]:
        """
        获取当前配置
        
        :return: 配置字典
        """
        return {
            'passive_mode': self.passive_mode,
            'check_interval': self.check_interval,
            'critical_ratio': self.critical_ratio,
            'energy_variance_threshold': self.energy_variance_threshold,
            'cooldown_period': self.cooldown_period,
            'predictive_window': self.predictive_window
        }


def compare_passive_modes(config_manager_class, create_scheduler_func, logger):
    """
    比较智能被动传能 vs 传统主动传能的性能
    
    :param config_manager_class: ConfigManager类
    :param create_scheduler_func: 创建调度器的函数
    :param logger: 日志记录器
    :return: 比较结果字典
    """
    logger.info("=" * 60)
    logger.info("开始比较智能被动传能 vs 传统主动传能")
    logger.info("=" * 60)
    
    modes_to_test = [
        ("智能被动传能(默认)", {"passive_mode": True, "check_interval": 10, "critical_ratio": 0.2}),
        ("智能被动传能(快速)", {"passive_mode": True, "check_interval": 5, "critical_ratio": 0.15}),
        ("智能被动传能(节能)", {"passive_mode": True, "check_interval": 20, "critical_ratio": 0.3}),
        ("传统主动传能(60分钟)", {"passive_mode": False}),
    ]
    
    results = {}
    
    for mode_name, mode_config in modes_to_test:
        logger.info(f"\n测试模式: {mode_name}")
        logger.info("-" * 60)
        
        # 创建配置
        config_manager = config_manager_class()
        config_manager.simulation_config.enable_energy_sharing = True
        config_manager.simulation_config.time_steps = 10080  # 测试7天
        
        # 应用模式配置
        for key, value in mode_config.items():
            setattr(config_manager.simulation_config, key, value)
        
        try:
            # 创建网络和调度器
            network = config_manager.create_network()
            scheduler = create_scheduler_func(config_manager)
            
            # 运行仿真
            simulation = config_manager.create_energy_simulation(network, scheduler)
            simulation.simulate()
            
            # 收集统计信息
            stats = simulation.print_statistics()
            
            # 获取被动传能管理器的统计信息
            passive_stats = {}
            if hasattr(simulation, 'passive_manager'):
                passive_stats = simulation.passive_manager.get_statistics()
            
            results[mode_name] = {
                'avg_variance': stats['avg_variance'],
                'total_loss_energy': stats['total_loss_energy'],
                'energy_efficiency': stats['total_received_energy'] / stats['total_sent_energy'] if stats['total_sent_energy'] > 0 else 0,
                'transfer_count': len([t for t in range(config_manager.simulation_config.time_steps) 
                                      if t in simulation.plans_by_time]),
                'passive_stats': passive_stats
            }
            logger.info(f"✓ {mode_name} 测试完成")
            
        except Exception as e:
            logger.error(f"✗ {mode_name} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
            results[mode_name] = None
    
    # 输出比较结果
    logger.info("\n" + "=" * 60)
    logger.info("传能模式性能比较结果:")
    logger.info("=" * 60)
    for mode_name, result in results.items():
        if result:
            logger.info(f"\n{mode_name}:")
            logger.info(f"  传能次数: {result['transfer_count']} 次")
            logger.info(f"  平均方差: {result['avg_variance']:.4f}")
            logger.info(f"  总能量损失: {result['total_loss_energy']:.4f} J")
            logger.info(f"  能量效率: {result['energy_efficiency']:.2%}")
            
            # 输出被动传能统计
            if result['passive_stats']:
                logger.info(f"  触发统计: {result['passive_stats']['trigger_count']} 次")
                if result['passive_stats']['trigger_reasons']:
                    logger.info(f"  触发原因分布: {result['passive_stats']['trigger_reasons']}")
        else:
            logger.info(f"\n{mode_name}: 测试失败")
    
    return results

