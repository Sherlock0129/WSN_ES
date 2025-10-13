"""
无线传感器网络仿真配置管理
统一管理所有仿真参数，支持从文件加载和动态配置
"""

import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class NodeConfig:
    """节点配置参数"""
    initial_energy: float = 40000.0
    low_threshold: float = 0.1
    high_threshold: float = 0.9
    capacity: float = 5200.0  # mAh
    voltage: float = 3.7  # V
    solar_efficiency: float = 0.2
    solar_area: float = 0.1  # m^2
    max_solar_irradiance: float = 1500.0  # W/m^2
    env_correction_factor: float = 1.0
    energy_char: float = 1000.0  # J
    energy_elec: float = 1e-4  # J per bit
    epsilon_amp: float = 1e-5  # J per bit per distance^2
    bit_rate: float = 1000000.0  # bits
    path_loss_exponent: float = 2.0
    energy_decay_rate: float = 5.0  # J per time step
    sensor_energy: float = 0.1  # J per time step


@dataclass
class NetworkConfig:
    """网络配置参数"""
    num_nodes: int = 25
    max_hops: int = 3
    distribution_mode: str = "random"  # "uniform" or "random"
    network_area_width: float = 10.0
    network_area_height: float = 10.0
    min_distance: float = 0.5
    random_seed: int = 129
    solar_node_ratio: float = 0.6  # 60% 节点有太阳能
    mobile_node_ratio: float = 0.1  # 10% 节点可移动


@dataclass
class SimulationConfig:
    """仿真配置参数"""
    time_steps: int = 10080  # 一周的分钟数
    energy_transfer_interval: int = 60  # 每60分钟执行一次能量传输
    use_fixed_network: bool = True
    fixed_seed: int = 130
    output_dir: str = "data"
    log_level: str = "INFO"
    
    # EnergySimulation 参数
    initial_K: int = 1  # 初始K值
    K_max: int = 24  # 最大K值
    hysteresis: float = 0.2  # 滞回阈值
    w_b: float = 0.8  # 均衡改进权重
    w_d: float = 0.8  # 有效送达量权重
    w_l: float = 1.5  # 损耗惩罚权重
    use_lookahead: bool = False  # 是否使用前瞻模拟


@dataclass
class SchedulerConfig:
    """调度器配置参数"""
    scheduler_type: str = "LyapunovScheduler"  # 默认调度器类型
    # LyapunovScheduler 参数
    lyapunov_v: float = 0.5
    lyapunov_k: int = 3
    # ClusterScheduler 参数
    cluster_round_period: int = 360
    cluster_p_ch: float = 0.05
    # PredictionScheduler 参数
    prediction_alpha: float = 0.6
    prediction_horizon: int = 60
    # PowerControlScheduler 参数
    power_target_eta: float = 0.25
    # 自适应K值参数
    adaptive_k_max: int = 24
    adaptive_hysteresis: float = 0.2
    adaptive_w_b: float = 0.8  # 均衡改进权重
    adaptive_w_d: float = 0.8  # 有效送达量权重
    adaptive_w_l: float = 1.5  # 损耗惩罚权重


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.node_config = NodeConfig()
        self.network_config = NetworkConfig()
        self.simulation_config = SimulationConfig()
        self.scheduler_config = SchedulerConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str) -> None:
        """从JSON文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 更新各个配置对象
            if 'node' in config_data:
                self._update_dataclass(self.node_config, config_data['node'])
            if 'network' in config_data:
                self._update_dataclass(self.network_config, config_data['network'])
            if 'simulation' in config_data:
                self._update_dataclass(self.simulation_config, config_data['simulation'])
            if 'scheduler' in config_data:
                self._update_dataclass(self.scheduler_config, config_data['scheduler'])
                
            print(f"配置已从 {config_file} 加载")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
            print("使用默认配置")
    
    def save_to_file(self, config_file: str) -> None:
        """保存配置到JSON文件"""
        try:
            config_data = {
                'node': asdict(self.node_config),
                'network': asdict(self.network_config),
                'simulation': asdict(self.simulation_config),
                'scheduler': asdict(self.scheduler_config)
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            print(f"配置已保存到 {config_file}")
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]) -> None:
        """更新dataclass对象的字段"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def get_network_config_dict(self) -> Dict[str, Any]:
        """获取网络配置字典（兼容现有代码）"""
        return {
            "num_nodes": self.network_config.num_nodes,
            "low_threshold": self.node_config.low_threshold,
            "high_threshold": self.node_config.high_threshold,
            "node_initial_energy": self.node_config.initial_energy,
            "max_hops": self.network_config.max_hops,
            "random_seed": self.network_config.random_seed,
            "distribution_mode": self.network_config.distribution_mode,
            "network_area": {
                "width": self.network_config.network_area_width,
                "height": self.network_config.network_area_height
            },
            "min_distance": self.network_config.min_distance,
            "output_dir": self.simulation_config.output_dir
        }
    
    def get_scheduler_params(self) -> Dict[str, Any]:
        """获取调度器参数字典"""
        scheduler_type = self.scheduler_config.scheduler_type
        
        if scheduler_type == "LyapunovScheduler":
            return {
                "V": self.scheduler_config.lyapunov_v,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        elif scheduler_type == "ClusterScheduler":
            return {
                "round_period": self.scheduler_config.cluster_round_period,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops,
                "p_ch": self.scheduler_config.cluster_p_ch
            }
        elif scheduler_type == "PredictionScheduler":
            return {
                "alpha": self.scheduler_config.prediction_alpha,
                "horizon_min": self.scheduler_config.prediction_horizon,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        elif scheduler_type == "PowerControlScheduler":
            return {
                "target_eta": self.scheduler_config.power_target_eta,
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        elif scheduler_type == "BaselineHeuristic":
            return {
                "K": self.scheduler_config.lyapunov_k,
                "max_hops": self.network_config.max_hops
            }
        else:
            raise ValueError(f"未知的调度器类型: {scheduler_type}")
    
    
    def __str__(self) -> str:
        """返回配置的字符串表示"""
        return json.dumps({
            'node': asdict(self.node_config),
            'network': asdict(self.network_config),
            'simulation': asdict(self.simulation_config),
            'scheduler': asdict(self.scheduler_config)
        }, indent=2, ensure_ascii=False)


# 全局配置实例
config_manager = ConfigManager()


def get_config() -> ConfigManager:
    """获取全局配置管理器实例"""
    return config_manager


def load_config(config_file: str) -> ConfigManager:
    """加载配置文件"""
    global config_manager
    config_manager = ConfigManager(config_file)
    return config_manager
