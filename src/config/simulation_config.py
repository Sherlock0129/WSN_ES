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
    
    # EnergySimulation K值相关参数
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


@dataclass
class ADCRConfig:
    """ADCR链路层配置参数"""
    # 核心算法参数
    round_period: int = 1440  # 重聚类周期（分钟）
    r_neighbor: float = 1.732  # 邻居检测半径（sqrt(3.0)）
    r_min_ch: float = 1.0  # 簇头间最小距离
    c_k: float = 1.2  # K值估计系数
    max_hops: int = 5  # 最大跳数
    plan_paths: bool = True  # 是否规划路径
    consume_energy: bool = True  # 是否消耗能量
    output_dir: str = "adcr"  # 输出目录
    
    # 簇头选择参数
    max_probability: float = 0.9  # 最大选择概率
    min_probability: float = 0.05  # 最小选择概率
    
    # 聚类成本函数参数
    distance_weight: float = 1.0  # 距离权重
    energy_weight: float = 0.2  # 能量权重
    
    # 通信能耗参数
    tx_rx_ratio: float = 0.5  # 发送接收能量比例
    sensor_energy: float = 0.1  # 感知能耗
    
    # 可视化参数
    image_width: int = 900  # 图像宽度
    image_height: int = 700  # 图像高度
    image_scale: int = 3  # 图像缩放
    node_marker_size: int = 7  # 节点标记大小
    ch_marker_size: int = 10  # 簇头标记大小
    vc_marker_size: int = 12  # 虚拟中心标记大小
    line_width: float = 1.0  # 线条宽度
    path_line_width: float = 2.0  # 路径线条宽度


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.node_config = NodeConfig()
        self.network_config = NetworkConfig()
        self.simulation_config = SimulationConfig()
        self.scheduler_config = SchedulerConfig()
        self.adcr_config = ADCRConfig()
        
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
            if 'adcr' in config_data:
                self._update_dataclass(self.adcr_config, config_data['adcr'])
                
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
                'scheduler': asdict(self.scheduler_config),
                'adcr': asdict(self.adcr_config)
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
    
    def create_network(self):
        """创建Network对象"""
        from core.network import Network
        return Network(
            num_nodes=self.network_config.num_nodes,
            low_threshold=self.node_config.low_threshold,
            high_threshold=self.node_config.high_threshold,
            node_initial_energy=self.node_config.initial_energy,
            max_hops=self.network_config.max_hops,
            distribution_mode=self.network_config.distribution_mode,
            network_area_width=self.network_config.network_area_width,
            network_area_height=self.network_config.network_area_height,
            min_distance=self.network_config.min_distance,
            random_seed=self.network_config.random_seed,
            solar_node_ratio=self.network_config.solar_node_ratio,
            mobile_node_ratio=self.network_config.mobile_node_ratio,
            output_dir=self.simulation_config.output_dir
        )
    
    def create_sensor_node(self, node_id: int, position: list, 
                          has_solar: bool = True, is_mobile: bool = False,
                          mobility_pattern: str = None, mobility_params: dict = None):
        """创建SensorNode对象"""
        from core.SensorNode import SensorNode
        return SensorNode(
            node_id=node_id,
            initial_energy=self.node_config.initial_energy,
            low_threshold=self.node_config.low_threshold,
            high_threshold=self.node_config.high_threshold,
            position=position,
            has_solar=has_solar,
            # 电池参数
            capacity=self.node_config.capacity,
            voltage=self.node_config.voltage,
            # 太阳能参数
            solar_efficiency=self.node_config.solar_efficiency,
            solar_area=self.node_config.solar_area,
            max_solar_irradiance=self.node_config.max_solar_irradiance,
            env_correction_factor=self.node_config.env_correction_factor,
            # 传输参数
            energy_char=self.node_config.energy_char,
            energy_elec=self.node_config.energy_elec,
            epsilon_amp=self.node_config.epsilon_amp,
            bit_rate=self.node_config.bit_rate,
            path_loss_exponent=self.node_config.path_loss_exponent,
            energy_decay_rate=self.node_config.energy_decay_rate,
            sensor_energy=self.node_config.sensor_energy,
            # 移动性参数
            is_mobile=is_mobile,
            mobility_pattern=mobility_pattern,
            mobility_params=mobility_params
        )
    
    def create_energy_simulation(self, network, scheduler=None):
        """创建EnergySimulation对象"""
        from core.energy_simulation import EnergySimulation
        return EnergySimulation(
            network=network,
            time_steps=self.simulation_config.time_steps,
            scheduler=scheduler,
            initial_K=self.simulation_config.initial_K,
            K_max=self.simulation_config.K_max,
            hysteresis=self.simulation_config.hysteresis,
            w_b=self.simulation_config.w_b,
            w_d=self.simulation_config.w_d,
            w_l=self.simulation_config.w_l,
            use_lookahead=self.simulation_config.use_lookahead
        )
    
    def create_adcr_link_layer(self, network):
        """创建ADCRLinkLayerVirtual对象"""
        from acdr.adcr_link_layer import ADCRLinkLayerVirtual
        return ADCRLinkLayerVirtual(
            network=network,
            # 核心算法参数
            round_period=self.adcr_config.round_period,
            r_neighbor=self.adcr_config.r_neighbor,
            r_min_ch=self.adcr_config.r_min_ch,
            c_k=self.adcr_config.c_k,
            max_hops=self.adcr_config.max_hops,
            plan_paths=self.adcr_config.plan_paths,
            consume_energy=self.adcr_config.consume_energy,
            output_dir=self.adcr_config.output_dir,
            # 簇头选择参数
            max_probability=self.adcr_config.max_probability,
            min_probability=self.adcr_config.min_probability,
            # 聚类成本函数参数
            distance_weight=self.adcr_config.distance_weight,
            energy_weight=self.adcr_config.energy_weight,
            # 通信能耗参数
            tx_rx_ratio=self.adcr_config.tx_rx_ratio,
            sensor_energy=self.adcr_config.sensor_energy,
            # 可视化参数
            image_width=self.adcr_config.image_width,
            image_height=self.adcr_config.image_height,
            image_scale=self.adcr_config.image_scale,
            node_marker_size=self.adcr_config.node_marker_size,
            ch_marker_size=self.adcr_config.ch_marker_size,
            vc_marker_size=self.adcr_config.vc_marker_size,
            line_width=self.adcr_config.line_width,
            path_line_width=self.adcr_config.path_line_width
        )
    
    def __str__(self) -> str:
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
