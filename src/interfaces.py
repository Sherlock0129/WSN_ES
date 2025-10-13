"""
无线传感器网络仿真接口层
定义清晰的模块接口，提高代码可维护性和可扩展性
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EnergyTransferPlan:
    """能量传输计划"""
    donor_id: int
    receiver_id: int
    path: List[int]  # 节点ID路径
    distance: float
    energy_sent: float = 0.0
    energy_delivered: float = 0.0
    energy_loss: float = 0.0
    efficiency: float = 0.0


@dataclass
class SimulationStats:
    """仿真统计信息"""
    time_step: int
    pre_std: float
    post_std: float
    delivered_total: float
    total_loss: float
    k_value: int
    active_plans: int


class INode(ABC):
    """传感器节点接口"""
    
    @abstractmethod
    def get_id(self) -> int:
        """获取节点ID"""
        pass
    
    @abstractmethod
    def get_position(self) -> Tuple[float, float]:
        """获取节点位置"""
        pass
    
    @abstractmethod
    def get_current_energy(self) -> float:
        """获取当前能量"""
        pass
    
    @abstractmethod
    def get_energy_capacity(self) -> float:
        """获取能量容量"""
        pass
    
    @abstractmethod
    def has_solar(self) -> bool:
        """是否有太阳能能力"""
        pass
    
    @abstractmethod
    def is_mobile(self) -> bool:
        """是否可移动"""
        pass
    
    @abstractmethod
    def distance_to(self, other_node: 'INode') -> float:
        """计算到其他节点的距离"""
        pass
    
    @abstractmethod
    def energy_transfer_efficiency(self, target_node: 'INode') -> float:
        """计算能量传输效率"""
        pass
    
    @abstractmethod
    def energy_consumption(self, target_node: Optional['INode'] = None, 
                         transfer_wet: bool = False) -> float:
        """计算能量消耗"""
        pass
    
    @abstractmethod
    def energy_harvest(self, time_step: int) -> float:
        """计算能量采集"""
        pass
    
    @abstractmethod
    def update_energy(self, time_step: int) -> Tuple[float, float]:
        """更新能量状态"""
        pass
    
    @abstractmethod
    def update_position(self, time_step: int) -> None:
        """更新位置"""
        pass


class INetwork(ABC):
    """网络接口"""
    
    @abstractmethod
    def get_nodes(self) -> List[INode]:
        """获取所有节点"""
        pass
    
    @abstractmethod
    def get_node_by_id(self, node_id: int) -> Optional[INode]:
        """根据ID获取节点"""
        pass
    
    @abstractmethod
    def get_num_nodes(self) -> int:
        """获取节点数量"""
        pass
    
    @abstractmethod
    def update_network_energy(self, time_step: int) -> None:
        """更新网络能量状态"""
        pass
    
    @abstractmethod
    def execute_energy_transfer(self, plans: List[EnergyTransferPlan]) -> None:
        """执行能量传输"""
        pass


class IScheduler(ABC):
    """调度器接口"""
    
    @abstractmethod
    def plan(self, network: INetwork, time_step: int) -> Tuple[List[EnergyTransferPlan], List[Dict]]:
        """制定能量传输计划"""
        pass
    
    @abstractmethod
    def post_step(self, network: INetwork, time_step: int, 
                 stats: SimulationStats) -> None:
        """步骤后处理"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取调度器名称"""
        pass


class IRouter(ABC):
    """路由器接口"""
    
    @abstractmethod
    def find_path(self, source_node: INode, dest_node: INode, 
                 max_hops: int = 5) -> Optional[List[INode]]:
        """查找路径"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取路由器名称"""
        pass


class ISimulator(ABC):
    """仿真器接口"""
    
    @abstractmethod
    def run_simulation(self) -> List[SimulationStats]:
        """运行仿真"""
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        pass
    
    @abstractmethod
    def save_results(self, filename: str) -> None:
        """保存结果"""
        pass


class ILogger(ABC):
    """日志接口"""
    
    @abstractmethod
    def info(self, message: str) -> None:
        """信息日志"""
        pass
    
    @abstractmethod
    def warning(self, message: str) -> None:
        """警告日志"""
        pass
    
    @abstractmethod
    def error(self, message: str) -> None:
        """错误日志"""
        pass
    
    @abstractmethod
    def debug(self, message: str) -> None:
        """调试日志"""
        pass


class IConfigProvider(ABC):
    """配置提供者接口"""
    
    @abstractmethod
    def get_node_config(self) -> Dict[str, Any]:
        """获取节点配置"""
        pass
    
    @abstractmethod
    def get_network_config(self) -> Dict[str, Any]:
        """获取网络配置"""
        pass
    
    @abstractmethod
    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置"""
        pass
    
    @abstractmethod
    def get_scheduler_config(self) -> Dict[str, Any]:
        """获取调度器配置"""
        pass


class IDataCollector(ABC):
    """数据收集器接口"""
    
    @abstractmethod
    def collect_node_data(self, node: INode, time_step: int) -> Dict[str, Any]:
        """收集节点数据"""
        pass
    
    @abstractmethod
    def collect_network_data(self, network: INetwork, time_step: int) -> Dict[str, Any]:
        """收集网络数据"""
        pass
    
    @abstractmethod
    def collect_simulation_data(self, stats: SimulationStats) -> Dict[str, Any]:
        """收集仿真数据"""
        pass
    
    @abstractmethod
    def save_data(self, filename: str) -> None:
        """保存数据"""
        pass


class IPlotter(ABC):
    """绘图器接口"""
    
    @abstractmethod
    def plot_node_distribution(self, nodes: List[INode], 
                              output_path: str = "data/node_distribution.png") -> None:
        """绘制节点分布"""
        pass
    
    @abstractmethod
    def plot_energy_over_time(self, stats: List[SimulationStats], 
                             output_path: str = "data/energy_over_time.png") -> None:
        """绘制能量随时间变化"""
        pass
    
    @abstractmethod
    def plot_energy_paths(self, network: INetwork, plans: List[EnergyTransferPlan], 
                          time_step: int, output_path: str = None) -> None:
        """绘制能量传输路径"""
        pass
    
    @abstractmethod
    def plot_k_values(self, stats: List[SimulationStats], 
                      output_path: str = "data/k_values.png") -> None:
        """绘制K值变化"""
        pass


# 工厂接口
class ISchedulerFactory(ABC):
    """调度器工厂接口"""
    
    @abstractmethod
    def create_scheduler(self, scheduler_type: str, **kwargs) -> IScheduler:
        """创建调度器"""
        pass
    
    @abstractmethod
    def get_available_schedulers(self) -> List[str]:
        """获取可用的调度器类型"""
        pass


class IRouterFactory(ABC):
    """路由器工厂接口"""
    
    @abstractmethod
    def create_router(self, router_type: str, **kwargs) -> IRouter:
        """创建路由器"""
        pass
    
    @abstractmethod
    def get_available_routers(self) -> List[str]:
        """获取可用的路由器类型"""
        pass


# 事件系统接口
class IEventBus(ABC):
    """事件总线接口"""
    
    @abstractmethod
    def subscribe(self, event_type: str, handler: callable) -> None:
        """订阅事件"""
        pass
    
    @abstractmethod
    def unsubscribe(self, event_type: str, handler: callable) -> None:
        """取消订阅"""
        pass
    
    @abstractmethod
    def publish(self, event_type: str, data: Any) -> None:
        """发布事件"""
        pass


class IEventHandler(ABC):
    """事件处理器接口"""
    
    @abstractmethod
    def handle(self, event_type: str, data: Any) -> None:
        """处理事件"""
        pass
