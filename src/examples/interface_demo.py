"""
演示如何让原有类真正实现接口
这是一个示例文件，展示接口的真正用法
"""

from src.interfaces import INode, INetwork, IScheduler, EnergyTransferPlan
from typing import List, Optional, Tuple
import numpy as np


class SensorNodeAdapter(INode):
    """传感器节点适配器 - 让原有SensorNode实现INode接口"""
    
    def __init__(self, original_node):
        self._node = original_node
    
    def get_id(self) -> int:
        return self._node.node_id
    
    def get_position(self) -> Tuple[float, float]:
        return tuple(self._node.position)
    
    def get_current_energy(self) -> float:
        return self._node.current_energy
    
    def get_energy_capacity(self) -> float:
        return self._node.capacity * self._node.V * 3600
    
    def has_solar(self) -> bool:
        return self._node.has_solar
    
    def is_mobile(self) -> bool:
        return getattr(self._node, 'is_mobile', False)
    
    def distance_to(self, other_node: INode) -> float:
        return self._node.distance_to(other_node._node)
    
    def energy_transfer_efficiency(self, target_node: INode) -> float:
        return self._node.energy_transfer_efficiency(target_node._node)
    
    def energy_consumption(self, target_node: Optional[INode] = None, 
                         transfer_wet: bool = False) -> float:
        target = target_node._node if target_node else None
        return self._node.energy_consumption(target, transfer_wet)
    
    def energy_harvest(self, time_step: int) -> float:
        return self._node.energy_harvest(time_step)
    
    def update_energy(self, time_step: int) -> Tuple[float, float]:
        return self._node.update_energy(time_step)
    
    def update_position(self, time_step: int) -> None:
        self._node.update_position(time_step)


class NetworkAdapter(INetwork):
    """网络适配器 - 让原有Network实现INetwork接口"""
    
    def __init__(self, original_network):
        self._network = original_network
        # 将原有节点包装为适配器
        self._nodes = [SensorNodeAdapter(node) for node in original_network.nodes]
    
    def get_nodes(self) -> List[INode]:
        return self._nodes
    
    def get_node_by_id(self, node_id: int) -> Optional[INode]:
        for node in self._nodes:
            if node.get_id() == node_id:
                return node
        return None
    
    def get_num_nodes(self) -> int:
        return self._network.num_nodes
    
    def update_network_energy(self, time_step: int) -> None:
        self._network.update_network_energy(time_step)
    
    def execute_energy_transfer(self, plans: List[EnergyTransferPlan]) -> None:
        # 将接口格式的计划转换为原有格式
        original_plans = []
        for plan in plans:
            original_plan = {
                "donor": self._network.get_node_by_id(plan.donor_id),
                "receiver": self._network.get_node_by_id(plan.receiver_id),
                "path": [self._network.get_node_by_id(node_id) for node_id in plan.path],
                "distance": plan.distance,
                "energy_sent": plan.energy_sent
            }
            original_plans.append(original_plan)
        
        self._network.execute_energy_transfer(original_plans)


class SchedulerAdapter(IScheduler):
    """调度器适配器 - 让原有调度器实现IScheduler接口"""
    
    def __init__(self, original_scheduler):
        self._scheduler = original_scheduler
    
    def plan(self, network: INetwork, time_step: int) -> Tuple[List[EnergyTransferPlan], List[dict]]:
        # 将接口格式的网络转换为原有格式
        original_network = self._get_original_network(network)
        
        # 调用原有调度器
        result = self._scheduler.plan(original_network, time_step)
        
        if isinstance(result, tuple):
            plans, candidates = result
        else:
            plans, candidates = result, []
        
        # 将原有格式的计划转换为接口格式
        interface_plans = []
        for plan in plans:
            interface_plan = EnergyTransferPlan(
                donor_id=plan["donor"].node_id,
                receiver_id=plan["receiver"].node_id,
                path=[node.node_id for node in plan["path"]],
                distance=plan["distance"],
                energy_sent=plan.get("energy_sent", 0.0),
                energy_delivered=plan.get("delivered", 0.0),
                energy_loss=plan.get("loss", 0.0),
                efficiency=plan.get("efficiency", 0.0)
            )
            interface_plans.append(interface_plan)
        
        return interface_plans, candidates
    
    def post_step(self, network: INetwork, time_step: int, stats) -> None:
        # 如果原有调度器有post_step方法，调用它
        if hasattr(self._scheduler, 'post_step'):
            original_network = self._get_original_network(network)
            self._scheduler.post_step(original_network, time_step, stats)
    
    def get_name(self) -> str:
        return self._scheduler.get_name()
    
    def _get_original_network(self, network: INetwork):
        """从接口网络获取原有网络对象"""
        if isinstance(network, NetworkAdapter):
            return network._network
        else:
            # 如果不是适配器，需要创建适配器
            raise ValueError("需要NetworkAdapter类型的网络")


# 使用示例
def demonstrate_interface_usage():
    """演示真正的接口使用"""
    from src.core.network import Network
    from src.scheduling.schedulers import LyapunovScheduler
    from src.config.simulation_config import ConfigManager
    
    # 1. 创建原有对象
    config_manager = ConfigManager()
    network_config = config_manager.get_network_config_dict()
    original_network = Network(network_config["num_nodes"], network_config)
    original_scheduler = LyapunovScheduler(V=0.5, K=3, max_hops=3)
    
    # 2. 包装为接口对象
    network: INetwork = NetworkAdapter(original_network)
    scheduler: IScheduler = SchedulerAdapter(original_scheduler)
    
    # 3. 使用接口方法
    print(f"网络节点数: {network.get_num_nodes()}")
    print(f"调度器名称: {scheduler.get_name()}")
    
    # 4. 制定计划
    plans, candidates = scheduler.plan(network, 0)
    print(f"制定了 {len(plans)} 个能量传输计划")
    
    # 5. 执行计划
    network.execute_energy_transfer(plans)
    
    return network, scheduler


if __name__ == "__main__":
    demonstrate_interface_usage()
