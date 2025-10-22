"""
结果记录和保存模块
负责仿真结果的记录、保存和管理
"""

import csv
from typing import List, Dict, Any, Optional
from utils.output_manager import OutputManager


class ResultManager:
    """结果管理器"""
    
    def __init__(self, session_dir: str):
        """
        初始化结果管理器
        
        Args:
            session_dir: 会话目录路径
        """
        self.session_dir = session_dir
        self.results: List[List[Dict[str, Any]]] = []
    
    def record_energy_status(self, network) -> None:
        """
        记录网络中所有节点的能量状态
        
        Args:
            network: 网络对象
        """
        energy_data = []
        for node in network.nodes:
            energy_data.append({
                "node_id": node.node_id,
                "current_energy": node.current_energy,
                "received_energy": sum(node.received_history),  # 所有接收能量的总和
                "transferred_energy": sum(node.transferred_history),  # 所有传输能量的总和
                "energy_history": node.energy_history[-1] if node.energy_history else None
            })
        self.results.append(energy_data)
    
    def save_results(self, filename: Optional[str] = None) -> str:
        """
        保存仿真结果到CSV文件
        
        Args:
            filename: 文件名，如果为None则使用默认路径
            
        Returns:
            保存的文件路径
        """
        if filename is None:
            filename = OutputManager.get_file_path(self.session_dir, 'simulation_results.csv')
        
        with open(filename, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # 写入表头
            writer.writerow([
                "Time Step", "Node ID", "Current Energy", 
                "Received Energy", "Transferred Energy", "Energy History"
            ])
            
            for t, step_result in enumerate(self.results):
                for node_data in step_result:
                    writer.writerow([
                        t + 1, 
                        node_data["node_id"], 
                        node_data["current_energy"],
                        node_data["received_energy"], 
                        node_data["transferred_energy"],
                        node_data["energy_history"]
                    ])

        print(f"Results saved to {filename}")
        return filename
    
    def display_results(self) -> None:
        """显示仿真结果"""
        print("\n--- Energy Simulation Results ---")
        for t, step_result in enumerate(self.results):
            print(f"\nTime step {t + 1}:")
            for node_data in step_result:
                print(f"Node {node_data['node_id']} - Current Energy: {node_data['current_energy']:.2f} Joules")
                print(f"  Received Energy: {node_data['received_energy']:.2f} Joules")
                print(f"  Transferred Energy: {node_data['transferred_energy']:.2f} Joules")
                if node_data['energy_history']:
                    print(f"  Energy History: {node_data['energy_history']}")
    
    def get_results(self) -> List[List[Dict[str, Any]]]:
        """
        获取仿真结果
        
        Returns:
            仿真结果列表
        """
        return self.results
    
    def clear_results(self) -> None:
        """清空结果数据"""
        self.results.clear()
    
    def get_summary_stats(self, network) -> Dict[str, Any]:
        """
        获取结果摘要统计
        
        Args:
            network: 网络对象
            
        Returns:
            摘要统计字典
        """
        if not self.results:
            return {}
        
        # 计算总体统计
        total_sent_energy = sum(sum(node.transferred_history) for node in network.nodes)
        total_received_energy = sum(sum(node.received_history) for node in network.nodes)
        total_loss_energy = total_sent_energy - total_received_energy
        
        # 计算效率
        efficiency = (total_received_energy / total_sent_energy * 100) if total_sent_energy > 0 else 0
        
        # 获取最终能量状态
        final_energies = []
        if self.results:
            for node_data in self.results[-1]:
                final_energies.append(node_data["current_energy"])
        
        return {
            'total_sent_energy': total_sent_energy,
            'total_received_energy': total_received_energy,
            'total_loss_energy': total_loss_energy,
            'efficiency': efficiency,
            'final_energies': final_energies,
            'num_time_steps': len(self.results),
            'num_nodes': len(network.nodes)
        }

