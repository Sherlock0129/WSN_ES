"""
仿真统计和可视化模块
负责统计计算、图表生成和数据可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.output_manager import OutputManager
from utils.gpu_compute import get_gpu_manager, compute_statistics_gpu
from utils.logger import get_statistics_logger


class SimulationStats:
    """仿真统计和可视化管理器"""
    
    def __init__(self, session_dir: str, use_gpu: bool = False):
        """
        初始化统计管理器
        
        Args:
            session_dir: 会话目录路径
            use_gpu: 是否使用GPU加速
        """
        self.session_dir = session_dir
        self.use_gpu = use_gpu
        self.gpu_manager = get_gpu_manager(use_gpu)
        
        # 能量统计
        self.energy_averages: List[float] = []
        self.energy_standards: List[float] = []
    
    def compute_step_stats(self, plans: List[Dict], pre_energies: np.ndarray, 
                          pre_received_total: float, network) -> Dict[str, float]:
        """
        计算单步统计信息
        
        Args:
            plans: 能量传输计划列表
            pre_energies: 传输前的节点能量数组
            pre_received_total: 传输前的总接收能量
            network: 网络对象
            
        Returns:
            包含统计信息的字典
        """
        # 计算本次传能后的统计量（基于真实执行后的节点状态/历史）
        post_energies = np.array([n.current_energy for n in network.nodes], dtype=float)
        
        # 使用GPU加速计算统计信息
        if self.use_gpu:
            pre_mean, pre_std, _ = compute_statistics_gpu(pre_energies, self.gpu_manager)
            post_mean, post_std, _ = compute_statistics_gpu(post_energies, self.gpu_manager)
        else:
            pre_std = float(np.std(pre_energies))
            post_std = float(np.std(post_energies))

        sent_total = sum(p["donor"].E_char for p in plans)  # 本轮各 donor 实际下发的名义能量
        post_received_total = sum(sum(n.received_history) for n in network.nodes)
        delivered_total = max(0.0, post_received_total - pre_received_total)
        total_loss = max(0.0, sent_total - delivered_total)

        return {
            "pre_std": pre_std,
            "post_std": post_std,
            "delivered_total": delivered_total,
            "total_loss": total_loss
        }
    
    def record_energy_stats(self, node_energies: List[float]) -> None:
        """
        记录能量统计信息
        
        Args:
            node_energies: 节点能量列表
        """
        if self.use_gpu:
            energy_avg, energy_std, _ = compute_statistics_gpu(node_energies, self.gpu_manager)
        else:
            energy_avg = np.mean(node_energies)
            energy_std = np.std(node_energies)
        
        self.energy_averages.append(energy_avg)
        self.energy_standards.append(energy_std)
    
    def print_statistics(self, network, additional_info: dict = None) -> Dict[str, float]:
        """
        打印仿真统计信息并保存到文件
        
        Args:
            network: 网络对象
            additional_info: 额外的统计信息（例如信息传输统计）
            
        Returns:
            统计信息字典
        """
        # 计算所有时间点方差的平均值
        avg_variance = np.mean(self.energy_standards) if self.energy_standards else 0

        # 计算总体的损失能量值
        # 总发送能量 = 所有节点传输的能量总和
        total_sent_energy = sum(sum(node.transferred_history) for node in network.nodes)

        # 总接收能量 = 所有节点接收的能量总和
        total_received_energy = sum(sum(node.received_history) for node in network.nodes)

        # 总损失能量 = 总发送能量 - 总接收能量
        total_loss_energy = total_sent_energy - total_received_energy

        # 构建统计信息字典
        stats = {
            'avg_variance': avg_variance,
            'total_loss_energy': total_loss_energy,
            'total_sent_energy': total_sent_energy,
            'total_received_energy': total_received_energy
        }

        # 使用 StatisticsLogger 打印并保存统计信息
        stats_logger = get_statistics_logger(self.session_dir)
        stats_logger.print_and_save_statistics(stats, network, additional_info)

        # 返回这些统计值，以便在其他地方使用
        return stats
    
    def plot_K_history(self, K_history: List[int], K_timestamps: List[datetime]) -> None:
        """
        绘制K值随时间变化的图表
        
        Args:
            K_history: K值历史列表
            K_timestamps: K值对应的时间戳列表
        """
        if not K_history:
            print("没有K值历史记录可供绘制")
            return

        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(K_timestamps, K_history, marker='o', linestyle='-')

        # 设置标题和标签
        plt.title('Dynamic K Values Over Time')
        plt.xlabel('Time')
        plt.ylabel('K Value')

        # 格式化x轴显示为日期
        try:
            date_formatter = DateFormatter('%a %H:%M')
            plt.gca().xaxis.set_major_formatter(date_formatter)
            plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
            plt.gcf().autofmt_xdate()  # 旋转日期标签
        except Exception as e:
            print(f"日期格式化出错: {e}，使用默认格式")

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 显示图表
        plt.tight_layout()
        k_value_plot_path = OutputManager.get_file_path(self.session_dir, 'K_value_over_time.png')
        plt.savefig(k_value_plot_path)
        plt.show()

        # 同时创建表格形式的数据
        self.create_K_table(K_history, K_timestamps)
    
    def create_K_table(self, K_history: List[int], K_timestamps: List[datetime]) -> None:
        """
        创建K值的表格数据
        
        Args:
            K_history: K值历史列表
            K_timestamps: K值对应的时间戳列表
        """
        # 创建DataFrame
        df = pd.DataFrame({
            'Time': [t.strftime('%Y-%m-%d %H:%M') for t in K_timestamps],
            'Day': [t.strftime('%A') for t in K_timestamps],
            'Hour': [t.hour for t in K_timestamps],
            'K_Value': K_history
        })

        # 按天分组并计算统计信息
        daily_stats = df.groupby('Day')['K_Value'].agg(['mean', 'min', 'max', 'std']).round(2)

        # 保存到CSV文件
        k_history_path = OutputManager.get_file_path(self.session_dir, 'K_value_history.csv')
        daily_stats_path = OutputManager.get_file_path(self.session_dir, 'K_value_daily_stats.csv')
        df.to_csv(k_history_path, index=False)
        daily_stats.to_csv(daily_stats_path)

        print("K值历史记录和统计信息已保存到CSV文件")
        print("\n每日K值统计:")
        print(daily_stats)
    
    def plot_energy_stats(self) -> None:
        """绘制每60个时间单位节点能量的平均值与方差"""
        if not self.energy_averages:
            print("没有能量统计数据可供绘制")
            return
            
        plt.figure(figsize=(12, 6))

        # 绘制平均值
        plt.subplot(2, 1, 1)
        plt.plot(range(0, len(self.energy_averages) * 60, 60), self.energy_averages, marker='o', color='b')
        plt.title("Average Node Energy Every 60 Time Units")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Average Energy (Joules)")
        plt.grid(True)

        # 绘制方差
        plt.subplot(2, 1, 2)
        plt.plot(range(0, len(self.energy_standards) * 60, 60), self.energy_standards, marker='o', color='r')
        plt.title("Variance of Node Energy Every 60 Time Units")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Energy Variance (Joules^2)")
        plt.grid(True)

        # 显示图表
        plt.tight_layout()
        plt.show()
    
    def plot_results(self, results: List[List[Dict]], time_steps: int, network) -> None:
        """
        绘制仿真结果图表
        Note: Physical center node (ID=0) is excluded from the plot.
        
        Args:
            results: 仿真结果数据
            time_steps: 时间步数
            network: 网络对象
        """
        time_steps_list = list(range(1, time_steps + 1))

        # 提取能量数据用于绘图（排除物理中心节点ID=0）
        node_ids = [node.node_id for node in network.nodes if node.node_id != 0]
        energy_data = {node_id: [] for node_id in node_ids}

        for step_result in results:
            for node_data in step_result:
                # 只记录非物理中心节点的数据
                if node_data["node_id"] != 0 and node_data["node_id"] in energy_data:
                    energy_data[node_data["node_id"]].append(node_data["current_energy"])

        # 绘制每个节点的能量数据
        plt.figure(figsize=(10, 6))
        for node_id, energy_values in energy_data.items():
            plt.plot(time_steps_list, energy_values, label=f"Node {node_id}")

        plt.title("Energy Consumption Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (Joules)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()
