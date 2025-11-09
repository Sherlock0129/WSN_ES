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
        
        # 反馈分数统计
        self.feedback_scores: List[Dict[str, Any]] = []  # 记录每次调度的反馈分数
        self.feedback_times: List[int] = []  # 记录对应的时间步
    
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
    
    def record_feedback_score(self, time_step: int, score: float, details: Dict[str, Any]) -> None:
        """
        记录反馈分数
        
        Args:
            time_step: 时间步
            score: 反馈分数
            details: 详细信息字典
        """
        self.feedback_times.append(time_step)
        feedback_record = {
            'time_step': time_step,
            'total_score': score,
            'balance_score': details.get('balance_score', 0.0),
            'survival_score': details.get('survival_score', 0.0),
            'efficiency_score': details.get('efficiency_score', 0.0),
            'energy_score': details.get('energy_score', 0.0),
            'impact': details.get('impact', '未知')
        }
        self.feedback_scores.append(feedback_record)
    
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
        
        # 添加反馈分数统计信息
        if self.feedback_scores:
            total_scores = [record['total_score'] for record in self.feedback_scores]
            positive_count = sum(1 for score in total_scores if score > 1)
            negative_count = sum(1 for score in total_scores if score < -1)
            neutral_count = len(total_scores) - positive_count - negative_count
            
            stats['feedback'] = {
                'avg_score': np.mean(total_scores),
                'max_score': np.max(total_scores),
                'min_score': np.min(total_scores),
                'std_score': np.std(total_scores),
                'positive_count': positive_count,
                'negative_count': negative_count,
                'neutral_count': neutral_count,
                'total_count': len(total_scores)
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
    
    def plot_feedback_scores(self) -> None:
        """
        绘制反馈分数随时间变化的图表
        
        展示：
        1. 总体反馈分数随时间变化
        2. 各维度分数的堆叠图
        3. 正/负影响的分布统计
        """
        if not self.feedback_scores:
            print("没有反馈分数数据可供绘制")
            return
        
        # 提取数据
        time_steps = [record['time_step'] for record in self.feedback_scores]
        total_scores = [record['total_score'] for record in self.feedback_scores]
        balance_scores = [record['balance_score'] for record in self.feedback_scores]
        survival_scores = [record['survival_score'] for record in self.feedback_scores]
        efficiency_scores = [record['efficiency_score'] for record in self.feedback_scores]
        energy_scores = [record['energy_score'] for record in self.feedback_scores]
        
        # 创建图表（3行1列）
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # 1. 总体反馈分数随时间变化
        ax1 = axes[0]
        ax1.plot(time_steps, total_scores, marker='o', linestyle='-', linewidth=2, markersize=4, color='blue')
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.axhline(y=5, color='green', linestyle=':', linewidth=1, alpha=0.5, label='显著改善阈值')
        ax1.axhline(y=-5, color='red', linestyle=':', linewidth=1, alpha=0.5, label='显著恶化阈值')
        ax1.fill_between(time_steps, 0, total_scores, 
                         where=[s >= 0 for s in total_scores], 
                         color='green', alpha=0.2, interpolate=True)
        ax1.fill_between(time_steps, 0, total_scores, 
                         where=[s < 0 for s in total_scores], 
                         color='red', alpha=0.2, interpolate=True)
        ax1.set_title('调度反馈分数随时间变化 (Overall Feedback Score)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('时间步 (Time Step)', fontsize=10)
        ax1.set_ylabel('反馈分数 (Score)', fontsize=10)
        ax1.legend(loc='best')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # 2. 各维度分数堆叠图
        ax2 = axes[1]
        ax2.plot(time_steps, balance_scores, marker='s', linestyle='-', 
                 linewidth=1.5, markersize=3, label='能量均衡性 (Balance)', alpha=0.8)
        ax2.plot(time_steps, survival_scores, marker='^', linestyle='-', 
                 linewidth=1.5, markersize=3, label='网络存活率 (Survival)', alpha=0.8)
        ax2.plot(time_steps, efficiency_scores, marker='o', linestyle='-', 
                 linewidth=1.5, markersize=3, label='传输效率 (Efficiency)', alpha=0.8)
        ax2.plot(time_steps, energy_scores, marker='d', linestyle='-', 
                 linewidth=1.5, markersize=3, label='能量水平 (Energy Level)', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_title('各维度分数变化 (Dimensional Scores)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间步 (Time Step)', fontsize=10)
        ax2.set_ylabel('分数 (Score)', fontsize=10)
        ax2.legend(loc='best', ncol=2)
        ax2.grid(True, linestyle='--', alpha=0.5)
        
        # 3. 正/负/中性影响分布统计（柱状图）
        ax3 = axes[2]
        positive_count = sum(1 for score in total_scores if score > 1)
        negative_count = sum(1 for score in total_scores if score < -1)
        neutral_count = len(total_scores) - positive_count - negative_count
        
        categories = ['正相关\n(Positive)', '中性\n(Neutral)', '负相关\n(Negative)']
        counts = [positive_count, neutral_count, negative_count]
        colors_bar = ['green', 'gray', 'red']
        
        bars = ax3.bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
        
        # 在柱子上标注数值和百分比
        total_count = len(total_scores)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            percentage = (count / total_count * 100) if total_count > 0 else 0
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(count)}\n({percentage:.1f}%)',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax3.set_title('调度影响分布统计 (Impact Distribution)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('次数 (Count)', fontsize=10)
        ax3.set_ylim(0, max(counts) * 1.2 if max(counts) > 0 else 1)
        ax3.grid(True, axis='y', linestyle='--', alpha=0.5)
        
        # 添加总体统计信息文本框
        avg_score = np.mean(total_scores)
        max_score = np.max(total_scores)
        min_score = np.min(total_scores)
        
        stats_text = f'统计摘要 (Statistics):\n'
        stats_text += f'平均分 (Avg): {avg_score:.2f}\n'
        stats_text += f'最高分 (Max): {max_score:.2f}\n'
        stats_text += f'最低分 (Min): {min_score:.2f}\n'
        stats_text += f'总次数 (Total): {total_count}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax3.text(0.98, 0.97, stats_text, transform=ax3.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right', bbox=props)
        
        # 调整布局并保存
        plt.tight_layout()
        feedback_plot_path = OutputManager.get_file_path(self.session_dir, 'feedback_scores.png')
        plt.savefig(feedback_plot_path, dpi=150)
        print(f"反馈分数图表已保存到: {feedback_plot_path}")
        plt.show()
        
        # 保存反馈分数数据到CSV
        df = pd.DataFrame(self.feedback_scores)
        feedback_csv_path = OutputManager.get_file_path(self.session_dir, 'feedback_scores.csv')
        df.to_csv(feedback_csv_path, index=False, encoding='utf-8-sig')
        print(f"反馈分数数据已保存到: {feedback_csv_path}")
