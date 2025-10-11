"""
数据收集器 - 统一管理仿真过程中的所有数据收集、存储和导出功能
"""

import csv
import json
import os
from datetime import datetime, timedelta
from collections import deque
import numpy as np


class DataCollector:
    """统一的数据收集器，负责收集、存储和导出仿真过程中的所有数据"""
    
    def __init__(self, output_dir="data", max_decision_records=20000, status_sample_period=60):
        """
        初始化数据收集器
        
        :param output_dir: 输出目录
        :param max_decision_records: 最大决策记录数（使用deque限制内存）
        :param status_sample_period: 状态采样周期（步数）
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 决策相关数据（限制内存）
        self.decision_records = deque(maxlen=max_decision_records)
        self.full_decision_history = []  # 完整历史用于导出
        
        # K值轨迹
        self.K_history = []
        self.K_timestamps = []
        
        # 能量统计时间序列
        self.energy_averages = []
        self.energy_standards = []
        
        # 每步状态数据（采样）
        self.status_records = []
        self.status_sample_period = status_sample_period
        
        # 运行元信息
        self.run_metadata = {}
        
    def reset(self, run_meta=None):
        """重置收集器，开始新的运行"""
        self.decision_records.clear()
        self.full_decision_history.clear()
        self.K_history.clear()
        self.K_timestamps.clear()
        self.energy_averages.clear()
        self.energy_standards.clear()
        self.status_records.clear()
        
        if run_meta:
            self.run_metadata = run_meta
            self.run_metadata['start_time'] = datetime.now().isoformat()
    
    def on_decision_before(self, t, network):
        """
        决策前快照 - 记录当前网络状态
        
        :param t: 时间步
        :param network: 网络对象
        :return: (pre_energies, pre_received_total) 用于后续统计计算
        """
        pre_energies = np.array([n.current_energy for n in network.nodes], dtype=float)
        pre_received_total = sum(sum(n.received_history) for n in network.nodes)
        
        # 计算并记录能量统计
        node_energies = [node.current_energy for node in network.nodes]
        energy_avg = np.mean(node_energies)
        energy_std = np.std(node_energies)
        
        self.energy_averages.append(energy_avg)
        self.energy_standards.append(energy_std)
        
        return pre_energies, pre_received_total
    
    def on_decision_after(self, t, network, plans, stats, K, current_time=None):
        """
        决策后记录 - 保存决策结果和统计信息
        
        :param t: 时间步
        :param network: 网络对象
        :param plans: 执行的能量传输计划
        :param stats: 统计信息字典
        :param K: 当前K值
        :param current_time: 当前时间（datetime对象）
        """
        # 序列化计划数据（避免存储复杂对象）
        serialized_plans = self._serialize_plans(plans)
        
        # 记录节点能量状态
        node_energies = {node.node_id: node.current_energy for node in network.nodes}
        
        # 构建决策记录
        record = {
            "time_step": t,
            "timestamp": current_time.isoformat() if current_time else datetime.now().isoformat(),
            "K_value": K,
            "plans": serialized_plans,
            "node_energies": node_energies,
            "stats": stats,
            "energy_avg": self.energy_averages[-1] if self.energy_averages else 0,
            "energy_std": self.energy_standards[-1] if self.energy_standards else 0
        }
        
        # 添加到记录中
        self.decision_records.append(record)
        self.full_decision_history.append(record)
        
        # 记录K值轨迹
        self.K_history.append(K)
        if current_time:
            self.K_timestamps.append(current_time)
        
        # 打印决策信息
        self._print_decision_info(K, stats)
    
    def on_step_sample(self, t, network):
        """
        每步状态采样 - 记录网络状态（按采样周期）
        
        :param t: 时间步
        :param network: 网络对象
        """
        if t % self.status_sample_period == 0:
            energy_data = []
            for node in network.nodes:
                energy_data.append({
                    "node_id": node.node_id,
                    "current_energy": node.current_energy,
                    "received_energy": sum(node.received_history),
                    "transferred_energy": sum(node.transferred_history),
                    "energy_history": node.energy_history[-1] if node.energy_history else None
                })
            
            self.status_records.append({
                "time_step": t,
                "timestamp": datetime.now().isoformat(),
                "nodes": energy_data
            })
    
    def _serialize_plans(self, plans):
        """序列化计划数据，避免存储复杂对象"""
        serialized = []
        for p in plans:
            s_plan = {
                "receiver_id": getattr(p.get("receiver"), "node_id", None),
                "donor_id": getattr(p.get("donor"), "node_id", None),
                "path_ids": [getattr(n, "node_id", None) for n in p.get("path", [])],
                "distance": p.get("distance"),
                "energy_sent": p.get("energy_sent", getattr(p.get("donor"), "E_char", 0.0))
            }
            serialized.append(s_plan)
        return serialized
    
    def _print_decision_info(self, K, stats):
        """打印决策信息"""
        print("dynamic_k={} pre_std={:.4f} post_std={:.4f} delivered={:.2f} loss={:.2f}".format(
            K, stats['pre_std'], stats['post_std'], stats['delivered_total'], stats['total_loss']
        ))
    
    def compute_step_stats(self, plans, pre_energies, pre_received_total, network=None):
        """
        计算步骤统计信息（从energy_simulation.py迁移过来）
        
        :param plans: 能量传输计划
        :param pre_energies: 决策前能量数组
        :param pre_received_total: 决策前总接收能量
        :param network: 网络对象（用于获取当前节点）
        :return: 统计信息字典
        """
        if network is None:
            return {
                "pre_std": 0.0,
                "post_std": 0.0,
                "delivered_total": 0.0,
                "total_loss": 0.0,
                "sent_total": 0.0,
                "num_plans": len(plans)
            }
        
        post_energies = np.array([n.current_energy for n in network.nodes], dtype=float)
        pre_std = float(np.std(pre_energies))
        post_std = float(np.std(post_energies))

        # 发送总量优先使用计划中的 energy_sent（如 PowerControlScheduler），否则退回 donor.E_char
        sent_total = 0.0
        for p in plans:
            if "energy_sent" in p:
                try:
                    sent_total += float(p["energy_sent"]) 
                except Exception:
                    sent_total += float(getattr(p.get("donor"), "E_char", 0.0))
            else:
                sent_total += float(getattr(p.get("donor"), "E_char", 0.0))
        
        post_received_total = sum(sum(n.received_history) for n in network.nodes)
        delivered_total = max(0.0, post_received_total - pre_received_total)
        total_loss = max(0.0, sent_total - delivered_total)

        return {
            "pre_std": pre_std,
            "post_std": post_std,
            "delivered_total": delivered_total,
            "total_loss": total_loss,
            "sent_total": sent_total,
            "num_plans": len(plans)
        }
    
    def print_statistics(self, network):
        """打印仿真统计信息"""
        avg_variance = np.mean(self.energy_standards) if self.energy_standards else 0
        print(f"\n所有时间点方差的平均值: {avg_variance:.4f}")

        total_sent_energy = sum(sum(node.transferred_history) for node in network.nodes)
        total_received_energy = sum(sum(node.received_history) for node in network.nodes)
        total_loss_energy = total_sent_energy - total_received_energy

        print(f"总体损失能量: {total_loss_energy:.4f} Joules")
        print(f"总发送能量: {total_sent_energy:.4f} Joules")
        print(f"总接收能量: {total_received_energy:.4f} Joules")
        print(f"能量传输效率: {(total_received_energy / total_sent_energy * 100 if total_sent_energy > 0 else 0):.2f}%")

        return {
            'avg_variance': avg_variance,
            'total_loss_energy': total_loss_energy,
            'total_sent_energy': total_sent_energy,
            'total_received_energy': total_received_energy
        }
    
    def create_K_table(self):
        """创建K值表格并导出CSV"""
        if not self.K_timestamps or not self.K_history:
            print("没有K值数据可导出")
            return
        
        import pandas as pd
        df = pd.DataFrame({
            'Time': [t.strftime('%Y-%m-%d %H:%M') for t in self.K_timestamps],
            'Day': [t.strftime('%A') for t in self.K_timestamps],
            'Hour': [t.hour for t in self.K_timestamps],
            'K_Value': self.K_history
        })

        daily_stats = df.groupby('Day')['K_Value'].agg(['mean', 'min', 'max', 'std']).round(2)

        df.to_csv(os.path.join(self.output_dir, 'K_value_history.csv'), index=False)
        daily_stats.to_csv(os.path.join(self.output_dir, 'K_value_daily_stats.csv'))

        print("K值历史记录和统计信息已保存到CSV文件")
        print("\n每日K值统计")
        print(daily_stats)
    
    def export_decision_summary_csv(self, filename="decision_summary.csv"):
        """导出决策汇总CSV"""
        if not self.full_decision_history:
            print("没有决策历史数据可导出")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        header = ["time_step", "K_value", "pre_std", "post_std", "delivered_total", "total_loss", "num_plans", "energy_avg", "energy_std"]
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for record in self.full_decision_history:
                row = {
                    "time_step": record["time_step"],
                    "K_value": record["K_value"],
                    "pre_std": record["stats"]["pre_std"],
                    "post_std": record["stats"]["post_std"],
                    "delivered_total": record["stats"]["delivered_total"],
                    "total_loss": record["stats"]["total_loss"],
                    "num_plans": record["stats"].get("num_plans", len(record["plans"])),
                    "energy_avg": record["energy_avg"],
                    "energy_std": record["energy_std"]
                }
                writer.writerow(row)
        
        print(f"决策汇总已导出到 {filepath}")
    
    def export_status_csv(self, filename="status_records.csv"):
        """导出状态记录CSV"""
        if not self.status_records:
            print("没有状态记录数据可导出")
            return
        
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Time Step", "Node ID", "Current Energy", "Received Energy", "Transferred Energy", "Energy History"])
            
            for record in self.status_records:
                for node_data in record["nodes"]:
                    writer.writerow([
                        record["time_step"],
                        node_data["node_id"],
                        node_data["current_energy"],
                        node_data["received_energy"],
                        node_data["transferred_energy"],
                        node_data["energy_history"]
                    ])
        
        print(f"状态记录已导出到 {filepath}")
    
    def export_json(self, filename="simulation_data.json"):
        """导出完整数据为JSON格式"""
        filepath = os.path.join(self.output_dir, filename)
        
        export_data = {
            "run_metadata": self.run_metadata,
            "decision_history": self.full_decision_history,
            "K_history": self.K_history,
            "K_timestamps": [t.isoformat() for t in self.K_timestamps],
            "energy_averages": self.energy_averages,
            "energy_standards": self.energy_standards,
            "status_records": self.status_records
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"完整仿真数据已导出到 {filepath}")
    
    def finalize(self):
        """完成数据收集，执行最终导出"""
        self.run_metadata['end_time'] = datetime.now().isoformat()
        
        # 自动导出关键数据
        self.export_decision_summary_csv()
        self.create_K_table()
        if self.status_records:
            self.export_status_csv()
        
        print(f"数据收集完成，所有文件保存在 {self.output_dir} 目录")
