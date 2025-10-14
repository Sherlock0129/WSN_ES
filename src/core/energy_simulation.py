
import csv
import os
import random
from turtle import pd
from datetime import datetime, timedelta
import numpy as np
from matplotlib import pyplot as plt

from scheduling import schedulers
from scheduling.lookahead import _eval_one_candidate, _compute_stats_for_network
from .energy_management import balance_energy
from utils.output_manager import OutputManager
try:
    from scheduling.schedulers import PowerControlScheduler
except Exception:
    # 允许没有 schedulers.py 时仍可运行（使用原 run_routing）
    PowerControlScheduler = None

class EnergySimulation:
    def __init__(self, network, time_steps, scheduler=None, 
                 # 自适应K值参数
                 initial_K=1, K_max=24, hysteresis=0.2, 
                 w_b=0.8, w_d=0.8, w_l=1.5,
                 # 其他参数
                 use_lookahead=False, output_dir="data"):
        """
        Initialize the energy simulation for the network.

        :param network: The network object that contains nodes and their parameters.
        :param time_steps: Total number of time steps to simulate.
        :param scheduler: Optional scheduler for energy transfer planning.
        :param initial_K: Initial K value for adaptive concurrency.
        :param K_max: Maximum K value for adaptive concurrency.
        :param hysteresis: Hysteresis threshold for K adaptation.
        :param w_b: Weight for balance improvement factor.
        :param w_d: Weight for delivery factor.
        :param w_l: Weight for loss penalty factor.
        :param use_lookahead: Whether to use lookahead simulation.
        """
        self.network = network
        self.time_steps = time_steps
        self.results = []  # To store the simulation results (energy, transfer, etc.)
        self.scheduler = scheduler  # ← 新增：可插拔调度器
        self.plans_by_time = {}

        # 自适应并发参数 - 从构造函数参数获取
        self.K = initial_K
        self.K_max = K_max
        self.last_reward = None
        self.last_direction = +1
        self.hysteresis = hysteresis  # 滞回阈值
        self.w_b = w_b  # 均衡改进权重
        self.w_d = w_d  # 有效送达量权重
        self.w_l = w_l  # 损耗惩罚权重

        # 记录K值历史
        self.K_history = []
        self.K_timestamps = []
        self.K_stats = []

        self.energy_averages = []  # 用于记录每个时间单位的能量平均值
        self.energy_Standards = []  # 用于记录每个时间单位的能量方差
        self.time_steps = time_steps  # 记录时间步

        self.cand = []

        # 是否使用前瞻模拟
        self.use_lookahead = use_lookahead
        self.output_dir = output_dir
        
        # 创建按日期命名的输出目录
        self.session_dir = OutputManager.get_session_dir(output_dir)

        # adaptK 日志控制（只在本次仿真内生效）
        self._adaptk_logged_header = False
        self._adaptk_last_t = None

    # def _reward(self, stats):
    #     return (self.w_b * (stats["pre_std"] - stats["post_std"])
    #             + self.w_d * stats["delivered_total"]
    #             - self.w_l * stats["total_loss"])

    def _reward(self, stats):
        """计算归一化后的奖励值"""
        # 归一化三个因子
        # 1. 均衡改进因子 (pre_std - post_std)
        # 2. 有效送达量因子 (delivered_total)
        # 3. 能量损耗因子 (total_loss)

        # 获取当前统计值
        improve = stats["pre_std"] - stats["post_std"]
        delivered = stats["delivered_total"]
        loss = stats["total_loss"]

        # 计算归一化因子（使用历史最大值或固定参考值）
        # 如果没有历史数据，使用当前值作为初始参考
        if not hasattr(self, 'max_improve'):
            self.max_improve = max(abs(improve), 1e-9)
            self.max_delivered = max(delivered, 1e-9)
            self.max_loss = max(loss, 1e-9)
        else:
            # 更新最大值
            self.max_improve = max(self.max_improve, abs(improve))
            self.max_delivered = max(self.max_delivered, delivered)
            self.max_loss = max(self.max_loss, loss)

        # 归一化每个因子
        norm_improve = improve / self.max_improve
        norm_delivered = delivered / self.max_delivered
        norm_loss = loss / self.max_loss

        # 计算加权奖励
        return  (
                self.w_b * norm_improve +
                self.w_d * norm_delivered -
                self.w_l * norm_loss
        )

    # def _adapt_K(self, stats):
    #     cur_reward = self._reward(stats)
    #     if self.last_reward is None:
    #         self.last_reward = cur_reward
    #         return
    #     improve = (cur_reward - self.last_reward) / (abs(self.last_reward) + 1e-9)
    #     if improve > self.hysteresis:
    #         new_K = self.K + self.last_direction
    #     elif improve < -self.hysteresis:
    #         self.last_direction *= -1
    #         new_K = self.K + self.last_direction
    #     else:
    #         new_K = self.K
    #     self.K = max(1, min(self.K_max, new_K))
    #     self.last_reward = cur_reward

    def _adapt_K(self, stats):
        # 记录调整前的K值
        old_K = self.K
        cur_reward = self._reward(stats)
        prev_reward = self.last_reward
        prev_dir = self.last_direction
        prev_K = self.K
        if prev_reward is None:
            self.last_reward = cur_reward
            return

        # =========== improve算法 =================#

        improve = cur_reward - prev_reward
        # improve = (cur_reward - prev_reward) / (abs(prev_reward) + 1e-9)

        # # 使用更稳健的分母，防止 prev≈0 导致极端比例
        # denom = max(abs(prev_reward), 1)
        # improve = (cur_reward - prev_reward) / denom
        #

        # 自适应步长：|improve| > 1 时，本次步长为 2，否则为 1
        # step = 2 if abs(improve) > 0.4 else 1
        step = 1

        if self.use_lookahead:
            try:
                new_K, _ = pick_k_via_lookahead(
                    self.network, self.scheduler,
                    getattr(self, "_adaptk_last_t", 0),
                    self.K, self.last_direction, improve, self.hysteresis, self.K_max,
                    horizon_minutes=60, reward_fn=self._reward
                )
            except Exception:
                if improve > self.hysteresis:
                    new_K = self.K + self.last_direction * step
                elif improve < -self.hysteresis:
                    self.last_direction *= -1
                    new_K = self.K + self.last_direction * step
                else:
                    new_K = self.K
        else:
            if improve > self.hysteresis:
                new_K = self.K + self.last_direction * step
            elif improve < -self.hysteresis:
                self.last_direction *= -1
                new_K = self.K + self.last_direction * step
            else:
                new_K = self.K


        self.K = max(1, min(self.K_max, new_K))


        # 记录反转后的方向（可能与 prev_dir 不同）
        dir_new = self.last_direction
        self.last_reward = cur_reward

        # # 追加：将单行汇总调试输出写入文件（携带 timestep）
        # try:
        #     log_path = "data/adaptK_log.txt"
        #     # 写 CSV 风格表头一次
        #     if not hasattr(self, "_adaptk_logged_header"):
        #         self._adaptk_logged_header = False
        #     if not self._adaptk_logged_header:
        #         with open(log_path, "w", encoding="utf-8") as f:
        #             f.write("timestep,pre_std,post_std,delivered,loss,reward_prev,reward_cur,improve,dir_prev,dir_new,K_old,K_new\n")
        #         self._adaptk_logged_header = True
        #     # 追加一行
        #     t_val = getattr(self, "_adaptk_last_t", None)
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write("{t},{p:.6f},{q:.6f},{d:.6f},{l:.6f},{rp:.6f},{rc:.6f},{im:.6f},{dirp},{dirn},{ko},{kn}\n".format(
        #             t=(-1 if t_val is None else t_val),
        #             p=stats["pre_std"], q=stats["post_std"], d=stats["delivered_total"], l=stats["total_loss"],
        #             rp=prev_reward, rc=cur_reward, im=improve, dirp=prev_dir, dirn=dir_new,
        #             ko=prev_K, kn=self.K
        #         ))
        # except Exception:
        #     pass

        self.K_stats.append({
            'old_K': old_K,
            'new_K': self.K,
            'improve': improve,
            'reward': cur_reward,
            'pre_std': stats["pre_std"],
            'post_std': stats["post_std"],
            'delivered': stats["delivered_total"],
            'loss': stats["total_loss"]
        })

    def print_statistics(self):
        """打印所有时间点方差的平均值以及循环结束后总体的损失能量值"""
        # 计算所有时间点方差的平均值
        avg_variance = np.mean(self.energy_Standards) if self.energy_Standards else 0
        print(f"\n所有时间点方差的平均值: {avg_variance:.4f}")

        # 计算总体的损失能量值
        # 总发送能量 = 所有节点传输的能量总和
        total_sent_energy = sum(sum(node.transferred_history) for node in self.network.nodes)

        # 总接收能量 = 所有节点接收的能量总和
        total_received_energy = sum(sum(node.received_history) for node in self.network.nodes)

        # 总损失能量 = 总发送能量 - 总接收能量
        total_loss_energy = total_sent_energy - total_received_energy

        print(f"总体损失能量值: {total_loss_energy:.4f} Joules")
        print(f"总发送能量: {total_sent_energy:.4f} Joules")
        print(f"总接收能量: {total_received_energy:.4f} Joules")
        print(f"能量传输效率: {(total_received_energy / total_sent_energy * 100 if total_sent_energy > 0 else 0):.2f}%")

        # 返回这些统计值，以便在其他地方使用
        return {
            'avg_variance': avg_variance,
            'total_loss_energy': total_loss_energy,
            'total_sent_energy': total_sent_energy,
            'total_received_energy': total_received_energy
        }

    def _compute_step_stats(self, plans, pre_energies, pre_received_total):
        # 计算本次传能后的统计量（基于真实执行后的节点状态/历史）
        post_energies = np.array([n.current_energy for n in self.network.nodes], dtype=float)
        pre_std = float(np.std(pre_energies))
        post_std = float(np.std(post_energies))

        sent_total = sum(p["donor"].E_char for p in plans)  # 本轮各 donor 实际下发的名义能量（execute 中按此发送）
        post_received_total = sum(sum(n.received_history) for n in self.network.nodes)
        delivered_total = max(0.0, post_received_total - pre_received_total)
        total_loss = max(0.0, sent_total - delivered_total)

        return {
            "pre_std": pre_std,
            "post_std": post_std,
            "delivered_total": delivered_total,
            "total_loss": total_loss
        }

    def simulate(self):
        start_time = datetime(2023, 1, 2)
        """Run the energy simulation for the specified number of time steps."""
        start_time = datetime(2023, 1, 2)
        for t in range(self.time_steps):
            print("\n--- Time step {} ---".format(t + 1))

            # Step 1: 更新节点能量（采集 + 衰减 + 位置）
            self.network.update_network_energy(t)

            # # Step 1.5: 汇总能量信息 + ADCR（虚拟中心，含通信能耗结算）
            # if hasattr(self.network, "adcr_link") and (self.network.adcr_link is not None):
            #     self.network.adcr_link.step(t)

            # Step 2: 每小时一次执行能量传输 + 自适应并发 K//notice
            if t % 60 == 0:
                current_time = start_time + timedelta(minutes=t)
                pre_energies = np.array([n.current_energy for n in self.network.nodes], dtype=float)
                pre_received_total = sum(sum(n.received_history) for n in self.network.nodes)
            
                node_energies = [node.current_energy for node in self.network.nodes]
                energy_avg = np.mean(node_energies)
                energy_std = np.std(node_energies)
            
                self.energy_averages.append(energy_avg)
                self.energy_Standards.append(energy_std)
            
                # ★ 优先使用外部调度器（若提供）
                if self.scheduler is not None:
                    # 同步自适应K给调度器（若其带 K）
                    if hasattr(self.scheduler, "K"):
                        try:
                            self.scheduler.K = self.K
                            # self.scheduler.K = 1
                        except Exception:
                            pass
            
                    result = self.scheduler.plan(self.network, t)
                    if isinstance(result, tuple):
                        plans, cand = result
                    else:
                        plans = result
                        cand = []
            
                    # PowerControlScheduler 有自定义执行器（按 energy_sent 执行）
                    if (PowerControlScheduler is not None) and isinstance(self.scheduler, PowerControlScheduler):
                        self.scheduler.execute_plans(self.network, plans)
                    else:
                        self.network.execute_energy_transfer(plans)
                else:
                    # 兼容旧逻辑：使用 network.run_routing()
                    plans = self.network.run_routing(t, max_donors_per_receiver=self.K)
                    self.network.execute_energy_transfer(plans)
            
                # 执行后自己算 stats
                stats = self._compute_step_stats(plans, pre_energies, pre_received_total)
            
                # 记录该时间步的计划、候选信息和节点能量，供可视化使用
                try:
                    # 收集所有节点的当前能量
                    node_energies = {node.node_id: node.current_energy for node in self.network.nodes}
                    self.plans_by_time[t] = {
                        "plans": plans, 
                        "candidates": cand,
                        "node_energies": node_energies
                    }
                except Exception:
                    pass
            
                print("K={} pre_std={:.4f} post_std={:.4f} delivered={:.2f} loss={:.2f}".format(
                    self.K, stats['pre_std'], stats['post_std'], stats['delivered_total'], stats['total_loss']
                ))
            
                # 自适应调 K
                self._adaptk_last_t = t
                self._adapt_K(stats)
            
                # 记录K值和对应时间
                self.K_history.append(self.K)
                self.K_timestamps.append(current_time)
            
            
                # 绘制每60个时间单位节点能量的平均值与方差
                # self.plot_energy_stats()

            # Step 3: 记录能量状态
            self.record_energy_status()
        # 模拟结束后绘制K值随时间变化的图表
        self.plot_K_history()
        self.print_statistics()

    def plot_K_history(self, mdates=None):
        """绘制K值随时间变化的图表"""
        import matplotlib.dates as mdates  # 在函数内部重新导入
        from matplotlib.dates import DateFormatter  # 也单独导入DateFormatter
        if not self.K_history:
            print("没有K值历史记录可供绘制")
            return

        # 创建图表
        plt.figure(figsize=(12, 6))
        plt.plot(self.K_timestamps, self.K_history, marker='o', linestyle='-')

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
            # 出错时使用默认格式

        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)

        # 显示图表
        plt.tight_layout()
        k_value_plot_path = OutputManager.get_file_path(self.session_dir, 'K_value_over_time.png')
        plt.savefig(k_value_plot_path)
        plt.show()

        # 同时创建表格形式的数据
        self.create_K_table()

    def create_K_table(self):
        """创建K值的表格数据"""
        # 创建DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'Time': [t.strftime('%Y-%m-%d %H:%M') for t in self.K_timestamps],
            'Day': [t.strftime('%A') for t in self.K_timestamps],
            'Hour': [t.hour for t in self.K_timestamps],
            'K_Value': self.K_history
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




    def plot_K_history(self, mdates=None):
        import plotly.graph_objects as go
        if not self.K_history:
            print("没有K值历史记录可供绘制")
            return

        # 横轴用分钟（每点间隔60）
        x_vals = [i * 60 for i in range(len(self.K_history))]
        # 如果更想用真实时间：x_vals = self.K_timestamps

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=self.K_history, mode='lines+markers', name='K',
            hovertemplate='Time: %{x} min<br>K: %{y}<extra></extra>'
            # 若用 datetime：hovertemplate='Time: %{x|%Y-%m-%d %H:%M}<br>K: %{y}<extra></extra>'
        ))

        fig.update_layout(
            title='Dynamic K Values Over Time',
            xaxis_title='Time (minutes)',
            yaxis_title='K Value',
            template='plotly_white',
            hovermode='closest'
        )

        # 如需保存静态图（需安装 kaleido）：pip install -U kaleido
        # fig.write_image('K_value_over_time.png', width=1000, height=600, scale=2)
        fig.show()

        # 同时创建表格形式的数据
        self.create_K_table()

    def create_K_table(self):
        """创建K值的表格数据"""
        # 创建DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'Time': [t.strftime('%Y-%m-%d %H:%M') for t in self.K_timestamps],
            'Day': [t.strftime('%A') for t in self.K_timestamps],
            'Hour': [t.hour for t in self.K_timestamps],
            'K_Value': self.K_history
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

    def record_energy_status(self):
        """Record the energy status of all nodes in the network."""
        energy_data = []
        for node in self.network.nodes:
            energy_data.append({
                "node_id": node.node_id,
                "current_energy": node.current_energy,
                "received_energy": sum(node.received_history),  # Sum of all received energy
                "transferred_energy": sum(node.transferred_history),  # Sum of all transferred energy
                "energy_history": node.energy_history[-1] if node.energy_history else None
            })
        self.results.append(energy_data)

    def save_results(self, filename=None):
        """Save the simulation results to a CSV file."""
        if filename is None:
            filename = OutputManager.get_file_path(self.session_dir, 'simulation_results.csv')
        
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["Time Step", "Node ID", "Current Energy", "Received Energy", "Transferred Energy", "Energy History"])
            for t, step_result in enumerate(self.results):
                for node_data in step_result:
                    writer.writerow([t + 1, node_data["node_id"], node_data["current_energy"],
                                     node_data["received_energy"], node_data["transferred_energy"],
                                     node_data["energy_history"]])

        print(f"Results saved to {filename}")

    def display_results(self):
        """Display the simulation results."""
        print("\n--- Energy Simulation Results ---")
        for t, step_result in enumerate(self.results):
            print(f"\nTime step {t + 1}:")
            for node_data in step_result:
                print(f"Node {node_data['node_id']} - Current Energy: {node_data['current_energy']:.2f} Joules")
                print(f"  Received Energy: {node_data['received_energy']:.2f} Joules")
                print(f"  Transferred Energy: {node_data['transferred_energy']:.2f} Joules")
                if node_data['energy_history']:
                    print(f"  Energy History: {node_data['energy_history']}")

    def save_results(self, filename="simulation_results.csv"):
        """Save the simulation results to a CSV file."""
        import csv
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(
                ["Time Step", "Node ID", "Current Energy", "Received Energy", "Transferred Energy", "Energy History"])
            for t, step_result in enumerate(self.results):
                for node_data in step_result:
                    writer.writerow([t + 1, node_data["node_id"], node_data["current_energy"],
                                     node_data["received_energy"], node_data["transferred_energy"],
                                     node_data["energy_history"]])

    def plot_results(self):
        """Plot the results of the energy simulation."""
        import matplotlib.pyplot as plt
        time_steps = list(range(1, self.time_steps + 1))

        # Extract energy data for plotting
        node_ids = [node.node_id for node in self.network.nodes]
        energy_data = {node_id: [] for node_id in node_ids}

        for step_result in self.results:
            for node_data in step_result:
                energy_data[node_data["node_id"]].append(node_data["current_energy"])

        # Plot energy data for each node
        plt.figure(figsize=(10, 6))
        for node_id, energy_values in energy_data.items():
            plt.plot(time_steps, energy_values, label=f"Node {node_id}")

        plt.title("Energy Consumption Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Energy (Joules)")
        plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))
        plt.grid(True)
        plt.show()

    def plot_energy_stats(self):
        """绘制每60个时间单位节点能量的平均值与方差"""
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
        plt.plot(range(0, len(self.energy_Standards) * 60, 60), self.energy_Standards, marker='o', color='r')
        plt.title("Variance of Node Energy Every 60 Time Units")
        plt.xlabel("Time (minutes)")
        plt.ylabel("Energy Variance (Joules^2)")
        plt.grid(True)

        # 显示图表
        plt.tight_layout()
        plt.show()