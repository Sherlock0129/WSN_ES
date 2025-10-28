
from datetime import datetime, timedelta
import numpy as np

from utils.output_manager import OutputManager
from dynamic_k.k_adaptation import KAdaptationManager
from .simulation_stats import SimulationStats
from .result_manager import ResultManager
from scheduling.passive_transfer import PassiveTransferManager

try:
    from scheduling.schedulers import PowerControlScheduler
except Exception:
    # 允许没有 schedulers.py 时仍可运行（使用原 run_routing）
    PowerControlScheduler = None

class EnergySimulation:
    def __init__(self, network, time_steps, scheduler=None, 
                 # 能量传输控制
                 enable_energy_sharing=True,
                 # 自适应K值参数
                 enable_k_adaptation=True, initial_K=1, K_max=24, hysteresis=0.2, 
                 w_b=0.8, w_d=0.8, w_l=1.5,
                 # 其他参数
                 use_lookahead=False, fixed_k=3, output_dir="data", use_gpu=False,
                 # 智能被动传能参数
                 passive_mode=True,  # 是否启用被动模式
                 check_interval=10,  # 检查间隔（分钟）
                 critical_ratio=0.2,  # 低能量节点临界比例
                 energy_variance_threshold=0.3,  # 能量方差阈值
                 cooldown_period=30,  # 冷却期（分钟）
                 predictive_window=60):  # 预测窗口（分钟）
        """
        Initialize the energy simulation for the network.

        :param network: The network object that contains nodes and their parameters.
        :param time_steps: Total number of time steps to simulate.
        :param scheduler: Optional scheduler for energy transfer planning.
        :param enable_k_adaptation: Whether to enable K value adaptation.
        :param initial_K: Initial K value for adaptive concurrency.
        :param K_max: Maximum K value for adaptive concurrency.
        :param hysteresis: Hysteresis threshold for K adaptation.
        :param w_b: Weight for a balance improvement factor.
        :param w_d: Weight for a delivery factor.
        :param w_l: Weight for a loss penalty factor.
        :param use_lookahead: Whether to use lookahead simulation.
        :param fixed_k: Fixed K value when not using adaptive K.
        :param use_gpu: Whether to use GPU acceleration.
        """
        self.network = network
        self.time_steps = time_steps
        self.scheduler = scheduler
        self.plans_by_time = {}
        self.use_gpu = use_gpu
        self.enable_energy_sharing = enable_energy_sharing
        self.enable_k_adaptation = enable_k_adaptation
        self.fixed_k = fixed_k

        # 创建智能被动传能管理器
        self.passive_manager = PassiveTransferManager(
            passive_mode=passive_mode,
            check_interval=check_interval,
            critical_ratio=critical_ratio,
            energy_variance_threshold=energy_variance_threshold,
            cooldown_period=cooldown_period,
            predictive_window=predictive_window
        )

        # 创建按日期命名的输出目录
        self.session_dir = OutputManager.get_session_dir(output_dir)

        # 初始化各个管理器
        self.k_adaptation = KAdaptationManager(
            initial_K=initial_K, K_max=K_max, hysteresis=hysteresis,
            w_b=w_b, w_d=w_d, w_l=w_l, use_lookahead=use_lookahead
        )
        
        # 如果不启用自适应K值，则设置为固定K值
        if not self.enable_k_adaptation:
            self.k_adaptation.K = self.fixed_k
            
        self.stats = SimulationStats(self.session_dir, use_gpu=use_gpu)
        self.result_manager = ResultManager(self.session_dir)

    @property
    def K(self):
        """获取当前K值"""
        return self.k_adaptation.K
    
    @K.setter
    def K(self, value):
        """设置K值"""
        self.k_adaptation.K = value

    def simulate(self):
        """Run the energy simulation for the specified number of time steps."""
        start_time = datetime(2023, 1, 2)
        
        for t in range(self.time_steps):
            # Step 1: 更新节点能量（采集 + 衰减 + 位置）
            self.network.update_network_energy(t)

            # Step 1.5: ADCR链路层处理（如果启用）
            if hasattr(self.network, 'adcr_link') and self.network.adcr_link is not None:
                self.network.adcr_link.step(t)

            # Step 2: 智能综合决策能量传输触发（使用被动传能管理器）
            should_trigger, trigger_reason = self.passive_manager.should_trigger_transfer(t, self.network)
            
            if should_trigger:
                # 更新上次传能时间
                self.passive_manager.update_last_transfer_time(t)
                
                # 输出触发信息
                mode_label = "智能被动传能" if self.passive_manager.passive_mode else "定时主动传能"
                if trigger_reason:
                    print(f"\n[{mode_label}] 时间步 {t}: {trigger_reason}")
                
                current_time = start_time + timedelta(minutes=t)
                pre_energies = np.array([n.current_energy for n in self.network.nodes], dtype=float)
                pre_received_total = sum(sum(n.received_history) for n in self.network.nodes)
            
                # 记录能量统计
                node_energies = [node.current_energy for node in self.network.nodes]
                self.stats.record_energy_stats(node_energies)
            
                # 检查是否启用能量传输
                if self.enable_energy_sharing:
                    # ★ 优先使用外部调度器（若提供）
                    if self.scheduler is not None:
                        # 【关键】更新NodeInfoManager中的节点能量信息
                        # 在每次调度之前，必须同步真实节点的当前能量到InfoNode
                        if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
                            self.scheduler.nim.batch_update_node_info(
                                nodes=self.network.nodes,
                                current_time=t
                            )
                        
                        # 同步自适应K给调度器（若其带 K）
                        if hasattr(self.scheduler, "K"):
                            try:
                                self.scheduler.K = self.K
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
                            self.network.execute_energy_transfer(plans, current_time=t)
                    else:
                        # 兼容旧逻辑：使用 network.run_routing()
                        plans = self.network.run_routing(t, max_donors_per_receiver=self.K)
                        self.network.execute_energy_transfer(plans, current_time=t)
                else:
                    # 能量传输被禁用，创建空的计划
                    plans = []
                    cand = []
            
                # 计算统计信息
                stats = self.stats.compute_step_stats(plans, pre_energies, pre_received_total, self.network)
            
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
            
                # 根据配置决定是否进行自适应调 K
                if self.enable_k_adaptation:
                # 自适应调 K
                    self.k_adaptation._adaptk_last_t = t
                    self.k_adaptation.adapt_K(stats, self.network, self.scheduler)
                else:
                    # 使用固定K值
                    self.k_adaptation.K = self.fixed_k
            
                # 记录K值和对应时间
                self.k_adaptation.record_K_value(current_time)

            # Step 3: 记录能量状态
            self.result_manager.record_energy_status(self.network)
        
        # 模拟结束后绘制K值随时间变化的图表
        K_history, K_timestamps, _ = self.k_adaptation.get_K_history()
        self.stats.plot_K_history(K_history, K_timestamps)
        self.stats.print_statistics(self.network)

    # 委托方法 - 将功能委托给相应的管理器
    def print_statistics(self):
        """打印仿真统计信息"""
        return self.stats.print_statistics(self.network)
    
    def save_results(self, filename=None):
        """保存仿真结果"""
        return self.result_manager.save_results(filename)

    def display_results(self):
        """显示仿真结果"""
        return self.result_manager.display_results()

    def plot_results(self):
        """绘制仿真结果"""
        return self.stats.plot_results(self.result_manager.get_results(), self.time_steps, self.network)

    def plot_energy_stats(self):
        """绘制能量统计图表"""


    # 委托方法 - 将功能委托给相应的管理器
    def print_statistics(self):
        """打印仿真统计信息"""
        return self.stats.print_statistics(self.network)
    
    def save_results(self, filename=None):
        """保存仿真结果"""
        return self.result_manager.save_results(filename)


    def display_results(self):

        """显示仿真结果"""
        return self.result_manager.display_results()


    def plot_results(self):

        """绘制仿真结果"""
        return self.stats.plot_results(self.result_manager.get_results(), self.time_steps, self.network)


    def plot_energy_stats(self):
        """绘制能量统计图表"""
        return self.stats.plot_energy_stats()

    def plot_K_history(self):
        """绘制K值历史图表"""
        if hasattr(self.k_adaptation, 'K_history') and hasattr(self.k_adaptation, 'K_timestamps'):
            return self.stats.plot_K_history(self.k_adaptation.K_history, self.k_adaptation.K_timestamps)
        else:
            print("没有K值历史记录可供绘制")
            return None
