
from datetime import datetime, timedelta
import numpy as np

from utils.output_manager import OutputManager
from dynamic_k.k_adaptation import KAdaptationManager
from .simulation_stats import SimulationStats
from .result_manager import ResultManager

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

        # 智能被动传能参数
        self.passive_mode = passive_mode
        self.check_interval = check_interval
        self.critical_ratio = critical_ratio
        self.energy_variance_threshold = energy_variance_threshold
        self.cooldown_period = cooldown_period
        self.predictive_window = predictive_window
        self.last_transfer_time = -cooldown_period  # 上次传能时间
        self.energy_history = []  # 记录能量历史用于预测

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

    def should_trigger_energy_transfer(self, t):
        """
        智能综合决策：是否应该触发能量传输
        
        综合考虑以下因素：
        1. 检查间隔：不是每个时间步都检查
        2. 冷却期：避免频繁传能造成振荡
        3. 低能量节点比例：达到临界比例才触发
        4. 能量方差：网络能量分布不均衡程度
        5. 预测性触发：基于能量消耗速率预测
        
        :param t: 当前时间步
        :return: (是否触发, 触发原因)
        """
        # 如果不是被动模式，使用传统的定时触发（每60分钟）
        if not self.passive_mode:
            return (t % 60 == 0), "定时触发"
        
        # 1. 检查间隔：只在指定间隔检查
        if t % self.check_interval != 0:
            return False, None
        
        # 2. 冷却期检查：避免过于频繁的传能
        if t - self.last_transfer_time < self.cooldown_period:
            return False, None
        
        nodes = self.network.nodes
        total_nodes = len(nodes)
        
        # 计算当前网络能量状态
        energies = np.array([node.current_energy for node in nodes])
        thresholds = np.array([node.low_threshold_energy for node in nodes])
        
        # 3. 低能量节点比例检查
        low_energy_nodes = energies < thresholds
        low_energy_ratio = np.sum(low_energy_nodes) / total_nodes
        
        # 4. 能量方差检查（归一化方差，考虑能量不均衡）
        if len(energies) > 0 and np.mean(energies) > 0:
            energy_cv = np.std(energies) / np.mean(energies)  # 变异系数
        else:
            energy_cv = 0
        
        # 5. 预测性检查：基于能量消耗速率
        predict_critical = False
        if len(self.energy_history) >= 2:
            # 计算平均能量下降速率
            recent_energies = self.energy_history[-min(5, len(self.energy_history)):]
            if len(recent_energies) >= 2:
                energy_rates = []
                for i in range(len(recent_energies) - 1):
                    rate = (recent_energies[i] - recent_energies[i + 1]) / self.check_interval
                    energy_rates.append(rate)
                avg_rate = np.mean(energy_rates)
                
                # 预测未来predictive_window分钟后的能量
                predicted_energies = energies - avg_rate * self.predictive_window
                predicted_low = np.sum(predicted_energies < thresholds) / total_nodes
                
                if predicted_low > self.critical_ratio * 0.5:  # 预测性阈值更宽松
                    predict_critical = True
        
        # 记录当前能量状态
        self.energy_history.append(np.mean(energies))
        if len(self.energy_history) > 20:  # 只保留最近20次记录
            self.energy_history.pop(0)
        
        # 综合决策逻辑
        reasons = []
        should_trigger = False
        
        # 条件1: 严重情况 - 低能量节点比例超过临界值
        if low_energy_ratio > self.critical_ratio:
            should_trigger = True
            reasons.append(f"低能量节点比例={low_energy_ratio:.2%}>{self.critical_ratio:.2%}")
        
        # 条件2: 能量分布严重不均
        if energy_cv > self.energy_variance_threshold:
            should_trigger = True
            reasons.append(f"能量变异系数={energy_cv:.3f}>{self.energy_variance_threshold:.3f}")
        
        # 条件3: 预测性触发
        if predict_critical:
            should_trigger = True
            reasons.append(f"预测{self.predictive_window}分钟后将出现能量危机")
        
        # 条件4: 紧急情况 - 存在极低能量节点（低于阈值的50%）
        critical_nodes = energies < (thresholds * 0.5)
        if np.any(critical_nodes):
            should_trigger = True
            critical_count = np.sum(critical_nodes)
            reasons.append(f"存在{critical_count}个极低能量节点（<阈值50%）")
        
        if should_trigger:
            reason_str = " | ".join(reasons)
            return True, reason_str
        
        return False, None

    def simulate(self):
        """Run the energy simulation for the specified number of time steps."""
        start_time = datetime(2023, 1, 2)
        
        for t in range(self.time_steps):
            # Step 1: 更新节点能量（采集 + 衰减 + 位置）
            self.network.update_network_energy(t)

            # Step 1.5: ADCR链路层处理（如果启用）
            if hasattr(self.network, 'adcr_link') and self.network.adcr_link is not None:
                self.network.adcr_link.step(t)

            # Step 2: 智能综合决策能量传输触发
            should_trigger, trigger_reason = self.should_trigger_energy_transfer(t)
            
            if should_trigger:
                # 更新上次传能时间
                self.last_transfer_time = t
                
                # 输出触发信息
                mode_label = "智能被动传能" if self.passive_mode else "定时主动传能"
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
                            self.network.execute_energy_transfer(plans)
                    else:
                        # 兼容旧逻辑：使用 network.run_routing()
                        plans = self.network.run_routing(t, max_donors_per_receiver=self.K)
                        self.network.execute_energy_transfer(plans)
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
