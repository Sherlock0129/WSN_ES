
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

# 导入EETOR配置设置函数
try:
    from routing.energy_transfer_routing import set_eetor_config
except ImportError:
    set_eetor_config = None

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
        
        # 设置被动传输管理器的NodeInfoManager（如果scheduler存在）
        # 这确保了被动传输决策使用虚拟能量层而不是直接访问SensorNode
        if self.scheduler is not None and hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
            self.passive_manager.node_info_manager = self.scheduler.nim
        
        for t in range(self.time_steps):
            # Step 1: 更新节点能量（采集 + 衰减 + 位置）
            self.network.update_network_energy(t)

            # Step 1.5: ADCR链路层处理（如果启用）
            if hasattr(self.network, 'adcr_link') and self.network.adcr_link is not None:
                self.network.adcr_link.step(t)

            # Step 1.6: 在检查触发条件前，先同步节点能量到虚拟能量层（确保被动传输管理器使用最新能量）
            # 这确保了被动传输决策基于最新的节点能量状态
            if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
                self.scheduler.nim.batch_update_node_info(
                    nodes=self.network.nodes,
                    current_time=t
                )

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
                pre_transferred_total = sum(sum(n.transferred_history) for n in self.network.nodes)
            
                # 记录能量统计
                node_energies = [node.current_energy for node in self.network.nodes]
                self.stats.record_energy_stats(node_energies)
                
                # 【反馈机制】收集调度前的网络状态
                pre_state = {
                    'energies': pre_energies,
                    'alive_nodes': sum(1 for e in pre_energies if e > 0),
                    'total_energy': float(np.sum(pre_energies)),
                    'std': float(np.std(pre_energies))
                }
            
                # 检查是否启用能量传输
                if self.enable_energy_sharing:
                    # ★ 优先使用外部调度器（若提供）
                    if self.scheduler is not None:
                        # 【关键】更新NodeInfoManager中的节点能量信息
                        # 在每次调度之前，必须同步真实节点的当前能量到InfoNode
                        if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
                            # 设置当前时间
                            self.scheduler.nim.current_time = t
                            # 仅当使用DurationAwareLyapunovScheduler时，才检查并更新锁定状态
                            from scheduling.schedulers import DurationAwareLyapunovScheduler
                            if isinstance(self.scheduler, DurationAwareLyapunovScheduler):
                                self.scheduler.nim.check_and_update_locks(t)
                            # 更新节点信息
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
            
                # 能量传输执行完成后，更新节点信息表（基于传输计划计算实际能量变化）
                if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
                    # 设置当前时间
                    self.scheduler.nim.current_time = t
                    # 首先：基于传输计划，更新参与传输的节点的能量（确定性计算）
                    # 中心节点知道所有传输路径和能量变化，可以精确计算
                    if plans:
                        self.scheduler.nim.apply_energy_transfer_changes(plans, current_time=t)
                    
                    # 然后：估算未上报节点的能量（衰减+太阳能采集）
                    self.scheduler.nim.estimate_all_nodes(current_time=t)
                    
                    # 机会主义信息传递：检查超时并强制上报（如果启用路径信息收集器）
                    if (hasattr(self.network, 'path_info_collector') and 
                        self.network.path_info_collector is not None and
                        hasattr(self.network.path_info_collector, 'enable_opportunistic_info_forwarding') and
                        self.network.path_info_collector.enable_opportunistic_info_forwarding):
                        max_wait_time = getattr(self.network.path_info_collector, 'max_wait_time', 10)
                        forced_count = self.scheduler.nim.check_timeout_and_force_report(
                            current_time=t,
                            max_wait_time=max_wait_time,
                            path_collector=self.network.path_info_collector,
                            network=self.network
                        )
                        if forced_count > 0:
                            print(f"[超时强制上报] 时间步 {t}: {forced_count} 个节点")
            
                # 计算统计信息
                stats = self.stats.compute_step_stats(plans, pre_energies, pre_received_total, pre_transferred_total, self.network)
                
                # 【反馈机制】收集调度后的网络状态并计算反馈分数
                post_energies = np.array([n.current_energy for n in self.network.nodes], dtype=float)
                post_state = {
                    'energies': post_energies,
                    'alive_nodes': sum(1 for e in post_energies if e > 0),
                    'total_energy': float(np.sum(post_energies)),
                    'std': float(np.std(post_energies))
                }
                
                # 计算反馈分数
                feedback_score = 0.0
                feedback_details = {}
                if self.scheduler is not None:
                    feedback_score, feedback_details = self.scheduler.compute_network_feedback_score(
                        pre_state, post_state, stats
                    )
                    # 记录反馈分数到统计信息
                    self.stats.record_feedback_score(t, feedback_score, feedback_details)
                    
                    # 打印反馈信息
                    impact = feedback_details.get('impact', '未知')
                    print(f"[反馈] 本次调度影响: {impact}, 综合分数: {feedback_score:.2f}")
                    
                    # 调用调度器的post_step方法（用于自适应调度器）
                    if hasattr(self.scheduler, 'post_step'):
                        feedback_data = {
                            'total_score': feedback_score,
                            'details': feedback_details
                        }
                        self.scheduler.post_step(self.network, t, feedback_data)
            
                # 记录该时间步的计划、候选信息和节点能量，供可视化使用
                try:
                    # 收集所有节点的当前能量
                    node_energies = {node.node_id: node.current_energy for node in self.network.nodes}
                    self.plans_by_time[t] = {
                        "plans": plans, 
                        "candidates": cand,
                        "node_energies": node_energies,
                        "feedback_score": feedback_score,  # 添加反馈分数
                        "feedback_details": feedback_details  # 添加详细信息
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
        
        # 模拟结束后绘制K值随时间变化的图表（训练模式下跳过）
        if not getattr(self, 'training_mode', False):
            K_history, K_timestamps, _ = self.k_adaptation.get_K_history()
            self.stats.plot_K_history(K_history, K_timestamps)
            
            # 绘制反馈分数图表（如果有记录）
            if self.stats.feedback_scores:
                self.stats.plot_feedback_scores()
        
        # 获取信息传输能量消耗统计（如果可用）
        info_transmission_stats = None
        if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
            # 获取统计信息
            info_transmission_stats = self.scheduler.nim.get_info_transmission_statistics()
            # 打印统计信息
            self.scheduler.nim.log_info_transmission_statistics()
        
        # 打印并保存统计信息（包含信息传输统计）
        self.stats.print_statistics(self.network, additional_info={
            'info_transmission': info_transmission_stats
        } if info_transmission_stats else None)

    # 委托方法 - 将功能委托给相应的管理器
    def print_statistics(self, additional_info: dict = None):
        """
        打印仿真统计信息
        
        Args:
            additional_info: 额外的统计信息（例如信息传输统计）
        """
        # 如果未提供 additional_info，尝试获取信息传输统计
        if additional_info is None:
            if hasattr(self.scheduler, 'nim') and self.scheduler.nim is not None:
                info_transmission_stats = self.scheduler.nim.get_info_transmission_statistics()
                if info_transmission_stats:
                    additional_info = {'info_transmission': info_transmission_stats}
        
        return self.stats.print_statistics(self.network, additional_info)
    
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
