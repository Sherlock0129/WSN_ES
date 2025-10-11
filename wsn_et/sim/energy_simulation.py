
from datetime import datetime, timedelta
import numpy as np
# 调整 energy_management 的导入到 core 包
from wsn_et.dynamic_k.reward import WeightedReward
from wsn_et.dynamic_k.adapt_k import HysteresisAdaptK
from .transfer_executor import TransferExecutor
from .data_collector import DataCollector
try:
    # 调整为从 scheduling 子包导入
    from ..scheduling.schedulers import PowerControlScheduler
except Exception:
    # 允许没有 schedulers.py 时仍可运行（使用 network.run_routing）
    PowerControlScheduler = None

class EnergySimulation:
    def __init__(self, network, time_steps, scheduler=None, reward=None, adapt_k=None, decision_period=60, data_collector=None):
        """
        Initialize the energy simulation for the network.

        :param network: The network object that contains nodes and their parameters.
        :param time_steps: Total number of time steps to simulate.
        :param scheduler: 调度器对象
        :param reward: 奖励函数对象
        :param adapt_k: K值自适应策略对象
        :param decision_period: 决策间隔（步数）
        :param data_collector: 数据收集器对象
        """
        self.network = network
        self.time_steps = time_steps
        self.scheduler = scheduler  # 可插拔：调度器
        self.transfer_executor = TransferExecutor()

        # 自适应并发参数（迁移到策略后仅保留 dynamic_k 当前值；其余交由策略持有）
        self.K = 1
        # 策略注入：奖励函数与自适应K；若未提供则使用默认实现
        self.reward_fn = reward or WeightedReward()
        self.adapt_k = adapt_k or HysteresisAdaptK(self.reward_fn, K_max=24, hysteresis=0.2)

        # 决策间隔
        self.decision_period = int(decision_period)

        # 数据收集器
        self.data_collector = data_collector or DataCollector()
        
        # 运行元信息
        self.run_metadata = {
            'time_steps': time_steps,
            'decision_period': decision_period,
            'scheduler_type': type(scheduler).__name__ if scheduler else 'None',
            'reward_type': type(self.reward_fn).__name__,
            'adapt_k_type': type(self.adapt_k).__name__
        }

        self._adaptk_last_t = None


    def simulate(self):
        """运行仿真"""
        # 初始化数据收集器
        self.data_collector.reset(self.run_metadata)
        
        start_time = datetime(2023, 1, 2)

        for t in range(self.time_steps):
            print("\n--- Time step {} ---".format(t + 1))

            self.network.update_network_energy(t)

            if t % self.decision_period == 0:
                current_time = start_time + timedelta(minutes=t)
                
                # 决策前快照（数据收集器接管）
                pre_energies, pre_received_total = self.data_collector.on_decision_before(t, self.network)
                
                if self.scheduler is not None:
                    if hasattr(self.scheduler, "dynamic_k"):
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
                
                    # 统一通过执行器执行计划
                    self.transfer_executor.execute(self.network, plans)
                else:
                    plans = self.network.run_routing(t, max_donors_per_receiver=self.K)
                    self.transfer_executor.execute(self.network, plans)
                
                # 计算统计信息（直接使用数据收集器）
                stats = self.data_collector.compute_step_stats(plans, pre_energies, pre_received_total, self.network)
                
                self._adaptk_last_t = t
                # 使用策略更新 dynamic_k
                try:
                    new_K = self.adapt_k.update(self.K, stats, t, network=self.network, scheduler=self.scheduler)
                    self.K = int(max(1, new_K))
                except Exception:
                    pass
                
                # 决策后记录（数据收集器接管）
                self.data_collector.on_decision_after(t, self.network, plans, stats, self.K, current_time)

            # 每步状态采样（数据收集器接管）
            self.data_collector.on_step_sample(t, self.network)
        
        # 完成数据收集
        self.data_collector.finalize()
        self.data_collector.print_statistics(self.network)

    def export_data(self, format='csv'):
        """导出数据"""
        if format == 'csv':
            self.data_collector.export_decision_summary_csv()
            self.data_collector.export_status_csv()
        elif format == 'json':
            self.data_collector.export_json()
        else:
            print(f"不支持的导出格式: {format}")
    
    def get_data_collector(self):
        """获取数据收集器实例"""
        return self.data_collector

