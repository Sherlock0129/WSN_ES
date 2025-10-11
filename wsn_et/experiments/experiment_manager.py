"""
统一的实验配置和网络管理模块
"""

import os
import random
import numpy as np
import copy
from typing import Dict, Any, Optional, List, Tuple

from wsn_et.core.network import Network
from wsn_et.sim.energy_simulation import EnergySimulation
from wsn_et.scheduling.schedulers import LyapunovScheduler


class ExperimentConfig:
    """实验配置管理"""
    
    def __init__(self):
        # 默认网络配置
        self.network_config = {
            "num_nodes": 25,
            "low_threshold": 0.1,
            "high_threshold": 0.9,
            "node_initial_energy": 40000,
            "max_hops": 3,
            "random_seed": 129,
            "distribution_mode": "random",
            "network_area": {
                "width": 10.0,
                "height": 10.0
            },
            "min_distance": 0.5
        }
        
        # 默认调度器配置
        self.scheduler_config = {
            "V": 0.5,
            "K": 3,
            "max_hops": 3
        }
        
        # 默认仿真配置
        self.simulation_config = {
            "time_steps": 10080,
            "decision_period": 60
        }
        
        # 默认输出配置
        self.output_config = {
            "results_file": "data/results.csv",
            "output_dir": "adcr",
            "base_data_dir": "data"
        }
        
        # 种子配置
        self.seed_config = {
            "fixed_network": True,
            "fixed_seed": 130
        }
    
    def update_network_config(self, **kwargs):
        """更新网络配置"""
        self.network_config.update(kwargs)
    
    def update_scheduler_config(self, **kwargs):
        """更新调度器配置"""
        self.scheduler_config.update(kwargs)
    
    def update_simulation_config(self, **kwargs):
        """更新仿真配置"""
        self.simulation_config.update(kwargs)
    
    def update_output_config(self, **kwargs):
        """更新输出配置"""
        self.output_config.update(kwargs)
    
    def update_seed_config(self, **kwargs):
        """更新种子配置"""
        self.seed_config.update(kwargs)


class NetworkManager:
    """网络管理器 - 统一管理网络创建和缓存"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._base_network_cached = None
    
    def _build_base_network(self) -> Network:
        """构建基础网络（缓存机制，只创建一次）"""
        if self._base_network_cached is None:
            if self.config.seed_config["fixed_network"]:
                random.seed(self.config.seed_config["fixed_seed"])
                np.random.seed(self.config.seed_config["fixed_seed"])
            
            self._base_network_cached = Network(
                num_nodes=self.config.network_config["num_nodes"],
                network_config=self.config.network_config
            )
        return self._base_network_cached
    
    def get_experiment_network(self) -> Network:
        """获取实验网络（使用缓存+深拷贝，保证一致性）"""
        base = self._build_base_network()
        return copy.deepcopy(base)
    
    def create_fresh_network(self) -> Network:
        """创建全新的网络（不使用缓存）"""
        if self.config.seed_config["fixed_network"]:
            random.seed(self.config.seed_config["fixed_seed"])
            np.random.seed(self.config.seed_config["fixed_seed"])
        
        return Network(
            num_nodes=self.config.network_config["num_nodes"],
            network_config=self.config.network_config
        )


class SimulationRunner:
    """仿真运行器 - 统一管理单次和并行仿真"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.network_manager = NetworkManager(config)
    
    def create_scheduler(self) -> LyapunovScheduler:
        """创建调度器"""
        return LyapunovScheduler(
            V=self.config.scheduler_config["V"],
            K=self.config.scheduler_config["K"],
            max_hops=self.config.scheduler_config["max_hops"]
        )
    
    def run_single_simulation(self, 
                            use_cached_network: bool = True,
                            output_dir: Optional[str] = None,
                            run_id: Optional[int] = None) -> Dict[str, Any]:
        """运行单次仿真"""
        try:
            # 创建网络
            if use_cached_network:
                network = self.network_manager.get_experiment_network()
            else:
                network = self.network_manager.create_fresh_network()
            
            # 创建调度器
            scheduler = self.create_scheduler()
            
            # 创建仿真
            simulation = EnergySimulation(
                network=network,
                time_steps=self.config.simulation_config["time_steps"],
                scheduler=scheduler,
                decision_period=self.config.simulation_config["decision_period"]
            )
            
            # 运行仿真
            simulation.simulate()
            
            # 收集结果
            result = {
                "status": "success",
                "simulation": simulation,
                "network": network,
                "scheduler": scheduler,
                "run_id": run_id,
                "output_dir": output_dir or self.config.output_config["output_dir"]
            }
            
            return result
            
        except Exception as e:
            return {
                "status": "failed",
                "error": str(e),
                "run_id": run_id
            }
    
    def run_parallel_simulations(self, 
                               num_runs: int = 10,
                               max_workers: int = 4,
                               use_same_seed: bool = True,
                               weight_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """运行并行仿真"""
        from .parallel_simulator import ParallelSimulator
        
        parallel_simulator = ParallelSimulator(self.config)
        return parallel_simulator.run_parallel_simulations(
            num_runs=num_runs,
            max_workers=max_workers,
            use_same_seed=use_same_seed,
            weight_config=weight_config
        )


def create_default_config() -> ExperimentConfig:
    """创建默认配置"""
    return ExperimentConfig()


def create_simulation_runner(config: Optional[ExperimentConfig] = None) -> SimulationRunner:
    """创建仿真运行器"""
    if config is None:
        config = create_default_config()
    return SimulationRunner(config)
