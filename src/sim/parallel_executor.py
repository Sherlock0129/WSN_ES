#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行仿真执行器
提供并行运行多次仿真的功能，支持权重扫描和种子对比实验
"""

import os
import sys
import time
import json
import random
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.simulation_config import ConfigManager
from utils.error_handling import logger, handle_exceptions


class ParallelSimulationExecutor:
    """并行仿真执行器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
        self.parallel_config = config_manager.parallel_config
        
    def run_parallel_simulations(self) -> List[Dict[str, Any]]:
        """执行并行仿真"""
        if not self.parallel_config.enabled:
            raise ValueError("并行模式未启用")
            
        logger.info(f"开始并行仿真: {self.parallel_config.num_runs} 次运行")
        logger.info(f"使用 {self.parallel_config.max_workers} 个并行进程")
        logger.info(f"种子模式: {'固定种子（对比实验）' if self.parallel_config.use_same_seed else '不同种子（独立实验）'}")
        
        if self.parallel_config.enable_weight_scan:
            logger.info(f"启用权重扫描: w_b从{self.parallel_config.w_b_start}开始，步长{self.parallel_config.w_b_step}")
        
        # 创建输出目录
        os.makedirs(self.parallel_config.output_base_dir, exist_ok=True)
        
        start_time = time.time()
        results = []
        
        # 使用进程池并行执行
        with ProcessPoolExecutor(max_workers=self.parallel_config.max_workers) as executor:
            # 提交所有任务
            futures = []
            for run_id in range(self.parallel_config.num_runs):
                # 计算当前运行的权重参数
                custom_weights = self._calculate_weights(run_id)
                
                future = executor.submit(
                    self._run_single_simulation,
                    run_id,
                    custom_weights
                )
                futures.append(future)
            
            # 收集结果
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    weights_info = result.get("reward_weights", {})
                    logger.info(f"✅ 运行 {result['run_id']} 完成 (耗时: {result['duration']:.2f}s)")
                    if weights_info:
                        logger.info(f"   权重: w_b={weights_info.get('w_b', 0):.1f}, "
                                   f"w_d={weights_info.get('w_d', 0):.1f}, "
                                   f"w_l={weights_info.get('w_l', 0):.1f}")
                else:
                    logger.error(f"❌ 运行 {result['run_id']} 失败: {result['error']}")
        
        end_time = time.time()
        
        # 统计结果
        successful_runs = [r for r in results if r["status"] == "success"]
        failed_runs = [r for r in results if r["status"] == "failed"]
        
        logger.info(f"=== 并行仿真完成 ===")
        logger.info(f"总耗时: {end_time - start_time:.2f}s")
        logger.info(f"成功: {len(successful_runs)}/{self.parallel_config.num_runs}")
        logger.info(f"失败: {len(failed_runs)}/{self.parallel_config.num_runs}")
        
        # 生成汇总报告
        if self.parallel_config.generate_summary and successful_runs:
            self._generate_summary_report(successful_runs)
        
        return results
    
    def _calculate_weights(self, run_id: int) -> Optional[Dict[str, float]]:
        """计算当前运行的权重参数"""
        if not self.parallel_config.enable_weight_scan:
            return None
            
        return {
            "w_b": self.parallel_config.w_b_start + run_id * self.parallel_config.w_b_step,
            "w_d": self.parallel_config.w_d_fixed,
            "w_l": self.parallel_config.w_l_fixed
        }
    
    def _run_single_simulation(self, run_id: int, custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """单次仿真执行（进程安全）"""
        try:
            # 创建独立的输出目录
            output_dir = os.path.join(self.parallel_config.output_base_dir, f"run_{run_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置种子
            if self.parallel_config.use_same_seed:
                # 对比实验：所有运行使用相同种子
                seed = self.parallel_config.base_seed
                random.seed(seed)
                np.random.seed(seed)
            else:
                # 独立实验：每次运行使用不同种子
                seed = self.parallel_config.base_seed + run_id
                random.seed(seed)
                np.random.seed(seed)
            
            # 创建独立的配置管理器实例
            config_manager = ConfigManager()
            
            # 创建网络
            network = config_manager.create_network()
            
            # 创建ADCR链路层
            network.adcr_link = config_manager.create_adcr_link_layer(network)
            
            # 创建调度器
            scheduler = self._create_scheduler(config_manager)
            
            # 创建仿真
            simulation = config_manager.create_energy_simulation(network, scheduler)
            
            # 应用自定义权重
            if custom_weights:
                simulation.w_b = custom_weights.get("w_b", simulation.w_b)
                simulation.w_d = custom_weights.get("w_d", simulation.w_d)
                simulation.w_l = custom_weights.get("w_l", simulation.w_l)
            
            # 运行仿真
            start_time = time.time()
            simulation.simulate()
            end_time = time.time()
            
            # 计算统计信息
            final_energies = [node.current_energy for node in network.nodes]
            avg_energy = np.mean(final_energies)
            std_energy = np.std(final_energies)
            
            # 计算能量传输统计
            total_sent = sum(node.transferred_energy for node in network.nodes)
            total_received = sum(node.received_energy for node in network.nodes)
            total_loss = total_sent - total_received
            efficiency = (total_received / total_sent * 100) if total_sent > 0 else 0
            
            # 保存单次运行结果
            if self.parallel_config.save_individual_results:
                self._save_individual_result(simulation, output_dir, run_id, custom_weights)
            
            return {
                "status": "success",
                "run_id": run_id,
                "seed": seed,
                "duration": end_time - start_time,
                "reward_weights": custom_weights,
                "stats": {
                    "avg_energy": avg_energy,
                    "std_energy": std_energy,
                    "total_sent": total_sent,
                    "total_received": total_received,
                    "total_loss": total_loss,
                    "efficiency": efficiency
                }
            }
            
        except Exception as e:
            return {
                "status": "failed",
                "run_id": run_id,
                "error": str(e),
                "duration": 0
            }
    
    def _create_scheduler(self, config_manager: ConfigManager):
        """创建调度器"""
        scheduler_type = config_manager.scheduler_config.scheduler_type
        
        if scheduler_type == "LyapunovScheduler":
            from scheduling.schedulers import LyapunovScheduler
            return LyapunovScheduler(
                V=config_manager.scheduler_config.lyapunov_v,
                K=config_manager.scheduler_config.lyapunov_k,
                max_hops=config_manager.network_config.max_hops
            )
        elif scheduler_type == "ClusterScheduler":
            from scheduling.schedulers import ClusterScheduler
            return ClusterScheduler(
                round_period=config_manager.scheduler_config.cluster_round_period,
                K=config_manager.scheduler_config.lyapunov_k,
                max_hops=config_manager.network_config.max_hops,
                p_ch=config_manager.scheduler_config.cluster_p_ch
            )
        elif scheduler_type == "PredictionScheduler":
            from scheduling.schedulers import PredictionScheduler
            return PredictionScheduler(
                alpha=config_manager.scheduler_config.prediction_alpha,
                horizon_min=config_manager.scheduler_config.prediction_horizon,
                K=config_manager.scheduler_config.lyapunov_k,
                max_hops=config_manager.network_config.max_hops
            )
        elif scheduler_type == "PowerControlScheduler":
            from scheduling.schedulers import PowerControlScheduler
            return PowerControlScheduler(
                target_eta=config_manager.scheduler_config.power_target_eta,
                K=config_manager.scheduler_config.lyapunov_k,
                max_hops=config_manager.network_config.max_hops
            )
        else:
            from scheduling.schedulers import BaselineHeuristic
            return BaselineHeuristic(
                K=config_manager.scheduler_config.lyapunov_k,
                max_hops=config_manager.network_config.max_hops
            )
    
    def _save_individual_result(self, simulation, output_dir: str, run_id: int, custom_weights: Optional[Dict[str, float]] = None):
        """保存单次运行结果"""
        try:
            # 保存K值历史
            if hasattr(simulation, 'K_history') and simulation.K_history:
                k_history_file = os.path.join(output_dir, f"K_value_history_run_{run_id}.csv")
                with open(k_history_file, 'w') as f:
                    f.write("time_step,K_value\n")
                    for t, k in simulation.K_history:
                        f.write(f"{t},{k}\n")
            
            # 保存权重信息
            if custom_weights:
                weights_file = os.path.join(output_dir, f"weights_run_{run_id}.json")
                with open(weights_file, 'w') as f:
                    json.dump(custom_weights, f, indent=2)
            
            # 保存仿真结果
            results_file = os.path.join(output_dir, f"simulation_results_run_{run_id}.csv")
            simulation.save_results(results_file)
            
        except Exception as e:
            logger.warning(f"保存运行 {run_id} 结果时出错: {e}")
    
    def _generate_summary_report(self, successful_runs: List[Dict[str, Any]]):
        """生成汇总报告"""
        try:
            summary_file = os.path.join(self.parallel_config.output_base_dir, "parallel_summary.json")
            
            # 计算汇总统计
            durations = [r["duration"] for r in successful_runs]
            efficiencies = [r["stats"]["efficiency"] for r in successful_runs]
            avg_energies = [r["stats"]["avg_energy"] for r in successful_runs]
            
            summary = {
                "timestamp": datetime.now().isoformat(),
                "total_runs": len(successful_runs),
                "parallel_config": {
                    "num_runs": self.parallel_config.num_runs,
                    "max_workers": self.parallel_config.max_workers,
                    "use_same_seed": self.parallel_config.use_same_seed,
                    "enable_weight_scan": self.parallel_config.enable_weight_scan
                },
                "statistics": {
                    "duration": {
                        "mean": np.mean(durations),
                        "std": np.std(durations),
                        "min": np.min(durations),
                        "max": np.max(durations)
                    },
                    "efficiency": {
                        "mean": np.mean(efficiencies),
                        "std": np.std(efficiencies),
                        "min": np.min(efficiencies),
                        "max": np.max(efficiencies)
                    },
                    "avg_energy": {
                        "mean": np.mean(avg_energies),
                        "std": np.std(avg_energies),
                        "min": np.min(avg_energies),
                        "max": np.max(avg_energies)
                    }
                },
                "runs": successful_runs
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"汇总报告已保存到: {summary_file}")
            
        except Exception as e:
            logger.error(f"生成汇总报告时出错: {e}")


if __name__ == "__main__":
    # 测试代码
    config_manager = ConfigManager()
    config_manager.parallel_config.enabled = True
    config_manager.parallel_config.num_runs = 3
    config_manager.parallel_config.max_workers = 2
    
    executor = ParallelSimulationExecutor(config_manager)
    results = executor.run_parallel_simulations()
    print(f"完成 {len(results)} 次运行")
