"""
并行仿真模块 - 从 parallel_main.py 提取的并行仿真功能
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import json
from datetime import datetime
import random
import numpy as np
from typing import Dict, Any, List, Optional

from wsn_et.core.network import Network
from wsn_et.scheduling.schedulers import LyapunovScheduler
from wsn_et.sim.energy_simulation import EnergySimulation
from .experiment_manager import ExperimentConfig


class ParallelSimulator:
    """并行仿真器"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
    
    def run_single_simulation(self, 
                            run_id: int, 
                            network_config: Dict[str, Any], 
                            scheduler_config: Dict[str, Any], 
                            output_base_dir: str = "data", 
                            use_same_seed: bool = True, 
                            reward_weights: Optional[Dict] = None) -> Dict[str, Any]:
        """运行单次仿真（用于并行执行）"""
        try:
            output_dir = os.path.join(output_base_dir, f"run_{run_id}")
            os.makedirs(output_dir, exist_ok=True)
            
            if use_same_seed:
                base_seed = self.config.seed_config["fixed_seed"]
                random.seed(base_seed)
                np.random.seed(base_seed)
                seed_info = f"固定种子 {base_seed}"
            else:
                seed = self.config.seed_config["fixed_seed"] + run_id
                random.seed(seed)
                np.random.seed(seed)
                seed_info = f"种子 {seed}"
            
            print(f"运行 {run_id}: {seed_info}")
            
            network = Network(num_nodes=25, network_config=network_config)
            scheduler = LyapunovScheduler(V=0.5, K=3, max_hops=3)
            simulation = EnergySimulation(network, time_steps=10080, scheduler=scheduler)
            
            if reward_weights:
                # 注意：这里需要适配新的 DataCollector 架构
                # 暂时跳过权重设置，因为原来的 w_b, w_d, w_l 属性已被删除
                print(f"  权重设置功能需要适配新的 DataCollector 架构")
            
            start_time = time.time()
            simulation.simulate()
            end_time = time.time()
            
            final_energies = [node.current_energy for node in network.nodes]
            avg_energy = np.mean(final_energies)
            std_energy = np.std(final_energies)
            min_energy = np.min(final_energies)
            
            # 适配新的 DataCollector 架构
            data_collector = simulation.get_data_collector()
            avg_variance = np.mean(data_collector.energy_standards) if data_collector.energy_standards else 0
            total_sent_energy = sum(sum(node.transferred_history) for node in network.nodes)
            total_received_energy = sum(sum(node.received_history) for node in network.nodes)
            total_loss_energy = total_sent_energy - total_received_energy
            energy_transfer_efficiency = (total_received_energy / total_sent_energy * 100 if total_sent_energy > 0 else 0)
            
            return {
                "run_id": run_id,
                "status": "success",
                "duration": end_time - start_time,
                "output_dir": output_dir,
                "final_energies": final_energies,
                "avg_energy": avg_energy,
                "std_energy": std_energy,
                "min_energy": min_energy,
                "seed_info": seed_info,
                "reward_weights": reward_weights,
                "num_nodes": len(network.nodes),
                "time_steps": simulation.time_steps,
                "avg_variance": avg_variance,
                "total_sent_energy": total_sent_energy,
                "total_received_energy": total_received_energy,
                "total_loss_energy": total_loss_energy,
                "energy_transfer_efficiency": energy_transfer_efficiency
            }
        except Exception as e:
            return {"run_id": run_id, "status": "failed", "error": str(e)}
    
    def generate_weight_combinations(self, weight_config: Dict[str, Any], num_runs: int) -> List[Dict[str, float]]:
        """生成权重组合"""
        combinations = []
        mode = weight_config.get("mode", "single")
        
        if mode == "single":
            vary_param = weight_config.get("vary_param", "w_l")
            start = weight_config.get("start", 0.5)
            step = weight_config.get("step", 0.2)
            
            for i in range(num_runs):
                current_value = start + i * step
                weights = {
                    "w_b": weight_config.get("w_b_fixed", 0.7),
                    "w_d": weight_config.get("w_d_fixed", 0.5),
                    "w_l": weight_config.get("w_l_fixed", 1.0)
                }
                weights[vary_param] = current_value
                combinations.append(weights)
        elif mode == "double":
            vary_params = weight_config.get("vary_params", ["w_b", "w_d"])
            param1, param2 = vary_params[0], vary_params[1]
            param1_start = weight_config.get(f"{param1}_start", 0.1)
            param1_step = weight_config.get(f"{param1}_step", 0.1)
            param1_count = weight_config.get(f"{param1}_count", 5)
            param2_start = weight_config.get(f"{param2}_start", 0.5)
            param2_step = weight_config.get(f"{param2}_step", 0.2)
            param2_count = weight_config.get(f"{param2}_count", 3)
            
            for i in range(param1_count):
                for j in range(param2_count):
                    weights = {
                        "w_b": weight_config.get("w_b_fixed", 0.8),
                        "w_d": weight_config.get("w_d_fixed", 0.8),
                        "w_l": weight_config.get("w_l_fixed", 1.5)
                    }
                    weights[param1] = param1_start + i * param1_step
                    weights[param2] = param2_start + j * param2_step
                    combinations.append(weights)
        elif mode == "triple":
            w_b_start = weight_config.get("w_b_start", 0.1)
            w_b_step = weight_config.get("w_b_step", 0.1)
            w_b_count = weight_config.get("w_b_count", 3)
            w_d_start = weight_config.get("w_d_start", 0.5)
            w_d_step = weight_config.get("w_d_step", 0.2)
            w_d_count = weight_config.get("w_d_count", 3)
            w_l_start = weight_config.get("w_l_start", 1.0)
            w_l_step = weight_config.get("w_l_step", 0.5)
            w_l_count = weight_config.get("w_l_count", 3)
            
            for i in range(w_b_count):
                for j in range(w_d_count):
                    for k in range(w_l_count):
                        weights = {
                            "w_b": w_b_start + i * w_b_step,
                            "w_d": w_d_start + j * w_d_step,
                            "w_l": w_l_start + k * w_l_step
                        }
                        combinations.append(weights)
        else:
            raise ValueError(f"不支持的权重配置模式: {mode}")
        
        return combinations
    
    def run_parallel_simulations(self, 
                                num_runs: int = 10, 
                                max_workers: int = 4, 
                                use_same_seed: bool = True, 
                                weight_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """运行并行仿真"""
        print(f"开始并行运行 {num_runs} 次仿真...")
        print(f"使用 {max_workers} 个并行进程")
        print(f"种子模式: {'固定种子（对比实验）' if use_same_seed else '不同种子（独立实验）'}")
        
        if weight_config is None:
            weight_config = {
                "mode": "single",
                "vary_param": "w_l",
                "start": 0.5,
                "step": 0.2,
                "w_b_fixed": 0.7,
                "w_d_fixed": 0.5,
                "w_l_fixed": 1.0
            }
        
        weight_combinations = self.generate_weight_combinations(weight_config, num_runs)
        
        start_time = time.time()
        results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for run_id in range(len(weight_combinations)):
                current_weights = weight_combinations[run_id]
                print(f"准备运行 {run_id}: 权重 w_b={current_weights['w_b']:.1f}, w_d={current_weights['w_d']:.1f}, w_l={current_weights['w_l']:.1f}")
                future = executor.submit(
                    self.run_single_simulation,
                    run_id,
                    self.config.network_config,
                    self.config.scheduler_config,
                    self.config.output_config["base_data_dir"],
                    use_same_seed,
                    current_weights
                )
                futures.append(future)
            
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                
                if result["status"] == "success":
                    weights_info = result["reward_weights"]
                    print(f"✅ 运行 {result['run_id']} 完成 (耗时: {result['duration']:.2f}s, 权重: w_b={weights_info['w_b']:.1f}, w_d={weights_info['w_d']:.1f}, w_l={weights_info['w_l']:.1f})")
                else:
                    print(f"❌ 运行 {result['run_id']} 失败: {result['error']}")
        
        end_time = time.time()
        successful_runs = [r for r in results if r["status"] == "success"]
        failed_runs = [r for r in results if r["status"] == "failed"]
        
        print("-" * 60)
        print(f"=== 运行完成 ===")
        print(f"总耗时: {end_time - start_time:.2f}s")
        print(f"成功: {len(successful_runs)}/{num_runs}")
        print(f"失败: {len(failed_runs)}/{num_runs}")
        
        return results
