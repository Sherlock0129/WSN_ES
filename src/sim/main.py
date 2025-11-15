import os
import random
import numpy as np
import copy

from core.energy_simulation import EnergySimulation
from core.network import Network
from scheduling import schedulers
from src.config.simulation_config import ConfigManager
from src.core.network import Network
from src.core.energy_simulation import EnergySimulation
from src.scheduling.schedulers import LyapunovScheduler
from src.viz.plotter import plot_node_distribution, plot_energy_over_time
from viz import plotter

# 创建一个网络实例
network_config = {
    "num_nodes": 25,
    "low_threshold": 0.1,
    "high_threshold": 0.9,
    "node_initial_energy": 40000,
    "max_hops": 3,
    "random_seed": 129,   # ← 固定随机数种子
    "distribution_mode": "random",  # 分布模式：uniform(均匀) 或 random(随机)
    "network_area": {     # 网络区域设置（用于随机分布）
        "width": 10.0,
        "height": 10.0
    },
    "min_distance": 0.5   # 节点间最小距离（避免重叠）
}
# 固定网络拓扑开关与种子
FIXED_NETWORK = True
FIXED_SEED = 130


_base_network_cached = None

def _build_base_network():
    global _base_network_cached
    if _base_network_cached is None:
        if FIXED_NETWORK:
            random.seed(FIXED_SEED)
            np.random.seed(FIXED_SEED)
        _base_network_cached = Network(num_nodes=network_config["num_nodes"], network_config=network_config)
    return _base_network_cached

def get_experiment_network():
    base = _build_base_network()
    return copy.deepcopy(base)
time_steps = 10080
# 创建能量仿真实例
sched = schedulers.LyapunovScheduler(V=0.5, K=3,max_hops=network_config["max_hops"])
# sched = schedulers.ClusterScheduler(round_period=360, K=3,max_hops=network_config["max_hops"])
# sched = schedulers.PredictionScheduler(alpha=0.6, horizon_min=60, K=3)
# sched = schedulers.PowerControlScheduler(target_eta=0.25, K=3)
# sched = schedulers.BaselineHeuristic(K=3)


# proposed method
# sched = None
network = get_experiment_network()
simulation = EnergySimulation(network, time_steps, scheduler=sched)

# 保存仿真结果到 CSV 文件
results_file = 'data/results.csv'
output_dir = 'adcr'
# 运行能量仿真并绘制图像
simulation.simulate()

# 保存详细的计划日志
from src.utils.logger import get_detailed_plan_logger
plan_logger = get_detailed_plan_logger(simulation.session_dir) # 使用仿真会话目录
plan_logger.save_simulation_plans(simulation)

# 记录整个模拟的所有plans到文件
def save_all_plans_to_file(simulation, output_file="all_plans.txt"):
    """将整个模拟的所有plans保存到文件"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== 整个模拟的能量传输计划记录 ===\n\n")

        # 按时间步排序
        sorted_times = sorted(simulation.plans_by_time.keys())

        for t in sorted_times:
            time_data = simulation.plans_by_time[t]
            if not time_data:  # 如果没有数据，跳过
                continue

            # 处理新的数据结构
            if isinstance(time_data, dict) and "plans" in time_data:
                plans = time_data["plans"]
                candidates = time_data.get("candidates", [])
                node_energies = time_data.get("node_energies", {})
            else:
                # 兼容旧格式
                plans = time_data
                candidates = []
                node_energies = {}
                
            if not plans:  # 如果没有plans，跳过
                continue
                
            f.write(f"t={t}\n")
            
            # 打印节点能量信息
            if node_energies:
                f.write("  [NODE_ENERGIES]\n")
                # 按节点ID排序打印
                for node_id in sorted(node_energies.keys()):
                    energy = node_energies[node_id]
                    f.write(f"    Node {node_id}: {energy:.2f}J\n")

            # 打印选中的plans
            for i, plan in enumerate(plans):
                donor = plan.get("donor")
                receiver = plan.get("receiver")
                path = plan.get("path", [])
                distance = plan.get("distance", 0.0)
                delivered = plan.get("delivered", 0.0)
                loss = plan.get("loss", 0.0)

                d_id = getattr(donor, "node_id", None)
                r_id = getattr(receiver, "node_id", None)
                path_ids = [str(getattr(n, "node_id", n)) for n in path]

                line = f"  [SELECTED] d={d_id}, r={r_id}, path={'->'.join(path_ids)}, dist={distance:.2f}, delivered={delivered:.2f}, loss={loss:.2f}\n"
                f.write(line)

            # 打印候选信息
            if candidates:
                f.write("  [CANDIDATES]\n")
                for score, d, r, path, dist, delivered, loss in candidates:
                    d_id = getattr(d, "node_id", None)
                    r_id = getattr(r, "node_id", None)
                    path_ids = [str(getattr(n, "node_id", n)) for n in path]
                    line = f"    score={score:.2f}, d={d_id}, r={r_id}, path={'->'.join(path_ids)}, dist={dist:.2f}, delivered={delivered:.2f}, loss={loss:.2f}\n"
                    f.write(line)

            f.write("\n")  # 两个时间步之间空一行

        f.write("=== 记录结束 ===\n")

    print(f"已保存 {len(sorted_times)} 个时间步的plans到 {output_file}")

# 调用保存方法
save_all_plans_to_file(simulation)

network.adcr_link.plot_clusters_and_paths(output_dir=output_dir)

# 画特定时间能量传输路径图
t = 6420
time_data = simulation.plans_by_time.get(t, {})
# 处理新的数据结构
if isinstance(time_data, dict) and "plans" in time_data:
    plans = time_data["plans"]
else:
    plans = time_data if isinstance(time_data, list) else []

# 测试：输出plans内容到txt文件中
out_path = f"plans_t{t}.txt"
with open(out_path, "w", encoding="utf-8") as f:
    for plan in plans:
        donor = plan.get("donor")
        receiver = plan.get("receiver")
        path = plan.get("path", [])
        d_id = getattr(donor, "node_id", None)
        r_id = getattr(receiver, "node_id", None)
        path_ids = [str(getattr(n, "node_id", n)) for n in path]
        line = f"d={d_id}, r={r_id}, path={'->'.join(path_ids)}\n"
        f.write(line)
plotter.plot_energy_paths_at_time(network, plans, t)

network.adcr_link.plot_clusters_and_paths(output_dir=output_dir)


# 绘制所有图像：节点分布、能量分布、能量变化等
# 绘制节点分布图
plotter.plot_node_distribution(network.nodes)

# 绘制能量分布图
# plotter.plot_energy_distribution(network.nodes, time_steps)  # 假设时间步为 10

# 绘制能量随时间变化的图
plotter.plot_energy_over_time(network.nodes, simulation.results)

# 绘制能量分布直方图
# plotter.plot_energy_histogram(network.nodes, time_steps)  # 假设时间步为 10

# 绘制能量传输历史图
# plotter.plot_energy_transfer_history(network.nodes)

# 保存仿真结果到 CSV
# simulation.save_results(results_file)
