"""
测试反馈机制的简短脚本
"""
import sys
sys.path.insert(0, 'src')

import random
import numpy as np

from src.core.energy_simulation import EnergySimulation
from src.core.network import Network
from src.scheduling.schedulers import LyapunovScheduler

# 创建网络配置
network_config = {
    "num_nodes": 10,  # 减少节点数以加快测试
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

# 固定随机种子
FIXED_SEED = 130
random.seed(FIXED_SEED)
np.random.seed(FIXED_SEED)

# 创建网络
print("创建网络...")
network = Network(num_nodes=network_config["num_nodes"], network_config=network_config)

# 创建调度器
print("创建调度器...")
sched = LyapunovScheduler(V=0.5, K=3, max_hops=network_config["max_hops"])

# 创建仿真实例（短时间测试：只运行1440分钟=1天）
print("创建仿真实例...")
time_steps = 1440
simulation = EnergySimulation(
    network, 
    time_steps, 
    scheduler=sched,
    enable_energy_sharing=True,
    enable_k_adaptation=False,  # 关闭K自适应以简化测试
    fixed_k=3,
    passive_mode=False  # 关闭被动模式，使用主动传能（每60分钟一次）
)

# 运行仿真
print(f"开始仿真（{time_steps}个时间步）...")
print("=" * 80)
simulation.simulate()
print("=" * 80)
print("仿真完成！")

# 打印反馈分数统计
if simulation.stats.feedback_scores:
    print("\n" + "=" * 80)
    print("反馈机制测试结果：")
    print("=" * 80)
    
    total_scores = [record['total_score'] for record in simulation.stats.feedback_scores]
    positive_count = sum(1 for score in total_scores if score > 1)
    negative_count = sum(1 for score in total_scores if score < -1)
    neutral_count = len(total_scores) - positive_count - negative_count
    
    print(f"总调度次数: {len(total_scores)}")
    print(f"平均反馈分数: {np.mean(total_scores):.2f}")
    print(f"最高分数: {np.max(total_scores):.2f}")
    print(f"最低分数: {np.min(total_scores):.2f}")
    print(f"标准差: {np.std(total_scores):.2f}")
    print(f"\n影响分布:")
    print(f"  正相关（改善）: {positive_count} 次 ({positive_count/len(total_scores)*100:.1f}%)")
    print(f"  中性（无明显影响）: {neutral_count} 次 ({neutral_count/len(total_scores)*100:.1f}%)")
    print(f"  负相关（恶化）: {negative_count} 次 ({negative_count/len(total_scores)*100:.1f}%)")
    
    print("\n前5次调度的反馈详情:")
    for i, record in enumerate(simulation.stats.feedback_scores[:5]):
        print(f"\n第{i+1}次调度 (t={record['time_step']}):")
        print(f"  总分: {record['total_score']:.2f} - {record['impact']}")
        print(f"  能量均衡: {record['balance_score']:.2f}")
        print(f"  网络存活: {record['survival_score']:.2f}")
        print(f"  传输效率: {record['efficiency_score']:.2f}")
        print(f"  能量水平: {record['energy_score']:.2f}")
    
    print("\n" + "=" * 80)
    print("反馈机制测试成功！✓")
    print("=" * 80)
else:
    print("\n警告：没有记录到反馈分数数据")


