# adaptive_lyapunov_example.py
# 自适应Lyapunov调度器使用示例

"""
本示例展示如何使用AdaptiveLyapunovScheduler进行能量调度仿真

核心特点：
1. 继承Lyapunov的理论优势
2. 根据反馈自动调整V参数
3. 多维度自适应（均衡性、效率、存活率）
4. 实时打印调整过程
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from network.sensor_network import SensorNetwork
from core.energy_simulation import EnergySimulation
from scheduling.schedulers import AdaptiveLyapunovScheduler, LyapunovScheduler
from node_info.node_info_manager import NodeInfoManager

def run_comparison():
    """对比标准Lyapunov和自适应Lyapunov"""
    
    print("\n" + "="*80)
    print("对比实验：标准Lyapunov vs 自适应Lyapunov")
    print("="*80 + "\n")
    
    # 实验配置
    config = {
        'num_nodes': 20,
        'area_size': 100,
        'simulation_time': 500,  # 500分钟
        'initial_energy': 5000,
        'K': 2,
        'max_hops': 3,
        'V': 0.5
    }
    
    results = {}
    
    # 实验1：标准Lyapunov调度器
    print("【实验1】标准Lyapunov调度器（V固定=0.5）")
    print("-" * 80)
    
    network1 = SensorNetwork(num_nodes=config['num_nodes'], 
                             area_size=config['area_size'],
                             initial_energy=config['initial_energy'])
    
    nim1 = NodeInfoManager(network1.nodes)
    scheduler1 = LyapunovScheduler(nim1, V=config['V'], K=config['K'], 
                                   max_hops=config['max_hops'])
    
    sim1 = EnergySimulation(network1, scheduler=scheduler1, 
                           enable_energy_sharing=True)
    
    sim1.run(duration=config['simulation_time'], step_interval=1, 
             print_interval=50)
    
    results['standard'] = sim1.stats.get_summary()
    print(f"\n实验1完成 - 最终存活节点: {results['standard']['final_alive_nodes']}")
    
    # 实验2：自适应Lyapunov调度器
    print("\n" + "="*80)
    print("【实验2】自适应Lyapunov调度器（V动态调整）")
    print("-" * 80)
    
    network2 = SensorNetwork(num_nodes=config['num_nodes'], 
                             area_size=config['area_size'],
                             initial_energy=config['initial_energy'])
    
    nim2 = NodeInfoManager(network2.nodes)
    scheduler2 = AdaptiveLyapunovScheduler(
        nim2, 
        V=config['V'],           # 初始V
        K=config['K'], 
        max_hops=config['max_hops'],
        window_size=10,          # 记忆最近10次反馈
        V_min=0.1,               # V的下限
        V_max=2.0,               # V的上限
        adjust_rate=0.15,        # 15%的调整速率
        sensitivity=2.0          # 触发调整的阈值
    )
    
    sim2 = EnergySimulation(network2, scheduler=scheduler2, 
                           enable_energy_sharing=True)
    
    sim2.run(duration=config['simulation_time'], step_interval=1, 
             print_interval=50)
    
    results['adaptive'] = sim2.stats.get_summary()
    print(f"\n实验2完成 - 最终存活节点: {results['adaptive']['final_alive_nodes']}")
    
    # 打印自适应调整摘要
    scheduler2.print_adaptation_summary()
    
    # 对比结果
    print("\n" + "="*80)
    print("实验对比结果")
    print("="*80)
    
    metrics = [
        ('final_alive_nodes', '最终存活节点'),
        ('avg_energy', '平均能量'),
        ('energy_std', '能量标准差'),
        ('total_delivered', '总送达能量'),
        ('total_loss', '总损耗能量'),
        ('avg_efficiency', '平均效率')
    ]
    
    print(f"{'指标':<20} {'标准Lyapunov':<20} {'自适应Lyapunov':<20} {'改善':<15}")
    print("-" * 80)
    
    for key, name in metrics:
        std_val = results['standard'].get(key, 0)
        ada_val = results['adaptive'].get(key, 0)
        
        if std_val != 0:
            improvement = ((ada_val - std_val) / std_val) * 100
            improve_str = f"{improvement:+.1f}%"
        else:
            improve_str = "N/A"
        
        print(f"{name:<20} {std_val:<20.2f} {ada_val:<20.2f} {improve_str:<15}")
    
    print("="*80 + "\n")
    
    # 获取自适应统计
    adapt_stats = scheduler2.get_adaptation_stats()
    print(f"自适应调整详情:")
    print(f"  - 初始V: {adapt_stats['initial_V']:.3f}")
    print(f"  - 最终V: {adapt_stats['current_V']:.3f}")
    print(f"  - 调整次数: {adapt_stats['total_adjustments']}")
    print(f"  - 平均反馈分数: {adapt_stats['avg_feedback']:.2f}")
    print()
    
    return results


def run_single_adaptive():
    """运行单个自适应Lyapunov仿真（详细输出）"""
    
    print("\n" + "="*80)
    print("自适应Lyapunov调度器 - 详细运行示例")
    print("="*80 + "\n")
    
    # 创建网络
    network = SensorNetwork(num_nodes=15, area_size=80, initial_energy=4000)
    print(f"创建网络: {len(network.nodes)}个节点，区域{network.area_size}×{network.area_size}m²")
    
    # 创建节点信息管理器
    nim = NodeInfoManager(network.nodes)
    
    # 创建自适应调度器
    scheduler = AdaptiveLyapunovScheduler(
        nim,
        V=0.5,              # 初始V参数
        K=2,                # 每个receiver最多2个donor
        max_hops=3,         # 最多3跳
        window_size=10,     # 滑动窗口大小
        V_min=0.1,          # V最小值
        V_max=2.0,          # V最大值
        adjust_rate=0.1,    # 调整速率10%
        sensitivity=2.0     # 敏感度阈值
    )
    
    print(f"调度器配置:")
    print(f"  - 初始V: {scheduler.V}")
    print(f"  - V范围: [{scheduler.V_min}, {scheduler.V_max}]")
    print(f"  - 调整速率: {scheduler.adjust_rate*100:.0f}%")
    print(f"  - 敏感度阈值: ±{scheduler.sensitivity}")
    print()
    
    # 创建仿真
    sim = EnergySimulation(network, scheduler=scheduler, enable_energy_sharing=True)
    
    # 运行仿真
    print("开始仿真...")
    print("(查看V参数的自适应调整)\n")
    
    sim.run(duration=300, step_interval=1, print_interval=30)
    
    # 打印摘要
    scheduler.print_adaptation_summary()
    
    # 获取统计信息
    stats = sim.stats.get_summary()
    print("仿真统计:")
    print(f"  - 最终存活节点: {stats.get('final_alive_nodes', 0)}")
    print(f"  - 平均能量: {stats.get('avg_energy', 0):.2f}")
    print(f"  - 能量标准差: {stats.get('energy_std', 0):.2f}")
    print(f"  - 总送达能量: {stats.get('total_delivered', 0):.2f}")
    print(f"  - 平均效率: {stats.get('avg_efficiency', 0):.2%}")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='自适应Lyapunov调度器示例')
    parser.add_argument('--mode', choices=['single', 'compare'], default='single',
                       help='运行模式: single=单次详细运行, compare=对比实验')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_single_adaptive()
    elif args.mode == 'compare':
        run_comparison()

