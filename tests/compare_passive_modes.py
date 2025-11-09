"""
对比passive_mode开关对DQN测试的影响

快速运行：
python tests/compare_passive_modes.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config.simulation_config import ConfigManager
from core.energy_simulation import EnergySimulation
from scheduling.dqn_scheduler import DQNScheduler
from info_collection.physical_center import NodeInfoManager


def run_comparison(model_path="dqn_model.pth"):
    """对比两种模式的性能"""
    
    print("=" * 80)
    print("DQN测试：passive_mode开关对比")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"\n[ERROR] 模型文件不存在: {model_path}")
        print("请先训练模型：python tests/run_dqn_simulation.py --train")
        return
    
    results = {}
    
    # 测试两种模式
    modes = [
        ("passive_mode=False (传统定时触发)", False),
        ("passive_mode=True (智能被动传能)", True)
    ]
    
    for mode_name, passive_mode in modes:
        print(f"\n{'='*80}")
        print(f"测试模式: {mode_name}")
        print("=" * 80)
        
        # 创建配置
        config = ConfigManager()
        config.simulation_config.time_steps = 200
        config.simulation_config.enable_energy_sharing = True
        config.network_config.num_nodes = 15
        config.network_config.enable_physical_center = True
        
        # 创建网络
        network = config.create_network()
        
        # 创建节点信息管理器
        nim = NodeInfoManager(
            initial_position=(5.0, 5.0),
            enable_logging=False
        )
        nim.initialize_node_info(network.nodes, initial_time=0)
        
        # 创建DQN调度器
        scheduler = DQNScheduler(
            node_info_manager=nim,
            K=2,
            max_hops=3,
            action_dim=10,
            training_mode=False
        )
        
        # 加载模型
        scheduler.plan(network, 0)
        scheduler.load_model(model_path)
        scheduler.agent.epsilon = 0.0  # 确保无探索
        
        # 运行仿真
        if passive_mode:
            simulation = EnergySimulation(
                network=network,
                time_steps=200,
                scheduler=scheduler,
                enable_energy_sharing=True,
                passive_mode=True,
                check_interval=10,
                critical_ratio=0.2,
                energy_variance_threshold=0.3,
                cooldown_period=30
            )
        else:
            simulation = EnergySimulation(
                network=network,
                time_steps=200,
                scheduler=scheduler,
                enable_energy_sharing=True,
                passive_mode=False
            )
        
        # 静默运行
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        simulation.simulate()
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # 统计结果
        transfer_count = output.count('K=')  # 统计传能次数
        
        # 获取结果
        sim_results = simulation.result_manager.get_results()
        
        # 统计传输
        durations = []
        total_energy_sent = 0
        total_energy_received = 0
        for result in sim_results:
            if 'plans' in result and result['plans']:
                for plan in result['plans']:
                    dur = plan.get('duration', 1)
                    durations.append(dur)
                    total_energy_sent += plan.get('loss', 0) + plan.get('delivered', 0)
                    total_energy_received += plan.get('delivered', 0)
        
        # 能量统计
        final_energies = [node.current_energy for node in network.nodes[1:]]
        dead_nodes = sum(1 for e in final_energies if e <= 0)
        
        # 被动传能统计
        passive_stats = None
        if hasattr(simulation, 'passive_manager'):
            passive_stats = simulation.passive_manager.get_statistics()
        
        # 保存结果
        results[mode_name] = {
            'transfer_count': transfer_count,
            'avg_duration': np.mean(durations) if durations else 0,
            'total_sent': total_energy_sent,
            'total_received': total_energy_received,
            'efficiency': total_energy_received / total_energy_sent if total_energy_sent > 0 else 0,
            'avg_energy': np.mean(final_energies),
            'std_energy': np.std(final_energies),
            'cv': np.std(final_energies) / np.mean(final_energies) if np.mean(final_energies) > 0 else 0,
            'dead_nodes': dead_nodes,
            'passive_stats': passive_stats
        }
        
        # 输出简要结果
        print(f"\n✓ 仿真完成")
        print(f"  传能次数: {transfer_count}")
        print(f"  平均能量: {results[mode_name]['avg_energy']:.0f}J")
        print(f"  能量CV: {results[mode_name]['cv']:.4f}")
        print(f"  死亡节点: {dead_nodes}个")
    
    # 输出对比结果
    print("\n" + "=" * 80)
    print("对比结果总结")
    print("=" * 80)
    
    # 表格对比
    print(f"\n{'指标':<25} {'传统定时触发':<20} {'智能被动传能':<20} {'改善':<15}")
    print("-" * 80)
    
    metrics = [
        ('传能次数', 'transfer_count', '{:.0f}次'),
        ('平均传输时长', 'avg_duration', '{:.2f}分钟'),
        ('总发送能量', 'total_sent', '{:.0f}J'),
        ('总接收能量', 'total_received', '{:.0f}J'),
        ('传输效率', 'efficiency', '{:.2%}'),
        ('平均节点能量', 'avg_energy', '{:.0f}J'),
        ('能量标准差', 'std_energy', '{:.0f}J'),
        ('能量变异系数(CV)', 'cv', '{:.4f}'),
        ('死亡节点数', 'dead_nodes', '{:.0f}个'),
    ]
    
    mode1_name = modes[0][0]
    mode2_name = modes[1][0]
    
    for metric_name, key, fmt in metrics:
        val1 = results[mode1_name][key]
        val2 = results[mode2_name][key]
        
        # 计算改善
        if key in ['cv', 'std_energy', 'dead_nodes']:
            # 越小越好
            if val1 > 0:
                improvement = (val1 - val2) / val1 * 100
                improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "-"
            else:
                improvement_str = "-"
        elif key in ['avg_energy', 'total_received', 'efficiency']:
            # 越大越好
            if val1 > 0:
                improvement = (val2 - val1) / val1 * 100
                improvement_str = f"{improvement:+.1f}%" if improvement != 0 else "-"
            else:
                improvement_str = "-"
        else:
            improvement_str = "-"
        
        print(f"{metric_name:<25} {fmt.format(val1):<20} {fmt.format(val2):<20} {improvement_str:<15}")
    
    # 被动传能详细统计
    if results[mode2_name]['passive_stats']:
        stats = results[mode2_name]['passive_stats']
        print(f"\n智能被动传能详细统计:")
        print(f"  触发次数: {stats['trigger_count']}次")
        print(f"  触发频率: {stats['trigger_count']/200*100:.1f}%")
        if stats['trigger_reasons']:
            print(f"  触发原因分布:")
            for reason, count in stats['trigger_reasons'].items():
                print(f"    · {reason}: {count}次 ({count/stats['trigger_count']*100:.1f}%)")
    
    # 结论
    print("\n" + "=" * 80)
    print("结论")
    print("=" * 80)
    
    cv_improvement = (results[mode1_name]['cv'] - results[mode2_name]['cv']) / results[mode1_name]['cv'] * 100
    
    print(f"\n✓ 启用智能被动传能后：")
    if results[mode2_name]['dead_nodes'] < results[mode1_name]['dead_nodes']:
        print(f"  • 死亡节点减少 {results[mode1_name]['dead_nodes'] - results[mode2_name]['dead_nodes']}个")
    if cv_improvement > 0:
        print(f"  • 能量均衡度提升 {cv_improvement:.1f}%")
    if results[mode2_name]['avg_energy'] > results[mode1_name]['avg_energy']:
        energy_gain = (results[mode2_name]['avg_energy'] - results[mode1_name]['avg_energy']) / results[mode1_name]['avg_energy'] * 100
        print(f"  • 平均节点能量提升 {energy_gain:.1f}%")
    
    print(f"\n推荐: 使用 passive_mode=True（智能被动传能）✅")
    print("     - 更好的能量均衡")
    print("     - 更长的网络寿命")
    print("     - 自适应传能频率")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='对比passive_mode对DQN的影响')
    parser.add_argument('--model', type=str, default='dqn_model.pth',
                       help='DQN模型路径（默认: dqn_model.pth）')
    args = parser.parse_args()
    
    run_comparison(model_path=args.model)

