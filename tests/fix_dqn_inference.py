"""
DQN推理模式修复工具

问题：训练后的DQN模型传能过于频繁，导致节点提前死亡
原因：
1. 加载模型时epsilon可能还很高，导致持续探索
2. training_mode未正确关闭
3. 没有使用被动模式控制传输频率

解决方案：
1. 强制设置epsilon=0（无探索）
2. 确保training_mode=False
3. 建议启用passive_mode
"""

import os
import sys
import torch

def fix_dqn_model(model_path, output_path=None):
    """
    修复DQN模型的epsilon值，强制设置为0（无探索）
    
    :param model_path: 原模型路径
    :param output_path: 输出路径（如果为None，会覆盖原文件）
    """
    if not os.path.exists(model_path):
        print(f"❌ 错误: 模型文件不存在: {model_path}")
        return False
    
    print("=" * 80)
    print("DQN推理模式修复工具")
    print("=" * 80)
    
    # 加载模型
    print(f"\n[1] 加载模型: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        print("  ✓ 模型加载成功")
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")
        return False
    
    # 检查epsilon
    old_epsilon = checkpoint.get('epsilon', 'N/A')
    print(f"\n[2] 当前epsilon值: {old_epsilon}")
    
    if old_epsilon == 'N/A':
        print("  ⚠️ 警告: 模型中没有epsilon字段")
    elif old_epsilon > 0.01:
        print(f"  ⚠️ 问题发现: epsilon={old_epsilon:.4f} 过高！")
        print(f"     这意味着{old_epsilon*100:.1f}%的时间会随机探索")
        print(f"     导致选择不合理的传输时长（1-10分钟随机）")
    else:
        print(f"  ✓ epsilon已经很低")
    
    # 修复epsilon
    print(f"\n[3] 修复epsilon")
    checkpoint['epsilon'] = 0.0  # 强制设置为0，完全不探索
    print(f"  ✓ 已设置 epsilon = 0.0 (无探索)")
    
    # 保存模型
    if output_path is None:
        output_path = model_path
        backup_path = model_path + ".backup"
        # 备份原文件
        import shutil
        shutil.copy(model_path, backup_path)
        print(f"\n[4] 备份原模型: {backup_path}")
    
    torch.save(checkpoint, output_path)
    print(f"\n[5] 保存修复后的模型: {output_path}")
    
    print("\n" + "=" * 80)
    print("✅ 修复完成！")
    print("=" * 80)
    
    print("\n使用说明:")
    print("1. 创建DQNScheduler时，确保设置: training_mode=False")
    print("2. 推荐启用被动模式: passive_mode=True")
    print("3. 现在epsilon=0，模型将始终选择最优动作，不再随机探索")
    
    return True


def check_dqn_config():
    """
    检查DQN配置是否正确
    """
    print("\n" + "=" * 80)
    print("DQN配置检查清单")
    print("=" * 80)
    
    print("\n请确认以下配置:")
    print("\n1. 创建DQNScheduler时:")
    print("   ✓ training_mode=False  # 必须设置为False")
    print("   ✓ epsilon_start=0.0    # 或者使用修复后的模型")
    print("   ✓ epsilon_end=0.0")
    
    print("\n2. 运行仿真时:")
    print("   ✓ passive_mode=True    # 推荐启用，控制传输频率")
    print("   ✓ check_interval=10    # 每10分钟检查一次，而非每分钟")
    
    print("\n3. 加载模型后:")
    print("   # 可选：手动强制设置epsilon为0")
    print("   scheduler.agent.epsilon = 0.0")
    
    print("\n示例代码:")
    print("-" * 80)
    print("""
# 创建DQN调度器（推理模式）
scheduler = DQNScheduler(
    node_info_manager=nim,
    K=2,
    max_hops=3,
    action_dim=10,
    training_mode=False,      # ← 关键！必须设置为False
    epsilon_start=0.0,        # ← 无探索
    epsilon_end=0.0
)

# 初始化并加载模型
scheduler.plan(network, 0)
scheduler.load_model("dqn_model.pth")

# 强制设置epsilon为0（双重保险）
scheduler.agent.epsilon = 0.0

# 运行仿真（推荐使用被动模式）
simulation = EnergySimulation(
    network=network,
    time_steps=10080,
    scheduler=scheduler,
    enable_energy_sharing=True,
    passive_mode=True,        # ← 推荐启用
    check_interval=10         # ← 控制传输频率
)
""")
    print("-" * 80)


def create_inference_example():
    """
    创建推理模式示例配置
    """
    example_code = """
\"\"\"
DQN推理模式使用示例（已修复传能过于频繁的问题）
\"\"\"

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.simulation_config import ConfigManager
from core.energy_simulation import EnergySimulation
from scheduling.dqn_scheduler import DQNScheduler
from info_collection.physical_center import NodeInfoManager


def run_dqn_inference(model_path="dqn_model.pth", time_steps=10080):
    \"\"\"
    使用训练好的DQN模型运行推理（已修复传能过频问题）
    
    修复要点：
    1. training_mode=False（关闭训练模式）
    2. epsilon=0.0（无探索）
    3. passive_mode=True（控制传输频率）
    \"\"\"
    print("=" * 80)
    print("DQN推理模式（已修复传能过频问题）")
    print("=" * 80)
    
    # 1. 创建配置
    config = ConfigManager()
    config.simulation_config.time_steps = time_steps
    config.simulation_config.enable_energy_sharing = True
    config.network_config.num_nodes = 15
    config.network_config.enable_physical_center = True
    
    # 2. 创建网络
    print("\\n[1] 创建网络...")
    network = config.create_network()
    print(f"  ✓ 节点数: {len(network.nodes)}")
    
    # 3. 创建节点信息管理器
    nim = NodeInfoManager(
        initial_position=(5.0, 5.0),
        enable_logging=False
    )
    nim.initialize_node_info(network.nodes, initial_time=0)
    
    # 4. 创建DQN调度器（推理模式）
    print("\\n[2] 创建DQN调度器（推理模式）...")
    scheduler = DQNScheduler(
        node_info_manager=nim,
        K=2,
        max_hops=3,
        action_dim=10,
        training_mode=False,      # ← 关键修复1: 关闭训练模式
        epsilon_start=0.0,        # ← 关键修复2: 无探索
        epsilon_end=0.0
    )
    print("  ✓ 训练模式: False")
    print("  ✓ 探索率: 0.0 (无探索)")
    
    # 5. 初始化并加载模型
    print("\\n[3] 加载训练好的模型...")
    scheduler.plan(network, 0)  # 初始化agent
    scheduler.load_model(model_path)
    
    # 6. 强制设置epsilon为0（双重保险）
    scheduler.agent.epsilon = 0.0
    print(f"  ✓ 模型已加载: {model_path}")
    print(f"  ✓ 当前epsilon: {scheduler.agent.epsilon}")
    
    # 7. 运行仿真（使用被动模式）
    print("\\n[4] 运行仿真...")
    print(f"  - 仿真步数: {time_steps}")
    print(f"  - 被动模式: True (控制传输频率)")  # ← 关键修复3
    print(f"  - 检查间隔: 10分钟")
    
    simulation = EnergySimulation(
        network=network,
        time_steps=time_steps,
        scheduler=scheduler,
        enable_energy_sharing=True,
        passive_mode=True,        # ← 关键修复3: 启用被动模式
        check_interval=10         # ← 每10分钟检查一次，而非每分钟
    )
    
    print("-" * 80)
    simulation.simulate()
    print("-" * 80)
    
    # 8. 统计结果
    print("\\n[5] 仿真统计:")
    results = simulation.result_manager.get_results()
    
    # 统计传输次数和时长
    total_transfers = 0
    total_energy_sent = 0
    durations = []
    
    for result in results:
        if 'plans' in result and result['plans']:
            total_transfers += len(result['plans'])
            for plan in result['plans']:
                duration = plan.get('duration', 1)
                durations.append(duration)
                total_energy_sent += duration * 500  # E_char=500
    
    import numpy as np
    print(f"  - 总传输次数: {total_transfers}")
    if durations:
        print(f"  - 平均传输时长: {np.mean(durations):.2f} 分钟")
        print(f"  - 时长范围: {min(durations)}-{max(durations)} 分钟")
        print(f"  - 总传输能量: {total_energy_sent:.0f}J")
    
    # 能量统计
    final_energies = [node.current_energy for node in network.nodes[1:]]
    alive_nodes = sum(1 for e in final_energies if e > 0)
    print(f"\\n  - 存活节点: {alive_nodes}/{len(final_energies)}")
    print(f"  - 平均能量: {np.mean(final_energies):.0f}J")
    print(f"  - 能量标准差: {np.std(final_energies):.0f}J")
    
    print("\\n" + "=" * 80)
    print("✅ 推理完成！")
    print("=" * 80)
    
    if alive_nodes < len(final_energies):
        print("\\n⚠️ 警告: 仍有节点死亡")
        print("建议:")
        print("1. 增加check_interval (如15或20分钟)")
        print("2. 检查训练质量（可能需要更多训练回合）")
        print("3. 调整K值（减少并发传输）")
    else:
        print("\\n✅ 所有节点存活，DQN工作正常！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='DQN推理模式（已修复）')
    parser.add_argument('--model', type=str, default='dqn_model.pth',
                       help='模型路径')
    parser.add_argument('--steps', type=int, default=10080,
                       help='仿真步数（默认10080=7天）')
    
    args = parser.parse_args()
    
    run_dqn_inference(model_path=args.model, time_steps=args.steps)
"""
    
    with open('dqn_inference_fixed.py', 'w', encoding='utf-8') as f:
        f.write(example_code)
    
    print("\n✅ 已创建推理示例: dqn_inference_fixed.py")
    print("\n使用方法:")
    print("  python dqn_inference_fixed.py --model dqn_model.pth --steps 10080")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='DQN推理模式修复工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 修复DQN模型的epsilon值
  python fix_dqn_inference.py --fix dqn_model.pth
  
  # 修复并保存到新文件
  python fix_dqn_inference.py --fix dqn_model.pth --output dqn_model_fixed.pth
  
  # 检查配置清单
  python fix_dqn_inference.py --check
  
  # 生成推理示例代码
  python fix_dqn_inference.py --example
        """
    )
    
    parser.add_argument('--fix', type=str, metavar='MODEL_PATH',
                       help='修复DQN模型的epsilon值')
    parser.add_argument('--output', type=str, metavar='OUTPUT_PATH',
                       help='输出路径（默认覆盖原文件）')
    parser.add_argument('--check', action='store_true',
                       help='显示配置检查清单')
    parser.add_argument('--example', action='store_true',
                       help='生成推理示例代码')
    
    args = parser.parse_args()
    
    if args.fix:
        fix_dqn_model(args.fix, args.output)
    
    if args.check or not any([args.fix, args.example]):
        check_dqn_config()
    
    if args.example:
        create_inference_example()


if __name__ == "__main__":
    main()


