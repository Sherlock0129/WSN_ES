"""
调度器配置诊断脚本
用于验证调度器类型和参数是否正确设置
"""

import sys
import os

# 添加项目根目录和src目录到路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config.simulation_config import ConfigManager

def main():
    print("=" * 70)
    print("调度器配置诊断")
    print("=" * 70)
    
    # 加载配置
    config_manager = ConfigManager()
    sched_config = config_manager.scheduler_config
    path_config = config_manager.path_collector_config
    
    print("\n【调度器配置】")
    print(f"  调度器类型: {sched_config.scheduler_type}")
    print(f"  enable_dqn: {sched_config.enable_dqn}")
    print(f"  enable_ddpg: {sched_config.enable_ddpg}")
    
    print("\n【传输时长参数】（仅DurationAwareLyapunovScheduler使用）")
    print(f"  最小时长: {sched_config.duration_min} 分钟")
    print(f"  最大时长: {sched_config.duration_max} 分钟")
    print(f"  AoI权重: {sched_config.duration_w_aoi}")
    print(f"  信息量权重: {sched_config.duration_w_info}")
    print(f"  信息采集速率: {sched_config.duration_info_rate} bits/分钟")
    
    print("\n【路径信息收集器配置】")
    print(f"  启用路径收集器: {path_config.enable_path_collector}")
    print(f"  机会主义信息传递: {path_config.enable_opportunistic_info_forwarding}")
    print(f"  延迟上报: {path_config.enable_delayed_reporting}")
    print(f"  最大等待时间: {path_config.max_wait_time} 分钟")
    
    print("\n【预期使用的调度器】")
    
    # 判断逻辑（与create_scheduler相同）
    if sched_config.enable_dqn:
        print("  ✓ DQN调度器（深度强化学习 - 离散动作空间）")
        print(f"    - 训练模式: {sched_config.dqn_training_mode}")
        print(f"    - 模型路径: {sched_config.dqn_model_path}")
        print(f"    - 动作空间: {sched_config.dqn_action_dim}个离散选项（1-{sched_config.dqn_action_dim}分钟）")
    elif sched_config.enable_ddpg:
        print("  ✓ DDPG调度器（深度强化学习 - 连续动作空间，自主探索）")
        print(f"    - 训练模式: {sched_config.ddpg_training_mode}")
        print(f"    - 模型路径: {sched_config.ddpg_model_path}")
        print(f"    - 动作范围: [{sched_config.ddpg_action_min:.1f}, {sched_config.ddpg_action_max:.1f}] 分钟")
        print(f"    - 特点: 可输出任意实数（如5.23分钟），完全自主探索")
    else:
        scheduler_type = sched_config.scheduler_type
        
        if scheduler_type == "LyapunovScheduler":
            print("  ✓ 标准 Lyapunov 调度器")
            print("    - 特点: 基于能量队列的机会传输")
        elif scheduler_type == "AdaptiveLyapunovScheduler":
            print("  ✓ 自适应参数 Lyapunov 调度器（推荐）")
            print("    - 特点: V参数自动调整，基于4维反馈")
            print(f"    - 初始V: {sched_config.adaptive_lyapunov_v}")
            print(f"    - V范围: [{sched_config.adaptive_v_min}, {sched_config.adaptive_v_max}]")
            print(f"    - 调整速率: {sched_config.adaptive_adjust_rate*100:.0f}%")
            print(f"    - 反馈窗口: {sched_config.adaptive_window_size}")
        elif scheduler_type == "AdaptiveDurationLyapunovScheduler":
            print("  ✓ 自适应时长 Lyapunov 调度器")
            print("    - 特点: 纯能量优化，选择最优传输时长")
        elif scheduler_type == "DurationAwareLyapunovScheduler":
            print("  ✓ 传输时长感知 Lyapunov 调度器")
            print("    - 特点: 综合考虑能量、AoI、信息量")
            print(f"    - 时长范围: {sched_config.duration_min}-{sched_config.duration_max} 分钟")
            print(f"    - 节点锁定: 启用（duration > 1时）")
            print(f"    - 支持时长感知可视化")
        elif scheduler_type == "ClusterScheduler":
            print("  ✓ 聚类调度器")
            print("    - 特点: 类似LEACH的簇内传输")
        elif scheduler_type == "PredictionScheduler":
            print("  ✓ 预测调度器")
            print("    - 特点: 基于能量趋势预测")
        elif scheduler_type == "PowerControlScheduler":
            print("  ✓ 功率控制调度器")
            print("    - 特点: 目标效率驱动的功率控制")
        elif scheduler_type == "BaselineHeuristic":
            print("  ✓ 基线启发式调度器")
            print("    - 特点: 简单的启发式策略")
        else:
            print(f"  ✗ 未知的调度器类型: {scheduler_type}")
    
    print("\n【配置验证】")
    
    # 验证1: 调度器类型是否正确设置
    if sched_config.scheduler_type == "DurationAwareLyapunovScheduler":
        print("  ✓ 调度器类型配置正确")
    else:
        print(f"  ⚠ 当前调度器类型: {sched_config.scheduler_type}")
        if sched_config.scheduler_type == "LyapunovScheduler":
            print("  ℹ 如需使用传输时长感知，请设置:")
            print("    scheduler_type: str = \"DurationAwareLyapunovScheduler\"")
    
    # 验证2: PathCollector配置提示
    if not path_config.enable_opportunistic_info_forwarding:
        print("  ℹ 机会主义信息传递已关闭")
        print("    - 信息会立即上报，不会等待搭便车")
        print("    - DurationAwareLyapunovScheduler的信息量奖励可能较低")
    
    if not path_config.enable_delayed_reporting:
        print("  ℹ 延迟上报已关闭")
        print("    - 信息会立即上报到物理中心")
        print("    - 不会等待累积更多信息")
    
    # 验证3: DQN/DDPG冲突检查
    if sched_config.enable_dqn or sched_config.enable_ddpg:
        if sched_config.scheduler_type != "LyapunovScheduler":
            print("  ⚠ 注意: DQN/DDPG已启用，scheduler_type将被忽略")
            print("    - DQN/DDPG的优先级高于scheduler_type")
    
    print("\n【参数获取测试】")
    try:
        params = config_manager.get_scheduler_params()
        print(f"  ✓ 成功获取调度器参数")
        print(f"    参数键: {list(params.keys())}")
        
        if 'min_duration' in params:
            print(f"    ✓ 包含DurationAwareLyapunovScheduler专用参数")
            print(f"      - min_duration: {params['min_duration']}")
            print(f"      - max_duration: {params['max_duration']}")
            print(f"      - w_aoi: {params['w_aoi']}")
            print(f"      - w_info: {params['w_info']}")
        else:
            print(f"    ℹ 未包含duration参数（可能使用其他调度器）")
    except Exception as e:
        print(f"  ✗ 获取参数失败: {e}")
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
    
    print("\n【建议】")
    if sched_config.scheduler_type == "DurationAwareLyapunovScheduler" and not sched_config.enable_dqn and not sched_config.enable_ddpg:
        print("  ✓ 配置正确，将使用DurationAwareLyapunovScheduler")
        print("  ✓ 运行仿真时会看到明确的日志:")
        print("    '✓ 使用传输时长感知 Lyapunov 调度器 (DurationAwareLyapunovScheduler)'")
        print("  ✓ 仿真结束后会生成专门的时长可视化图表")
    elif sched_config.enable_dqn:
        print("  ⚠ DQN已启用，将使用DQN调度器而非DurationAwareLyapunovScheduler")
        print("  💡 如需使用DurationAwareLyapunovScheduler，请设置:")
        print("    enable_dqn: bool = False")
    elif sched_config.scheduler_type == "LyapunovScheduler":
        print("  ⚠ 当前使用标准LyapunovScheduler")
        print("  💡 如需使用传输时长感知，请修改配置:")
        print("    scheduler_type: str = \"DurationAwareLyapunovScheduler\"")
    
    print("\n【验证方法】")
    print("  运行仿真: python src/sim/refactored_main.py")
    print("  查看日志中的调度器类型确认")

if __name__ == "__main__":
    main()

