"""
AdaptiveDurationAwareLyapunovScheduler配置示例

这是一个高级调度器，结合了两个维度的优化：
1. 自适应参数调整（根据反馈动态调整V参数）
2. 传输时长优化（综合考虑能量、AoI、信息量）

核心特性：
- 继承DurationAwareLyapunovScheduler的传输时长优化
- 增加AdaptiveLyapunovScheduler的自适应参数调整
- 多维度自适应：响应均衡性、效率、存活率等多个指标
- 带记忆的平滑调整：避免参数震荡
- 节点锁定机制：传输时长>1时自动启用

适用场景：
- 需要长期运行且网络环境动态变化的场景
- 需要在能量效率和传输效率之间自动平衡的场景
- 需要优化传输时长以平衡能量、AoI和信息量的场景
"""

from dataclasses import dataclass
from src.config.simulation_config import (
    SimulationConfig,
    NetworkConfig,
    NodeConfig,
    SchedulerConfig,
    ConfigManager
)


def create_adaptive_duration_aware_config():
    """创建AdaptiveDurationAwareLyapunovScheduler配置"""
    
    # 调度器配置
    scheduler_config = SchedulerConfig(
        # 使用AdaptiveDurationAwareLyapunovScheduler
        scheduler_type="AdaptiveDurationAwareLyapunovScheduler",
        
        # 初始Lyapunov参数（会自动调整）
        adaptive_lyapunov_v=0.5,          # 初始V参数
        adaptive_lyapunov_k=3,            # 每个receiver最多的donor数
        
        # 自适应参数调整参数
        adaptive_window_size=10,          # 反馈窗口大小（记忆最近10次）
        adaptive_v_min=0.1,               # V的最小值
        adaptive_v_max=2.0,               # V的最大值
        adaptive_adjust_rate=0.1,         # 参数调整速率（10%增减）
        adaptive_sensitivity=2.0,         # 反馈敏感度（阈值）
        
        # 传输时长优化参数
        duration_min=1,                   # 最小传输时长（分钟）
        duration_max=5,                   # 最大传输时长（分钟）
        duration_w_aoi=0.02,             # AoI惩罚权重（较小，减小对长传输的惩罚）
        duration_w_info=0.1,             # 信息量奖励权重（较大，鼓励信息搭便车）
        duration_info_rate=10000.0       # 信息采集速率（bits/分钟）
    )
    
    # 网络配置
    network_config = NetworkConfig(
        num_nodes=100,                    # 节点数量
        network_area_width=500.0,         # 网络宽度（米）
        network_area_height=500.0,        # 网络高度（米）
        max_hops=3,                       # 最大跳数
        distribution_mode="center_dense", # 节点分布模式
        min_distance=10.0,                # 节点最小间距
        solar_node_ratio=0.3,             # 太阳能节点比例
        mobile_node_ratio=0.1,            # 移动节点比例
        random_seed=42                    # 随机种子
    )
    
    # 节点配置
    node_config = NodeConfig(
        initial_energy=1000.0,            # 初始能量
        low_threshold=200.0,              # 低能量阈值
        high_threshold=800.0,             # 高能量阈值
        E_char=300.0                      # 特征能量传输量
    )
    
    # 仿真配置
    simulation_config = SimulationConfig(
        total_time=1440 * 7,              # 总仿真时间（7天）
        step_duration=60,                 # 时间步长（1小时）
        use_gpu_acceleration=False,       # GPU加速（可选）
        output_dir="data"                 # 输出目录
    )
    
    # 创建配置管理器
    config_manager = ConfigManager(
        simulation_config=simulation_config,
        network_config=network_config,
        node_config=node_config,
        scheduler_config=scheduler_config
    )
    
    return config_manager


# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = create_adaptive_duration_aware_config()
    
    # 保存配置到JSON文件
    config.save_to_json("adaptive_duration_aware_config.json")
    
    print("✓ 配置已保存到 adaptive_duration_aware_config.json")
    print("\n调度器配置摘要:")
    print(f"  调度器类型: {config.scheduler_config.scheduler_type}")
    print(f"  初始V: {config.scheduler_config.adaptive_lyapunov_v}")
    print(f"  V范围: [{config.scheduler_config.adaptive_v_min}, {config.scheduler_config.adaptive_v_max}]")
    print(f"  调整速率: {config.scheduler_config.adaptive_adjust_rate*100:.0f}%")
    print(f"  反馈窗口: {config.scheduler_config.adaptive_window_size}")
    print(f"  时长范围: {config.scheduler_config.duration_min}-{config.scheduler_config.duration_max} 分钟")
    print(f"  AoI权重: {config.scheduler_config.duration_w_aoi}")
    print(f"  信息量权重: {config.scheduler_config.duration_w_info}")
    print(f"\n特性: V参数自适应调整 + 传输时长优化 + 节点锁定机制")
    
    # 运行仿真示例（需要取消注释）
    # from sim.refactored_main import run_simulation
    # run_simulation(config_file="adaptive_duration_aware_config.json")

