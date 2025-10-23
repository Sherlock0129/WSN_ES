"""
智能被动传能系统测试脚本
测试不同配置下的触发行为
"""

import sys
import os
sys.path.insert(0, 'src')

from config.simulation_config import ConfigManager

def test_passive_mode():
    """测试智能被动传能模式"""
    print("=" * 60)
    print("测试1: 智能被动传能模式（默认配置）")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 设置智能被动传能参数
    config_manager.simulation_config.passive_mode = True
    config_manager.simulation_config.check_interval = 10
    config_manager.simulation_config.critical_ratio = 0.2
    config_manager.simulation_config.energy_variance_threshold = 0.3
    config_manager.simulation_config.cooldown_period = 30
    config_manager.simulation_config.predictive_window = 60
    
    # 启用能量传输
    config_manager.simulation_config.enable_energy_sharing = True
    config_manager.simulation_config.time_steps = 500  # 测试500分钟
    
    # 创建网络
    network = config_manager.create_network()
    print(f"✓ 创建网络: {len(network.nodes)} 个节点")
    
    # 创建调度器
    from scheduling.schedulers import LyapunovScheduler
    scheduler_params = config_manager.get_scheduler_params()
    scheduler = LyapunovScheduler(**scheduler_params)
    print(f"✓ 创建调度器: {scheduler.get_name()}")
    
    # 创建仿真
    simulation = config_manager.create_energy_simulation(network, scheduler)
    print(f"✓ 创建仿真对象")
    print(f"  - 被动模式: {simulation.passive_mode}")
    print(f"  - 检查间隔: {simulation.check_interval} 分钟")
    print(f"  - 临界比例: {simulation.critical_ratio}")
    print(f"  - 能量方差阈值: {simulation.energy_variance_threshold}")
    print(f"  - 冷却期: {simulation.cooldown_period} 分钟")
    print(f"  - 预测窗口: {simulation.predictive_window} 分钟")
    
    # 运行仿真
    print("\n开始仿真...")
    simulation.simulate()
    print("\n✓ 仿真完成")

def test_active_mode():
    """测试传统主动传能模式"""
    print("\n" + "=" * 60)
    print("测试2: 传统主动传能模式（定时60分钟）")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 设置为主动模式
    config_manager.simulation_config.passive_mode = False
    config_manager.simulation_config.enable_energy_sharing = True
    config_manager.simulation_config.time_steps = 500
    
    # 创建网络和仿真
    network = config_manager.create_network()
    from scheduling.schedulers import LyapunovScheduler
    scheduler = LyapunovScheduler(**config_manager.get_scheduler_params())
    simulation = config_manager.create_energy_simulation(network, scheduler)
    
    print(f"✓ 创建网络: {len(network.nodes)} 个节点")
    print(f"✓ 被动模式: {simulation.passive_mode} (将使用定时触发)")
    
    print("\n开始仿真...")
    simulation.simulate()
    print("\n✓ 仿真完成")

def test_aggressive_config():
    """测试激进配置（快速响应）"""
    print("\n" + "=" * 60)
    print("测试3: 激进配置（快速响应型）")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 激进配置
    config_manager.simulation_config.passive_mode = True
    config_manager.simulation_config.check_interval = 5  # 更频繁检查
    config_manager.simulation_config.critical_ratio = 0.15  # 更低的触发阈值
    config_manager.simulation_config.energy_variance_threshold = 0.25
    config_manager.simulation_config.cooldown_period = 15  # 更短的冷却期
    config_manager.simulation_config.predictive_window = 30
    config_manager.simulation_config.enable_energy_sharing = True
    config_manager.simulation_config.time_steps = 500
    
    network = config_manager.create_network()
    from scheduling.schedulers import LyapunovScheduler
    scheduler = LyapunovScheduler(**config_manager.get_scheduler_params())
    simulation = config_manager.create_energy_simulation(network, scheduler)
    
    print(f"✓ 创建网络: {len(network.nodes)} 个节点")
    print(f"  - 检查间隔: {simulation.check_interval} 分钟 (快速)")
    print(f"  - 临界比例: {simulation.critical_ratio} (严格)")
    print(f"  - 冷却期: {simulation.cooldown_period} 分钟 (短)")
    
    print("\n开始仿真...")
    simulation.simulate()
    print("\n✓ 仿真完成")

def test_conservative_config():
    """测试保守配置（节能型）"""
    print("\n" + "=" * 60)
    print("测试4: 保守配置（节能型）")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 保守配置
    config_manager.simulation_config.passive_mode = True
    config_manager.simulation_config.check_interval = 20  # 更少检查
    config_manager.simulation_config.critical_ratio = 0.3  # 更高的触发阈值
    config_manager.simulation_config.energy_variance_threshold = 0.4
    config_manager.simulation_config.cooldown_period = 60  # 更长的冷却期
    config_manager.simulation_config.predictive_window = 90
    config_manager.simulation_config.enable_energy_sharing = True
    config_manager.simulation_config.time_steps = 500
    
    network = config_manager.create_network()
    from scheduling.schedulers import LyapunovScheduler
    scheduler = LyapunovScheduler(**config_manager.get_scheduler_params())
    simulation = config_manager.create_energy_simulation(network, scheduler)
    
    print(f"✓ 创建网络: {len(network.nodes)} 个节点")
    print(f"  - 检查间隔: {simulation.check_interval} 分钟 (慢)")
    print(f"  - 临界比例: {simulation.critical_ratio} (宽松)")
    print(f"  - 冷却期: {simulation.cooldown_period} 分钟 (长)")
    
    print("\n开始仿真...")
    simulation.simulate()
    print("\n✓ 仿真完成")

if __name__ == "__main__":
    print("智能被动传能系统测试")
    print("=" * 60)
    
    # 运行测试
    try:
        test_passive_mode()
        test_active_mode()
        test_aggressive_config()
        test_conservative_config()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

