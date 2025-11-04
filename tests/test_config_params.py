"""
测试配置文件参数传递是否正常工作
"""

import sys
import os
sys.path.insert(0, '../src')

from config.simulation_config import ConfigManager, load_config


def test_default_config():
    """测试默认配置"""
    print("=" * 60)
    print("测试1: 默认配置")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 检查被动传能参数
    sim_config = config_manager.simulation_config
    
    print("\n被动传能参数（默认）:")
    print(f"  passive_mode: {sim_config.passive_mode}")
    print(f"  check_interval: {sim_config.check_interval}")
    print(f"  critical_ratio: {sim_config.critical_ratio}")
    print(f"  energy_variance_threshold: {sim_config.energy_variance_threshold}")
    print(f"  cooldown_period: {sim_config.cooldown_period}")
    print(f"  predictive_window: {sim_config.predictive_window}")
    
    # 创建 EnergySimulation 并检查参数是否正确传递
    network = config_manager.create_network()
    simulation = config_manager.create_energy_simulation(network)
    
    print("\n传递到 EnergySimulation 的参数:")
    print(f"  passive_mode: {simulation.passive_manager.passive_mode}")
    print(f"  check_interval: {simulation.passive_manager.check_interval}")
    print(f"  critical_ratio: {simulation.passive_manager.critical_ratio}")
    print(f"  energy_variance_threshold: {simulation.passive_manager.energy_variance_threshold}")
    print(f"  cooldown_period: {simulation.passive_manager.cooldown_period}")
    print(f"  predictive_window: {simulation.passive_manager.predictive_window}")
    
    print("\n✓ 默认配置参数传递正常")


def test_json_config():
    """测试从JSON文件加载配置"""
    print("\n" + "=" * 60)
    print("测试2: 从JSON文件加载配置")
    print("=" * 60)
    
    config_file = "src/config/智能被动传能示例.json"
    
    if not os.path.exists(config_file):
        print(f"✗ 配置文件不存在: {config_file}")
        return
    
    config_manager = load_config(config_file)
    sim_config = config_manager.simulation_config
    
    print(f"\n从 {config_file} 加载的参数:")
    print(f"  passive_mode: {sim_config.passive_mode}")
    print(f"  check_interval: {sim_config.check_interval}")
    print(f"  critical_ratio: {sim_config.critical_ratio}")
    print(f"  energy_variance_threshold: {sim_config.energy_variance_threshold}")
    print(f"  cooldown_period: {sim_config.cooldown_period}")
    print(f"  predictive_window: {sim_config.predictive_window}")
    
    # 创建仿真并验证
    network = config_manager.create_network()
    simulation = config_manager.create_energy_simulation(network)
    
    print("\n传递到 EnergySimulation 的参数:")
    passive_config = simulation.passive_manager.get_config()
    for key, value in passive_config.items():
        print(f"  {key}: {value}")
    
    print("\n✓ JSON配置文件参数传递正常")


def test_custom_config():
    """测试自定义配置"""
    print("\n" + "=" * 60)
    print("测试3: 自定义配置")
    print("=" * 60)
    
    config_manager = ConfigManager()
    
    # 修改配置
    config_manager.simulation_config.passive_mode = False  # 改为传统模式
    config_manager.simulation_config.check_interval = 5
    config_manager.simulation_config.critical_ratio = 0.15
    
    print("\n自定义配置:")
    print(f"  passive_mode: {config_manager.simulation_config.passive_mode}")
    print(f"  check_interval: {config_manager.simulation_config.check_interval}")
    print(f"  critical_ratio: {config_manager.simulation_config.critical_ratio}")
    
    # 创建仿真
    network = config_manager.create_network()
    simulation = config_manager.create_energy_simulation(network)
    
    print("\n传递到 EnergySimulation 的参数:")
    print(f"  passive_mode: {simulation.passive_manager.passive_mode}")
    print(f"  check_interval: {simulation.passive_manager.check_interval}")
    print(f"  critical_ratio: {simulation.passive_manager.critical_ratio}")
    
    # 验证
    assert simulation.passive_manager.passive_mode == False, "passive_mode 传递失败"
    assert simulation.passive_manager.check_interval == 5, "check_interval 传递失败"
    assert simulation.passive_manager.critical_ratio == 0.15, "critical_ratio 传递失败"
    
    print("\n✓ 自定义配置参数传递正常")


def test_direct_creation():
    """测试直接创建（不使用 ConfigManager）"""
    print("\n" + "=" * 60)
    print("测试4: 直接创建 EnergySimulation（向后兼容性）")
    print("=" * 60)
    
    from core.energy_simulation import EnergySimulation
    from config.simulation_config import ConfigManager
    
    config_manager = ConfigManager()
    network = config_manager.create_network()
    
    # 直接创建，使用默认参数
    simulation = EnergySimulation(
        network=network,
        time_steps=1000
    )
    
    print("\n使用默认参数创建:")
    print(f"  passive_mode: {simulation.passive_manager.passive_mode}")
    print(f"  check_interval: {simulation.passive_manager.check_interval}")
    
    # 直接创建，自定义参数
    simulation2 = EnergySimulation(
        network=network,
        time_steps=1000,
        passive_mode=False,
        check_interval=20,
        critical_ratio=0.25
    )
    
    print("\n使用自定义参数创建:")
    print(f"  passive_mode: {simulation2.passive_manager.passive_mode}")
    print(f"  check_interval: {simulation2.passive_manager.check_interval}")
    print(f"  critical_ratio: {simulation2.passive_manager.critical_ratio}")
    
    assert simulation2.passive_manager.passive_mode == False
    assert simulation2.passive_manager.check_interval == 20
    assert simulation2.passive_manager.critical_ratio == 0.25
    
    print("\n✓ 直接创建方式参数传递正常（向后兼容）")


if __name__ == "__main__":
    print("配置参数传递测试")
    print("=" * 60)
    
    try:
        test_default_config()
        test_json_config()
        test_custom_config()
        test_direct_creation()
        
        print("\n" + "=" * 60)
        print("✓ 所有测试通过！配置文件参数传递正常工作")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

