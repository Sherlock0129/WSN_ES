# test_path_collector_csv.py
# -*- coding: utf-8 -*-
"""
测试PathCollector独立虚拟中心的CSV归档功能
"""
import sys
import os
sys.path.insert(0, 'src')

from config.simulation_config import ConfigManager


def test_path_collector_csv_generation():
    """测试PathCollector生成CSV文件"""
    print("\n" + "=" * 60)
    print("测试：PathCollector独立虚拟中心CSV生成")
    print("=" * 60)
    
    # 1. 创建配置（禁用ADCR，启用PathCollector）
    config_manager = ConfigManager()
    config_manager.simulation_config.enable_adcr_link_layer = False  # 禁用ADCR
    config_manager.path_collector_config.enable_path_collector = True  # 启用PathCollector
    config_manager.simulation_config.simulation_days = 1  # 只运行1天快速测试
    config_manager.simulation_config.time_step_minutes = 60  # 1小时步长
    
    print("\n[配置]")
    print(f"  ADCR启用: {config_manager.simulation_config.enable_adcr_link_layer}")
    print(f"  PathCollector启用: {config_manager.path_collector_config.enable_path_collector}")
    print(f"  模拟天数: {config_manager.simulation_config.simulation_days}")
    
    # 2. 创建网络
    print("\n[创建网络]")
    network = config_manager.create_network()
    print(f"  节点数: {len(network.nodes)}")
    
    # 3. 创建PathCollector（独立虚拟中心）
    print("\n[创建PathCollector]")
    from acdr.virtual_center import VirtualCenter
    vc = VirtualCenter(enable_logging=False)
    vc.initialize_node_info(network.nodes, initial_time=0)
    network.path_info_collector = config_manager.create_path_collector(vc)
    print(f"  虚拟中心: 独立创建")
    print(f"  初始archive_path: {vc.archive_path}")
    
    # 4. 创建调度器（使用ConfigManager）
    print("\n[创建调度器]")
    from sim.refactored_main import create_scheduler
    scheduler = create_scheduler(config_manager)
    print(f"  调度器: {scheduler.get_name()}")
    
    # 5. 创建仿真
    simulation = config_manager.create_energy_simulation(network, scheduler)
    print(f"  输出目录: {simulation.session_dir}")
    
    # 6. 设置虚拟中心归档路径（模拟refactored_main的逻辑）
    print("\n[设置归档路径]")
    if hasattr(network, 'adcr_link') and network.adcr_link is not None:
        print("  使用ADCR虚拟中心")
        network.adcr_link.set_archive_path(simulation.session_dir)
    elif hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        print("  使用PathCollector独立虚拟中心")
        archive_path = os.path.join(simulation.session_dir, "virtual_center_node_info.csv")
        network.path_info_collector.vc.archive_path = archive_path
        print(f"  设置后archive_path: {archive_path}")
    
    # 7. 运行仿真
    print("\n[运行仿真]")
    simulation.simulate()
    print("  仿真完成")
    
    # 8. 强制刷新归档
    print("\n[刷新归档]")
    if hasattr(network, 'adcr_link') and network.adcr_link is not None:
        network.adcr_link.vc.force_flush_archive()
        print("  ADCR虚拟中心归档已刷新")
    elif hasattr(network, 'path_info_collector') and network.path_info_collector is not None:
        network.path_info_collector.vc.force_flush_archive()
        print("  PathCollector虚拟中心归档已刷新")
    
    # 9. 检查CSV文件
    print("\n[检查CSV文件]")
    csv_path = os.path.join(simulation.session_dir, "virtual_center_node_info.csv")
    if os.path.exists(csv_path):
        print(f"  [OK] CSV文件已生成: {csv_path}")
        
        # 读取并显示前几行
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"  文件大小: {len(lines)} 行")
            print("\n  前5行:")
            for i, line in enumerate(lines[:5]):
                print(f"    {i+1}: {line.strip()}")
        
        print("\n[SUCCESS] 测试通过！PathCollector独立虚拟中心可以生成CSV！")
        return True
    else:
        print(f"  [FAILED] CSV文件未生成！")
        print(f"  预期路径: {csv_path}")
        return False


if __name__ == "__main__":
    try:
        success = test_path_collector_csv_generation()
        if not success:
            sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

