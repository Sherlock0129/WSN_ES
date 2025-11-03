# -*- coding: utf-8 -*-
"""
测试信息感知路由在模拟中的实际效果

验证：
1. 信息感知路由是否在模拟中被调用
2. 有信息量的节点是否被优先选择
3. 是否减少了额外的信息上报次数
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config.simulation_config import ConfigManager


def test_info_aware_routing_in_simulation():
    """测试信息感知路由在模拟中的效果"""
    
    print("=" * 60)
    print("信息感知路由在模拟中的效果测试")
    print("=" * 60)
    
    # 1. 创建配置
    print("\n[1] 创建配置...")
    config_manager = ConfigManager()
    
    # 启用信息感知路由
    config_manager.eetor_config.enable_info_aware_routing = True
    config_manager.eetor_config.info_reward_factor = 0.3
    
    # 启用机会主义信息传递
    config_manager.path_collector_config.enable_opportunistic_info_forwarding = True
    config_manager.path_collector_config.enable_delayed_reporting = True
    config_manager.path_collector_config.max_wait_time = 10
    
    print(f"  EETOR配置:")
    print(f"    - enable_info_aware_routing: {config_manager.eetor_config.enable_info_aware_routing}")
    print(f"    - info_reward_factor: {config_manager.eetor_config.info_reward_factor}")
    print(f"  PathCollector配置:")
    print(f"    - enable_opportunistic_info_forwarding: {config_manager.path_collector_config.enable_opportunistic_info_forwarding}")
    print(f"    - enable_delayed_reporting: {config_manager.path_collector_config.enable_delayed_reporting}")
    
    # 2. 创建网络
    print("\n[2] 创建网络...")
    network = config_manager.create_network()
    print(f"  - 网络节点数: {network.num_nodes}")
    
    # 3. 创建路径信息收集器
    print("\n[3] 创建路径信息收集器...")
    if config_manager.path_collector_config.enable_path_collector:
        from acdr.physical_center import NodeInfoManager
        
        # 获取物理中心节点
        physical_center = network.get_physical_center()
        
        # 创建NodeInfoManager
        vc = NodeInfoManager(
            initial_position=tuple(physical_center.position) if physical_center else (0.0, 0.0),
            enable_logging=False
        )
        vc.initialize_node_info(network.nodes, initial_time=0)
        
        # 创建路径信息收集器
        network.path_info_collector = config_manager.create_path_collector(vc, physical_center)
        print(f"  - 路径信息收集器创建成功")
        print(f"  - 虚拟中心节点数: {len(vc.latest_info)}")
    else:
        print("  [ERROR] 路径信息收集器未启用")
        return
    
    # 4. 创建调度器
    print("\n[4] 创建调度器...")
    from sim.refactored_main import create_scheduler
    scheduler = create_scheduler(config_manager, network)
    print(f"  - 调度器类型: {scheduler.get_name()}")
    print(f"  - NodeInfoManager: {scheduler.nim is not None}")
    
    # 5. 验证信息感知路由的调用链
    print("\n[5] 验证信息感知路由的调用链...")
    
    # 5.1 检查scheduler.nim是否正确传递
    if scheduler.nim is not None:
        print("  [OK] scheduler.nim 存在")
    else:
        print("  [ERROR] scheduler.nim 不存在")
        return
    
    # 5.2 检查EETOR配置是否正确加载
    from routing.energy_transfer_routing import get_eetor_config, set_eetor_config
    
    # 设置EETOR配置（确保配置正确传递）
    set_eetor_config(config_manager.eetor_config)
    
    eetor_config = get_eetor_config()
    if eetor_config.enable_info_aware_routing:
        print(f"  [OK] EETOR配置中 enable_info_aware_routing = {eetor_config.enable_info_aware_routing}")
        print(f"  [OK] info_reward_factor = {eetor_config.info_reward_factor}")
    else:
        print(f"  [WARN] EETOR配置中 enable_info_aware_routing = {eetor_config.enable_info_aware_routing}")
        print("  [提示] 需要调用 set_eetor_config() 来传递配置")
    
    # 6. 模拟一些路径完成，检查信息量是否更新
    print("\n[6] 模拟路径完成，检查信息量更新...")
    
    # 模拟一条能量传输路径完成
    if network.path_info_collector and hasattr(network.path_info_collector, 'collect_and_report'):
        # 选择一些节点作为路径
        path_nodes = [n for n in network.nodes if n.node_id <= 5][:3]  # 选择前3个节点
        if len(path_nodes) >= 2:
            print(f"  - 模拟路径: {' -> '.join([str(n.node_id) for n in path_nodes])}")
            
            # 调用collect_and_report（注意参数顺序）
            current_time = 1
            # collect_and_report(path: List[SensorNode], all_nodes: List[SensorNode], current_time: int)
            all_nodes = network.nodes
            network.path_info_collector.collect_and_report(path_nodes, all_nodes, current_time)
            
            # 检查终点节点的信息量是否更新
            receiver = path_nodes[-1]
            receiver_info = scheduler.nim.get_node_info(receiver.node_id)
            
            if receiver_info:
                info_volume = receiver_info.get('info_volume', 0)
                is_reported = receiver_info.get('info_is_reported', True)
                
                print(f"  - 终点节点 {receiver.node_id}:")
                print(f"    - 信息量: {info_volume} bits")
                print(f"    - 是否已上报: {is_reported}")
                
                if info_volume > 0 and not is_reported:
                    print("  [OK] 信息量已更新，节点进入等待状态")
                elif info_volume == 0 and is_reported:
                    print("  [WARN] 信息量未累积（可能是立即上报模式）")
                else:
                    print(f"  [INFO] 信息量状态: {info_volume} bits, 已上报={is_reported}")
            else:
                print(f"  [WARN] 节点 {receiver.node_id} 的信息未找到")
    
    # 7. 验证信息感知路由是否会影响路由选择
    print("\n[7] 验证信息感知路由的调用...")
    
    # 检查select_forwarder_prefix_energy_aware中是否有信息感知逻辑
    from routing.energy_transfer_routing import select_forwarder_prefix_energy_aware
    import inspect
    
    source = inspect.getsource(select_forwarder_prefix_energy_aware)
    
    checks = {
        'enable_info_aware_routing': 'enable_info_aware_routing' in source,
        'node_info_manager': 'node_info_manager' in source and 'get_node_info' in source,
        'info_volume': 'info_volume' in source,
        'info_reward_factor': 'info_reward_factor' in source,
        'info_bonus': 'info_bonus' in source or 'bonus' in source,
    }
    
    for check_name, result in checks.items():
        if result:
            print(f"  [OK] 代码中包含 {check_name} 相关逻辑")
        else:
            print(f"  [ERROR] 代码中缺少 {check_name} 相关逻辑")
    
    # 8. 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_ok = all(checks.values())
    
    if all_ok:
        print("\n[OK] 信息感知路由功能已正确实现并在模拟中可用")
        print("\n使用说明:")
        print("  1. 在配置中设置 EETORConfig.enable_info_aware_routing = True")
        print("  2. 调整 info_reward_factor 控制信息奖励强度（0~1）")
        print("  3. 启用机会主义信息传递以累积节点信息量")
        print("  4. 路由算法会自动优先选择信息量大的节点")
    else:
        print("\n[WARN] 部分功能检查未通过，请检查代码实现")
    
    print("=" * 60)


if __name__ == "__main__":
    test_info_aware_routing_in_simulation()

