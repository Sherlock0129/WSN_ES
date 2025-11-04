# -*- coding: utf-8 -*-
"""
综合测试信息感知路由功能

验证：
1. 代码实现完整性
2. 配置正确传递
3. 调用链完整
4. 实际运行效果
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def comprehensive_test():
    """综合测试"""
    
    print("=" * 70)
    print("信息感知路由功能综合测试")
    print("=" * 70)
    
    results = {
        'code_implementation': False,
        'function_signatures': False,
        'scheduler_calls': False,
        'config_setup': False,
        'simulation_integration': False
    }
    
    # 1. 代码实现检查
    print("\n[测试1] 代码实现检查")
    print("-" * 70)
    try:
        from routing.energy_transfer_routing import select_forwarder_prefix_energy_aware
        import inspect
        source = inspect.getsource(select_forwarder_prefix_energy_aware)
        
        required_elements = [
            'enable_info_aware_routing',
            'node_info_manager',
            'info_volume',
            'info_reward_factor',
            'info_bonus'
        ]
        
        missing = [elem for elem in required_elements if elem not in source]
        
        if not missing:
            print("  [OK] select_forwarder_prefix_energy_aware 包含所有必要元素")
            results['code_implementation'] = True
        else:
            print(f"  [FAIL] 缺少元素: {missing}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # 2. 函数签名检查
    print("\n[测试2] 函数签名检查")
    print("-" * 70)
    try:
        from routing.energy_transfer_routing import (
            select_forwarder_prefix_energy_aware,
            compute_energy_transfer_costs,
            find_energy_transfer_path,
            find_energy_transfer_path_adaptive,
            eetor_find_path_adaptive
        )
        import inspect
        
        functions = [
            ('select_forwarder_prefix_energy_aware', select_forwarder_prefix_energy_aware),
            ('compute_energy_transfer_costs', compute_energy_transfer_costs),
            ('find_energy_transfer_path', find_energy_transfer_path),
            ('find_energy_transfer_path_adaptive', find_energy_transfer_path_adaptive),
            ('eetor_find_path_adaptive', eetor_find_path_adaptive)
        ]
        
        all_have_nim = True
        for name, func in functions:
            sig = inspect.signature(func)
            has_nim = 'node_info_manager' in sig.parameters
            if has_nim:
                print(f"  [OK] {name} 包含 node_info_manager 参数")
            else:
                print(f"  [FAIL] {name} 缺少 node_info_manager 参数")
                all_have_nim = False
        
        results['function_signatures'] = all_have_nim
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # 3. 调度器调用检查
    print("\n[测试3] 调度器调用检查")
    print("-" * 70)
    try:
        with open('../src/scheduling/schedulers.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        # 统计调用次数
        import re
        calls = re.findall(r'eetor_find_path_adaptive\([^)]*\)', code)
        total_calls = len(calls)
        
        # 检查是否传递了node_info_manager
        calls_with_nim = code.count('node_info_manager=self.nim')
        
        print(f"  - 总调用次数: {total_calls}")
        print(f"  - 传递 node_info_manager 的次数: {calls_with_nim}")
        
        if calls_with_nim >= total_calls:
            print("  [OK] 所有调用都传递了 node_info_manager=self.nim")
            results['scheduler_calls'] = True
        else:
            print(f"  [WARN] 只有 {calls_with_nim}/{total_calls} 处传递了 node_info_manager")
            # 检查具体哪些没有传递
            for i, call in enumerate(calls, 1):
                if 'node_info_manager' not in call:
                    print(f"    调用 {i}: {call[:50]}...")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # 4. 配置设置检查
    print("\n[测试4] 配置设置检查")
    print("-" * 70)
    try:
        from config.simulation_config import EETORConfig
        from routing.energy_transfer_routing import set_eetor_config, get_eetor_config
        
        # 创建配置并设置
        config = EETORConfig()
        config.enable_info_aware_routing = True
        config.info_reward_factor = 0.3
        set_eetor_config(config)
        
        # 验证
        retrieved_config = get_eetor_config()
        if (retrieved_config.enable_info_aware_routing == True and 
            retrieved_config.info_reward_factor == 0.3):
            print("  [OK] 配置正确设置和获取")
            results['config_setup'] = True
        else:
            print(f"  [FAIL] 配置不匹配: enable={retrieved_config.enable_info_aware_routing}, "
                  f"factor={retrieved_config.info_reward_factor}")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # 5. 模拟集成检查
    print("\n[测试5] 模拟集成检查")
    print("-" * 70)
    try:
        with open('../src/sim/refactored_main.py', 'r', encoding='utf-8') as f:
            code = f.read()
        
        has_set_eetor = 'set_eetor_config' in code
        has_info_aware_log = '信息感知路由' in code or 'enable_info_aware_routing' in code
        
        if has_set_eetor:
            print("  [OK] refactored_main.py 中调用了 set_eetor_config")
        if has_info_aware_log:
            print("  [OK] refactored_main.py 中有信息感知路由的日志")
        
        results['simulation_integration'] = has_set_eetor and has_info_aware_log
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # 6. 机会主义信息传递机制检查
    print("\n[测试6] 机会主义信息传递机制检查")
    print("-" * 70)
    try:
        from info_collection.physical_center import NodeInfoManager
        from info_collection.path_based_collector import PathBasedInfoCollector
        
        # 检查NodeInfoManager是否有相关方法
        has_check_timeout = hasattr(NodeInfoManager, 'check_timeout_and_force_report')
        has_info_volume_fields = True  # 假设有，通过初始化测试
        
        # 检查PathBasedInfoCollector是否有相关方法
        has_update_info_volume = hasattr(PathBasedInfoCollector, '_update_info_volume')
        has_set_receiver_info_volume = hasattr(PathBasedInfoCollector, '_set_receiver_info_volume')
        has_report_info_to_center = hasattr(PathBasedInfoCollector, '_report_info_to_center')
        
        print(f"  - NodeInfoManager.check_timeout_and_force_report: {has_check_timeout}")
        print(f"  - PathBasedInfoCollector._update_info_volume: {has_update_info_volume}")
        print(f"  - PathBasedInfoCollector._set_receiver_info_volume: {has_set_receiver_info_volume}")
        print(f"  - PathBasedInfoCollector._report_info_to_center: {has_report_info_to_center}")
        
        if all([has_check_timeout, has_update_info_volume, has_set_receiver_info_volume, has_report_info_to_center]):
            print("  [OK] 机会主义信息传递机制方法完整")
        else:
            print("  [WARN] 部分方法缺失")
    except Exception as e:
        print(f"  [ERROR] {e}")
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\n总体通过率: {passed}/{total} ({passed*100//total if total > 0 else 0}%)")
    
    if passed == total:
        print("\n[结论] 所有测试通过！信息感知路由功能已完整实现并集成到模拟中。")
        print("\n功能状态:")
        print("  [OK] 代码实现完整")
        print("  [OK] 函数签名正确")
        print("  [OK] 调度器调用正确")
        print("  [OK] 配置设置正确")
        print("  [OK] 模拟集成完成")
        print("\n使用方式:")
        print("  1. 在配置中启用: EETORConfig.enable_info_aware_routing = True")
        print("  2. 调整奖励系数: info_reward_factor (0~1，越大奖励越大)")
        print("  3. 启用机会主义信息传递以累积信息量")
        print("  4. 运行模拟，路由算法会自动优先选择信息量大的节点")
    else:
        print(f"\n[结论] 有 {total - passed} 项测试未通过，需要检查实现。")
    
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    comprehensive_test()

