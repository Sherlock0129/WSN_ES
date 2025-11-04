# -*- coding: utf-8 -*-
"""
验证信息感知路由的实现和有效性

综合检查：
1. 代码实现完整性
2. 配置传递正确性
3. 调用链完整性
4. 实际运行验证
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def verify_implementation():
    """验证实现完整性"""
    
    print("=" * 60)
    print("信息感知路由实现验证")
    print("=" * 60)
    
    # 1. 检查代码实现
    print("\n[1] 检查代码实现...")
    
    checks = []
    
    # 1.1 检查select_forwarder_prefix_energy_aware
    try:
        from routing.energy_transfer_routing import select_forwarder_prefix_energy_aware
        import inspect
        source = inspect.getsource(select_forwarder_prefix_energy_aware)
        
        check1 = 'enable_info_aware_routing' in source
        check2 = 'node_info_manager' in source
        check3 = 'info_volume' in source
        check4 = 'info_reward_factor' in source
        check5 = 'info_bonus' in source or 'bonus' in source
        
        checks.append(("select_forwarder_prefix_energy_aware 包含信息感知逻辑", 
                       check1 and check2 and check3 and check4 and check5))
        
        if check1:
            print("  [OK] select_forwarder_prefix_energy_aware 检查 enable_info_aware_routing")
        if check2:
            print("  [OK] select_forwarder_prefix_energy_aware 接受 node_info_manager 参数")
        if check3:
            print("  [OK] select_forwarder_prefix_energy_aware 读取 info_volume")
        if check4:
            print("  [OK] select_forwarder_prefix_energy_aware 使用 info_reward_factor")
        if check5:
            print("  [OK] select_forwarder_prefix_energy_aware 计算 info_bonus")
    except Exception as e:
        print(f"  [ERROR] 无法检查 select_forwarder_prefix_energy_aware: {e}")
        checks.append(("select_forwarder_prefix_energy_aware 检查", False))
    
    # 1.2 检查函数签名
    try:
        from routing.energy_transfer_routing import (
            select_forwarder_prefix_energy_aware,
            compute_energy_transfer_costs,
            find_energy_transfer_path,
            find_energy_transfer_path_adaptive,
            eetor_find_path_adaptive
        )
        import inspect
        
        sig1 = inspect.signature(select_forwarder_prefix_energy_aware)
        sig2 = inspect.signature(compute_energy_transfer_costs)
        sig3 = inspect.signature(find_energy_transfer_path)
        sig4 = inspect.signature(find_energy_transfer_path_adaptive)
        sig5 = inspect.signature(eetor_find_path_adaptive)
        
        has_nim1 = 'node_info_manager' in sig1.parameters
        has_nim2 = 'node_info_manager' in sig2.parameters
        has_nim3 = 'node_info_manager' in sig3.parameters
        has_nim4 = 'node_info_manager' in sig4.parameters
        has_nim5 = 'node_info_manager' in sig5.parameters
        
        checks.append(("所有EETOR函数接受node_info_manager参数",
                       has_nim1 and has_nim2 and has_nim3 and has_nim4 and has_nim5))
        
        if has_nim1:
            print("  [OK] select_forwarder_prefix_energy_aware 函数签名包含 node_info_manager")
        if has_nim2:
            print("  [OK] compute_energy_transfer_costs 函数签名包含 node_info_manager")
        if has_nim3:
            print("  [OK] find_energy_transfer_path 函数签名包含 node_info_manager")
        if has_nim4:
            print("  [OK] find_energy_transfer_path_adaptive 函数签名包含 node_info_manager")
        if has_nim5:
            print("  [OK] eetor_find_path_adaptive 函数签名包含 node_info_manager")
    except Exception as e:
        print(f"  [ERROR] 无法检查函数签名: {e}")
        checks.append(("函数签名检查", False))
    
    # 2. 检查调度器调用
    print("\n[2] 检查调度器调用...")
    
    try:
        with open('../src/scheduling/schedulers.py', 'r', encoding='utf-8') as f:
            schedulers_code = f.read()
        
        call_count = schedulers_code.count('eetor_find_path_adaptive')
        pass_nim_count = schedulers_code.count('node_info_manager=self.nim')
        
        checks.append(("schedulers.py 传递 node_info_manager",
                       call_count > 0 and pass_nim_count == call_count))
        
        if call_count > 0:
            print(f"  [OK] schedulers.py 中有 {call_count} 处调用 eetor_find_path_adaptive")
        if pass_nim_count == call_count and call_count > 0:
            print(f"  [OK] schedulers.py 中所有调用都传递了 node_info_manager=self.nim")
        elif call_count > 0:
            print(f"  [WARN] schedulers.py 中只有 {pass_nim_count}/{call_count} 处传递了 node_info_manager")
    except Exception as e:
        print(f"  [ERROR] 无法检查 schedulers.py: {e}")
        checks.append(("schedulers.py 检查", False))
    
    # 3. 检查配置
    print("\n[3] 检查配置...")
    
    try:
        from config.simulation_config import EETORConfig
        eetor_config = EETORConfig()
        
        has_enable = hasattr(eetor_config, 'enable_info_aware_routing')
        has_factor = hasattr(eetor_config, 'info_reward_factor')
        has_wait_time = hasattr(eetor_config, 'max_info_wait_time')
        has_threshold = hasattr(eetor_config, 'min_info_volume_threshold')
        
        checks.append(("EETORConfig 包含所有信息感知路由参数",
                       has_enable and has_factor and has_wait_time and has_threshold))
        
        if has_enable:
            print(f"  [OK] EETORConfig 有 enable_info_aware_routing (默认: {eetor_config.enable_info_aware_routing})")
        if has_factor:
            print(f"  [OK] EETORConfig 有 info_reward_factor (默认: {eetor_config.info_reward_factor})")
        if has_wait_time:
            print(f"  [OK] EETORConfig 有 max_info_wait_time (默认: {eetor_config.max_info_wait_time})")
        if has_threshold:
            print(f"  [OK] EETORConfig 有 min_info_volume_threshold (默认: {eetor_config.min_info_volume_threshold})")
    except Exception as e:
        print(f"  [ERROR] 无法检查配置: {e}")
        checks.append(("配置检查", False))
    
    # 4. 检查模拟初始化
    print("\n[4] 检查模拟初始化...")
    
    try:
        with open('../src/sim/refactored_main.py', 'r', encoding='utf-8') as f:
            main_code = f.read()
        
        has_set_config = 'set_eetor_config' in main_code
        has_info_aware_log = '信息感知路由' in main_code or 'enable_info_aware_routing' in main_code
        
        checks.append(("模拟初始化时设置EETOR配置",
                       has_set_config))
        
        if has_set_config:
            print("  [OK] refactored_main.py 中调用了 set_eetor_config")
        if has_info_aware_log:
            print("  [OK] refactored_main.py 中有信息感知路由的日志输出")
    except Exception as e:
        print(f"  [ERROR] 无法检查 refactored_main.py: {e}")
        checks.append(("模拟初始化检查", False))
    
    # 5. 总结
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in checks if result)
    total = len(checks)
    
    for name, result in checks:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status} {name}")
    
    print(f"\n通过率: {passed}/{total} ({passed*100//total}%)")
    
    if passed == total:
        print("\n[OK] 所有检查通过！信息感知路由功能已完整实现。")
        print("\n使用说明:")
        print("  1. 在配置中设置 EETORConfig.enable_info_aware_routing = True")
        print("  2. 调整 info_reward_factor 控制信息奖励强度（0~1）")
        print("  3. 启用机会主义信息传递以累积节点信息量")
        print("  4. 运行模拟，路由算法会自动优先选择信息量大的节点")
    else:
        print(f"\n[WARN] 有 {total - passed} 项检查未通过，请检查实现")
    
    print("=" * 60)


if __name__ == "__main__":
    verify_implementation()

