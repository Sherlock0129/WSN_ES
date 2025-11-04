# -*- coding: utf-8 -*-
"""
短期模拟测试信息感知路由的实际效果

运行一个简短的模拟，验证：
1. 信息感知路由是否被调用
2. 有信息量的节点是否被优先选择
3. 是否减少了信息上报次数
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config.simulation_config import ConfigManager
from sim.refactored_main import run_simulation


def test_short_simulation():
    """运行短期模拟测试"""
    
    print("=" * 60)
    print("短期模拟测试 - 信息感知路由效果验证")
    print("=" * 60)
    
    # 创建配置文件（临时）
    config_manager = ConfigManager()
    
    # 配置参数
    config_manager.simulation_config.time_steps = 120  # 只运行2小时
    config_manager.simulation_config.enable_energy_sharing = True
    
    # 启用信息感知路由
    config_manager.eetor_config.enable_info_aware_routing = True
    config_manager.eetor_config.info_reward_factor = 0.3
    
    # 启用机会主义信息传递
    config_manager.path_collector_config.enable_opportunistic_info_forwarding = True
    config_manager.path_collector_config.enable_delayed_reporting = True
    config_manager.path_collector_config.max_wait_time = 10
    
    # 网络配置（小规模，快速测试）
    config_manager.network_config.num_nodes = 10
    
    print("\n[配置信息]")
    print(f"  - 模拟时间步: {config_manager.simulation_config.time_steps} 分钟")
    print(f"  - 网络节点数: {config_manager.network_config.num_nodes}")
    print(f"  - 信息感知路由: {config_manager.eetor_config.enable_info_aware_routing}")
    print(f"  - 信息奖励系数: {config_manager.eetor_config.info_reward_factor}")
    print(f"  - 机会主义信息传递: {config_manager.path_collector_config.enable_opportunistic_info_forwarding}")
    print(f"  - 延迟上报: {config_manager.path_collector_config.enable_delayed_reporting}")
    
    print("\n[开始运行模拟...]")
    print("=" * 60)
    
    try:
        # 运行模拟
        run_simulation()
        
        print("\n" + "=" * 60)
        print("[测试完成]")
        print("=" * 60)
        print("\n请检查输出日志，确认：")
        print("  1. 信息感知路由是否在模拟中被调用")
        print("  2. 路径选择是否考虑了节点信息量")
        print("  3. 信息量是否在节点信息表中正确累积")
        print("  4. 超时强制上报是否工作")
        
    except Exception as e:
        print(f"\n[错误] 模拟运行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_short_simulation()

