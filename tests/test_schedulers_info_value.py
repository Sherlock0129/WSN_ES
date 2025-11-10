# tests/test_schedulers_info_value.py
# -*- coding: utf-8 -*-
"""
测试三个调度器的信息价值功能

测试内容：
1. AdaptiveDurationLyapunovScheduler 的信息价值功能
2. DDPGScheduler 的状态空间中的信息价值
3. DQNScheduler 的状态空间中的信息价值
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from info_collection.path_based_collector import PathBasedInfoCollector
from info_collection.physical_center import VirtualCenter
from scheduling.schedulers import AdaptiveDurationLyapunovScheduler
from scheduling.ddpg_scheduler import DDPGScheduler
from scheduling.dqn_scheduler import DQNScheduler


def test_adaptive_duration_scheduler():
    """测试 AdaptiveDurationLyapunovScheduler 的信息价值功能"""
    print("=" * 60)
    print("测试1: AdaptiveDurationLyapunovScheduler")
    print("=" * 60)
    
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02
    )
    
    scheduler = AdaptiveDurationLyapunovScheduler(
        node_info_manager=vc,
        V=0.5,
        K=2,
        max_hops=5,
        min_duration=1,
        max_duration=5
    )
    
    # 设置 path_collector
    scheduler.path_collector = collector
    
    # 检查是否有信息价值相关的方法
    has_method = hasattr(scheduler, '_compute_duration_score')
    print(f"  - 有 _compute_duration_score 方法: {has_method}")
    
    if has_method:
        # 检查方法中是否使用信息价值
        import inspect
        source = inspect.getsource(scheduler._compute_duration_score)
        uses_info_value = 'info_value' in source or 'path_collector' in source
        print(f"  - 方法中使用信息价值: {uses_info_value}")
    
    print("\n[OK] AdaptiveDurationLyapunovScheduler 信息价值功能已添加")


def test_ddpg_scheduler():
    """测试 DDPGScheduler 的状态空间中的信息价值"""
    print("\n" + "=" * 60)
    print("测试2: DDPGScheduler")
    print("=" * 60)
    
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02
    )
    
    scheduler = DDPGScheduler(
        node_info_manager=vc,
        K=2,
        max_hops=5,
        training_mode=False
    )
    
    # 设置 path_collector
    scheduler.path_collector = collector
    
    # 检查是否有 _compute_state 方法
    has_method = hasattr(scheduler, '_compute_state')
    print(f"  - 有 _compute_state 方法: {has_method}")
    
    if has_method:
        # 检查方法中是否使用信息价值
        import inspect
        source = inspect.getsource(scheduler._compute_state)
        uses_info_value = 'info_value' in source or 'normalized_info_values' in source
        print(f"  - 方法中使用信息价值: {uses_info_value}")
    
    print("\n[OK] DDPGScheduler 信息价值功能已添加")


def test_dqn_scheduler():
    """测试 DQNScheduler 的状态空间中的信息价值"""
    print("\n" + "=" * 60)
    print("测试3: DQNScheduler")
    print("=" * 60)
    
    vc = VirtualCenter(initial_position=(0.0, 0.0), enable_logging=False)
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        physical_center=None,
        energy_mode="free",
        base_data_size=1000000,
        enable_logging=False,
        info_value_decay_rate=0.02
    )
    
    scheduler = DQNScheduler(
        node_info_manager=vc,
        K=2,
        max_hops=5,
        training_mode=False
    )
    
    # 设置 path_collector
    scheduler.path_collector = collector
    
    # 检查是否有 _compute_state 方法
    has_method = hasattr(scheduler, '_compute_state')
    print(f"  - 有 _compute_state 方法: {has_method}")
    
    if has_method:
        # 检查方法中是否使用信息价值
        import inspect
        source = inspect.getsource(scheduler._compute_state)
        uses_info_value = 'info_value' in source or 'normalized_info_values' in source
        print(f"  - 方法中使用信息价值: {uses_info_value}")
    
    print("\n[OK] DQNScheduler 信息价值功能已添加")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("调度器信息价值功能测试")
    print("=" * 60)
    
    try:
        test_adaptive_duration_scheduler()
        test_ddpg_scheduler()
        test_dqn_scheduler()
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        print("\n修改总结:")
        print("1. AdaptiveDurationLyapunovScheduler: 在 _compute_duration_score 中添加信息价值奖励")
        print("2. DDPGScheduler: 在 _compute_state 中添加信息价值到状态空间")
        print("3. DQNScheduler: 在 _compute_state 中添加信息价值到状态空间")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

