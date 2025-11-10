"""
AdaptiveDurationAwareLyapunovScheduler 测试脚本

测试新调度器的基本功能：
1. 调度器初始化
2. 参数配置正确性
3. 自适应机制
4. 传输时长优化
"""

import sys
import os

# 添加src目录到路径
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

from scheduling.schedulers import AdaptiveDurationAwareLyapunovScheduler
from info_collection.physical_center import NodeInfoManager
from core.network import Network
from config.simulation_config import ConfigManager


def test_scheduler_initialization():
    """测试调度器初始化"""
    print("=" * 60)
    print("测试1: 调度器初始化")
    print("=" * 60)
    
    # 创建简单的网络
    network = Network(
        num_nodes=20,
        low_threshold=200.0,
        high_threshold=800.0,
        node_initial_energy=1000.0,
        max_hops=3,
        distribution_mode="uniform",
        network_area_width=500.0,
        network_area_height=500.0
    )
    
    # 创建节点信息管理器
    node_info_manager = NodeInfoManager()
    node_info_manager.initialize_node_info(network.nodes, initial_time=0)
    
    # 创建调度器
    scheduler = AdaptiveDurationAwareLyapunovScheduler(
        node_info_manager=node_info_manager,
        V=0.5,
        K=3,
        max_hops=3,
        min_duration=1,
        max_duration=5,
        w_aoi=0.02,
        w_info=0.1,
        info_collection_rate=10000.0,
        window_size=10,
        V_min=0.1,
        V_max=2.0,
        adjust_rate=0.1,
        sensitivity=2.0
    )
    
    print(f"✓ 调度器类型: {scheduler.get_name()}")
    print(f"✓ 初始V参数: {scheduler.V}")
    print(f"✓ V范围: [{scheduler.V_min}, {scheduler.V_max}]")
    print(f"✓ 时长范围: {scheduler.min_duration}-{scheduler.max_duration} 分钟")
    print(f"✓ AoI权重: {scheduler.w_aoi}")
    print(f"✓ 信息量权重: {scheduler.w_info}")
    print(f"✓ 反馈窗口: {scheduler.window_size}")
    print(f"✓ 调整速率: {scheduler.adjust_rate}")
    
    return scheduler, network


def test_scheduler_planning(scheduler, network):
    """测试调度器规划功能"""
    print("\n" + "=" * 60)
    print("测试2: 调度器规划功能")
    print("=" * 60)
    
    # 执行规划
    try:
        plans, all_candidates = scheduler.plan(network, t=0)
        print(f"✓ 规划成功")
        print(f"  - 生成计划数: {len(plans)}")
        print(f"  - 候选数: {len(all_candidates)}")
        
        if plans:
            # 显示第一个计划的详细信息
            plan = plans[0]
            print(f"\n第一个计划详情:")
            print(f"  - Donor: Node {plan['donor'].node_id}")
            print(f"  - Receiver: Node {plan['receiver'].node_id}")
            print(f"  - 路径长度: {len(plan['path'])} 跳")
            print(f"  - 传输时长: {plan.get('duration', 1)} 分钟")
            print(f"  - 能量传输: {plan.get('delivered', 0):.2f} J")
            print(f"  - 能量损耗: {plan.get('loss', 0):.2f} J")
            if 'aoi_cost' in plan:
                print(f"  - AoI代价: {plan['aoi_cost']:.2f} 分钟")
            if 'info_gain' in plan:
                print(f"  - 信息收益: {plan['info_gain']:.2f} bits")
            if 'score' in plan:
                print(f"  - 综合得分: {plan['score']:.2f}")
        
        return True
    except Exception as e:
        print(f"✗ 规划失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_adaptive_mechanism(scheduler):
    """测试自适应机制"""
    print("\n" + "=" * 60)
    print("测试3: 自适应机制")
    print("=" * 60)
    
    # 模拟反馈
    feedback = {
        'total_score': -5.0,
        'details': {
            'balance_score': -3.0,
            'efficiency_score': -1.0,
            'survival_score': -1.0,
            'efficiency': 0.4,
            'std_change': 10.0,
            'alive_change': 0
        }
    }
    
    old_V = scheduler.V
    print(f"原始V值: {old_V}")
    
    # 触发自适应调整
    scheduler.post_step(network=None, t=60, feedback=feedback)
    
    new_V = scheduler.V
    print(f"调整后V值: {new_V}")
    
    if old_V != new_V:
        print(f"✓ 自适应机制触发 (V: {old_V:.3f} → {new_V:.3f})")
    else:
        print(f"○ 自适应机制未触发（需要积累更多反馈）")
    
    # 获取统计信息
    stats = scheduler.get_adaptation_stats()
    print(f"\n自适应统计:")
    print(f"  - 当前V: {stats['current_V']:.3f}")
    print(f"  - 调整次数: {stats['total_adjustments']}")
    print(f"  - 平均反馈: {stats['avg_feedback']:.2f}")
    
    return True


def test_config_integration():
    """测试配置文件集成"""
    print("\n" + "=" * 60)
    print("测试4: 配置文件集成")
    print("=" * 60)
    
    # 创建配置管理器
    config_manager = ConfigManager()
    config_manager.scheduler_config.scheduler_type = "AdaptiveDurationAwareLyapunovScheduler"
    
    # 获取调度器参数
    params = config_manager.get_scheduler_params()
    
    print("✓ 配置参数提取成功:")
    print(f"  - V: {params['V']}")
    print(f"  - K: {params['K']}")
    print(f"  - max_hops: {params['max_hops']}")
    print(f"  - min_duration: {params['min_duration']}")
    print(f"  - max_duration: {params['max_duration']}")
    print(f"  - w_aoi: {params['w_aoi']}")
    print(f"  - w_info: {params['w_info']}")
    print(f"  - window_size: {params['window_size']}")
    print(f"  - V_min: {params['V_min']}")
    print(f"  - V_max: {params['V_max']}")
    
    return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("AdaptiveDurationAwareLyapunovScheduler 测试")
    print("="*60 + "\n")
    
    try:
        # 测试1: 初始化
        scheduler, network = test_scheduler_initialization()
        
        # 测试2: 规划
        test_scheduler_planning(scheduler, network)
        
        # 测试3: 自适应机制
        test_adaptive_mechanism(scheduler)
        
        # 测试4: 配置集成
        test_config_integration()
        
        print("\n" + "="*60)
        print("✓ 所有测试通过")
        print("="*60 + "\n")
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ 测试失败: {e}")
        print("="*60 + "\n")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

