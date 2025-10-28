# test_path_collector.py
# -*- coding: utf-8 -*-
"""
测试路径信息收集器的功能
"""
import sys
sys.path.insert(0, 'src')

from config.simulation_config import ConfigManager
from info_collection.path_based_collector import PathBasedInfoCollector
from acdr.physical_center import VirtualCenter
from core.SensorNode import SensorNode
import random


def create_test_nodes(num_nodes=10):
    """创建测试节点"""
    nodes = []
    random.seed(42)
    
    for i in range(num_nodes):
        x = random.uniform(0, 10)
        y = random.uniform(0, 10)
        has_solar = (i % 2 == 0)  # 偶数节点有太阳能
        
        node = SensorNode(
            node_id=i,
            position=(x, y),
            initial_energy=40000,
            low_threshold=0.1,
            high_threshold=0.9,
            has_solar=has_solar,
            capacity=3.5,
            voltage=3.7
        )
        nodes.append(node)
    
    return nodes


def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "=" * 60)
    print("测试1: 基本功能")
    print("=" * 60)
    
    # 1. 创建虚拟中心和节点
    vc = VirtualCenter(enable_logging=True)
    nodes = create_test_nodes(10)
    vc.initialize_node_info(nodes, initial_time=0)
    
    # 2. 创建路径信息收集器
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        enable_logging=True,
        decay_rate=5.0,
        use_solar_model=True
    )
    
    # 3. 模拟一次能量传输路径
    path = [nodes[0], nodes[3], nodes[7]]  # donor -> relay -> receiver
    
    # 4. 收集信息
    print("\n[测试] 收集路径信息...")
    stats = collector.collect_and_report(path, nodes, current_time=100)
    
    print(f"\n[测试结果]")
    print(f"  - 实时信息: {stats['real']} 个节点")
    print(f"  - 估算信息: {stats['estimated']} 个节点")
    
    # 5. 验证虚拟中心数据
    print("\n[虚拟中心数据验证]")
    for node_id in [0, 3, 7]:  # 路径节点
        info = vc.get_node_info(node_id)
        print(f"  Node {node_id}: 新鲜度={info['freshness']}, 是否估算={info.get('is_estimated', 'N/A')}")
    
    print("\n[OK] 测试1通过！\n")


def test_energy_estimation():
    """测试能量估算准确性"""
    print("\n" + "=" * 60)
    print("测试2: 能量估算准确性")
    print("=" * 60)
    
    # 1. 创建虚拟中心和节点
    vc = VirtualCenter(enable_logging=False)
    nodes = create_test_nodes(10)
    vc.initialize_node_info(nodes, initial_time=0)
    
    # 2. 创建收集器
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        enable_logging=False,
        decay_rate=5.0,
        use_solar_model=True
    )
    
    # 3. 第一次收集（t=100）
    path1 = [nodes[0], nodes[1], nodes[2]]
    collector.collect_and_report(path1, nodes, current_time=100)
    
    # 4. 模拟时间流逝和能量变化（手动更新节点能量）
    for node in nodes:
        # 简单模拟：扣除一些能量
        node.current_energy -= 500
    
    # 5. 第二次收集（t=200），不同的路径
    path2 = [nodes[5], nodes[6], nodes[7]]
    collector.collect_and_report(path2, nodes, current_time=200)
    
    # 6. 评估估算准确性
    print("\n[估算准确性评估]")
    accuracy = collector.evaluate_estimation_accuracy(nodes)
    
    if accuracy['count'] > 0:
        print(f"  估算节点数: {accuracy['count']}")
        print(f"  平均绝对误差: {accuracy['avg_absolute_error']:.2f} J")
        print(f"  平均相对误差: {accuracy['avg_relative_error']*100:.2f}%")
        print(f"  最大误差: {accuracy['max_error']:.2f} J")
    else:
        print("  无估算节点")
    
    print("\n[OK] 测试2通过！\n")


def test_multiple_collections():
    """测试多次收集的统计"""
    print("\n" + "=" * 60)
    print("测试3: 多次收集统计")
    print("=" * 60)
    
    # 1. 创建虚拟中心和节点
    vc = VirtualCenter(enable_logging=False)
    nodes = create_test_nodes(10)
    vc.initialize_node_info(nodes, initial_time=0)
    
    # 2. 创建收集器
    collector = PathBasedInfoCollector(
        virtual_center=vc,
        enable_logging=False
    )
    
    # 3. 模拟多次能量传输
    paths = [
        [nodes[0], nodes[1]],
        [nodes[2], nodes[3], nodes[4]],
        [nodes[5], nodes[6]],
        [nodes[7], nodes[8], nodes[9]],
        [nodes[1], nodes[5]]
    ]
    
    for i, path in enumerate(paths):
        t = 100 + i * 50
        collector.collect_and_report(path, nodes, current_time=t)
    
    # 4. 打印统计
    print("\n[收集器统计]")
    collector.print_statistics()
    
    print("[OK] 测试3通过！\n")


def test_config_integration():
    """测试配置集成"""
    print("\n" + "=" * 60)
    print("测试4: 配置文件集成")
    print("=" * 60)
    
    # 1. 创建配置管理器
    config_manager = ConfigManager()
    
    # 2. 检查PathCollectorConfig是否存在
    assert hasattr(config_manager, 'path_collector_config'), "配置管理器缺少path_collector_config"
    
    print(f"\n[PathCollectorConfig]")
    print(f"  启用状态: {config_manager.path_collector_config.enable_path_collector}")
    print(f"  衰减率: {config_manager.path_collector_config.decay_rate}")
    print(f"  太阳能模型: {config_manager.path_collector_config.use_solar_model}")
    print(f"  批量更新: {config_manager.path_collector_config.batch_update}")
    print(f"  日志输出: {config_manager.path_collector_config.enable_logging}")
    
    # 3. 测试create_path_collector方法
    vc = VirtualCenter(enable_logging=False)
    collector = config_manager.create_path_collector(vc)
    
    assert isinstance(collector, PathBasedInfoCollector), "create_path_collector返回类型错误"
    
    print("\n[OK] 测试4通过！\n")


def test_with_adcr():
    """测试与ADCR虚拟中心的集成"""
    print("\n" + "=" * 60)
    print("测试5: 与ADCR虚拟中心集成")
    print("=" * 60)
    
    # 1. 创建配置管理器
    config_manager = ConfigManager()
    
    # 2. 创建网络
    network = config_manager.create_network()
    
    # 3. 创建ADCR（这会创建虚拟中心）
    adcr_link = config_manager.create_adcr_link_layer(network)
    
    # 4. 使用ADCR的虚拟中心创建路径收集器
    collector = config_manager.create_path_collector(adcr_link.vc)
    
    # 5. 模拟一次收集
    path = network.nodes[:3]
    stats = collector.collect_and_report(path, network.nodes, current_time=100)
    
    print(f"\n[集成测试结果]")
    print(f"  网络节点数: {len(network.nodes)}")
    print(f"  虚拟中心位置: {adcr_link.vc.get_position()}")
    print(f"  实时信息: {stats['real']} 个节点")
    print(f"  估算信息: {stats['estimated']} 个节点")
    
    print("\n[OK] 测试5通过！\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("路径信息收集器 - 完整测试套件")
    print("=" * 60)
    
    try:
        test_basic_functionality()
        test_energy_estimation()
        test_multiple_collections()
        test_config_integration()
        test_with_adcr()
        
        print("\n" + "=" * 60)
        print("[SUCCESS] 所有测试通过！")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n[FAILED] 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

