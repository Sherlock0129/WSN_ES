#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
虚拟中心重构验证测试

测试要点：
1. VirtualCenter 类可以正常导入
2. ADCRLinkLayerVirtual 可以正常导入
3. VirtualCenter 基本功能正常
"""

import sys
import os

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试 1: 导入模块")
    print("=" * 60)
    
    try:
        from info_collection.physical_center import VirtualCenter, create_virtual_center
        print("✓ VirtualCenter 导入成功")
    except ImportError as e:
        print(f"✗ VirtualCenter 导入失败: {e}")
        return False
    
    try:
        from info_collection.adcr_link_layer import ADCRLinkLayerVirtual
        print("✓ ADCRLinkLayerVirtual 导入成功")
    except ImportError as e:
        print(f"✗ ADCRLinkLayerVirtual 导入失败: {e}")
        return False
    
    return True


def test_virtual_center_basic():
    """测试 VirtualCenter 基本功能"""
    print("\n" + "=" * 60)
    print("测试 2: VirtualCenter 基本功能")
    print("=" * 60)
    
    from info_collection.physical_center import VirtualCenter
    
    # 测试创建
    vc = VirtualCenter(initial_position=(5.0, 5.0), enable_logging=False)
    print(f"✓ VirtualCenter 创建成功: {vc}")
    
    # 测试位置获取
    pos = vc.get_position()
    assert pos == (5.0, 5.0), f"位置错误: {pos}"
    print(f"✓ 位置获取正确: {pos}")
    
    # 测试距离计算
    dist = vc.distance_to((8.0, 9.0))
    expected = ((8-5)**2 + (9-5)**2) ** 0.5  # 5.0
    assert abs(dist - expected) < 0.001, f"距离计算错误: {dist} != {expected}"
    print(f"✓ 距离计算正确: {dist:.3f}")
    
    # 测试位置设置
    vc.set_position(10.0, 10.0)
    assert vc.get_position() == (10.0, 10.0)
    print(f"✓ 位置设置正确: {vc.get_position()}")
    
    return True


def test_virtual_center_with_nodes():
    """测试 VirtualCenter 与节点交互"""
    print("\n" + "=" * 60)
    print("测试 3: VirtualCenter 与节点交互")
    print("=" * 60)
    
    from info_collection.physical_center import VirtualCenter
    from core.SensorNode import SensorNode
    
    # 创建虚拟中心
    vc = VirtualCenter(enable_logging=False)
    
    # 创建测试节点
    nodes = [
        SensorNode(node_id=0, position=(1.0, 1.0), initial_energy=1000),
        SensorNode(node_id=1, position=(5.0, 5.0), initial_energy=1000),
        SensorNode(node_id=2, position=(9.0, 9.0), initial_energy=1000),
    ]
    
    # 测试位置更新（几何中心）
    vc.update_position(nodes)
    expected_center = ((1+5+9)/3, (1+5+9)/3)  # (5.0, 5.0)
    actual_center = vc.get_position()
    assert abs(actual_center[0] - expected_center[0]) < 0.001
    assert abs(actual_center[1] - expected_center[1]) < 0.001
    print(f"✓ 几何中心计算正确: {actual_center}")
    
    # 测试锚点选择（应该选择node 1，因为它在中心）
    anchor = vc.find_anchor(nodes)
    assert anchor is not None
    assert anchor.node_id == 1, f"锚点选择错误: {anchor.node_id}"
    print(f"✓ 锚点选择正确: Node {anchor.node_id}")
    
    return True


def test_integration():
    """测试集成"""
    print("\n" + "=" * 60)
    print("测试 4: 集成测试")
    print("=" * 60)

    from config.simulation_config import ConfigManager
    
    # 创建配置
    config = ConfigManager()
    
    # 创建网络（少量节点用于测试）
    network = config.create_network()
    print(f"✓ 网络创建成功: {len(network.nodes)} 个节点")
    
    # 创建 ADCR
    adcr = config.create_adcr_link_layer(network)
    print(f"✓ ADCR 创建成功")
    
    # 检查 VirtualCenter 是否正确初始化
    assert hasattr(adcr, 'vc'), "ADCR 没有 vc 属性"
    print(f"✓ VirtualCenter 已正确集成到 ADCR 中")
    
    # 检查虚拟中心位置
    vc_pos = adcr.vc.get_position()
    print(f"✓ 虚拟中心初始位置: {vc_pos}")
    
    return True


def main():
    """主测试函数"""
    print("\n" + "█" * 60)
    print(" " * 15 + "虚拟中心重构验证测试")
    print("█" * 60 + "\n")
    
    tests = [
        ("导入测试", test_imports),
        ("基本功能测试", test_virtual_center_basic),
        ("节点交互测试", test_virtual_center_with_nodes),
        ("集成测试", test_integration),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n✗ {test_name} 失败")
        except Exception as e:
            failed += 1
            print(f"\n✗ {test_name} 异常: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 所有测试通过！虚拟中心重构完成！")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查")
        return 1


if __name__ == "__main__":
    sys.exit(main())

