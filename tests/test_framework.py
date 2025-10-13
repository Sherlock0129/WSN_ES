"""
无线传感器网络仿真测试框架
提供单元测试、集成测试和性能测试功能
"""

import unittest
import time
import os
import sys
from typing import List, Dict, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.simulation_config import ConfigManager, NodeConfig, NetworkConfig
from src.utils.error_handling import logger, error_handler, ValidationError, Validator
from src.interfaces import INode, INetwork, IScheduler, EnergyTransferPlan


class TestBase(unittest.TestCase):
    """测试基类"""
    
    def setUp(self):
        """测试前准备"""
        self.config_manager = ConfigManager()
        logger.info(f"开始测试: {self._testMethodName}")
    
    def tearDown(self):
        """测试后清理"""
        logger.info(f"完成测试: {self._testMethodName}")


class TestConfigManager(TestBase):
    """配置管理器测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = ConfigManager()
        
        # 测试节点配置
        self.assertEqual(config.node_config.initial_energy, 40000.0)
        self.assertEqual(config.node_config.low_threshold, 0.1)
        self.assertEqual(config.node_config.high_threshold, 0.9)
        
        # 测试网络配置
        self.assertEqual(config.network_config.num_nodes, 25)
        self.assertEqual(config.network_config.max_hops, 3)
        
        # 测试仿真配置
        self.assertEqual(config.simulation_config.time_steps, 10080)
    
    def test_config_loading(self):
        """测试配置加载"""
        # 创建测试配置文件
        test_config = {
            "node": {
                "initial_energy": 50000.0,
                "low_threshold": 0.2
            },
            "network": {
                "num_nodes": 50,
                "max_hops": 5
            }
        }
        
        # 保存测试配置
        import json
        test_file = "test_config.json"
        with open(test_file, 'w') as f:
            json.dump(test_config, f)
        
        try:
            # 加载配置
            config = ConfigManager(test_file)
            
            # 验证配置
            self.assertEqual(config.node_config.initial_energy, 50000.0)
            self.assertEqual(config.node_config.low_threshold, 0.2)
            self.assertEqual(config.network_config.num_nodes, 50)
            self.assertEqual(config.network_config.max_hops, 5)
            
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)
    
    def test_network_config_dict(self):
        """测试网络配置字典"""
        config = ConfigManager()
        network_config = config.get_network_config_dict()
        
        self.assertIn("num_nodes", network_config)
        self.assertIn("low_threshold", network_config)
        self.assertIn("high_threshold", network_config)
        self.assertIn("distribution_mode", network_config)
        
        self.assertEqual(network_config["num_nodes"], 25)
        self.assertEqual(network_config["low_threshold"], 0.1)


class TestValidator(TestBase):
    """参数验证器测试"""
    
    def test_validate_positive_number(self):
        """测试正数验证"""
        # 有效正数
        self.assertEqual(Validator.validate_positive_number(5.0, "test"), 5.0)
        self.assertEqual(Validator.validate_positive_number(1, "test"), 1.0)
        
        # 无效输入
        with self.assertRaises(ValidationError):
            Validator.validate_positive_number(-1, "test")
        
        with self.assertRaises(ValidationError):
            Validator.validate_positive_number(0, "test")
        
        with self.assertRaises(ValidationError):
            Validator.validate_positive_number("invalid", "test")
    
    def test_validate_percentage(self):
        """测试百分比验证"""
        # 有效百分比
        self.assertEqual(Validator.validate_percentage(0.5, "test"), 0.5)
        self.assertEqual(Validator.validate_percentage(0, "test"), 0.0)
        self.assertEqual(Validator.validate_percentage(1, "test"), 1.0)
        
        # 无效输入
        with self.assertRaises(ValidationError):
            Validator.validate_percentage(-0.1, "test")
        
        with self.assertRaises(ValidationError):
            Validator.validate_percentage(1.1, "test")
    
    def test_validate_integer_range(self):
        """测试整数范围验证"""
        # 有效整数
        self.assertEqual(Validator.validate_integer_range(5, "test", 0, 10), 5)
        self.assertEqual(Validator.validate_integer_range(0, "test", 0, 10), 0)
        self.assertEqual(Validator.validate_integer_range(10, "test", 0, 10), 10)
        
        # 无效输入
        with self.assertRaises(ValidationError):
            Validator.validate_integer_range(-1, "test", 0, 10)
        
        with self.assertRaises(ValidationError):
            Validator.validate_integer_range(11, "test", 0, 10)


class MockNode(INode):
    """模拟节点类用于测试"""
    
    def __init__(self, node_id: int, position: tuple = (0, 0), 
                 energy: float = 1000.0, has_solar: bool = True):
        self._id = node_id
        self._position = position
        self._energy = energy
        self._has_solar = has_solar
        self._is_mobile = False
    
    def get_id(self) -> int:
        return self._id
    
    def get_position(self) -> tuple:
        return self._position
    
    def get_current_energy(self) -> float:
        return self._energy
    
    def get_energy_capacity(self) -> float:
        return 10000.0
    
    def has_solar(self) -> bool:
        return self._has_solar
    
    def is_mobile(self) -> bool:
        return self._is_mobile
    
    def distance_to(self, other_node: INode) -> float:
        x1, y1 = self._position
        x2, y2 = other_node.get_position()
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def energy_transfer_efficiency(self, target_node: INode) -> float:
        distance = self.distance_to(target_node)
        return max(0.0, min(1.0, 0.6 / (distance ** 2)))
    
    def energy_consumption(self, target_node: Optional[INode] = None, 
                          transfer_wet: bool = False) -> float:
        return 10.0
    
    def energy_harvest(self, time_step: int) -> float:
        return 5.0 if self._has_solar else 0.0
    
    def update_energy(self, time_step: int) -> tuple:
        harvested = self.energy_harvest(time_step)
        decayed = 5.0
        self._energy = max(0, self._energy + harvested - decayed)
        return harvested, decayed
    
    def update_position(self, time_step: int) -> None:
        pass


class MockNetwork(INetwork):
    """模拟网络类用于测试"""
    
    def __init__(self, nodes: List[INode]):
        self._nodes = nodes
    
    def get_nodes(self) -> List[INode]:
        return self._nodes
    
    def get_node_by_id(self, node_id: int) -> Optional[INode]:
        for node in self._nodes:
            if node.get_id() == node_id:
                return node
        return None
    
    def get_num_nodes(self) -> int:
        return len(self._nodes)
    
    def update_network_energy(self, time_step: int) -> None:
        for node in self._nodes:
            node.update_energy(time_step)
    
    def execute_energy_transfer(self, plans: List[EnergyTransferPlan]) -> None:
        for plan in plans:
            donor = self.get_node_by_id(plan.donor_id)
            receiver = self.get_node_by_id(plan.receiver_id)
            if donor and receiver:
                # 模拟能量传输
                efficiency = donor.energy_transfer_efficiency(receiver)
                energy_transferred = plan.energy_sent * efficiency
                donor._energy -= plan.energy_sent
                receiver._energy += energy_transferred


class TestMockClasses(TestBase):
    """模拟类测试"""
    
    def setUp(self):
        super().setUp()
        self.nodes = [
            MockNode(0, (0, 0), 1000.0, True),
            MockNode(1, (1, 1), 500.0, False),
            MockNode(2, (2, 2), 1500.0, True)
        ]
        self.network = MockNetwork(self.nodes)
    
    def test_mock_node(self):
        """测试模拟节点"""
        node = self.nodes[0]
        
        self.assertEqual(node.get_id(), 0)
        self.assertEqual(node.get_position(), (0, 0))
        self.assertEqual(node.get_current_energy(), 1000.0)
        self.assertTrue(node.has_solar())
        self.assertFalse(node.is_mobile())
        
        # 测试距离计算
        other_node = self.nodes[1]
        distance = node.distance_to(other_node)
        self.assertAlmostEqual(distance, np.sqrt(2), places=5)
        
        # 测试能量更新
        harvested, decayed = node.update_energy(0)
        self.assertEqual(harvested, 5.0)  # 有太阳能
        self.assertEqual(decayed, 5.0)
    
    def test_mock_network(self):
        """测试模拟网络"""
        self.assertEqual(self.network.get_num_nodes(), 3)
        
        # 测试节点获取
        node = self.network.get_node_by_id(1)
        self.assertIsNotNone(node)
        self.assertEqual(node.get_id(), 1)
        
        # 测试不存在的节点
        node = self.network.get_node_by_id(999)
        self.assertIsNone(node)
        
        # 测试能量更新
        initial_energy = self.nodes[0].get_current_energy()
        self.network.update_network_energy(0)
        # 能量应该保持不变（采集=衰减）


class TestEnergyTransferPlan(TestBase):
    """能量传输计划测试"""
    
    def test_energy_transfer_plan(self):
        """测试能量传输计划"""
        plan = EnergyTransferPlan(
            donor_id=0,
            receiver_id=1,
            path=[0, 1],
            distance=1.0,
            energy_sent=100.0,
            energy_delivered=80.0,
            energy_loss=20.0,
            efficiency=0.8
        )
        
        self.assertEqual(plan.donor_id, 0)
        self.assertEqual(plan.receiver_id, 1)
        self.assertEqual(plan.path, [0, 1])
        self.assertEqual(plan.distance, 1.0)
        self.assertEqual(plan.energy_sent, 100.0)
        self.assertEqual(plan.energy_delivered, 80.0)
        self.assertEqual(plan.energy_loss, 20.0)
        self.assertEqual(plan.efficiency, 0.8)


class PerformanceTest(TestBase):
    """性能测试"""
    
    def test_large_network_performance(self):
        """测试大型网络性能"""
        # 创建大型网络
        nodes = []
        for i in range(100):
            node = MockNode(i, (i % 10, i // 10), 1000.0)
            nodes.append(node)
        
        network = MockNetwork(nodes)
        
        # 测试能量更新性能
        start_time = time.time()
        network.update_network_energy(0)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"100个节点能量更新耗时: {duration:.4f}秒")
        
        # 性能断言（应该小于1秒）
        self.assertLess(duration, 1.0)
    
    def test_distance_calculation_performance(self):
        """测试距离计算性能"""
        nodes = [MockNode(i, (i, i)) for i in range(50)]
        
        start_time = time.time()
        for i in range(1000):
            node1 = nodes[i % len(nodes)]
            node2 = nodes[(i + 1) % len(nodes)]
            node1.distance_to(node2)
        end_time = time.time()
        
        duration = end_time - start_time
        logger.info(f"1000次距离计算耗时: {duration:.4f}秒")
        
        # 性能断言
        self.assertLess(duration, 0.1)


class IntegrationTest(TestBase):
    """集成测试"""
    
    def test_simulation_integration(self):
        """测试仿真集成"""
        # 创建测试网络
        nodes = [
            MockNode(0, (0, 0), 2000.0, True),   # 高能量太阳能节点
            MockNode(1, (1, 0), 500.0, False),  # 低能量非太阳能节点
            MockNode(2, (2, 0), 1000.0, True)   # 中等能量太阳能节点
        ]
        network = MockNetwork(nodes)
        
        # 模拟能量传输计划
        plans = [
            EnergyTransferPlan(
                donor_id=0,
                receiver_id=1,
                path=[0, 1],
                distance=1.0,
                energy_sent=100.0
            )
        ]
        
        # 执行能量传输
        initial_donor_energy = nodes[0].get_current_energy()
        initial_receiver_energy = nodes[1].get_current_energy()
        
        network.execute_energy_transfer(plans)
        
        # 验证能量传输结果
        final_donor_energy = nodes[0].get_current_energy()
        final_receiver_energy = nodes[1].get_current_energy()
        
        self.assertLess(final_donor_energy, initial_donor_energy)
        self.assertGreater(final_receiver_energy, initial_receiver_energy)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestConfigManager,
        TestValidator,
        TestMockClasses,
        TestEnergyTransferPlan,
        PerformanceTest,
        IntegrationTest
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 输出测试结果
    logger.info(f"测试运行完成: {result.testsRun} 个测试")
    logger.info(f"失败: {len(result.failures)} 个")
    logger.info(f"错误: {len(result.errors)} 个")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # 设置日志级别
    logger.set_level(logger.LogLevel.INFO)
    
    # 运行测试
    success = run_tests()
    
    if success:
        print("\n✅ 所有测试通过!")
        sys.exit(0)
    else:
        print("\n❌ 部分测试失败!")
        sys.exit(1)
