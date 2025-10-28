#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
无线传感器网络仿真日志系统
提供统一的日志记录功能
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional
from enum import Enum
from contextlib import contextmanager
from .output_manager import OutputManager


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Logger:
    """统一日志管理器"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """设置日志系统"""
        # 创建按日期命名的日志目录
        date_str = datetime.now().strftime('%Y%m%d')
        log_dir = os.path.join("logs", date_str)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建主日志器
        self.logger = logging.getLogger("WSN_Simulation")
        self.logger.setLevel(logging.DEBUG)
        
        # 清除已有的处理器
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 文件处理器
        log_file = os.path.join(log_dir, "simulation.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 错误文件处理器
        error_file = os.path.join(log_dir, "errors.log")
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, extra=kwargs)
    
    def set_level(self, level: LogLevel):
        """设置日志级别"""
        self.logger.setLevel(getattr(logging, level.value))
    
    def add_file_handler(self, filename: str, level: LogLevel = LogLevel.INFO):
        """添加文件处理器"""
        handler = logging.FileHandler(filename, encoding='utf-8')
        handler.setLevel(getattr(logging, level.value))
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)


# 全局日志实例
logger = Logger()


def log_function_call(func):
    """函数调用日志装饰器"""
    def wrapper(*args, **kwargs):
        logger.debug(f"调用函数: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"函数 {func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"函数 {func.__name__} 执行失败: {str(e)}")
            raise
    return wrapper


def log_execution_time(func):
    """执行时间日志装饰器"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"函数 {func.__name__} 执行时间: {duration:.4f}秒")
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"函数 {func.__name__} 执行失败 (耗时: {duration:.4f}秒): {str(e)}")
            raise
    return wrapper


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: dict = {}
        self.start_times: dict = {}
    
    def start_timer(self, name: str):
        """开始计时"""
        self.start_times[name] = datetime.now()
    
    def end_timer(self, name: str) -> float:
        """结束计时"""
        if name not in self.start_times:
            logger.warning(f"计时器 {name} 未开始")
            return 0.0
        
        duration = (datetime.now() - self.start_times[name]).total_seconds()
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        del self.start_times[name]
        return duration
    
    def get_metrics(self) -> dict:
        """获取性能指标"""
        result = {}
        for name, times in self.metrics.items():
            if times:
                result[name] = {
                    'count': len(times),
                    'total': sum(times),
                    'average': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        return result
    
    def reset(self):
        """重置指标"""
        self.metrics.clear()
        self.start_times.clear()


# 全局性能监控器
performance_monitor = PerformanceMonitor()


@contextmanager
def monitor_performance(name: str):
    """性能监控上下文管理器"""
    performance_monitor.start_timer(name)
    try:
        yield
    finally:
        duration = performance_monitor.end_timer(name)
        logger.debug(f"性能监控 [{name}]: {duration:.4f}秒")


class DetailedPlanLogger:
    """详细计划日志记录器"""
    
    def __init__(self, session_dir: str = None):
        """
        初始化详细计划日志记录器
        
        Args:
            session_dir: 会话目录，如果为None则使用当前时间戳目录
        """
        if session_dir is None:
            session_dir = OutputManager.get_session_dir("data")
        
        self.session_dir = session_dir
        self.plans_file = OutputManager.get_file_path(session_dir, "plans.txt")
        
        # 确保目录存在
        OutputManager.ensure_dir_exists(session_dir)
        
        logger.info(f"详细计划日志将保存到: {self.plans_file}")
    
    def save_simulation_plans(self, simulation):
        """
        保存整个仿真的所有计划到文件
        
        Args:
            simulation: EnergySimulation实例，包含plans_by_time数据
        """
        try:
            with open(self.plans_file, "w", encoding="utf-8") as f:
                f.write("=== 无线传感器网络能量传输详细计划记录 ===\n")
                f.write(f"仿真时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总时间步: {simulation.time_steps}\n")
                f.write(f"节点数量: {len(simulation.network.nodes)}\n")
                f.write("=" * 60 + "\n\n")
                
                # 按时间步排序
                sorted_times = sorted(simulation.plans_by_time.keys())
                
                for t in sorted_times:
                    time_data = simulation.plans_by_time[t]
                    if not time_data:  # 如果没有数据，跳过
                        continue
                    
                    # 处理新的数据结构
                    if isinstance(time_data, dict) and "plans" in time_data:
                        plans = time_data["plans"]
                        candidates = time_data.get("candidates", [])
                        node_energies = time_data.get("node_energies", {})
                    else:
                        # 兼容旧格式
                        plans = time_data
                        candidates = []
                        node_energies = {}
                    
                    # 即使没有plans也记录节点能量
                    f.write(f"时间步 t={t}\n")
                    f.write("-" * 40 + "\n")
                    
                    # 打印节点能量信息
                    if node_energies:
                        f.write("节点能量状态:\n")
                        # 按节点ID排序打印
                        for node_id in sorted(node_energies.keys()):
                            energy = node_energies[node_id]
                            f.write(f"  Node {node_id:2d}: {energy:8.2f}J\n")
                        f.write("\n")
                    
                    # 打印候选计划
                    if candidates:
                        f.write("候选计划 (Candidates):\n")
                        for i, candidate in enumerate(candidates):
                            if len(candidate) >= 7:
                                score, donor, receiver, path, distance, delivered, loss = candidate[:7]
                                
                                d_id = getattr(donor, "node_id", None)
                                r_id = getattr(receiver, "node_id", None)
                                path_ids = [str(getattr(n, "node_id", n)) for n in path]
                                path_str = "->".join(path_ids)
                                
                                f.write(f"  [{i+1:2d}] 评分:{score:6.2f} | "
                                       f"路径: {path_str} | "
                                       f"距离:{distance:5.2f}m | "
                                       f"传输:{delivered:7.2f}J | "
                                       f"损失:{loss:7.2f}J\n")
                        f.write("\n")
                    
                    # 打印选中的计划
                    if plans:
                        f.write("选中计划 (Selected Plans):\n")
                        for i, plan in enumerate(plans):
                            donor = plan.get("donor")
                            receiver = plan.get("receiver")
                            path = plan.get("path", [])
                            distance = plan.get("distance", 0.0)
                            delivered = plan.get("delivered", 0.0)
                            loss = plan.get("loss", 0.0)
                            
                            d_id = getattr(donor, "node_id", None)
                            r_id = getattr(receiver, "node_id", None)
                            path_ids = [str(getattr(n, "node_id", n)) for n in path]
                            path_str = "->".join(path_ids)
                            
                            f.write(f"  [{i+1:2d}] 路径: {path_str} | "
                                   f"距离:{distance:5.2f}m | "
                                   f"传输:{delivered:7.2f}J | "
                                   f"损失:{loss:7.2f}J\n")
                    else:
                        f.write("选中计划: 无\n")
                    
                    f.write("\n" + "=" * 60 + "\n\n")
                
                f.write("=== 记录结束 ===\n")
            
            logger.info(f"详细计划记录已保存到: {self.plans_file}")
            logger.info(f"记录了 {len(sorted_times)} 个时间步的计划信息")
            
        except Exception as e:
            logger.error(f"保存详细计划记录失败: {str(e)}")
            raise
    
    def save_single_timestep_plans(self, time_step: int, time_data, output_file: str = None):
        """
        保存单个时间步的计划到文件
        
        Args:
            time_step: 时间步
            time_data: 时间步数据
            output_file: 输出文件名，如果为None则使用默认名称
        """
        if output_file is None:
            output_file = f"plans_t{time_step}.txt"
        
        file_path = OutputManager.get_file_path(self.session_dir, output_file)
        
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"=== 时间步 {time_step} 的详细计划记录 ===\n")
                f.write(f"记录时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n\n")
                
                # 处理数据结构
                if isinstance(time_data, dict) and "plans" in time_data:
                    plans = time_data["plans"]
                    candidates = time_data.get("candidates", [])
                    node_energies = time_data.get("node_energies", {})
                else:
                    plans = time_data if isinstance(time_data, list) else []
                    candidates = []
                    node_energies = {}
                
                # 节点能量
                if node_energies:
                    f.write("节点能量状态:\n")
                    for node_id in sorted(node_energies.keys()):
                        energy = node_energies[node_id]
                        f.write(f"  Node {node_id:2d}: {energy:8.2f}J\n")
                    f.write("\n")
                
                # 候选计划
                if candidates:
                    f.write("候选计划:\n")
                    for i, candidate in enumerate(candidates):
                        if len(candidate) >= 7:
                            score, donor, receiver, path, distance, delivered, loss = candidate[:7]
                            d_id = getattr(donor, "node_id", None)
                            r_id = getattr(receiver, "node_id", None)
                            path_ids = [str(getattr(n, "node_id", n)) for n in path]
                            path_str = "->".join(path_ids)
                            
                            f.write(f"  [{i+1:2d}] 评分:{score:6.2f} | "
                                   f"路径: {path_str} | "
                                   f"传输:{delivered:7.2f}J\n")
                    f.write("\n")
                
                # 选中计划
                if plans:
                    f.write("选中计划:\n")
                    for i, plan in enumerate(plans):
                        donor = plan.get("donor")
                        receiver = plan.get("receiver")
                        path = plan.get("path", [])
                        delivered = plan.get("delivered", 0.0)
                        
                        d_id = getattr(donor, "node_id", None)
                        r_id = getattr(receiver, "node_id", None)
                        path_ids = [str(getattr(n, "node_id", n)) for n in path]
                        path_str = "->".join(path_ids)
                        
                        f.write(f"  [{i+1:2d}] 路径: {path_str} | "
                               f"传输:{delivered:7.2f}J\n")
                else:
                    f.write("选中计划: 无\n")
                
                f.write("\n=== 记录结束 ===\n")
            
            logger.info(f"时间步 {time_step} 的计划记录已保存到: {file_path}")
            
        except Exception as e:
            logger.error(f"保存时间步 {time_step} 计划记录失败: {str(e)}")
            raise


class StatisticsLogger:
    """仿真统计信息日志记录器"""
    
    def __init__(self, session_dir: str = None):
        """
        初始化统计信息日志记录器
        
        Args:
            session_dir: 会话目录，如果为None则使用当前时间戳目录
        """
        if session_dir is None:
            session_dir = OutputManager.get_session_dir("data")
        
        self.session_dir = session_dir
        self.stats_file = OutputManager.get_file_path(session_dir, "simulation_statistics.txt")
        self.stats_json_file = OutputManager.get_file_path(session_dir, "simulation_statistics.json")
        
        # 确保目录存在
        OutputManager.ensure_dir_exists(session_dir)
        
        logger.info(f"统计信息日志将保存到: {self.stats_file}")
    
    def save_statistics(self, stats: dict, network=None, additional_info: dict = None):
        """
        保存统计信息到文件（文本格式和JSON格式）
        
        Args:
            stats: 统计信息字典，包含以下键：
                - avg_variance: 所有时间点方差的平均值
                - total_loss_energy: 总体损失能量值
                - total_sent_energy: 总发送能量
                - total_received_energy: 总接收能量
                - efficiency: 能量传输效率（可选）
            network: Network实例（可选，用于获取额外信息）
            additional_info: 额外的统计信息（可选）
        """
        try:
            # 计算效率
            efficiency = (stats['total_received_energy'] / stats['total_sent_energy'] * 100 
                         if stats['total_sent_energy'] > 0 else 0)
            stats['efficiency'] = efficiency
            
            # 保存为文本格式（易读）
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("无线传感器网络能量传输仿真统计报告\n")
                f.write("=" * 70 + "\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"会话目录: {self.session_dir}\n")
                f.write("\n")
                
                # 基本统计信息
                f.write("-" * 70 + "\n")
                f.write("【能量传输统计】\n")
                f.write("-" * 70 + "\n")
                f.write(f"总发送能量:         {stats['total_sent_energy']:>15.4f} Joules\n")
                f.write(f"总接收能量:         {stats['total_received_energy']:>15.4f} Joules\n")
                f.write(f"总体损失能量:       {stats['total_loss_energy']:>15.4f} Joules\n")
                f.write(f"能量传输效率:       {stats['efficiency']:>15.2f}%\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("【能量分布统计】\n")
                f.write("-" * 70 + "\n")
                f.write(f"所有时间点方差的平均值: {stats['avg_variance']:>11.4f}\n")
                f.write("\n")
                
                # 网络节点信息
                if network is not None:
                    f.write("-" * 70 + "\n")
                    f.write("【网络节点信息】\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"节点总数:           {len(network.nodes):>15d}\n")
                    
                    # 计算节点能量统计
                    node_energies = [node.current_energy for node in network.nodes]
                    f.write(f"节点平均能量:       {sum(node_energies)/len(node_energies):>15.4f} Joules\n")
                    f.write(f"节点最大能量:       {max(node_energies):>15.4f} Joules\n")
                    f.write(f"节点最小能量:       {min(node_energies):>15.4f} Joules\n")
                    
                    # 计算存活节点数（能量>0）
                    alive_nodes = sum(1 for e in node_energies if e > 0)
                    f.write(f"存活节点数:         {alive_nodes:>15d}\n")
                    f.write(f"死亡节点数:         {len(node_energies) - alive_nodes:>15d}\n")
                    f.write("\n")
                    
                    # 节点传输历史统计
                    total_transfers = sum(len(node.transferred_history) for node in network.nodes)
                    total_receptions = sum(len(node.received_history) for node in network.nodes)
                    f.write(f"总传输次数:         {total_transfers:>15d}\n")
                    f.write(f"总接收次数:         {total_receptions:>15d}\n")
                    f.write("\n")
                
                # 额外信息
                if additional_info:
                    f.write("-" * 70 + "\n")
                    f.write("【额外信息】\n")
                    f.write("-" * 70 + "\n")
                    for key, value in additional_info.items():
                        if isinstance(value, float):
                            f.write(f"{key}: {value:>15.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    f.write("\n")
                
                f.write("=" * 70 + "\n")
                f.write("报告结束\n")
                f.write("=" * 70 + "\n")
            
            # 保存为JSON格式（便于程序处理）
            import json
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'session_dir': self.session_dir,
                'statistics': stats
            }
            
            if network is not None:
                node_energies = [node.current_energy for node in network.nodes]
                json_data['network_info'] = {
                    'total_nodes': len(network.nodes),
                    'avg_energy': sum(node_energies) / len(node_energies),
                    'max_energy': max(node_energies),
                    'min_energy': min(node_energies),
                    'alive_nodes': sum(1 for e in node_energies if e > 0),
                    'dead_nodes': len(node_energies) - sum(1 for e in node_energies if e > 0),
                    'total_transfers': sum(len(node.transferred_history) for node in network.nodes),
                    'total_receptions': sum(len(node.received_history) for node in network.nodes)
                }
            
            if additional_info:
                json_data['additional_info'] = additional_info
            
            with open(self.stats_json_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)
            
            logger.info(f"统计信息已保存到文本文件: {self.stats_file}")
            logger.info(f"统计信息已保存到JSON文件: {self.stats_json_file}")
            
        except Exception as e:
            logger.error(f"保存统计信息失败: {str(e)}")
            raise
    
    def print_and_save_statistics(self, stats: dict, network=None, additional_info: dict = None):
        """
        打印并保存统计信息
        
        Args:
            stats: 统计信息字典
            network: Network实例（可选）
            additional_info: 额外的统计信息（可选）
        """
        # 计算效率
        efficiency = (stats['total_received_energy'] / stats['total_sent_energy'] * 100 
                     if stats['total_sent_energy'] > 0 else 0)
        
        # 打印到控制台
        print("\n" + "=" * 70)
        print("【能量传输统计】")
        print("=" * 70)
        print(f"所有时间点方差的平均值: {stats['avg_variance']:.4f}")
        print(f"总体损失能量值: {stats['total_loss_energy']:.4f} Joules")
        print(f"总发送能量: {stats['total_sent_energy']:.4f} Joules")
        print(f"总接收能量: {stats['total_received_energy']:.4f} Joules")
        print(f"能量传输效率: {efficiency:.2f}%")
        
        if network is not None:
            node_energies = [node.current_energy for node in network.nodes]
            alive_nodes = sum(1 for e in node_energies if e > 0)
            print(f"\n【网络状态】")
            print(f"节点总数: {len(network.nodes)}")
            print(f"存活节点: {alive_nodes}")
            print(f"死亡节点: {len(network.nodes) - alive_nodes}")
        
        print("=" * 70 + "\n")
        
        # 保存到文件
        self.save_statistics(stats, network, additional_info)


# 全局详细计划日志记录器实例
detailed_plan_logger = None

def get_detailed_plan_logger(session_dir: str = None) -> DetailedPlanLogger:
    """获取详细计划日志记录器实例"""
    # 每次都创建新的实例，确保使用正确的session_dir
    return DetailedPlanLogger(session_dir)


def get_statistics_logger(session_dir: str = None) -> StatisticsLogger:
    """获取统计信息日志记录器实例"""
    # 每次都创建新的实例，确保使用正确的session_dir
    return StatisticsLogger(session_dir)

