"""
无线传感器网络仿真错误处理和日志系统
提供统一的错误处理和日志记录功能
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from contextlib import contextmanager


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class SimulationError(Exception):
    """仿真基础异常类"""
    pass


class NetworkError(SimulationError):
    """网络相关异常"""
    pass


class NodeError(SimulationError):
    """节点相关异常"""
    pass


class RoutingError(SimulationError):
    """路由相关异常"""
    pass


class SchedulerError(SimulationError):
    """调度器相关异常"""
    pass


class ConfigError(SimulationError):
    """配置相关异常"""
    pass


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


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.error_counts: Dict[str, int] = {}
        self.error_history: list = []
    
    def handle_error(self, error: Exception, context: str = "", 
                    recoverable: bool = True) -> bool:
        """处理错误"""
        error_type = type(error).__name__
        error_msg = str(error)
        
        # 记录错误
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_history.append({
            'timestamp': datetime.now(),
            'type': error_type,
            'message': error_msg,
            'context': context,
            'recoverable': recoverable
        })
        
        # 记录日志
        if recoverable:
            logger.warning(f"可恢复错误 [{context}]: {error_type} - {error_msg}")
        else:
            logger.error(f"严重错误 [{context}]: {error_type} - {error_msg}")
        
        return recoverable
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        return {
            'total_errors': len(self.error_history),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:] if self.error_history else []
        }
    
    def clear_history(self):
        """清除错误历史"""
        self.error_counts.clear()
        self.error_history.clear()


# 全局错误处理器
error_handler = ErrorHandler()


@contextmanager
def handle_exceptions(context: str = "", recoverable: bool = True):
    """异常处理上下文管理器"""
    try:
        yield
    except Exception as e:
        error_handler.handle_error(e, context, recoverable)
        if not recoverable:
            raise


def safe_execute(func, *args, context: str = "", 
                recoverable: bool = True, default=None, **kwargs):
    """安全执行函数"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(e, context, recoverable)
        return default


class ValidationError(SimulationError):
    """验证错误"""
    pass


class Validator:
    """参数验证器"""
    
    @staticmethod
    def validate_positive_number(value: float, name: str) -> float:
        """验证正数"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} 必须是数字类型")
        if value <= 0:
            raise ValidationError(f"{name} 必须大于0")
        return float(value)
    
    @staticmethod
    def validate_percentage(value: float, name: str) -> float:
        """验证百分比"""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{name} 必须是数字类型")
        if not 0 <= value <= 1:
            raise ValidationError(f"{name} 必须在0和1之间")
        return float(value)
    
    @staticmethod
    def validate_integer_range(value: int, name: str, min_val: int = 0, 
                             max_val: int = None) -> int:
        """验证整数范围"""
        if not isinstance(value, int):
            raise ValidationError(f"{name} 必须是整数")
        if value < min_val:
            raise ValidationError(f"{name} 不能小于 {min_val}")
        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} 不能大于 {max_val}")
        return value
    
    @staticmethod
    def validate_string_choice(value: str, name: str, choices: list) -> str:
        """验证字符串选择"""
        if not isinstance(value, str):
            raise ValidationError(f"{name} 必须是字符串")
        if value not in choices:
            raise ValidationError(f"{name} 必须是以下之一: {choices}")
        return value


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, list] = {}
        self.start_times: Dict[str, datetime] = {}
    
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
    
    def get_metrics(self) -> Dict[str, Dict[str, float]]:
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
