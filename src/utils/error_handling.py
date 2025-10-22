"""
无线传感器网络仿真错误处理系统
提供统一的错误处理和异常管理功能
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from contextlib import contextmanager
from .logger import logger


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


class ValidationError(SimulationError):
    """验证错误"""
    pass


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
