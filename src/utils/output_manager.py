#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出管理工具类
统一管理文件输出路径和目录创建
"""

import os
from datetime import datetime
from typing import Optional, Union


class OutputManager:
    """输出管理器 - 统一管理文件输出路径和目录创建"""
    
    @staticmethod
    def get_session_dir(base_dir: str = "data") -> str:
        """
        获取按日期+时间命名的会话目录
        
        Args:
            base_dir: 基础目录，默认为"data"
            
        Returns:
            str: 会话目录路径，格式为 base_dir/YYYYMMDD_HHMMSS
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(base_dir, timestamp_str)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir
    
    @staticmethod
    def get_parallel_run_dir(base_dir: str, run_id: int) -> str:
        """
        获取并行运行的目录
        
        Args:
            base_dir: 基础目录
            run_id: 运行ID
            
        Returns:
            str: 运行目录路径，格式为 base_dir/YYYYMMDD_HHMMSS/run_X
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(base_dir, timestamp_str, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    @staticmethod
    def get_file_path(session_dir: str, filename: str) -> str:
        """
        获取文件完整路径
        
        Args:
            session_dir: 会话目录
            filename: 文件名
            
        Returns:
            str: 完整文件路径
        """
        return os.path.join(session_dir, filename)
    
    @staticmethod
    def get_adcr_dir(session_dir: str) -> str:
        """
        获取ADCR子目录
        
        Args:
            session_dir: 会话目录
            
        Returns:
            str: ADCR目录路径
        """
        adcr_dir = os.path.join(session_dir, 'adcr')
        os.makedirs(adcr_dir, exist_ok=True)
        return adcr_dir
    
    @staticmethod
    def ensure_dir_exists(directory: str) -> str:
        """
        确保目录存在
        
        Args:
            directory: 目录路径
            
        Returns:
            str: 目录路径
        """
        os.makedirs(directory, exist_ok=True)
        return directory
    
    @staticmethod
    def get_timestamped_filename(base_name: str, extension: str = None) -> str:
        """
        获取带时间戳的文件名
        
        Args:
            base_name: 基础文件名
            extension: 文件扩展名（可选）
            
        Returns:
            str: 带时间戳的文件名
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if extension:
            return f"{base_name}_{timestamp}.{extension}"
        else:
            return f"{base_name}_{timestamp}"
    
    @staticmethod
    def get_log_dir() -> str:
        """
        获取按日期+时间命名的日志目录
        
        Returns:
            str: 日志目录路径，格式为 logs/YYYYMMDD_HHMMSS
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join("logs", timestamp_str)
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    @staticmethod
    def get_log_file(log_type: str = "simulation") -> str:
        """
        获取日志文件路径
        
        Args:
            log_type: 日志类型（simulation, errors等）
            
        Returns:
            str: 日志文件路径，格式为 logs/YYYYMMDD_HHMMSS/log_type.log
        """
        log_dir = OutputManager.get_log_dir()
        return os.path.join(log_dir, f"{log_type}.log")


class FilePathBuilder:
    """文件路径构建器 - 提供链式API构建文件路径"""
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = base_dir
        self.session_dir = None
        self.sub_dir = None
    
    def session(self, date_str: Optional[str] = None) -> 'FilePathBuilder':
        """设置会话目录"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        self.session_dir = os.path.join(self.base_dir, date_str)
        os.makedirs(self.session_dir, exist_ok=True)
        return self
    
    def subdir(self, sub_dir_name: str) -> 'FilePathBuilder':
        """设置子目录"""
        if self.session_dir is None:
            self.session()
        self.sub_dir = os.path.join(self.session_dir, sub_dir_name)
        os.makedirs(self.sub_dir, exist_ok=True)
        return self
    
    def file(self, filename: str) -> str:
        """构建文件路径"""
        if self.sub_dir:
            return os.path.join(self.sub_dir, filename)
        elif self.session_dir:
            return os.path.join(self.session_dir, filename)
        else:
            return os.path.join(self.base_dir, filename)
    
    def build(self) -> str:
        """构建最终路径"""
        if self.sub_dir:
            return self.sub_dir
        elif self.session_dir:
            return self.session_dir
        else:
            return self.base_dir


# 便捷函数
def get_session_dir(base_dir: str = "data") -> str:
    """获取会话目录的便捷函数"""
    return OutputManager.get_session_dir(base_dir)


def get_file_path(session_dir: str, filename: str) -> str:
    """获取文件路径的便捷函数"""
    return OutputManager.get_file_path(session_dir, filename)


def ensure_dir_exists(directory: str) -> str:
    """确保目录存在的便捷函数"""
    return OutputManager.ensure_dir_exists(directory)


# 使用示例
if __name__ == "__main__":
    # 基本用法
    session_dir = OutputManager.get_session_dir()
    file_path = OutputManager.get_file_path(session_dir, "test.csv")
    print(f"文件路径: {file_path}")
    
    # 链式API用法
    file_path = (FilePathBuilder("data")
                 .session()
                 .subdir("adcr")
                 .file("adcr_info_paths.png"))
    print(f"ADCR文件路径: {file_path}")
    
    # 便捷函数用法
    session_dir = get_session_dir()
    file_path = get_file_path(session_dir, "simulation_results.csv")
    print(f"仿真结果路径: {file_path}")
