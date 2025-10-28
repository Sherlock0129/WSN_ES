# src/info_collection/__init__.py
# -*- coding: utf-8 -*-
"""
信息收集模块

包含不同的信息收集策略：
- PathBasedCollector: 基于能量传输路径的机会主义收集
- (未来可扩展) TimerBasedCollector: 基于定时器的周期性收集
"""

from .path_based_collector import PathBasedInfoCollector, create_path_based_collector

__all__ = [
    'PathBasedInfoCollector',
    'create_path_based_collector'
]

