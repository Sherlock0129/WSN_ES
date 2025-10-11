#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并行仿真接口 - 提供并行仿真的便捷入口
"""

import multiprocessing as mp
from wsn_et.experiments.experiment_manager import create_default_config
from wsn_et.experiments.main import run_parallel_simulation


def main():
    """并行仿真的主入口"""
    MAX_WORKERS = 5
    
    # 创建默认配置
    config = create_default_config()
    
    print("=" * 60)
    print("并行仿真启动器")
    print("=" * 60)
    print(f"并行进程数: {MAX_WORKERS}")
    print(f"网络配置: {config.network_config}")
    print(f"调度器配置: {config.scheduler_config}")
    print("=" * 60)
    
    # 运行并行仿真
    results = run_parallel_simulation(
        config=config,
        num_runs=10,
        max_workers=MAX_WORKERS
    )
    
    print("\n所有运行完成！")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

