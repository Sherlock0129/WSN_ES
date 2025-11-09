"""
AdaptiveLyapunovScheduler 配置示例

展示如何在配置中使用自适应Lyapunov调度器
"""

from dataclasses import dataclass

# 方式1：使用默认参数（推荐）
@dataclass
class SchedulerConfig:
    scheduler_type: str = "AdaptiveLyapunovScheduler"  # 启用自适应调度器
    
    # 其他参数使用默认值即可
    # adaptive_lyapunov_v: float = 0.5         # 初始V（会自动调整）
    # adaptive_lyapunov_k: int = 3             # K值
    # adaptive_window_size: int = 10           # 反馈窗口
    # adaptive_v_min: float = 0.1              # V最小值
    # adaptive_v_max: float = 2.0              # V最大值
    # adaptive_adjust_rate: float = 0.1        # 调整速率10%
    # adaptive_sensitivity: float = 2.0        # 敏感度阈值


# 方式2：自定义参数（高级用户）
@dataclass
class SchedulerConfigCustom:
    scheduler_type: str = "AdaptiveLyapunovScheduler"
    
    # 自适应参数
    adaptive_lyapunov_v: float = 0.6         # 更大的初始V（更保守）
    adaptive_lyapunov_k: int = 4             # 更多donor
    adaptive_window_size: int = 15           # 更大的窗口（更平滑）
    adaptive_v_min: float = 0.2              # 更大的下限
    adaptive_v_max: float = 1.5              # 更小的上限
    adaptive_adjust_rate: float = 0.15       # 更快的调整速率
    adaptive_sensitivity: float = 1.5        # 更敏感的触发


# 方式3：针对不同场景的推荐配置

# 场景1：网络负载不均衡（推荐更积极的调整）
@dataclass
class UnbalancedNetworkConfig:
    scheduler_type: str = "AdaptiveLyapunovScheduler"
    adaptive_lyapunov_v: float = 0.4         # 较小的初始V
    adaptive_v_min: float = 0.1              # 允许更小的V
    adaptive_v_max: float = 2.0              
    adaptive_adjust_rate: float = 0.15       # 更快响应
    adaptive_sensitivity: float = 1.5        # 更敏感


# 场景2：网络相对稳定（推荐保守调整）
@dataclass
class StableNetworkConfig:
    scheduler_type: str = "AdaptiveLyapunovScheduler"
    adaptive_lyapunov_v: float = 0.5         
    adaptive_v_min: float = 0.2              # 不允许太小
    adaptive_v_max: float = 1.5              # 不允许太大
    adaptive_adjust_rate: float = 0.05       # 更慢调整
    adaptive_sensitivity: float = 3.0        # 更不敏感


# 场景3：长时间运行（推荐大窗口）
@dataclass
class LongRunConfig:
    scheduler_type: str = "AdaptiveLyapunovScheduler"
    adaptive_lyapunov_v: float = 0.5
    adaptive_window_size: int = 20           # 大窗口
    adaptive_adjust_rate: float = 0.08       # 缓慢调整
    adaptive_sensitivity: float = 2.5        


# 场景4：快速实验/调试（推荐小窗口）
@dataclass
class QuickTestConfig:
    scheduler_type: str = "AdaptiveLyapunovScheduler"
    adaptive_lyapunov_v: float = 0.5
    adaptive_window_size: int = 5            # 小窗口
    adaptive_adjust_rate: float = 0.2        # 快速调整
    adaptive_sensitivity: float = 1.0        # 很敏感


# 使用说明
"""
在 src/config/simulation_config.py 中：

1. 基础使用（默认参数）：
   @dataclass
   class SchedulerConfig:
       scheduler_type: str = "AdaptiveLyapunovScheduler"  # 就这一行！

2. 自定义参数：
   @dataclass
   class SchedulerConfig:
       scheduler_type: str = "AdaptiveLyapunovScheduler"
       
       # 根据需要修改以下参数
       adaptive_lyapunov_v: float = 0.6
       adaptive_adjust_rate: float = 0.15
       # ...

3. 运行仿真：
   python src/sim/refactored_main.py

4. 查看结果：
   仿真结束后会自动打印自适应调整摘要

5. 参数调优指南：

   如果V参数震荡：
   - 减小 adaptive_adjust_rate (0.05-0.08)
   - 增大 adaptive_sensitivity (3.0-4.0)
   - 增大 adaptive_window_size (15-20)

   如果V从不调整：
   - 减小 adaptive_sensitivity (1.0-1.5)
   - 增大 adaptive_adjust_rate (0.15-0.2)

   如果V触及边界：
   - 放宽 adaptive_v_min 和 adaptive_v_max

6. 对比实验：
   # 标准Lyapunov
   scheduler_type: str = "LyapunovScheduler"
   
   # 自适应Lyapunov
   scheduler_type: str = "AdaptiveLyapunovScheduler"
   
   运行两次，对比性能差异

7. 常见问题：

   Q: 如何知道V调整是否合理？
   A: 查看仿真日志中的调整信息和最终摘要

   Q: 什么场景不适合用自适应版本？
   A: 极短仿真(<50步)或需要严格可复现性的对照实验

   Q: 如何导出调整历史？
   A: scheduler.get_adaptation_stats() 返回完整统计

8. 推荐配置组合：

   # 典型WSN场景（20-50节点）
   adaptive_lyapunov_v: float = 0.5
   adaptive_lyapunov_k: int = 3
   adaptive_window_size: int = 10
   adaptive_adjust_rate: float = 0.1
   adaptive_sensitivity: float = 2.0

   # 大规模网络（>50节点）
   adaptive_lyapunov_v: float = 0.6
   adaptive_lyapunov_k: int = 4
   adaptive_window_size: int = 15
   adaptive_adjust_rate: float = 0.08
   adaptive_sensitivity: float = 2.5
"""

