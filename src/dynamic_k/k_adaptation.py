"""
K值自适应管理模块
负责K值的动态调整和自适应算法
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime


class KAdaptationManager:
    """K值自适应管理器"""
    
    def __init__(self, initial_K: int = 1, K_max: int = 24, hysteresis: float = 0.2,
                 w_b: float = 0.8, w_d: float = 0.8, w_l: float = 1.5,
                 use_lookahead: bool = False):
        """
        初始化K值自适应管理器
        
        Args:
            initial_K: 初始K值
            K_max: 最大K值
            hysteresis: 滞回阈值
            w_b: 均衡改进权重
            w_d: 有效送达量权重
            w_l: 损耗惩罚权重
            use_lookahead: 是否使用前瞻模拟
        """
        self.K = initial_K
        self.K_max = K_max
        self.hysteresis = hysteresis
        self.w_b = w_b
        self.w_d = w_d
        self.w_l = w_l
        self.use_lookahead = use_lookahead
        
        # 状态变量
        self.last_reward = None
        self.last_direction = +1
        
        # 历史记录
        self.K_history: List[int] = []
        self.K_timestamps: List[datetime] = []
        self.K_stats: List[Dict[str, Any]] = []
        
        # 归一化因子
        self.max_improve = None
        self.max_delivered = None
        self.max_loss = None
        
        # 前瞻模拟相关
        self._adaptk_last_t = None
    
    def _reward(self, stats: Dict[str, float]) -> float:
        """
        计算归一化后的奖励值
        
        Args:
            stats: 包含 pre_std, post_std, delivered_total, total_loss 的统计字典
            
        Returns:
            归一化后的奖励值
        """
        # 获取当前统计值
        improve = stats["pre_std"] - stats["post_std"]
        delivered = stats["delivered_total"]
        loss = stats["total_loss"]

        # 计算归一化因子（使用历史最大值或固定参考值）
        if self.max_improve is None:
            self.max_improve = max(abs(improve), 1e-9)
            self.max_delivered = max(delivered, 1e-9)
            self.max_loss = max(loss, 1e-9)
        else:
            # 更新最大值
            self.max_improve = max(self.max_improve, abs(improve))
            self.max_delivered = max(self.max_delivered, delivered)
            self.max_loss = max(self.max_loss, loss)

        # 归一化每个因子
        norm_improve = improve / self.max_improve
        norm_delivered = delivered / self.max_delivered
        norm_loss = loss / self.max_loss

        # 计算加权奖励
        return (
            self.w_b * norm_improve +
            self.w_d * norm_delivered -
            self.w_l * norm_loss
        )
    
    def adapt_K(self, stats: Dict[str, float], network=None, scheduler=None) -> None:
        """
        自适应调整K值
        
        Args:
            stats: 统计信息字典
            network: 网络对象（用于前瞻模拟）
            scheduler: 调度器对象（用于前瞻模拟）
        """
        # 记录调整前的K值
        old_K = self.K
        cur_reward = self._reward(stats)
        prev_reward = self.last_reward
        prev_dir = self.last_direction
        prev_K = self.K
        
        if prev_reward is None:
            self.last_reward = cur_reward
            return

        # 计算改进量
        improve = cur_reward - prev_reward
        step = 1

        if self.use_lookahead:
            try:
                # 尝试使用前瞻模拟
                from scheduling.lookahead import pick_k_via_lookahead
                new_K, _ = pick_k_via_lookahead(
                    network, scheduler,
                    self._adaptk_last_t or 0,
                    self.K, self.last_direction, improve, self.hysteresis, self.K_max,
                    horizon_minutes=60, reward_fn=self._reward
                )
            except Exception:
                # 前瞻模拟失败，使用基础算法
                if improve > self.hysteresis:
                    new_K = self.K + self.last_direction * step
                elif improve < -self.hysteresis:
                    self.last_direction *= -1
                    new_K = self.K + self.last_direction * step
                else:
                    new_K = self.K
        else:
            # 基础自适应算法
            if improve > self.hysteresis:
                new_K = self.K + self.last_direction * step
            elif improve < -self.hysteresis:
                self.last_direction *= -1
                new_K = self.K + self.last_direction * step
            else:
                new_K = self.K

        # 限制K值范围
        self.K = max(1, min(self.K_max, new_K))
        
        # 记录反转后的方向
        dir_new = self.last_direction
        self.last_reward = cur_reward

        # 记录统计信息
        self.K_stats.append({
            'old_K': old_K,
            'new_K': self.K,
            'improve': improve,
            'reward': cur_reward,
            'pre_std': stats["pre_std"],
            'post_std': stats["post_std"],
            'delivered': stats["delivered_total"],
            'loss': stats["total_loss"]
        })
    
    def record_K_value(self, timestamp: datetime) -> None:
        """
        记录当前K值和时间戳
        
        Args:
            timestamp: 当前时间戳
        """
        self.K_history.append(self.K)
        self.K_timestamps.append(timestamp)
    
    def get_K_history(self) -> tuple:
        """
        获取K值历史记录
        
        Returns:
            (K_history, K_timestamps, K_stats) 元组
        """
        return self.K_history, self.K_timestamps, self.K_stats
    
    def reset(self) -> None:
        """重置K值自适应管理器状态"""
        self.K = 1
        self.last_reward = None
        self.last_direction = +1
        self.K_history.clear()
        self.K_timestamps.clear()
        self.K_stats.clear()
        self.max_improve = None
        self.max_delivered = None
        self.max_loss = None
        self._adaptk_last_t = None
