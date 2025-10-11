import math


class RewardFunction(object):
    def compute(self, stats):
        """
        计算单步奖励。
        期望输入 stats 字段:
          - pre_std, post_std, delivered_total, total_loss
        返回 float。
        """
        raise NotImplementedError


class WeightedReward(RewardFunction):
    def __init__(self, w_b=0.8, w_d=0.8, w_l=1.5):
        self.w_b = float(w_b)
        self.w_d = float(w_d)
        self.w_l = float(w_l)
        # 归一化跟踪值，避免除零
        self._max_improve = 1e-9
        self._max_delivered = 1e-9
        self._max_loss = 1e-9

    def compute(self, stats):
        improve = float(stats.get("pre_std", 0.0)) - float(stats.get("post_std", 0.0))
        delivered = float(stats.get("delivered_total", 0.0))
        loss = float(stats.get("total_loss", 0.0))

        # 更新归一化上界
        self._max_improve = max(self._max_improve, abs(improve))
        self._max_delivered = max(self._max_delivered, delivered)
        self._max_loss = max(self._max_loss, loss)

        # 归一化
        norm_improve = improve / self._max_improve
        norm_delivered = delivered / self._max_delivered
        norm_loss = loss / self._max_loss

        return (self.w_b * norm_improve) + (self.w_d * norm_delivered) - (self.w_l * norm_loss)


