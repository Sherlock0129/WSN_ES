from wsn_et.dynamic_k.reward import RewardFunction

try:
    # 可选：仅在使用前瞻策略时需要
    from wsn_et.dynamic_k.lookahead import pick_k_via_lookahead
except Exception:
    pick_k_via_lookahead = None


class AdaptKPolicy(object):
    def reset(self, K_init=1):
        raise NotImplementedError

    def update(self, prev_K, stats, t, network=None, scheduler=None):
        """
        返回新的 dynamic_k 值（已裁剪到合法范围）。
        """
        raise NotImplementedError


class HysteresisAdaptK(AdaptKPolicy):
    def __init__(self, reward_fn, K_max=24, hysteresis=0.2, step=1):
        if not isinstance(reward_fn, RewardFunction):
            # 允许鸭子类型：只要有 compute(stats) 即可
            pass
        self.reward_fn = reward_fn
        self.K_max = int(K_max)
        self.hysteresis = float(hysteresis)
        self.step = int(step)
        self._last_reward = None
        self._direction = +1

    def reset(self, K_init=1):
        self._last_reward = None
        self._direction = +1

    def update(self, prev_K, stats, t, network=None, scheduler=None):
        cur = float(self.reward_fn.compute(stats))
        if self._last_reward is None:
            self._last_reward = cur
            return int(prev_K)
        improve = cur - self._last_reward
        new_K = int(prev_K)
        if improve > self.hysteresis:
            new_K = prev_K + self._direction * self.step
        elif improve < -self.hysteresis:
            self._direction *= -1
            new_K = prev_K + self._direction * self.step
        self._last_reward = cur
        if new_K < 1:
            new_K = 1
        if new_K > self.K_max:
            new_K = self.K_max
        return int(new_K)


class LookaheadAdaptK(AdaptKPolicy):
    def __init__(self, reward_fn, K_max=24, hysteresis=0.2, horizon_minutes=60):
        self.reward_fn = reward_fn
        self.K_max = int(K_max)
        self.hysteresis = float(hysteresis)
        self.horizon_minutes = int(horizon_minutes)
        self._last_reward = None
        self._direction = +1
        self._last_t = 0

    def reset(self, K_init=1):
        self._last_reward = None
        self._direction = +1
        self._last_t = 0

    def update(self, prev_K, stats, t, network=None, scheduler=None):
        cur = float(self.reward_fn.compute(stats))
        if self._last_reward is None:
            self._last_reward = cur
            self._last_t = t
            return int(prev_K)
        improve = cur - self._last_reward

        new_K = int(prev_K)
        used_lookahead = False
        if pick_k_via_lookahead is not None and (network is not None) and (scheduler is not None):
            try:
                new_K, _ = pick_k_via_lookahead(
                    network, scheduler, self._last_t, prev_K, self._direction,
                    improve, self.hysteresis, self.K_max,
                    horizon_minutes=self.horizon_minutes, reward_fn=self.reward_fn.compute
                )
                used_lookahead = True
            except Exception:
                used_lookahead = False

        if not used_lookahead:
            if improve > self.hysteresis:
                new_K = prev_K + self._direction
            elif improve < -self.hysteresis:
                self._direction *= -1
                new_K = prev_K + self._direction

        self._last_t = t
        self._last_reward = cur
        if new_K < 1:
            new_K = 1
        if new_K > self.K_max:
            new_K = self.K_max
        return int(new_K)


