import copy
import numpy as np

def _compute_stats_for_network(net, plans, pre_energies, pre_received_total, pre_transferred_total):
    post_energies = np.array([n.current_energy for n in net.nodes], dtype=float)
    pre_std = float(np.std(pre_energies))
    post_std = float(np.std(post_energies))
    # 实际发送量：从transferred_history获取（与主流程保持一致）
    post_transferred_total = sum(sum(n.transferred_history) for n in net.nodes)
    sent_total = max(0.0, post_transferred_total - pre_transferred_total)
    post_received_total = sum(sum(n.received_history) for n in net.nodes)
    delivered_total = max(0.0, post_received_total - pre_received_total)
    total_loss = max(0.0, sent_total - delivered_total)
    return pre_std, post_std, delivered_total, total_loss

def _eval_one_candidate(network, scheduler, K_value, t, horizon_minutes, reward_fn):
    # 深拷贝现场，避免副作用
    net_copy = copy.deepcopy(network)
    sched_copy = None
    if scheduler is not None:
        try:
            sched_copy = copy.deepcopy(scheduler)
        except Exception:
            sched_copy = scheduler  # 回退到共享引用（尽量避免随机副作用）

    # 先推进未来 horizon_minutes 分钟的能量演化，不做传能
    for m in range(1, horizon_minutes + 1):
        net_copy.update_network_energy(t + m)

    # 到 t+horizon 进行一次传能评估
    pre_energies = np.array([n.current_energy for n in net_copy.nodes], dtype=float)
    pre_received_total = sum(sum(n.received_history) for n in net_copy.nodes)
    pre_transferred_total = sum(sum(n.transferred_history) for n in net_copy.nodes)

    # 计划与执行（与主流程口径一致）
    if sched_copy is not None:
        if hasattr(sched_copy, "K"):
            try:
                sched_copy.K = K_value
            except Exception:
                pass
        result = sched_copy.plan(net_copy, t + horizon_minutes)
        if isinstance(result, tuple):
            plans, cand = result
        else:
            plans = result
            cand = []
        try:
            from .schedulers import PowerControlScheduler
            if isinstance(sched_copy, PowerControlScheduler):
                sched_copy.execute_plans(net_copy, plans)
            else:
                net_copy.execute_energy_transfer(plans)
        except Exception:
            net_copy.execute_energy_transfer(plans)
    else:
        plans = net_copy.run_routing(t + horizon_minutes, max_donors_per_receiver=K_value)
        net_copy.execute_energy_transfer(plans)

    pre_std, post_std, delivered_total, total_loss = _compute_stats_for_network(
        net_copy, plans, pre_energies, pre_received_total, pre_transferred_total
    )
    stats = {
        "pre_std": pre_std,
        "post_std": post_std,
        "delivered_total": delivered_total,
        "total_loss": total_loss,
    }
    reward = float(reward_fn(stats))
    return reward

def pick_k_via_lookahead(network, scheduler, t, current_K, direction, improve, hysteresis, K_max,
                         horizon_minutes=60, reward_fn=lambda s: 0.0):
    # 基于 improve 与滞回判定方向
    base_dir = 1 if direction >= 0 else -1
    if improve > hysteresis:
        d = base_dir
        candidates = [current_K, current_K + d, current_K + 2 * d, current_K + 3 * d]
    elif improve < -hysteresis:
        d = -base_dir
        candidates = [current_K, current_K + d, current_K + 2 * d, current_K + 3 * d]
    else:
        d = base_dir
        candidates = [current_K, current_K + d, current_K + 2 * d, current_K - d, current_K - 2 * d]
    # 去重并裁剪到区间 [1, K_max]
    candidates = sorted({max(1, min(K_max, k)) for k in candidates})

    best_k, best_reward = current_K, -1e18
    for k in candidates:
        try:
            r = _eval_one_candidate(network, scheduler, k, t, horizon_minutes, reward_fn)
        except Exception:
            r = -1e18
        if r > best_reward:
            best_reward = r
            best_k = k
    return best_k, best_reward


