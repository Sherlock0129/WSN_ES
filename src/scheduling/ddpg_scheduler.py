"""
DDPG深度强化学习调度器

将WSN节点间能量共享建模为马尔可夫决策过程（MDP）：
- 状态空间：节点能量、位置、距离等
- 动作空间：传输时长（连续，1-5分钟）
- 奖励函数：能量均衡、网络存活、传输效率
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
from typing import List, Dict, Tuple, Optional

from scheduling.schedulers import BaseScheduler
from routing.energy_transfer_routing import eetor_find_path_adaptive


# ==================== Neural Networks ====================

class Actor(nn.Module):
    """
    Actor网络：策略网络，输出动作（传输时长）
    
    输入：状态向量
    输出：传输时长（连续值，可配置范围）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 action_min=1.0, action_max=10.0):
        super(Actor, self).__init__()
        
        self.action_min = action_min
        self.action_max = action_max
        self.action_range = action_max - action_min
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # He初始化（适合ReLU）
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        # 输出层小权重初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # 使用tanh将输出限制在[-1,1]，然后映射到[action_min, action_max]
        x = torch.tanh(self.fc3(x))
        # 动作范围：[action_min, action_max]分钟（可配置）
        action = self.action_min + (x + 1.0) / 2.0 * self.action_range
        return action


class Critic(nn.Module):
    """
    Critic网络：价值网络，评估状态-动作对的Q值
    
    输入：状态向量 + 动作向量
    输出：Q值（标量）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        
        # 状态处理层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        
        # 状态-动作融合层
        self.fc2 = nn.Linear(hidden_dim + action_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # He初始化（适合ReLU）
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        # 输出层小权重初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        # 将状态特征和动作拼接
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


# ==================== Replay Buffer ====================

class ReplayBuffer:
    """
    经验回放缓冲区
    
    存储 (state, action, reward, next_state, done) 转移样本
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch]).reshape(-1, 1)
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch]).reshape(-1, 1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


# ==================== Ornstein-Uhlenbeck Noise ====================

class OUNoise:
    """
    Ornstein-Uhlenbeck噪声：用于探索
    
    生成时间相关的噪声，有助于连续动作空间的探索
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        """生成噪声样本"""
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state


# ==================== DDPG Agent ====================

class DDPGAgent:
    """
    DDPG智能体
    
    包含：
    - Actor（策略网络）和Target Actor
    - Critic（价值网络）和Target Critic
    - 经验回放缓冲区
    - OU噪声生成器
    """
    def __init__(self, state_dim, action_dim, 
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.001,
                 buffer_capacity=10000,
                 hidden_dim=256,
                 action_min=1.0, action_max=10.0):
        """
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度
        :param actor_lr: Actor学习率
        :param critic_lr: Critic学习率
        :param gamma: 折扣因子
        :param tau: 软更新系数
        :param buffer_capacity: 经验回放缓冲区容量
        :param hidden_dim: 隐藏层维度
        :param action_min: 动作最小值（分钟）
        :param action_max: 动作最大值（分钟）
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DDPG] 使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"[DDPG] GPU设备名称: {torch.cuda.get_device_name(0)}")
            print(f"[DDPG] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self.action_min = action_min
        self.action_max = action_max
        self.gamma = gamma
        self.tau = tau
        
        # Actor网络（传入action范围）
        self.actor = Actor(state_dim, action_dim, hidden_dim, action_min, action_max).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, hidden_dim, action_min, action_max).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # OU噪声
        self.noise = OUNoise(action_dim)
    
    def select_action(self, state, add_noise=True):
        """
        选择动作
        
        :param state: 当前状态
        :param add_noise: 是否添加探索噪声
        :return: 动作（传输时长）
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()
        
        if add_noise:
            noise = self.noise.sample()
            action = action + noise
            # 限制在配置的动作范围内
            action = np.clip(action, self.action_min, self.action_max)
        
        return action
    
    def update(self, batch_size=64):
        """
        更新网络参数
        
        :param batch_size: 批次大小
        """
        if len(self.replay_buffer) < batch_size:
            return None, None
        
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ==================== 更新Critic ====================
        # 计算目标Q值
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 当前Q值
        current_q = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ==================== 更新Actor ====================
        # Actor损失：最大化Q值
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ==================== 软更新目标网络 ====================
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return actor_loss.item(), critic_loss.item()
    
    def _soft_update(self, source, target):
        """
        软更新目标网络
        
        target = tau * source + (1 - tau) * target
        """
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ==================== DDPG Scheduler ====================

class DDPGScheduler(BaseScheduler):
    """
    基于DDPG深度强化学习的能量传输调度器
    
    将WSN能量共享建模为马尔可夫决策过程（MDP）：
    - 状态：节点能量、位置、能量方差等
    - 动作：传输时长（连续，1-5分钟）
    - 奖励：能量均衡改善、传输效率、网络存活
    """
    def __init__(self, node_info_manager, K=2, max_hops=5,
                 state_dim=None, action_dim=1,
                 actor_lr=1e-4, critic_lr=1e-3,
                 gamma=0.99, tau=0.001,
                 buffer_capacity=10000,
                 training_mode=True,
                 action_min=1.0, action_max=10.0):
        """
        :param node_info_manager: 节点信息管理器
        :param K: 每个receiver最多接受的donor数量
        :param max_hops: 最大跳数
        :param state_dim: 状态空间维度（自动计算）
        :param action_dim: 动作空间维度（每个决策输出1个值：传输时长）
        :param actor_lr: Actor学习率
        :param critic_lr: Critic学习率
        :param gamma: 折扣因子
        :param tau: 软更新系数
        :param buffer_capacity: 经验回放缓冲区容量
        :param training_mode: 是否处于训练模式
        :param action_min: 动作最小值（分钟），让DDPG自己探索最优时长
        :param action_max: 动作最大值（分钟），扩大动作空间
        """
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        
        self.training_mode = training_mode
        self.action_min = action_min
        self.action_max = action_max
        
        # 状态维度（将在第一次plan时确定）
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # DDPG智能体（延迟初始化）
        self.agent = None
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_capacity = buffer_capacity
        
        # 历史状态（用于经验回放）
        self.prev_state = None
        self.prev_action = None
        self.prev_energies = None
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        self.actor_losses = []
        self.critic_losses = []
    
    def _compute_state(self, network, t) -> np.ndarray:
        """
        计算当前状态向量
        
        状态包括：
        1. 所有节点的归一化能量
        2. 能量方差（网络均衡性）
        3. 低能量节点比例
        4. 平均能量
        5. 时间步（归一化）
        
        :return: 状态向量
        """
        info_nodes = self.nim.get_info_nodes()
        nodes = self._filter_regular_nodes(info_nodes)
        
        if len(nodes) == 0:
            return np.zeros(self.state_dim or 10)
        
        # 节点能量（归一化）
        energies = np.array([n.current_energy for n in nodes], dtype=float)
        max_energy = 50000.0  # 假设最大能量
        normalized_energies = energies / max_energy
        
        # 能量统计
        mean_energy = np.mean(normalized_energies)
        std_energy = np.std(normalized_energies)
        min_energy = np.min(normalized_energies)
        
        # 低能量节点比例（能量<平均值的50%）
        low_energy_ratio = np.sum(energies < mean_energy * max_energy * 0.5) / len(nodes)
        
        # 时间步（归一化，假设最大10080分钟=7天）
        normalized_time = t / 10080.0
        
        # 构建状态向量
        state = np.concatenate([
            normalized_energies,  # 各节点能量
            [mean_energy],        # 平均能量
            [std_energy],         # 能量标准差
            [min_energy],         # 最小能量
            [low_energy_ratio],   # 低能量节点比例
            [normalized_time]     # 时间步
        ])
        
        return state
    
    def _compute_reward(self, prev_energies, current_energies, plans) -> float:
        """
        计算奖励函数（归一化版本，范围约为-50到+50）
        
        奖励组成：
        1. 能量均衡改善奖励（方差减小）
        2. 传输效率奖励（高效传输）
        3. 低能量节点惩罚（节点濒死）
        4. 网络存活奖励（无节点死亡）
        
        :param prev_energies: 上一步的节点能量
        :param current_energies: 当前节点能量
        :param plans: 传输计划
        :return: 奖励值（归一化到合理范围）
        """
        if prev_energies is None or len(prev_energies) == 0:
            return 0.0
        
        max_energy = 50000.0  # 归一化基准
        
        # 1. 能量均衡奖励（归一化方差）
        prev_std_norm = np.std(prev_energies) / max_energy
        current_std_norm = np.std(current_energies) / max_energy
        balance_reward = (prev_std_norm - current_std_norm) * 10.0  # -10到+10
        
        # 2. 传输效率奖励
        efficiency_reward = 0.0
        if plans:
            total_delivered = sum([p.get('delivered', 0) for p in plans])
            total_loss = sum([p.get('loss', 0) for p in plans])
            if total_delivered + total_loss > 0:
                efficiency = total_delivered / (total_delivered + total_loss)
                efficiency_reward = efficiency * 2.0  # 0到2
        
        # 3. 低能量节点惩罚（归一化）
        mean_energy = np.mean(current_energies)
        low_energy_ratio = np.sum(current_energies < mean_energy * 0.3) / len(current_energies)
        low_energy_penalty = -low_energy_ratio * 5.0  # -5到0
        
        # 4. 死亡节点惩罚（严重惩罚）
        dead_ratio = np.sum(current_energies <= 0) / len(current_energies)
        death_penalty = -dead_ratio * 20.0  # -20到0
        
        # 5. 网络存活奖励
        survival_reward = 1.0 if dead_ratio == 0 else 0.0
        
        # 6. 最小能量奖励（鼓励提升最弱节点）
        min_energy_norm = np.min(current_energies) / max_energy
        min_energy_reward = min_energy_norm * 2.0  # 0到2
        
        # 总奖励（范围约为-30到+15）
        total_reward = (balance_reward + efficiency_reward + 
                       low_energy_penalty + death_penalty + 
                       survival_reward + min_energy_reward)
        
        # 裁剪到合理范围
        total_reward = np.clip(total_reward, -50.0, 50.0)
        
        return total_reward
    
    def _path_eta(self, path):
        """计算路径总效率"""
        eta = 1.0
        for i in range(len(path) - 1):
            eta *= path[i].energy_transfer_efficiency(path[i + 1])
        return max(1e-6, min(1.0, eta))
    
    def plan(self, network, t):
        """
        规划能量传输（使用DDPG策略）
        
        :param network: 网络对象
        :param t: 当前时间步
        :return: (plans, candidates)
        """
        # 计算当前状态
        current_state = self._compute_state(network, t)
        
        # 初始化DDPG智能体（第一次调用时）
        if self.agent is None:
            self.state_dim = len(current_state)
            self.agent = DDPGAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                actor_lr=self.actor_lr,
                critic_lr=self.critic_lr,
                gamma=self.gamma,
                tau=self.tau,
                buffer_capacity=self.buffer_capacity,
                action_min=self.action_min,
                action_max=self.action_max
            )
            print(f"[DDPG] 动作空间范围: [{self.action_min:.1f}, {self.action_max:.1f}] 分钟")
        
        # 获取节点信息
        info_nodes = self.nim.get_info_nodes()
        id2node = {n.node_id: n for n in network.nodes}
        nodes = self._filter_regular_nodes(info_nodes)
        
        if len(nodes) == 0:
            return [], []
        
        # 计算能量统计
        E = np.array([n.current_energy for n in nodes], dtype=float)
        E_bar = float(E.mean())
        Q = dict((n, max(0.0, E_bar - n.current_energy)) for n in nodes)
        
        # 筛选donors和receivers
        receivers = sorted([n for n in nodes if Q[n] > 0], key=lambda x: Q[x], reverse=True)
        donors = [n for n in nodes if n.current_energy > E_bar]
        
        # 记录当前能量（用于计算奖励）
        current_energies = np.array([node.current_energy for node in network.nodes[1:]], dtype=float)
        
        # ==================== 经验回放 ====================
        if self.prev_state is not None and self.training_mode:
            # 计算奖励
            prev_energies = self.prev_energies
            reward = self._compute_reward(prev_energies, current_energies, None)
            
            # 添加到经验回放
            done = False  # 通常每步不是终止状态
            self.agent.replay_buffer.push(
                self.prev_state, self.prev_action, reward, current_state, done
            )
            
            # 更新网络
            if self.step_count % 4 == 0:  # 每4步更新一次
                actor_loss, critic_loss = self.agent.update(batch_size=64)
                if actor_loss is not None:
                    self.actor_losses.append(actor_loss)
                    self.critic_losses.append(critic_loss)
        
        # 保存当前状态
        self.prev_state = current_state
        self.prev_energies = current_energies
        self.step_count += 1
        
        # ==================== 生成传输计划 ====================
        used = set()
        plans = []
        all_candidates = []
        
        for r in receivers:
            cand = []
            for d in donors:
                if d in used or d is r:
                    continue
                
                dist = r.distance_to(d)
                
                # 使用自适应路径查找
                path = eetor_find_path_adaptive(nodes, d, r, max_hops=self.max_hops,
                                                 node_info_manager=self.nim)
                if path is None:
                    continue
                
                eta = self._path_eta(path)
                
                # 效率低于10%的传输直接放弃
                if eta < 0.1:
                    continue
                
                # 使用DDPG策略选择传输时长（自动探索最优值）
                # 这里简化：对每个donor-receiver对使用相同的策略
                duration_action = self.agent.select_action(current_state, add_noise=self.training_mode)
                duration = float(duration_action[0])  # 取第一个动作值
                # 动作已在Actor网络中限制到[action_min, action_max]
                # 四舍五入到整数或保留小数（根据需要）
                duration = max(1.0, min(self.action_max, duration))  # 安全限制
                
                # 计算能量传输
                E_char = getattr(d, "E_char", 300.0)
                energy_sent_total = duration * E_char
                energy_delivered = energy_sent_total * eta
                energy_loss = energy_sent_total - energy_delivered
                
                # Lyapunov得分（用于排序）
                Q_normalized = Q[r] / E_bar if E_bar > 0 else 0
                score = energy_delivered * Q_normalized - 0.5 * energy_loss
                
                cand.append((score, d, r, path, dist, energy_delivered, energy_loss, duration))
            
            if not cand:
                continue
            
            cand.sort(key=lambda x: x[0], reverse=True)
            all_candidates.extend(cand)
            quota = self.K
            
            for sc, d, rr, path, dist, delivered, loss, duration in cand:
                if quota <= 0:
                    break
                
                # 转换回真实节点对象
                receiver = id2node[rr.node_id]
                donor = id2node[d.node_id]
                real_path = [id2node[n.node_id] for n in path]
                
                plans.append({
                    "receiver": receiver,
                    "donor": donor,
                    "path": real_path,
                    "distance": dist,
                    "delivered": delivered,
                    "loss": loss,
                    "duration": duration,  # DDPG决策的传输时长
                    "score": sc
                })
                used.add(d)
                quota -= 1
        
        # 保存当前动作（用于经验回放）
        if plans:
            # 简化：使用第一个计划的时长作为代表
            self.prev_action = np.array([plans[0]['duration']], dtype=float)
        else:
            self.prev_action = np.array([1.0], dtype=float)
        
        return plans, all_candidates
    
    def save_model(self, filepath):
        """保存DDPG模型"""
        if self.agent is not None:
            self.agent.save(filepath)
            print(f"[DDPG] 模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载DDPG模型"""
        if self.agent is not None:
            self.agent.load(filepath)
            print(f"[DDPG] 模型已加载: {filepath}")
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'buffer_size': len(self.agent.replay_buffer) if self.agent else 0
        }

