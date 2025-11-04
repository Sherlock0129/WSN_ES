"""
DQN深度强化学习调度器（离散动作空间）

将WSN节点间能量共享建模为马尔可夫决策过程（MDP）：
- 状态空间：节点能量、位置、距离等
- 动作空间：传输时长（离散，1-10分钟）
- 奖励函数：能量均衡、网络存活、传输效率

优势：
- 离散动作更容易训练
- 计算效率更高
- 收敛更快更稳定
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


# ==================== Q-Network ====================

class QNetwork(nn.Module):
    """
    Q网络：估计每个动作的Q值
    
    输入：状态向量
    输出：10个动作的Q值（对应1-10分钟）
    """
    def __init__(self, state_dim, action_dim=10, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        
        # He初始化（适合ReLU激活）
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        # 输出层使用小权重初始化
        nn.init.uniform_(self.fc4.weight, -3e-3, 3e-3)
        nn.init.constant_(self.fc4.bias, 0)
    
    def forward(self, state):
        """
        前向传播
        
        :param state: 状态向量 (batch_size, state_dim)
        :return: Q值向量 (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_values = self.fc4(x)
        return q_values


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


# ==================== DQN Agent ====================

class DQNAgent:
    """
    DQN智能体
    
    包含：
    - Q网络和Target Q网络
    - 经验回放缓冲区
    - ε-greedy探索策略
    """
    def __init__(self, state_dim, action_dim=10,
                 lr=1e-3, gamma=0.99, tau=0.005,
                 buffer_capacity=10000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 hidden_dim=256):
        """
        :param state_dim: 状态空间维度
        :param action_dim: 动作空间维度（10个离散动作：1-10分钟）
        :param lr: 学习率
        :param gamma: 折扣因子
        :param tau: 软更新系数
        :param buffer_capacity: 经验回放缓冲区容量
        :param epsilon_start: 初始探索率
        :param epsilon_end: 最终探索率
        :param epsilon_decay: 探索率衰减
        :param hidden_dim: 隐藏层维度
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DQN] 使用设备: {self.device}")
        if torch.cuda.is_available():
            print(f"[DQN] GPU设备名称: {torch.cuda.get_device_name(0)}")
            print(f"[DQN] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        
        # ε-greedy探索参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Q网络
        self.q_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        # 训练统计
        self.update_count = 0
    
    def select_action(self, state, training=True):
        """
        选择动作（ε-greedy策略）
        
        :param state: 当前状态
        :param training: 是否处于训练模式
        :return: 动作索引（0-9，对应1-10分钟）
        """
        # ε-greedy探索
        if training and random.random() < self.epsilon:
            # 随机探索
            action = random.randint(0, self.action_dim - 1)
        else:
            # 贪婪选择
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                action = q_values.argmax(dim=1).item()
        
        return action
    
    def update(self, batch_size=64):
        """
        更新Q网络
        
        :param batch_size: 批次大小
        :return: 损失值
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 从经验回放缓冲区采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # 计算当前Q值
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # 计算目标Q值（Double DQN）
        with torch.no_grad():
            # 使用Q网络选择动作
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            # 使用Target网络评估Q值
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 计算损失
        loss = F.mse_loss(current_q, target_q)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # 软更新目标网络
        self._soft_update()
        
        # 更新计数
        self.update_count += 1
        
        # 衰减探索率
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def _soft_update(self):
        """
        软更新目标网络
        
        target = tau * q_network + (1 - tau) * target
        """
        for target_param, param in zip(self.target_network.parameters(), 
                                       self.q_network.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)


# ==================== DQN Scheduler ====================

class DQNScheduler(BaseScheduler):
    """
    基于DQN深度强化学习的能量传输调度器（离散动作空间）
    
    将WSN能量共享建模为马尔可夫决策过程（MDP）：
    - 状态：节点能量、位置、能量方差等
    - 动作：传输时长（离散，1-10分钟，共10个动作）
    - 奖励：能量均衡改善、传输效率、网络存活
    
    优势：
    - 离散动作更容易训练
    - 计算效率高（比DDPG快）
    - 收敛更快更稳定
    """
    def __init__(self, node_info_manager, K=2, max_hops=5,
                 action_dim=10,  # 10个离散动作：1-10分钟
                 lr=1e-3, gamma=0.99, tau=0.005,
                 buffer_capacity=10000,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 training_mode=True):
        """
        :param node_info_manager: 节点信息管理器
        :param K: 每个receiver最多接受的donor数量
        :param max_hops: 最大跳数
        :param action_dim: 动作空间维度（10 = 1-10分钟）
        :param lr: 学习率
        :param gamma: 折扣因子
        :param tau: 软更新系数
        :param buffer_capacity: 经验回放缓冲区容量
        :param epsilon_start: 初始探索率
        :param epsilon_end: 最终探索率
        :param epsilon_decay: 探索率衰减
        :param training_mode: 是否处于训练模式
        """
        BaseScheduler.__init__(self, node_info_manager, K, max_hops)
        
        self.training_mode = training_mode
        self.action_dim = action_dim  # 10个离散动作
        
        # 状态维度（将在第一次plan时确定）
        self.state_dim = None
        
        # DQN智能体（延迟初始化）
        self.agent = None
        self.lr = lr
        self.gamma = gamma
        self.tau = tau
        self.buffer_capacity = buffer_capacity
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 历史状态（用于经验回放）
        self.prev_state = None
        self.prev_action = None
        self.prev_energies = None
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        self.losses = []
    
    def _compute_state(self, network, t) -> np.ndarray:
        """
        计算当前状态向量
        
        状态包括：
        1. 所有节点的归一化能量
        2. 能量方差（网络均衡性）
        3. 低能量节点比例
        4. 平均能量
        5. 最小能量
        6. 时间步（归一化）
        
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
        normalized_time = min(t / 10080.0, 1.0)
        
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
        计算奖励函数（归一化版本，范围约为-10到+10）
        
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
        规划能量传输（使用DQN策略）
        
        :param network: 网络对象
        :param t: 当前时间步
        :return: (plans, candidates)
        """
        # 计算当前状态
        current_state = self._compute_state(network, t)
        
        # 初始化DQN智能体（第一次调用时）
        if self.agent is None:
            self.state_dim = len(current_state)
            self.agent = DQNAgent(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                lr=self.lr,
                gamma=self.gamma,
                tau=self.tau,
                buffer_capacity=self.buffer_capacity,
                epsilon_start=self.epsilon_start,
                epsilon_end=self.epsilon_end,
                epsilon_decay=self.epsilon_decay
            )
        
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
            reward = self._compute_reward(self.prev_energies, current_energies, None)
            
            # 添加到经验回放
            done = False  # 通常每步不是终止状态
            self.agent.replay_buffer.push(
                self.prev_state, self.prev_action, reward, current_state, done
            )
            
            # 更新网络
            if self.step_count % 4 == 0:  # 每4步更新一次
                loss = self.agent.update(batch_size=64)
                if loss is not None:
                    self.losses.append(loss)
        
        # 保存当前状态
        self.prev_state = current_state
        self.prev_energies = current_energies
        self.step_count += 1
        
        # ==================== 使用DQN选择动作 ====================
        action = self.agent.select_action(current_state, training=self.training_mode)
        duration = action + 1  # 动作0-9对应时长1-10分钟
        
        # 保存当前动作
        self.prev_action = action
        
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
                
                # 使用DQN选择的时长
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
            
            for sc, d, rr, path, dist, delivered, loss, dur in cand:
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
                    "duration": dur,  # DQN决策的传输时长
                    "score": sc
                })
                used.add(d)
                quota -= 1
        
        return plans, all_candidates
    
    def save_model(self, filepath):
        """保存DQN模型"""
        if self.agent is not None:
            self.agent.save(filepath)
            print(f"[DQN] 模型已保存到: {filepath}")
    
    def load_model(self, filepath):
        """加载DQN模型"""
        if self.agent is not None:
            self.agent.load(filepath)
            print(f"[DQN] 模型已加载: {filepath}")
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return {
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'buffer_size': len(self.agent.replay_buffer) if self.agent else 0,
            'epsilon': self.agent.epsilon if self.agent else 0,
            'update_count': self.agent.update_count if self.agent else 0
        }

