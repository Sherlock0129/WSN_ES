"""
节点对帕累托边界可视化工具

从节点对视角理解帕累托边界：横轴为节点a的综合效能，纵轴为节点b的综合效能。
展示不同机制下，节点间资源分配的权衡与边界外移。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import os


def compute_node_utility(node_data: Dict, weights: Dict[str, float] = None) -> float:
    """
    计算节点综合效能
    
    Args:
        node_data: 节点数据字典，包含：
            - final_energy: 最终能量
            - initial_energy: 初始能量
            - received_count: 被服务次数（或从received_history长度计算）
            - max_possible_services: 最大可能服务次数（总时间步数）
            - avg_aoei: 平均AOEI值
            - max_aoei: 最大AOEI值（用于归一化）
            - is_alive: 是否存活
        weights: 权重字典 {'energy': w1, 'service': w2, 'freshness': w3, 'survival': w4}
    
    Returns:
        节点综合效能值 (0~1)
    """
    if weights is None:
        weights = {'energy': 0.4, 'service': 0.3, 'freshness': 0.2, 'survival': 0.1}
    
    # 能量状态比
    energy_ratio = node_data.get('final_energy', 0) / max(node_data.get('initial_energy', 1), 1)
    energy_ratio = min(1.0, max(0.0, energy_ratio))
    
    # 服务频度比
    received_count = node_data.get('received_count', 0)
    max_services = node_data.get('max_possible_services', 1)
    service_ratio = received_count / max(max_services, 1)
    service_ratio = min(1.0, max(0.0, service_ratio))
    
    # 信息新鲜度（AOEI越小越新鲜，归一化到[0,1]）
    avg_aoei = node_data.get('avg_aoei', 0)
    max_aoei = node_data.get('max_aoei', 100)  # 默认最大值
    freshness = 1.0 - min(1.0, avg_aoei / max(max_aoei, 1))
    
    # 生存状态
    survival = 1.0 if node_data.get('is_alive', True) else 0.0
    
    # 加权综合
    utility = (weights['energy'] * energy_ratio +
               weights['service'] * service_ratio +
               weights['freshness'] * freshness +
               weights['survival'] * survival)
    
    return utility


def extract_node_data_from_simulation(simulation_dir: str, node_id: int) -> Dict:
    """
    从仿真结果目录中提取节点数据
    
    Args:
        simulation_dir: 仿真结果目录（包含simulation_statistics.txt或JSON）
        node_id: 节点ID
    
    Returns:
        节点数据字典
    """
    node_data = {
        'node_id': node_id,
        'initial_energy': 0,
        'final_energy': 0,
        'received_count': 0,
        'avg_aoei': 0,
        'is_alive': True
    }
    
    # 尝试从JSON文件读取
    json_path = os.path.join(simulation_dir, 'simulation_data.json')
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            nodes_data = data.get('nodes_data', [])
            for node in nodes_data:
                if node.get('node_id') == node_id:
                    node_data['initial_energy'] = node.get('initial_energy', 0)
                    node_data['final_energy'] = node.get('final_energy', 0)
                    # 从received_history长度估算服务次数
                    received_history = node.get('received_history', [])
                    node_data['received_count'] = len([e for e in received_history if e > 0])
                    break
    
    # 尝试从统计文件读取
    stats_path = os.path.join(simulation_dir, 'simulation_statistics.txt')
    if os.path.exists(stats_path):
        # 这里可以解析文本文件（如果需要）
        pass
    
    # 设置默认值
    if node_data['initial_energy'] == 0:
        node_data['initial_energy'] = 40000.0  # 默认初始能量
    
    node_data['max_possible_services'] = 1000  # 默认最大服务次数（可从配置读取）
    node_data['max_aoei'] = 100.0  # 默认最大AOEI
    
    return node_data


def pareto_frontier_2d(points: np.ndarray) -> np.ndarray:
    """
    计算二维帕累托前沿（两轴均"越大越好"）
    
    Args:
        points: shape (N, 2) 的数组，每行为 (x, y)
    
    Returns:
        前沿点数组，按x升序排列
    """
    if len(points) == 0:
        return np.array([])
    
    # 按x升序排序
    sorted_indices = np.argsort(points[:, 0])
    sorted_points = points[sorted_indices]
    
    frontier = []
    best_y = -np.inf
    
    for x, y in sorted_points:
        if y > best_y:
            frontier.append([x, y])
            best_y = y
    
    return np.array(frontier) if frontier else np.array([])


def plot_node_pair_pareto(
    results_data: List[Dict],
    node_a_id: int,
    node_b_id: int,
    methods: List[str] = None,
    output_path: str = "node_pair_pareto.pdf",
    weights: Dict[str, float] = None
):
    """
    绘制节点对帕累托边界图
    
    Args:
        results_data: 结果数据列表，每个元素为：
            {
                'method': 'baseline_name',
                'simulation_dir': 'path/to/simulation',
                'node_data': {node_id: node_data_dict, ...}  # 可选，若提供则直接使用
            }
        node_a_id: 节点a的ID（横轴）
        node_b_id: 节点b的ID（纵轴）
        methods: 方法名称列表（用于图例）
        output_path: 输出文件路径
        weights: 节点效能权重
    """
    plt.figure(figsize=(8, 6))
    
    if methods is None:
        methods = [r.get('method', f'method_{i}') for i, r in enumerate(results_data)]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    markers = ['o', 's', '^', 'd', 'v', 'P', 'X', '*']
    
    all_points = []  # 用于计算全体前沿
    
    for idx, result in enumerate(results_data):
        method = result.get('method', f'method_{idx}')
        sim_dir = result.get('simulation_dir', '')
        
        # 提取节点数据
        if 'node_data' in result:
            node_a_data = result['node_data'].get(node_a_id, {})
            node_b_data = result['node_data'].get(node_b_id, {})
        else:
            node_a_data = extract_node_data_from_simulation(sim_dir, node_a_id)
            node_b_data = extract_node_data_from_simulation(sim_dir, node_b_id)
        
        # 计算效能
        utility_a = compute_node_utility(node_a_data, weights)
        utility_b = compute_node_utility(node_b_data, weights)
        
        # 绘制散点
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        plt.scatter(utility_a, utility_b, s=100, alpha=0.7, 
                   label=method, color=color, marker=marker, edgecolors='black', linewidths=1)
        
        all_points.append([utility_a, utility_b])
    
    # 计算并绘制全体前沿
    if len(all_points) > 0:
        all_points_array = np.array(all_points)
        frontier = pareto_frontier_2d(all_points_array)
        if len(frontier) >= 2:
            plt.plot(frontier[:, 0], frontier[:, 1], 
                    color='crimson', linewidth=2.5, linestyle='--', 
                    alpha=0.8, label='Pareto Frontier (All)')
    
    plt.xlabel(f'节点 {node_a_id} 综合效能 $U_a$', fontsize=12)
    plt.ylabel(f'节点 {node_b_id} 综合效能 $U_b$', fontsize=12)
    plt.title(f'节点对帕累托边界：节点 {node_a_id} vs 节点 {node_b_id}', fontsize=14)
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"节点对帕累托图已保存到: {output_path}")
    plt.close()


def batch_plot_node_pairs(
    results_data: List[Dict],
    node_pairs: List[Tuple[int, int]],
    methods: List[str] = None,
    output_dir: str = "paper/figures",
    weights: Dict[str, float] = None
):
    """
    批量绘制多个节点对的帕累托图
    
    Args:
        results_data: 结果数据列表
        node_pairs: 节点对列表，如 [(1, 2), (3, 4), (5, 6)]
        methods: 方法名称列表
        output_dir: 输出目录
        weights: 节点效能权重
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for node_a, node_b in node_pairs:
        output_path = os.path.join(output_dir, f"node_pair_pareto_{node_a}_{node_b}.pdf")
        plot_node_pair_pareto(
            results_data, node_a, node_b, methods, output_path, weights
        )


# 使用示例
if __name__ == "__main__":
    # 示例：从多个仿真结果中提取节点对数据
    results_data = [
        {
            'method': '无共享',
            'simulation_dir': 'data/20251111_233904',
            # 或直接提供节点数据：
            # 'node_data': {
            #     1: {'final_energy': 35000, 'initial_energy': 40000, ...},
            #     2: {'final_energy': 38000, 'initial_energy': 40000, ...}
            # }
        },
        {
            'method': '本文机制',
            'simulation_dir': 'data/20251112_001629',
        }
    ]
    
    # 选择代表性的节点对（例如：低能节点 vs 高能节点，或太阳能节点 vs 普通节点）
    node_pairs = [(1, 2), (3, 4)]  # 根据实际网络选择
    
    batch_plot_node_pairs(results_data, node_pairs, output_dir="paper/figures")

