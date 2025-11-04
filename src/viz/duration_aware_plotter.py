"""
传输时长感知的可视化模块
为DurationAwareLyapunovScheduler提供专门的可视化功能，体现持续传输的过程
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any
from utils.output_manager import OutputManager


def plot_energy_transfer_timeline(plans, network, t, session_dir):
    """
    绘制能量传输时间线图，体现持续传输的过程
    
    对于每个传输计划，显示：
    - 开始时间：t
    - 结束时间：t + duration
    - 传输路径：donor -> ... -> receiver
    - 每分钟传输的能量
    
    Args:
        plans: 传输计划列表（包含duration字段）
        network: 网络对象
        t: 当前时间步
        session_dir: 输出目录
    """
    if not plans:
        print("[可视化] 无传输计划，跳过时间线图")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    # ===== 上图：传输时间线（甘特图） =====
    colors = plt.cm.tab20(np.linspace(0, 1, len(plans)))
    
    for idx, plan in enumerate(plans):
        donor = plan['donor']
        receiver = plan['receiver']
        duration = plan.get('duration', 1)
        delivered = plan.get('delivered', 0)
        
        # 绘制时间条
        start_time = t
        end_time = t + duration
        
        ax1.barh(idx, duration, left=start_time, height=0.8, 
                color=colors[idx], alpha=0.7, edgecolor='black')
        
        # 标注传输信息
        label = f"D{donor.node_id}→R{receiver.node_id}: {delivered:.0f}J"
        ax1.text(start_time + duration/2, idx, label, 
                ha='center', va='center', fontsize=9, weight='bold')
        
        # 标注每分钟传输量
        E_char = getattr(donor, "E_char", 300.0)
        ax1.text(start_time - 0.5, idx, f"{E_char:.0f}J/min", 
                ha='right', va='center', fontsize=8, style='italic')
    
    ax1.set_xlabel('Time (minutes)', fontsize=12)
    ax1.set_ylabel('Transfer Index', fontsize=12)
    ax1.set_title(f'Energy Transfer Timeline at t={t} (Duration-Aware)', fontsize=14, weight='bold')
    ax1.set_yticks(range(len(plans)))
    ax1.set_yticklabels([f"Transfer {i+1}" for i in range(len(plans))])
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.axvline(t, color='red', linestyle='--', linewidth=2, label=f'Current time: t={t}')
    ax1.legend()
    
    # ===== 下图：每分钟能量流动 =====
    # 统计每分钟的总能量流动
    max_time = t + max(plan.get('duration', 1) for plan in plans)
    time_range = range(t, max_time + 1)
    energy_flow = {time: 0 for time in time_range}
    
    for plan in plans:
        donor = plan['donor']
        duration = plan.get('duration', 1)
        E_char = getattr(donor, "E_char", 300.0)
        
        for dt in range(duration):
            energy_flow[t + dt] += E_char
    
    times = list(energy_flow.keys())
    flows = list(energy_flow.values())
    
    ax2.bar(times, flows, width=0.8, color='steelblue', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    ax2.set_ylabel('Total Energy Flow (J/min)', fontsize=12)
    ax2.set_title('Total Energy Flow per Minute', fontsize=14, weight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axvline(t, color='red', linestyle='--', linewidth=2, label=f'Current time: t={t}')
    ax2.legend()
    
    # 在每个柱子上标注数值
    for time, flow in zip(times, flows):
        if flow > 0:
            ax2.text(time, flow + max(flows)*0.02, f"{flow:.0f}J", 
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    save_path = OutputManager.get_file_path(session_dir, f'transfer_timeline_t{t}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 传输时间线图已保存: {save_path}")


def plot_energy_transfer_with_duration(plans, network, t, session_dir):
    """
    绘制带传输时长标注的能量传输路径图
    
    在路径上显示：
    - 箭头：传输方向
    - 线宽：传输能量大小
    - 标签：传输时长（如"5min"）
    - 颜色：不同receiver用不同颜色
    
    Args:
        plans: 传输计划列表
        network: 网络对象
        t: 当前时间步
        session_dir: 输出目录
    """
    if not plans:
        print("[可视化] 无传输计划，跳过路径图")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 绘制所有节点
    xs = [n.position[0] for n in network.nodes]
    ys = [n.position[1] for n in network.nodes]
    ids = [n.node_id for n in network.nodes]
    
    # 节点大小根据能量显示
    energies = [n.current_energy for n in network.nodes]
    max_energy = max(energies) if energies else 1
    sizes = [50 + 200 * (e / max_energy) for e in energies]
    
    scatter = ax.scatter(xs, ys, c=energies, s=sizes, cmap='YlOrRd', 
                        edgecolors='black', linewidths=1.5, zorder=3, alpha=0.8)
    
    # 添加节点ID标签
    for x, y, nid in zip(xs, ys, ids):
        ax.text(x, y, str(nid), fontsize=9, ha='center', va='center', 
               weight='bold', color='white' if nid == 0 else 'black')
    
    # 为不同receiver分配颜色
    import itertools
    color_cycle = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                                   '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
    receiver_to_color = {}
    
    # 绘制传输路径
    for plan in plans:
        receiver = plan['receiver']
        donor = plan['donor']
        path = plan.get('path', [])
        duration = plan.get('duration', 1)
        delivered = plan.get('delivered', 0)
        
        if len(path) < 2:
            continue
        
        # 为receiver分配颜色
        rid = receiver.node_id
        if rid not in receiver_to_color:
            receiver_to_color[rid] = next(color_cycle)
        color = receiver_to_color[rid]
        
        # 线宽根据传输能量
        E_char = getattr(donor, "E_char", 300.0)
        total_energy = duration * E_char
        linewidth = 2.0 + 4.0 * (duration / 5.0)  # duration越大，线越粗
        
        # 逐段绘制路径
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]
            
            ax, ay = node_a.position[0], node_a.position[1]
            bx, by = node_b.position[0], node_b.position[1]
            
            # 绘制箭头
            arrow = FancyArrowPatch((ax, ay), (bx, by), 
                                   arrowstyle='-|>', mutation_scale=15,
                                   linewidth=linewidth, color=color, 
                                   alpha=0.6, zorder=2)
            ax.add_patch(arrow)
            
            # 在第一段路径上标注传输时长
            if i == 0:
                mid_x = (ax + bx) / 2
                mid_y = (ay + by) / 2
                
                # 背景框
                bbox_props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                                 edgecolor=color, linewidth=2, alpha=0.9)
                
                # 标注文字
                label = f"{duration}min\n{total_energy:.0f}J"
                ax.text(mid_x, mid_y, label, fontsize=9, weight='bold',
                       ha='center', va='center', bbox=bbox_props, color=color)
    
    # 图例
    legend_elements = [plt.Line2D([0], [0], color=color, linewidth=3, 
                                 label=f'To Node {rid}')
                      for rid, color in receiver_to_color.items()]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 配置
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Energy Transfer Paths with Duration at t={t}', 
                fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 颜色条（节点能量）
    cbar = plt.colorbar(scatter, ax=ax, label='Node Energy (J)')
    
    plt.tight_layout()
    save_path = OutputManager.get_file_path(session_dir, f'transfer_paths_duration_t{t}.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[可视化] 带时长的传输路径图已保存: {save_path}")


def create_energy_flow_animation(simulation, output_path=None):
    """
    创建能量传输动画，逐帧显示持续传输的过程
    
    对于duration=5的传输，会显示5帧，每帧显示当前正在传输的能量
    
    Args:
        simulation: EnergySimulation对象（包含plans_by_time）
        output_path: 动画保存路径
    """
    if not hasattr(simulation, 'plans_by_time') or not simulation.plans_by_time:
        print("[可视化] 无传输记录，无法创建动画")
        return
    
    print("[可视化] 创建能量传输动画...")
    
    # 收集所有传输事件
    transfer_events = []  # (start_time, end_time, plan)
    
    for t, data in simulation.plans_by_time.items():
        plans = data.get('plans', [])
        for plan in plans:
            duration = plan.get('duration', 1)
            transfer_events.append({
                'start': t,
                'end': t + duration,
                'plan': plan
            })
    
    if not transfer_events:
        print("[可视化] 无传输事件，无法创建动画")
        return
    
    # 确定时间范围
    max_time = max(event['end'] for event in transfer_events)
    time_range = range(0, max_time + 1)
    
    # 创建动画
    fig, ax = plt.subplots(figsize=(12, 10))
    
    def update_frame(t):
        ax.clear()
        
        # 绘制节点
        network = simulation.network
        xs = [n.position[0] for n in network.nodes]
        ys = [n.position[1] for n in network.nodes]
        
        # 获取t时刻的节点能量
        node_energies = simulation.plans_by_time.get(t, {}).get('node_energies', {})
        if not node_energies:
            # 使用当前能量
            energies = [n.current_energy for n in network.nodes]
        else:
            energies = [node_energies.get(n.node_id, n.current_energy) 
                       for n in network.nodes]
        
        max_energy = max(energies) if energies else 1
        sizes = [50 + 200 * (e / max_energy) for e in energies]
        
        scatter = ax.scatter(xs, ys, c=energies, s=sizes, cmap='YlOrRd',
                           edgecolors='black', linewidths=1.5, alpha=0.8, zorder=3)
        
        # 绘制当前正在传输的路径
        active_transfers = [event for event in transfer_events 
                          if event['start'] <= t < event['end']]
        
        for event in active_transfers:
            plan = event['plan']
            path = plan.get('path', [])
            duration = plan.get('duration', 1)
            
            if len(path) < 2:
                continue
            
            # 计算当前传输进度
            progress = (t - event['start']) / duration
            
            # 绘制路径（颜色渐变表示进度）
            alpha = 0.3 + 0.7 * progress
            
            for i in range(len(path) - 1):
                node_a = path[i]
                node_b = path[i + 1]
                
                ax, ay = node_a.position[0], node_a.position[1]
                bx, by = node_b.position[0], node_b.position[1]
                
                arrow = FancyArrowPatch((ax, ay), (bx, by),
                                       arrowstyle='-|>', mutation_scale=15,
                                       linewidth=3, color='red',
                                       alpha=alpha, zorder=2)
                ax.add_patch(arrow)
                
                # 在第一段上标注进度
                if i == 0:
                    mid_x = (ax + bx) / 2
                    mid_y = (ay + by) / 2
                    elapsed = t - event['start'] + 1
                    label = f"{elapsed}/{duration}min"
                    ax.text(mid_x, mid_y, label, fontsize=10, weight='bold',
                           ha='center', va='center',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_title(f'Energy Transfer Animation - Time: {t} min\n'
                    f'Active Transfers: {len(active_transfers)}',
                    fontsize=14, weight='bold')
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        plt.colorbar(scatter, ax=ax, label='Node Energy (J)')
    
    anim = animation.FuncAnimation(fig, update_frame, frames=time_range,
                                  interval=500, repeat=True)
    
    if output_path is None:
        output_path = OutputManager.get_file_path(simulation.session_dir, 
                                                  'energy_transfer_animation.gif')
    
    anim.save(output_path, writer='pillow', fps=2)
    plt.close()
    print(f"[可视化] 能量传输动画已保存: {output_path}")


def plot_duration_statistics(simulation, session_dir):
    """
    绘制传输时长统计图
    
    显示：
    - 各种传输时长的使用频率
    - 不同时长的平均传输效率
    - 不同时长的能量传输总量
    
    Args:
        simulation: EnergySimulation对象
        session_dir: 输出目录
    """
    if not hasattr(simulation, 'plans_by_time') or not simulation.plans_by_time:
        print("[可视化] 无传输记录，无法绘制统计图")
        return
    
    # 收集数据
    duration_stats = {}  # {duration: {'count': N, 'total_energy': E, 'efficiencies': [...]}}
    
    for t, data in simulation.plans_by_time.items():
        plans = data.get('plans', [])
        for plan in plans:
            duration = plan.get('duration', 1)
            delivered = plan.get('delivered', 0)
            loss = plan.get('loss', 0)
            
            if duration not in duration_stats:
                duration_stats[duration] = {
                    'count': 0,
                    'total_energy': 0,
                    'efficiencies': []
                }
            
            duration_stats[duration]['count'] += 1
            duration_stats[duration]['total_energy'] += delivered
            
            if delivered + loss > 0:
                efficiency = delivered / (delivered + loss)
                duration_stats[duration]['efficiencies'].append(efficiency)
    
    if not duration_stats:
        print("[可视化] 无传输数据，无法绘制统计图")
        return
    
    # 创建子图
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Duration Usage Frequency', 'Average Efficiency by Duration',
                       'Total Energy by Duration', 'Energy per Transfer by Duration'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    durations = sorted(duration_stats.keys())
    counts = [duration_stats[d]['count'] for d in durations]
    avg_efficiencies = [np.mean(duration_stats[d]['efficiencies']) 
                       if duration_stats[d]['efficiencies'] else 0 
                       for d in durations]
    total_energies = [duration_stats[d]['total_energy'] for d in durations]
    avg_energies = [duration_stats[d]['total_energy'] / duration_stats[d]['count'] 
                   for d in durations]
    
    # 子图1：使用频率
    fig.add_trace(
        go.Bar(x=[f"{d}min" for d in durations], y=counts, 
               marker_color='steelblue', name='Count'),
        row=1, col=1
    )
    
    # 子图2：平均效率
    fig.add_trace(
        go.Bar(x=[f"{d}min" for d in durations], y=avg_efficiencies,
               marker_color='green', name='Efficiency'),
        row=1, col=2
    )
    
    # 子图3：总能量
    fig.add_trace(
        go.Bar(x=[f"{d}min" for d in durations], y=total_energies,
               marker_color='orange', name='Total Energy'),
        row=2, col=1
    )
    
    # 子图4：单次平均能量
    fig.add_trace(
        go.Bar(x=[f"{d}min" for d in durations], y=avg_energies,
               marker_color='red', name='Avg Energy'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Duration", row=1, col=1)
    fig.update_xaxes(title_text="Duration", row=1, col=2)
    fig.update_xaxes(title_text="Duration", row=2, col=1)
    fig.update_xaxes(title_text="Duration", row=2, col=2)
    
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency", row=1, col=2)
    fig.update_yaxes(title_text="Energy (J)", row=2, col=1)
    fig.update_yaxes(title_text="Energy (J)", row=2, col=2)
    
    fig.update_layout(
        title_text="Duration-Aware Energy Transfer Statistics",
        height=800,
        showlegend=False
    )
    
    save_path = OutputManager.get_file_path(session_dir, 'duration_statistics.html')
    fig.write_html(save_path)
    print(f"[可视化] 传输时长统计图已保存: {save_path}")

