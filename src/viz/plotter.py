import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from utils.output_manager import OutputManager

# 小工具：确保目录存在
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_node_distribution(
    nodes,
    output_dir="data",
    show_paths=True,
    path_len=None,
    session_dir=None,
    title_font_size: int = 12,
    axis_label_font_size: int = 10,
    tick_font_size: int = 9,
    legend_font_size: int = 12,
):
    """
    Plot 2D node distribution, distinguishing solar/non-solar nodes with optional mobile node paths.
    Uses Matplotlib to produce a static IEEE-style figure with equal axis scaling.

    Args:
        nodes: List[SensorNode]
        output_dir: Directory to save the image
        show_paths: Whether to plot mobile node trajectories (requires node.position_history)
        path_len: Show only the last N points of each mobile node's trajectory; None for all
    """
    _ensure_dir(output_dir)

    # Separate nodes into solar and non-solar groups
    solar_x, solar_y, solar_ids = [], [], []
    nosolar_x, nosolar_y, nosolar_ids = [], [], []

    for node in nodes:
        x, y = node.position[0], node.position[1]
        has_solar = getattr(node, "has_solar", getattr(node, "solar_enabled", True))
        if has_solar:
            solar_x.append(x); solar_y.append(y); solar_ids.append(node.node_id)
        else:
            nosolar_x.append(x); nosolar_y.append(y); nosolar_ids.append(node.node_id)

    fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

    ax.scatter(
        solar_x, solar_y, c='#f5b301', s=45, marker='o', label='Solar nodes', edgecolors='black', linewidths=0.4
    )
    ax.scatter(
        nosolar_x, nosolar_y, c='#666666', s=55, marker='^', label='Non-solar nodes', edgecolors='black', linewidths=0.4
    )

    for x, y, node_id in zip(solar_x, solar_y, solar_ids[:100]):
        ax.text(x, y, str(node_id), fontsize=tick_font_size, ha='left', va='bottom', family='Arial')
    for x, y, node_id in zip(nosolar_x, nosolar_y, nosolar_ids[:100]):
        ax.text(x, y, str(node_id), fontsize=tick_font_size, ha='left', va='bottom', family='Arial')

    # Mobile paths
    if show_paths:
        for node in nodes:
            if getattr(node, "is_mobile", False):
                history = getattr(node, "position_history", None)
                if history and len(history) > 1:
                    if path_len is not None:
                        history = history[-path_len:]
                    hx = [p[0] for p in history]
                    hy = [p[1] for p in history]
                    ax.plot(
                        hx, hy, '-', linewidth=1.0, color='black', alpha=0.5, label='_nolegend_'
                    )

    # Equal axis scaling
    all_x = solar_x + nosolar_x
    all_y = solar_y + nosolar_y
    if all_x and all_y:
        x_range = [min(all_x), max(all_x)]
        y_range = [min(all_y), max(all_y)]
        max_span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        x_range = [min(all_x) - 0.1 * max_span, max(all_x) + 0.1 * max_span]
        y_range = [min(all_y) - 0.1 * max_span, max(all_y) + 0.1 * max_span]
    else:
        x_range, y_range = [-1, 1], [-1, 1]

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title("Node distribution in 2D space", fontsize=title_font_size, family='Arial')
    ax.set_xlabel("X position (m)", fontsize=axis_label_font_size, family='Arial')
    ax.set_ylabel("Y position (m)", fontsize=axis_label_font_size, family='Arial')
    ax.grid(True, color=(0, 0, 0, 0.25), linestyle='--', linewidth=0.6, alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_family('Arial')

    handles, labels = ax.get_legend_handles_labels()
    # Deduplicate path legend entries
    dedup = {}
    for h, l in zip(handles, labels):
        if l not in dedup:
            dedup[l] = h
    legend = ax.legend(
        dedup.values(),
        dedup.keys(),
        loc='upper right',
        fontsize=legend_font_size,
        frameon=False,
        borderpad=0.4,
        labelspacing=0.6
    )
    if legend is not None:
        for text in legend.get_texts():
            text.set_family('Arial')
            text.set_fontsize(legend_font_size)

    # 使用传入的会话目录或创建新的
    if session_dir is None:
        session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, 'node_distribution.png')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"节点分布图已保存到: {save_path}")
    # IEEE Caption: Fig. 1. Node distribution in 2D space.

def plot_energy_distribution(nodes, time_step, output_dir="data"):
    """
    Plot the energy distribution of nodes at a specific time step using Plotly and save in IEEE style.
    """
    _ensure_dir(output_dir)

    node_ids = [node.node_id for node in nodes]
    energy_values = [node.current_energy for node in nodes]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=node_ids, y=energy_values, marker_color='orange',
            hovertemplate='Node ID: %{x}<br>Energy: %{y} J<extra></extra>'
        )
    )
    fig.update_layout(
        title=dict(text=f"Node energy distribution at time step {time_step}", font=dict(size=10, family='Arial')),
        xaxis_title="Node ID", yaxis_title="Energy (J)",
        font=dict(family='Arial', size=8), showlegend=False, template='plotly_white',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial'))
    )

    # 创建按日期命名的输出目录
    session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, f'energy_distribution_t{time_step}.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    fig.show()
    # IEEE Caption: Fig. X. Node energy distribution at time step {time_step}.

def plot_energy_paths_at_time(network, plans, t, output_path=None):
    """在节点分布图上叠加某时间步的能量传输路径（多跳折线+箭头）。

    Args:
        network: Network 对象（提供 nodes 列表及节点 position/node_id 等）
        plans: schedulers.plan(...) 返回的计划列表（包含 receiver/donor/path/distance/energy_sent）
        t: 时间步（用于标题/文件名）
        output_path: 可选保存路径；None 则默认 data/energy_paths_t{t}.png
    """
    _ensure_dir("data")

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    # 画节点散点
    xs = [n.position[0] for n in network.nodes]
    ys = [n.position[1] for n in network.nodes]
    ids = [n.node_id for n in network.nodes]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c='#666666', s=20, zorder=1)
    for x, y, nid in zip(xs, ys, ids):
        plt.text(x, y, str(nid), fontsize=7, ha='left', va='bottom')

    # 为不同接收端分配颜色
    import itertools
    color_cycle = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    receiver_to_color = {}

    # 叠加路径
    for plan in plans:
        r = plan.get('receiver'); d = plan.get('donor'); path = plan.get('path')
        if r is None or d is None or not path or len(path) < 2:
            continue
        rid = getattr(r, 'node_id', None)
        if rid not in receiver_to_color:
            receiver_to_color[rid] = next(color_cycle)
        color = receiver_to_color[rid]

        # 线宽与能量（可选）
        energy_sent = plan.get('energy_sent', getattr(d, 'E_char', 300.0))
        lw = 1.0 + 2.0 * (float(energy_sent) / (float(getattr(d, 'E_char', 300.0)) + 1e-9))

        # 逐段画箭头
        for a, b in zip(path, path[1:]):
            ax, ay = a.position[0], a.position[1]
            bx, by = b.position[0], b.position[1]
            arr = FancyArrowPatch((ax, ay), (bx, by), arrowstyle='-|>', mutation_scale=8,
                                  linewidth=lw, color=color, zorder=2)
            plt.gca().add_patch(arr)

    plt.title(f'Energy transfer paths at t={t}')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.axis('equal'); plt.grid(True, alpha=0.3)

    # 创建按日期命名的输出目录
    session_dir = OutputManager.get_session_dir("data")
    
    out = output_path or OutputManager.get_file_path(session_dir, f'energy_paths_t{t}.png')
    plt.tight_layout(); plt.savefig(out, dpi=200); plt.show()

def _interpolate_energy_changes(energy_data, time_steps, step_duration=60, threshold=500.0):
    """
    对能量大幅跳变进行插值，模拟duration>1的逐分钟传输
    
    Args:
        energy_data: {node_id: [energy_values]}
        time_steps: 原始时间步列表  
        step_duration: 每个时间步的持续时间（分钟）
        threshold: 触发插值的能量变化阈值（J）
    
    Returns:
        (interpolated_time_steps, interpolated_energy_data)
    """
    # 第一遍：检测所有需要插值的位置和duration
    interpolation_map = {}  # {step_index: max_duration}
    
    for node_id, energies in energy_data.items():
        for i in range(len(energies) - 1):
            energy_change = abs(energies[i+1] - energies[i])
            if energy_change > threshold:
                # 估算可能的duration（假设每分钟传输约100-300J）
                estimated_duration = min(step_duration, max(2, int(energy_change / 200)))
                if i in interpolation_map:
                    interpolation_map[i] = max(interpolation_map[i], estimated_duration)
                else:
                    interpolation_map[i] = estimated_duration
    
    if not interpolation_map:
        # 不需要插值
        return time_steps, energy_data
    
    # 第二遍：为所有节点在相同位置插值
    interpolated_data = {}
    interpolated_time = []
    
    for node_id, energies in energy_data.items():
        interpolated_values = []
        
        for i in range(len(energies)):
            if i == 0:
                # 第一个点直接添加
                interpolated_values.append(energies[i])
                if len(interpolated_time) == 0:
                    interpolated_time.append(time_steps[i])
            else:
                # 检查是否需要插值
                if (i-1) in interpolation_map:
                    # 需要插值
                    duration = interpolation_map[i-1]
                    energy_change = energies[i] - energies[i-1]
                    step_per_minute = energy_change / duration
                    
                    # 生成中间点
                    for minute in range(1, duration):
                        interpolated_values.append(energies[i-1] + step_per_minute * minute)
                        if node_id == list(energy_data.keys())[0]:
                            # 生成中间时间点
                            t_start = time_steps[i-1]
                            t_end = time_steps[i]
                            t_interp = t_start + (t_end - t_start) * minute / duration
                            interpolated_time.append(t_interp)
                
                # 添加当前点
                interpolated_values.append(energies[i])
                if node_id == list(energy_data.keys())[0]:
                    interpolated_time.append(time_steps[i])
        
        interpolated_data[node_id] = interpolated_values
    
    return interpolated_time, interpolated_data


def plot_energy_over_time(nodes, results, output_dir="data", session_dir=None, interpolate_duration=True):
    """
    Plot the energy change of each node over time using Plotly and save in IEEE style.
    Note: Physical center node (ID=0) is excluded from the plot.
    
    Args:
        interpolate_duration: 是否对duration>1的传输进行插值（逐分钟显示）
    """
    _ensure_dir(output_dir)

    time_steps = list(range(1, len(results) + 1))

    # 排除物理中心节点（ID=0）
    regular_nodes = [node for node in nodes if node.node_id != 0]
    
    energy_data = {node.node_id: [] for node in regular_nodes}
    for _, step_result in enumerate(results):
        for node_data in step_result:
            # 只记录非物理中心节点的数据
            if node_data["node_id"] != 0 and node_data["node_id"] in energy_data:
                energy_data[node_data["node_id"]].append(node_data["current_energy"])
    
    # 如果启用插值，检测能量大幅跳变并进行平滑处理
    if interpolate_duration:
        time_steps, energy_data = _interpolate_energy_changes(energy_data, time_steps)
    
    fig = go.Figure()
    colors = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
        'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
        'rgb(188, 189, 34)', 'rgb(23, 190, 207)', 'rgb(174, 199, 232)', 'rgb(255, 187, 120)',
        'rgb(152, 223, 138)', 'rgb(255, 152, 150)', 'rgb(197, 176, 213)', 'rgb(196, 156, 148)',
        'rgb(247, 182, 210)', 'rgb(199, 199, 199)', 'rgb(219, 219, 141)', 'rgb(158, 218, 229)'
    ]

    for idx, (node_id, energy_values) in enumerate(energy_data.items()):
        color = colors[idx % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=time_steps, y=energy_values, mode='lines', name=f"Node {node_id}",
                line=dict(color=color),
                hovertemplate='Node ID: ' + str(node_id) + '<br>Time Step: %{x}<br>Energy: %{y} J<extra></extra>'
            )
        )

    fig.update_layout(
        title=dict(text="Energy change over time for each node", font=dict(size=40, family='Arial')),
        xaxis_title="Time step", yaxis_title="Energy (J)",
        font=dict(family='Arial', size=32),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', font=dict(size=32, family='Arial')),
        showlegend=True, hovermode='closest', template='plotly_white', margin=dict(r=200),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=32, family='Arial')),
                   tickfont=dict(size=28, family='Arial')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=32, family='Arial')),
                   tickfont=dict(size=28, family='Arial'))
    )

    # 使用传入的会话目录或创建新的
    if session_dir is None:
        session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, 'energy_over_time.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    fig.show()
    print(f"能量变化图已保存到: {save_path}")
    # IEEE Caption: Fig. X. Energy change over time for each node.

def plot_center_node_energy(nodes, results, output_dir="data", session_dir=None):
    """
    Plot the energy change of the physical center node (Node 0) over time using Plotly.
    """
    _ensure_dir(output_dir)

    time_steps = list(range(1, len(results) + 1))

    # 找到物理中心节点（ID=0）
    center_node = next((node for node in nodes if node.node_id == 0), None)
    
    if center_node is None:
        print("警告: 未找到物理中心节点 (ID=0)，跳过绘图")
        return
    
    # 提取物理中心节点的能量数据
    energy_values = []
    for _, step_result in enumerate(results):
        for node_data in step_result:
            if node_data["node_id"] == 0:
                energy_values.append(node_data["current_energy"])
                break

    if not energy_values:
        print("警告: 未找到物理中心节点的能量数据，跳过绘图")
        return

    fig = go.Figure()
    
    # 使用深蓝色绘制物理中心节点的能量变化
    fig.add_trace(
        go.Scatter(
            x=time_steps, y=energy_values, mode='lines', name="Node 0",
            line=dict(color='rgb(31, 119, 180)'),
            hovertemplate='Node ID: ' + str(0) + '<br>Time Step: %{x}<br>Energy: %{y} J<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(text="Physical Center Node Energy Over Time", font=dict(size=10, family='Arial')),
        xaxis_title="Time step", yaxis_title="Energy (J)",
        font=dict(family='Arial', size=8),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', font=dict(size=8, family='Arial')),
        showlegend=True, hovermode='closest', template='plotly_white', margin=dict(r=150),
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial'))
    )

    # 使用传入的会话目录或创建新的
    if session_dir is None:
        session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, 'center_node_energy_over_time.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    fig.show()
    print(f"物理中心节点能量变化图已保存到: {save_path}")
    # IEEE Caption: Fig. X. Energy change over time for the physical center node.

def plot_energy_histogram(nodes, time_step, output_dir="data"):
    """
    Plot a histogram of node energy levels at a given time step using Plotly and save in IEEE style.
    """
    _ensure_dir(output_dir)

    energy_values = [node.current_energy for node in nodes]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=energy_values, nbinsx=20, marker_color='green', opacity=0.7,
            hovertemplate='Energy: %{x} J<br>Frequency: %{y}<extra></extra>'
        )
    )
    fig.update_layout(
        title=dict(text=f"Energy distribution histogram at time step {time_step}", font=dict(size=10, family='Arial')),
        xaxis_title="Energy (J)", yaxis_title="Frequency",
        font=dict(family='Arial', size=8), showlegend=False, template='plotly_white',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial')),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial'))
    )

    # 创建按日期命名的输出目录
    session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, f'energy_histogram_t{time_step}.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    fig.show()
    # IEEE Caption: Fig. X. Energy distribution histogram at time step {time_step}.

def plot_routing_path(path, output_dir="data"):
    """
    Plot the routing path selected by the OECR algorithm using Plotly with equal axis scaling and save in IEEE style.
    """
    _ensure_dir(output_dir)

    x_coords = [node.position[0] for node in path]
    y_coords = [node.position[1] for node in path]
    node_ids = [str(node.node_id) for node in path]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_coords, y=y_coords, mode='lines+markers+text', name='Routing path',
            line=dict(color='purple', width=2), marker=dict(size=8),
            text=node_ids, textposition='middle right', textfont=dict(size=8, family='Arial'),
            hovertemplate='Node ID: %{text}<br>X: %{x} m<br>Y: %{y} m<extra></extra>'
        )
    )

    if x_coords and y_coords:
        x_range = [min(x_coords), max(x_coords)]
        y_range = [min(y_coords), max(y_coords)]
        max_span = max(x_range[1] - x_range[0], y_range[1] - y_range[0])
        x_range = [min(x_coords) - 0.1 * max_span, max(x_coords) + 0.1 * max_span]
        y_range = [min(y_coords) - 0.1 * max_span, max(y_coords) + 0.1 * max_span]
    else:
        x_range, y_range = [-1, 1], [-1, 1]

    fig.update_layout(
        title=dict(text="Optimal routing path (OECR)", font=dict(size=10, family='Arial')),
        xaxis_title="X position (m)", yaxis_title="Y position (m)",
        font=dict(family='Arial', size=8),
        legend=dict(x=1.05, y=1, xanchor='left', yanchor='top', font=dict(size=8, family='Arial')),
        showlegend=True, hovermode='closest', template='plotly_white', margin=dict(r=150),
        xaxis=dict(range=x_range, showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   scaleanchor='y', scaleratio=1,
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial')),
        yaxis=dict(range=y_range, showgrid=True, gridcolor='rgba(0, 0, 0, 0.3)',
                   title=dict(font=dict(size=8, family='Arial')),
                   tickfont=dict(size=8, family='Arial'))
    )

    # 创建按日期命名的输出目录
    session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, 'routing_path.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    fig.show()
    # IEEE Caption: Fig. X. Optimal routing path using OECR algorithm.

def plot_energy_transfer_history(nodes, output_dir="data"):
    """
    按“总传输能量”降序排序绘制柱状图（传输越多越靠左），并以 IEEE 风格保存。
    """
    _ensure_dir(output_dir)

    node_ids = [node.node_id for node in nodes]
    received_energy = [sum(getattr(node, "received_history", []) or [0]) for node in nodes]
    transferred_energy = [sum(getattr(node, "transferred_history", []) or [0]) for node in nodes]

    idx_sorted = sorted(range(len(nodes)), key=lambda i: transferred_energy[i], reverse=True)
    node_ids = [node_ids[i] for i in idx_sorted]
    received_energy = [received_energy[i] for i in idx_sorted]
    transferred_energy = [transferred_energy[i] for i in idx_sorted]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(i) for i in node_ids], y=received_energy, name='Received energy',
            marker_color='blue',
            hovertemplate='Node ID: %{x}<br>Received: %{y:.2f} J<extra></extra>'
        )
    )
    fig.add_trace(
        go.Bar(
            x=[str(i) for i in node_ids], y=transferred_energy, name='Transferred energy',
            marker_color='red',
            hovertemplate='Node ID: %{x}<br>Transferred: %{y:.2f} J<extra></extra>'
        )
    )

    fig.update_layout(
        title=dict(text="Energy transfer history for each node", font=dict(size=10, family='Arial')),
        xaxis_title="Node ID (sorted by transferred energy, desc.)", yaxis_title="Energy (J)",
        font=dict(family='Arial', size=8),
        legend=dict(x=1.02, y=1, xanchor='left', yanchor='top', font=dict(size=8), borderwidth=0),
        barmode='group', template='plotly_white',
        margin=dict(l=60, r=160, t=40, b=50)
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(0,0,0,0.25)', tickangle=0)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(0,0,0,0.25)', zeroline=False)

    # 创建按日期命名的输出目录
    session_dir = OutputManager.get_session_dir(output_dir)
    
    save_path = OutputManager.get_file_path(session_dir, 'energy_transfer_history.png')
    fig.write_image(save_path, width=1040, height=780, scale=2)
    fig.show()
    # IEEE Caption: Fig. X. Energy transfer history for each node (sorted by transferred energy, left is larger).


def plot_adcr_clusters_and_paths(adcr_link_layer, output_dir="data", session_dir=None):
    """
    Plot ADCR clustering results and information paths to virtual center.
    This is a wrapper function that calls the ADCR link layer's built-in plotting method.
    
    Args:
        adcr_link_layer: ADCRLinkLayerVirtual instance
        output_dir: Base directory to save the image
        session_dir: Specific session directory (if provided, uses this instead of output_dir)
    """
    if adcr_link_layer is None:
        print("[ADCR-Plot] ADCR link layer not available, skipping plot")
        return
    
    # Use session_dir if provided, otherwise use output_dir
    target_dir = session_dir if session_dir is not None else output_dir
    
    try:
        # Call the ADCR link layer's built-in plotting method
        adcr_link_layer.plot_clusters_and_paths(output_dir=target_dir)
        print(f"[ADCR-Plot] ADCR clustering and paths plot saved to {target_dir}")
    except Exception as e:
        print(f"[ADCR-Plot] Failed to generate ADCR plot: {e}")
