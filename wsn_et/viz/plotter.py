import os
import numpy as np
import plotly.graph_objects as go

# 小工具：确保目录存在
def _ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def plot_node_distribution(nodes, output_dir="data", show_paths=True, path_len=None, show_plot=True):
    """
    Plot 2D node distribution, distinguishing solar/non-solar nodes with optional mobile node paths.
    Uses Plotly for interactivity, ensures equal axis scaling, and saves in IEEE style.

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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=solar_x, y=solar_y, mode='markers+text', name='Solar nodes',
            marker=dict(color='#f5b301', symbol='circle', size=8),
            text=[str(node_id) for node_id in solar_ids[:100]],
            textposition='bottom right', textfont=dict(size=8, family='Arial'),
            hovertemplate='Node ID: %{text}<br>X: %{x} m<br>Y: %{y} m<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=nosolar_x, y=nosolar_y, mode='markers+text', name='Non-solar nodes',
            marker=dict(color='#666666', symbol='triangle-up', size=8),
            text=[str(node_id) for node_id in nosolar_ids[:100]],
            textposition='bottom right', textfont=dict(size=8, family='Arial'),
            hovertemplate='Node ID: %{text}<br>X: %{x} m<br>Y: %{y} m<extra></extra>'
        )
    )

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
                    fig.add_trace(
                        go.Scatter(
                            x=hx, y=hy, mode='lines', name=f'Path Node {node.node_id}',
                            line=dict(width=1, color='rgba(0, 0, 0, 0.5)'),
                            hovertemplate=f'Node ID: {node.node_id}<br>X: %{x} m<br>Y: %{y} m<extra></extra>',
                            showlegend=False
                        )
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

    fig.update_layout(
        title=dict(text="Node distribution in 2D space", font=dict(size=10, family='Arial')),
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

    save_path = os.path.join(output_dir, 'node_distribution.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    if show_plot:
        fig.show()
    # IEEE Caption: Fig. 1. Node distribution in 2D space.

def plot_energy_distribution(nodes, time_step, output_dir="data", show_plot=True):
    """
    Plot the energy distribution of nodes at a specific time step using Plotly and save in IEEE style.
    """
    _ensure_dir(output_dir)

    node_ids = [node.node_id for node in nodes]
    energy_values = [node.current_energy for node in nodes]
    # 省略: 其余与原实现一致（为控制补丁长度，仅保留必要函数）

def plot_energy_paths_at_time(network, plans, t, output_path=None, show_plot=True):
    """在节点分布图上叠加某时间步的能量传输路径（多跳折线+箭头）。"""
    _ensure_dir("data")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    xs = [n.position[0] for n in network.nodes]
    ys = [n.position[1] for n in network.nodes]
    ids = [n.node_id for n in network.nodes]

    plt.figure(figsize=(8, 6))
    plt.scatter(xs, ys, c='#666666', s=20, zorder=1)
    for x, y, nid in zip(xs, ys, ids):
        plt.text(x, y, str(nid), fontsize=7, ha='left', va='bottom')

    import itertools
    color_cycle = itertools.cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])
    receiver_to_color = {}

    for plan in plans:
        r = plan.get('receiver'); d = plan.get('donor'); path = plan.get('path')
        if r is None or d is None or not path or len(path) < 2:
            continue
        rid = getattr(r, 'node_id', None)
        if rid not in receiver_to_color:
            receiver_to_color[rid] = next(color_cycle)
        color = receiver_to_color[rid]

        energy_sent = plan.get('energy_sent', getattr(d, 'E_char', 300.0))
        lw = 1.0 + 2.0 * (float(energy_sent) / (float(getattr(d, 'E_char', 300.0)) + 1e-9))

        for a, b in zip(path, path[1:]):
            ax, ay = a.position[0], a.position[1]
            bx, by = b.position[0], b.position[1]
            arr = FancyArrowPatch((ax, ay), (bx, by), arrowstyle='-|>', mutation_scale=8,
                                  linewidth=lw, color=color, zorder=2)
            plt.gca().add_patch(arr)

    plt.title(f'Energy transfer paths at t={t}')
    plt.xlabel('X (m)'); plt.ylabel('Y (m)')
    plt.axis('equal'); plt.grid(True, alpha=0.3)

    out = output_path or f'data/energy_paths_t{t}.png'
    plt.tight_layout(); plt.savefig(out, dpi=200)
    if show_plot:
        plt.show()

def plot_energy_over_time(nodes, results, output_dir="data", show_plot=True):
    """
    Plot the energy change of each node over time using Plotly and save in IEEE style.
    """
    _ensure_dir(output_dir)
    time_steps = list(range(1, len(results) + 1))

    energy_data = {node.node_id: [] for node in nodes}
    for _, step_result in enumerate(results):
        for node_data in step_result:
            energy_data[node_data["node_id"]].append(node_data["current_energy"])

    fig = go.Figure()
    colors = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
        'rgb(148, 103, 189)', 'rgb(140, 86, 75)', 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
        'rgb(188, 189, 34)', 'rgb(23, 190, 207)'
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
        title=dict(text="Energy change over time for each node", font=dict(size=10, family='Arial')),
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

    save_path = os.path.join(output_dir, 'energy_over_time.png')
    fig.write_image(save_path, width=800, height=600, scale=3)
    if show_plot:
        fig.show()

