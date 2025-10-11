#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
wb权重参数灵敏度分析可视化脚本
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np

def parse_weight_data(file_path, weight_param="wd"):
    data = {
        weight_param: [],
        'variance': [],
        'efficiency': [],
        'min_energy': []
    }
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    current_section = None
    for line in lines:
        line = line.strip()
        if '📊 所有时间点方差的平均' in line:
            current_section = 'variance'; continue
        elif '能量传输效率' in line:
            current_section = 'efficiency'; continue
        elif '🔋 最终最小能' in line:
            current_section = 'min_energy'; continue
        elif line and ':' in line and current_section:
            parts = line.split(':')
            if len(parts) >= 2:
                wb_value = float(parts[0].strip())
                value_part = parts[1].strip().split('±')[0].strip()
                if current_section == 'variance':
                    value = float(value_part)
                elif current_section == 'efficiency':
                    value = float(value_part.replace('%', ''))
                elif current_section == 'min_energy':
                    value = float(value_part.replace('J', ''))
                if wb_value not in data[weight_param]:
                    data[weight_param].append(wb_value)
                    data['variance'].append(0)
                    data['efficiency'].append(0)
                    data['min_energy'].append(0)
                idx = data[weight_param].index(wb_value)
                data[current_section][idx] = value
    df = pd.DataFrame(data)
    df = df.sort_values(weight_param).reset_index(drop=True)
    return df

def create_weight_analysis_plot(data_file_path, output_file=None, weight_param="wd"):
    df = parse_weight_data(data_file_path, weight_param)
    variance_norm = 1 - (df['variance'] - df['variance'].min()) / (df['variance'].max() - df['variance'].min() + 1e-9)
    efficiency_norm = (df['efficiency'] - df['efficiency'].min()) / (df['efficiency'].max() - df['efficiency'].min() + 1e-9)
    min_energy_norm = (df['min_energy'] - df['min_energy'].min()) / (df['min_energy'].max() - df['min_energy'].min() + 1e-9)
    fig = go.Figure()
    large_font_size = 24
    title_font_size = 24
    fig.add_trace(
        go.Scatter(
            x=df[weight_param], y=variance_norm,
            mode='lines+markers', name='Energy Variance (Normalized, Higher Better)',
            line=dict(color='#FF6B6B', width=5), marker=dict(size=10, color='#FF6B6B'),
            customdata=df['variance'],
            hovertemplate=f'<b>{weight_param} Weight:</b> %{{x}}<br><b>Energy Variance (Normalized):</b> %{{y:.3f}}<br><b>Original Variance:</b> %{{customdata:.2f}}<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[weight_param], y=efficiency_norm,
            mode='lines+markers', name='Transfer Efficiency (Normalized, Higher Better)',
            line=dict(color='#4ECDC4', width=5), marker=dict(size=10, color='#4ECDC4'),
            customdata=df['efficiency'],
            hovertemplate=f'<b>{weight_param} Weight:</b> %{{x}}<br><b>Transfer Efficiency (Normalized):</b> %{{y:.3f}}<br><b>Original Efficiency:</b> %{{customdata:.2f}}%<extra></extra>'
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[weight_param], y=min_energy_norm,
            mode='lines+markers', name='Minimum Energy (Normalized, Higher Better)',
            line=dict(color='#45B7D1', width=5), marker=dict(size=10, color='#45B7D1'),
            customdata=df['min_energy'],
            hovertemplate=f'<b>{weight_param} Weight:</b> %{{x}}<br><b>Minimum Energy (Normalized):</b> %{{y:.3f}}<br><b>Original Energy:</b> %{{customdata:.2f}}J<extra></extra>'
        )
    )
    composite_scores = 1 * variance_norm + 1 * efficiency_norm + 1 * min_energy_norm
    max_composite_idx = composite_scores.idxmax()
    max_composite_weight = df.loc[max_composite_idx, weight_param]
    max_composite_score = composite_scores.iloc[max_composite_idx]
    fig.add_vline(
        x=max_composite_weight, line_dash="dash", line_color="red", line_width=3, opacity=0.8,
        annotation_text=f"Optimal {weight_param} = {max_composite_weight}", annotation_position="top",
        annotation_font_size=20, annotation_font_color="red"
    )
    fig.add_annotation(
        x=max_composite_weight, y=0,
        text=f"Best Overall<br>{weight_param}={max_composite_weight}<br>Score={max_composite_score:.3f}",
        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=3, arrowcolor="red",
        ax=0, ay=20, font=dict(size=20, color="red"),
        bgcolor="rgba(255, 255, 255, 0.7)", bordercolor="red", borderwidth=2
    )
    fig.update_layout(
        title={'text': f'{weight_param.upper()} Weight Parameter Sensitivity Analysis - Three Metrics Comparison', 'x': 0.5, 'xanchor': 'center', 'font': {'size': title_font_size + 4, 'color': '#2C3E50'}},
        xaxis_title=f"{weight_param.upper()} Weight Value",
        yaxis_title="Normalized Metric Value (0-1, Higher Better)",
        font=dict(size=large_font_size), height=700, showlegend=True,
        legend=dict(x=0.5, y=-0.25, xanchor='center', yanchor='top', orientation='h', bgcolor="rgba(255,255,255,0.8)", bordercolor="black", borderwidth=1, font=dict(size=30)),
        plot_bgcolor='white', paper_bgcolor='white', margin=dict(b=150)
    )
    fig.update_xaxes(title_font=dict(size=large_font_size), tickfont=dict(size=large_font_size - 2), showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(title_font=dict(size=large_font_size), tickfont=dict(size=large_font_size - 2), showgrid=True, gridwidth=1, gridcolor='lightgray', range=[0, 1.1])
    if output_file:
        pdf_file = output_file.replace('.html', '.pdf')
        fig.write_image(pdf_file, format='pdf', width=1200, height=700, scale=2)
        print(f"Chart saved to: {pdf_file}")
    fig.show()
    print("\n" + "="*60)
    print(f"{weight_param.upper()} Weight Analysis Statistical Results")
    print("="*60)
    print(f"\nBest Overall Performance (Composite Score):")
    print(f"   {weight_param} = {max_composite_weight}")
    print(f"   Composite Score = {max_composite_score:.3f}")
    print(f"   Energy Variance = {df.loc[max_composite_idx, 'variance']:.2f}")
    print(f"   Transfer Efficiency = {df.loc[max_composite_idx, 'efficiency']:.2f}%")
    print(f"   Minimum Energy = {df.loc[max_composite_idx, 'min_energy']:.2f}J")
    min_variance_idx = df['variance'].idxmin()
    max_efficiency_idx = df['efficiency'].idxmax()
    max_min_energy_idx = df['min_energy'].idxmax()
    print(f"\nIndividual Metric Optimal Values (for reference):")
    print(f"   Energy Variance Optimal: {weight_param} = {df.loc[min_variance_idx, weight_param]}, Variance = {df.loc[min_variance_idx, 'variance']:.2f}")
    print(f"   Transfer Efficiency Optimal: {weight_param} = {df.loc[max_efficiency_idx, weight_param]}, Efficiency = {df.loc[max_efficiency_idx, 'efficiency']:.2f}%")
    print(f"   Minimum Energy Optimal: {weight_param} = {df.loc[max_min_energy_idx, weight_param]}, Min Energy = {df.loc[max_min_energy_idx, 'min_energy']:.2f}J")
    print(f"\nComprehensive Recommendation:")
    print(f"   Based on composite score, recommend {weight_param} = {max_composite_weight}")
    return fig

if __name__ == "__main__":
    WEIGHT_PARAM = "wl"
    data_file = f"../data/reward_function_test/{WEIGHT_PARAM}.txt"
    output_file = f"../data/{WEIGHT_PARAM}_analysis_plot.pdf"
    print(f"Starting {WEIGHT_PARAM} weight analysis chart generation...")
    fig1 = create_weight_analysis_plot(
        data_file, 
        output_file=output_file,
        weight_param=WEIGHT_PARAM
    )
    print(f"\n{WEIGHT_PARAM.upper()} chart generated successfully!")

