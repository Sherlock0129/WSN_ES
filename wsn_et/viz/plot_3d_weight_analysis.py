#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三权重参数3D可视化分析脚本 - 重构版本
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
import re
import os
from typing import Tuple, List, Dict, Optional


class DataParser:
    def __init__(self):
        self.data_structure = {
        'w_b': [],
        'w_d': [],
        'w_l': [],
        'variance': [],
        'efficiency': [],
        'min_energy': []
    }
    
    def parse_triple_weight_data(self, file_path: str) -> pd.DataFrame:
        print(f"📊 开始解析数据文件 {file_path}")
        data = {key: [] for key in self.data_structure.keys()}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {file_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ 读取文件时出错: {e}")
            return pd.DataFrame()
        current_section = None
        parsed_count = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if '📊 所有时间点方差的平均值' in line:
                current_section = 'variance'; continue
            elif '📈 能量传输效率' in line or '效率' in line and '%' in line:
                current_section = 'efficiency'; continue
            elif '🔋 最终最小能量' in line or '最小能量' in line:
                current_section = 'min_energy'; continue
            elif any(keyword in line for keyword in ['📤 总发送能量', '📥 总接收能量', '💸 总损失能量', '🔋 最终平均能量', '⏱️ 运行时间:']):
                current_section = None; continue
            if line and ':' in line and current_section:
                if self._parse_data_line(line, data, current_section):
                    parsed_count += 1
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values(['w_b', 'w_d', 'w_l']).reset_index(drop=True)
            print(f"✅ 成功解析 {len(df)} 个权重组合，累计 {parsed_count} 行数据")
        else:
            print("⚠️ 警告: 未解析到任何数据")
        return df
    
    def _parse_data_line(self, line: str, data: Dict, current_section: str) -> bool:
        try:
            parts = line.split(':')
            if len(parts) < 2:
                return False
            weight_part = parts[0].strip()
            weight_match = re.findall(r'w_b=([\d.]+), w_d=([\d.]+), w_l=([\d.]+)', weight_part)
            if not weight_match:
                return False
            w_b, w_d, w_l = map(float, weight_match[0])
            value_part = parts[1].strip().split('±')[0].strip()
            value = self._parse_numeric_value(value_part, current_section)
            if value is None:
                return False
            existing_idx = self._find_existing_combination(data, w_b, w_d, w_l)
            if existing_idx is not None:
                data[current_section][existing_idx] = value
            else:
                data['w_b'].append(w_b)
                data['w_d'].append(w_d)
                data['w_l'].append(w_l)
                data['variance'].append(0)
                data['efficiency'].append(0)
                data['min_energy'].append(0)
                data[current_section][-1] = value
            return True
        except (ValueError, IndexError) as e:
            print(f"⚠️ 解析行数据时出错: {line[:50]}... - {e}")
            return False
    
    def _find_existing_combination(self, data: Dict, w_b: float, w_d: float, w_l: float) -> Optional[int]:
        for i, (b, d, l) in enumerate(zip(data['w_b'], data['w_d'], data['w_l'])):
            if abs(b - w_b) < 1e-6 and abs(d - w_d) < 1e-6 and abs(l - w_l) < 1e-6:
                return i
        return None
    
    def _parse_numeric_value(self, value_str: str, section_type: str) -> Optional[float]:
        try:
            clean_str = value_str.strip()
            if section_type == 'variance':
                return float(clean_str)
            elif section_type == 'efficiency':
                clean_str = clean_str.replace('%', '').strip()
                return float(clean_str)
            elif section_type == 'min_energy':
                clean_str = clean_str.rstrip('Jj').strip()
                return float(clean_str)
            else:
                return None
        except (ValueError, TypeError):
            return None


class WeightAnalyzer:
    def __init__(self):
        self.composite_weights = {
            'variance': 1.0,
            'efficiency': 1.0,
            'min_energy': 1.0
        }
    
    def analyze_weights(self, df: pd.DataFrame) -> Dict:
        if df.empty:
            print("❌ 错误: 数据为空，无法进行分析")
            return {}
        print(f"🔍 开始分析 {len(df)} 个权重组合")
        normalized_data = self._normalize_metrics(df)
        composite_scores = self._calculate_composite_scores(normalized_data)
        optimal_solution = self._find_optimal_solution(df, composite_scores)
        statistics = self._calculate_statistics(df, composite_scores)
        return {
            'normalized_data': normalized_data,
            'composite_scores': composite_scores,
            'optimal_solution': optimal_solution,
            'statistics': statistics,
            'data_summary': self._get_data_summary(df)
        }
    
    def _normalize_metrics(self, df: pd.DataFrame) -> Dict:
        variance = df['variance'].to_numpy(dtype=float)
        efficiency = df['efficiency'].to_numpy(dtype=float)
        min_energy = df['min_energy'].to_numpy(dtype=float)
        variance_norm = 1 - (variance - variance.min()) / (variance.max() - variance.min() + 1e-9)
        efficiency_norm = (efficiency - efficiency.min()) / (efficiency.max() - efficiency.min() + 1e-9)
        min_energy_norm = (min_energy - min_energy.min()) / (min_energy.max() - min_energy.min() + 1e-9)
        return {
            'variance_norm': variance_norm,
            'efficiency_norm': efficiency_norm,
            'min_energy_norm': min_energy_norm
        }
    
    def _calculate_composite_scores(self, normalized_data: Dict) -> np.ndarray:
        w = self.composite_weights
        score = (w['variance'] * normalized_data['variance_norm'] +
                 w['efficiency'] * normalized_data['efficiency_norm'] +
                 w['min_energy'] * normalized_data['min_energy_norm'])
        return score
    
    def _find_optimal_solution(self, df: pd.DataFrame, scores: np.ndarray) -> Dict:
        idx = int(np.argmax(scores))
        return {
            'w_b': float(df.loc[idx, 'w_b']),
            'w_d': float(df.loc[idx, 'w_d']),
            'w_l': float(df.loc[idx, 'w_l']),
            'score': float(scores[idx])
        }
    
    def _calculate_statistics(self, df: pd.DataFrame, scores: np.ndarray) -> Dict:
        top_indices = np.argsort(scores)[::-1][:5]
        top5 = [{
            'rank': i + 1,
            'w_b': float(df.loc[idx, 'w_b']),
            'w_d': float(df.loc[idx, 'w_d']),
            'w_l': float(df.loc[idx, 'w_l']),
            'score': float(scores[idx])
        } for i, idx in enumerate(top_indices)]
        return {
            'top5_combinations': top5,
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores))
        }
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict:
        return {
            'count': int(len(df)),
            'w_b_range': (float(df['w_b'].min()), float(df['w_b'].max())),
            'w_d_range': (float(df['w_d'].min()), float(df['w_d'].max())),
            'w_l_range': (float(df['w_l'].min()), float(df['w_l'].max()))
        }


class VisualizationEngine:
    def __init__(self):
        self.save_formats = ['html']
    
    def _make_3d_fig(self, df: pd.DataFrame, z_values: np.ndarray, title: str, colorscale='Viridis'):
        fig = go.Figure(data=[go.Scatter3d(
            x=df['w_b'], y=df['w_d'], z=z_values,
            mode='markers',
            marker=dict(size=4, color=z_values, colorscale=colorscale, showscale=True)
        )])
        fig.update_layout(
            title=title,
            scene=dict(xaxis_title='w_b', yaxis_title='w_d', zaxis_title='Metric'),
            template='plotly_white'
        )
        return fig
    
    def create_composite_score_3d_plot(self, df: pd.DataFrame, analysis_result: Dict, output_file: Optional[str] = None):
        scores = analysis_result['composite_scores']
        fig = self._make_3d_fig(df, scores, 'Composite Score over (w_b, w_d, w_l)')
        if output_file:
            try:
                fig.write_html(output_file)
            except Exception as e:
                print(f"保存HTML失败: {e}")
        return fig


class ReportGenerator:
    def generate_analysis_report(self, analysis_result: Dict) -> str:
        optimal_solution = analysis_result['optimal_solution']
        summary = analysis_result['data_summary']
        report = []
        report.append("=" * 60)
        report.append("三权重参数 3D 分析报告")
        report.append("=" * 60)
        report.append(f"样本数: {summary['count']}")
        report.append(f"w_b 范围: {summary['w_b_range']}")
        report.append(f"w_d 范围: {summary['w_d_range']}")
        report.append(f"w_l 范围: {summary['w_l_range']}")
        report.append("")
        report.append("最优解:")
        report.append(f"  w_b={optimal_solution['w_b']:.2f}, w_d={optimal_solution['w_d']:.2f}, w_l={optimal_solution['w_l']:.2f}, 分数={optimal_solution['score']:.3f}")
        return "\n".join(report)
    
    def print_analysis_report(self, analysis_result: Dict):
        print(self.generate_analysis_report(analysis_result))


class MainController:
    def __init__(self):
        self.data_parser = DataParser()
        self.weight_analyzer = WeightAnalyzer()
        self.visualization_engine = VisualizationEngine()
        self.report_generator = ReportGenerator()
    
    def run_complete_analysis(self, data_file_path: str, output_dir: Optional[str] = None) -> Dict:
        print("🚀 开始三权重参数3D分析...")
        print("=" * 60)
        print("📊 步骤1: 解析实验数据")
        df = self.data_parser.parse_triple_weight_data(data_file_path)
        if df.empty:
            print("❌ 数据解析失败，分析终止")
            return {}
        print("\n🔍 步骤2: 分析权重参数效果")
        analysis_result = self.weight_analyzer.analyze_weights(df)
        print("\n🎨 步骤3: 生成3D可视化图表")
        main_fig = self.visualization_engine.create_composite_score_3d_plot(
            df, analysis_result, 
            output_file=f"{output_dir}/3d_weight_analysis.html" if output_dir else None
        )
        print("\n📋 步骤4: 生成分析报告")
        self.report_generator.print_analysis_report(analysis_result)
        print("\n🖼️ 步骤5: 显示图表")
        main_fig.show()
        print("\n✅ 分析完成")
        return {
            'data': df,
            'analysis': analysis_result,
            'main_figure': main_fig,
        }


def main():
    data_file = "../data/reward_function_test/3.txt"
    output_dir = "../data"
    save_formats = ['html']
    print("🎯 三权重参数3D可视化分析")
    print("=" * 60)
    print(f"📁 数据文件: {data_file}")
    print(f"📁 输出目录: {output_dir}")
    print(f"💾 保存格式: {', '.join(save_formats)}")
    print("=" * 60)
    controller = MainController()
    controller.visualization_engine.save_formats = save_formats
    results = controller.run_complete_analysis(data_file, output_dir)
    if results:
        print("\n🎉 所有任务完成！")
        print(f"📊 分析了 {len(results['data'])} 个权重组合")
        print(f"🎨 生成了 1 个3D图表（复合分数）")
    else:
        print("\n❌ 分析失败，请检查数据文件路径和格式")


if __name__ == "__main__":
    main()

