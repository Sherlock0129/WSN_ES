# 如何添加机制设计章节的示例图

本文档说明如何在机制设计章节中添加示例图片。

## 图片位置

所有图片应放在 `paper_p/sections/figures/` 目录下。

## 已添加的图片占位符

### 1. M1：AOEI优先级机制

**文件名**：`aoei_adaptive_threshold.png`  
**位置**：第85-91行  
**说明**：动态AOI上限调整示意图
- 横轴：信息量
- 纵轴：AOI上限
- 内容：展示信息量越大，AOI上限越小，促使节点更快上报

**建议内容**：
- 绘制一条下降曲线，表示 $A_{\max,i}(t) = A_{\max,0} / (1 + I_i(t) / \gamma)$
- 标注几个关键点（如信息量为0、100、500时的AOI上限值）
- 添加图例说明

### 2. M2：InfoNode数字孪生机制

**文件名**：`infonode_energy_estimation.png`  
**位置**：第221-227行  
**说明**：InfoNode理论能量计算流程图

**建议内容**：
- 流程图形式，展示以下步骤：
  1. 输入：上次已知能量 $E_i(t_0)$、时间间隔 $\Delta t$
  2. 计算能量消耗：$E_{\text{cons}} = E_{\text{sen}} \cdot \Delta t + E_{\text{com}} \cdot \Delta t$
  3. 计算能量采集：$E_{\text{harvest}} = \eta_{\text{solar}} \cdot A_i \cdot G(t) \cdot \Delta t$
  4. 计算传输相关能量：$E_{\text{send},i}$ 和 $E_{\text{recv},i}$
  5. 输出：估算能量 $E_i^{\text{est}}(t)$
- 使用流程图符号（矩形表示处理步骤，菱形表示判断，箭头表示流程方向）

### 3. M3：ALDP自适应时长规划机制

**文件名**：`aldp_score_function.png`  
**位置**：第399-405行  
**说明**：ALDP评分函数随时长变化示意图

**建议内容**：
- 横轴：传输时长 $\tau$（1-5分钟）
- 纵轴：综合评分 $\text{Score}(\tau)$
- 绘制以下曲线：
  - 能量收益 $B_{\text{energy}}(\tau)$（上升曲线）
  - 损耗惩罚 $P_{\text{loss}}(\tau)$（上升曲线）
  - 时效惩罚 $P_{\text{aoi}}(\tau)$（上升曲线）
  - 信息奖励 $R_{\text{info}}(\tau)$（上升曲线）
  - 综合评分 $\text{Score}(\tau)$（可能有峰值）
- 标注最优时长 $\tau^*$ 的位置
- 添加图例说明各条曲线的含义

### 4. M4：EETOR机会主义上报机制

**文件名**：`eetor_path_selection.png`  
**位置**：第485-491行  
**说明**：EETOR路径选择与信息收集示意图

**建议内容**：
- 网络拓扑图，包含：
  - 源节点（标注为"Source"）
  - 目标节点（标注为"Target"）
  - 中继节点（标注为"Relay 1"、"Relay 2"等）
  - 其他节点（标注为"Node 1"、"Node 2"等）
- 用粗线表示选定的能量传输路径
- 用虚线箭头表示信息收集方向
- 在路径上的节点标注信息量 $B_{v_i}$ 和信息年龄 $A_{v_i}$
- 添加图例说明能量传输路径和信息收集路径

## 图片制作方法

### 工具分配
- **函数图**（M1, M3）：使用 Python 脚本自动生成
  - 运行 `generate_function_plots.py` 即可生成
- **流程图**（M2, M4）：使用 Draw.io 手动制作
  - 参考 `drawio_flowcharts.md` 中的详细说明

### 快速开始

#### 1. 生成函数图（Python）
```bash
cd paper_p/sections/figures
python generate_function_plots.py
```
这将自动生成：
- `aoei_adaptive_threshold.png` (M1)
- `aldp_score_function.png` (M3)

#### 2. 制作流程图（Draw.io）
参考 `drawio_flowcharts.md` 中的详细步骤制作：
- `infonode_energy_estimation.png` (M2)
- `eetor_path_selection.png` (M4)

### 图片格式要求
- **格式**：PNG 或 PDF（推荐PDF，矢量图质量更好）
- **分辨率**：至少 300 DPI（用于打印）
- **尺寸**：根据 LaTeX 中的 `width=0.9\linewidth` 设置，图片宽度应适合单栏或双栏显示

### 详细说明

#### 函数图（Python脚本）
- **脚本文件**: `generate_function_plots.py`
- **使用方法**: 直接运行脚本即可生成所有函数图
- **输出**: PNG和PDF两种格式（PDF为矢量图，质量更好）

#### 流程图（Draw.io）
- **说明文档**: `drawio_flowcharts.md`
- **使用方法**: 按照文档中的步骤在Draw.io中制作
- **输出**: PNG格式，300 DPI

## 添加图片到LaTeX

图片已经通过以下代码添加到LaTeX文档中：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{sections/figures/图片文件名.png}
\caption{图片说明}
\label{fig:图片标签}
\end{figure}
```

只需要：
1. 将生成的图片文件放到 `paper_p/sections/figures/` 目录下
2. 确保文件名与LaTeX代码中的文件名一致
3. 重新编译LaTeX文档即可

## 注意事项

1. **图片质量**：确保图片清晰，文字可读
2. **字体大小**：图片中的文字应足够大，建议至少10pt
3. **颜色**：如果论文是黑白打印，确保图片在灰度模式下仍然清晰
4. **引用**：在正文中可以通过 `\ref{fig:图片标签}` 引用图片

