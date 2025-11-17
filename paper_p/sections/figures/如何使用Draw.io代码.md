# 如何使用Draw.io代码

## 方法1：直接导入（推荐）

1. 打开 Draw.io: https://app.diagrams.net/
2. 点击 **文件 → 打开**
3. 选择对应的 `.drawio` 文件：
   - `infonode_energy_estimation.drawio` - InfoNode流程图
   - `eetor_path_selection.drawio` - EETOR网络拓扑图
4. 文件会自动加载，你可以直接编辑
5. 编辑完成后，**文件 → 导出为 → PNG**
   - 分辨率：300 DPI
   - 背景：白色
   - 保存到：`paper_p/sections/figures/` 目录

## 方法2：复制XML代码

1. 打开 Draw.io: https://app.diagrams.net/
2. 点击 **文件 → 新建 → 空白图表**
3. 按 `Ctrl+E` 打开XML编辑器
4. 复制对应的 `.drawio` 文件中的XML内容
5. 粘贴到XML编辑器中
6. 点击"应用"按钮
7. 图表会自动生成

## 文件说明

### 1. infonode_energy_estimation.drawio
- **用途**：InfoNode理论能量计算流程图
- **位置**：M2节，第221-227行
- **导出文件名**：`infonode_energy_estimation.png`

### 2. eetor_path_selection.drawio
- **用途**：EETOR路径选择与信息收集示意图
- **位置**：M4节，第485-491行
- **导出文件名**：`eetor_path_selection.png`

## 编辑建议

### 流程图（InfoNode）
- 可以调整节点大小和位置
- 可以修改文字内容
- 可以更改颜色方案
- 确保箭头方向正确（从上到下）

### 网络拓扑图（EETOR）
- 可以调整节点位置
- 可以修改信息标注内容
- 可以调整线条粗细和颜色
- 确保能量传输路径（红色）和信息收集路径（蓝色虚线）区分明显

## 导出设置

导出PNG时，建议设置：
- **分辨率**：300 DPI（用于打印）
- **背景**：白色
- **边框**：包含边框
- **缩放**：100%

## 注意事项

1. 导出前检查所有文字是否清晰可读
2. 确保颜色对比度足够（考虑黑白打印）
3. 检查图片尺寸是否适合LaTeX（宽度约0.9\linewidth）
4. 保存源文件（.drawio格式）以便后续修改




