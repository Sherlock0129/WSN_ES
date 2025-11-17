# S2论文模块化结构说明

**创建时间**: 2025-11-10  
**目的**: 将长篇论文按章节拆分，便于管理和编辑

---

## 📂 文件结构

```
paper/
├── s2_main.tex                    # 原始单文件版本
├── s2_main_modular.tex            # 模块化主文件（新）
└── sections/                      # 章节目录（新）
    ├── README.md                  # 本文档
    ├── 01_introduction.tex        # §1 引言
    ├── 02_related_work.tex        # §2 相关工作
    ├── 03_modeling.tex            # §3 建模
    ├── 04_mechanism_design.tex    # §4 机制设计
    ├── 05_experiments.tex         # §5 实验
    ├── 06_discussion.tex          # §6 讨论
    ├── 07_conclusion.tex          # §7 结论
    └── bibliography.tex           # 参考文献
```

---

## 🚀 使用方法

### 方法1：编译模块化版本（推荐）

**编译主文件**:
```bash
cd paper
pdflatex s2_main_modular.tex
pdflatex s2_main_modular.tex  # 运行两次以解析引用
```

**输出**: `s2_main_modular.pdf`

### 方法2：继续使用原始单文件

如果需要回到单文件版本，直接编译：
```bash
pdflatex s2_main.tex
```

---

## ✏️ 编辑章节

### 单章节编辑
现在可以单独编辑各章节文件，无需在长文件中滚动查找：

- **修改引言**: 编辑 `sections/01_introduction.tex`
- **修改实验部分**: 编辑 `sections/05_experiments.tex`
- **添加参考文献**: 编辑 `sections/bibliography.tex`

### 章节内容对应

| 文件 | 章节名称 | 行数 | 内容概要 |
|------|---------|------|---------|
| `01_introduction.tex` | Introduction | ~20行 | 研究背景、动机、贡献 |
| `02_related_work.tex` | Related Work | ~5行 | 四条主线、三方面缺口 |
| `03_modeling.tex` | Modeling | ~120行 | 节点建模、经济学解释、博弈论、目标约束 |
| `04_mechanism_design.tex` | Problems and Mechanism Design | ~120行 | E1-M1到E6-M6、命题陈述 |
| `05_experiments.tex` | Experiments | ~100行 | 实验设置、参数表、基线、消融、结果 |
| `06_discussion.tex` | Discussion | ~15行 | 可解释性、互补性、复杂度、推广边界 |
| `07_conclusion.tex` | Conclusion | ~15行 | 总结、未来工作 |
| `bibliography.tex` | Bibliography | ~35行 | 参考文献列表 |

---

## 🔧 维护建议

### 同步更新
如果修改了某个章节文件，需要：

1. **重新编译主文件**:
   ```bash
   pdflatex s2_main_modular.tex
   ```

2. **查看编译错误**: LaTeX会指出具体哪个章节文件有问题

### 合并回单文件（可选）
如果需要将所有修改合并回单文件版本：

```bash
# Linux/Mac
cat sections/01_*.tex sections/02_*.tex ... > merged_content.tex

# Windows PowerShell
Get-Content sections\*.tex | Set-Content merged_content.tex
```

然后手动复制到`s2_main.tex`的相应位置。

---

## ✅ 模块化的优势

1. **易于编辑**: 每个章节独立，避免在长文件中滚动
2. **版本控制友好**: Git diff更清晰，冲突更少
3. **协作友好**: 多人可同时编辑不同章节
4. **逻辑清晰**: 文件结构与论文结构一致
5. **重用性**: 章节可在不同文档中复用

---

## 📝 注意事项

### 编译要求
- 模块化版本需要在`paper/`目录下编译
- 确保`sections/`目录存在且包含所有章节文件
- LaTeX会自动解析`\input{}`命令

### 相对路径
所有章节文件中的引用（如`\cite{}`）不需要修改，因为它们在主文件的上下文中编译。

### 图表和公式编号
- 公式编号、表格编号、图编号会自动按章节顺序编号
- 交叉引用（`\ref{}`）正常工作

---

## 🔄 从单文件迁移的变化

**s2_main.tex（原始）**:
```latex
\section{Introduction}
... 引言内容 ...

\section{Related Work}
... 相关工作内容 ...
```

**s2_main_modular.tex（模块化）**:
```latex
\input{sections/01_introduction}
\input{sections/02_related_work}
```

**优势**: 主文件只有~80行，结构清晰；各章节文件独立管理。

---

## 📊 文件统计

| 文件 | 大小 | 行数 |
|------|------|------|
| `s2_main.tex` | ~33 KB | 569行 |
| `s2_main_modular.tex` | ~2 KB | 80行 |
| `01_introduction.tex` | ~2.5 KB | ~20行 |
| `02_related_work.tex` | ~2.5 KB | ~5行 |
| `03_modeling.tex` | ~7 KB | ~120行 |
| `04_mechanism_design.tex` | ~6 KB | ~120行 |
| `05_experiments.tex` | ~7 KB | ~100行 |
| `06_discussion.tex` | ~2 KB | ~15行 |
| `07_conclusion.tex` | ~1.5 KB | ~15行 |
| `bibliography.tex` | ~2 KB | ~35行 |

**总计**: 模块化后更易于管理，每个文件都在可控范围内。

---

**维护者**: AI Assistant  
**最后更新**: 2025-11-10

