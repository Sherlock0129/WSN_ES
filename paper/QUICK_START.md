# S2论文快速开始指南

## 🚀 立即开始

### 编译模块化版本（推荐）
```powershell
cd D:\University\WSN_ES\paper
pdflatex s2_main_modular.tex
pdflatex s2_main_modular.tex
```

**输出**: `s2_main_modular.pdf` (8页)

---

## 📂 文件结构一览

```
paper/
├── s2_main.tex                    # 原始单文件 (569行) - 保留备份
├── s2_main_modular.tex            # 模块化主文件 (80行) ⭐ 日常使用
└── sections/                      # 章节目录
    ├── 01_introduction.tex        # §1 引言 (3.7KB)
    ├── 02_related_work.tex        # §2 相关工作 (3KB)
    ├── 03_modeling.tex            # §3 建模 (11KB)
    ├── 04_mechanism_design.tex    # §4 机制设计 (11KB)
    ├── 05_experiments.tex         # §5 实验 (14KB)
    ├── 06_discussion.tex          # §6 讨论 (2.5KB)
    ├── 07_conclusion.tex          # §7 结论 (1.8KB)
    └── bibliography.tex           # 参考文献 (2.1KB)
```

---

## ✏️ 快速编辑

### 修改某个章节
1. 打开对应文件（如 `sections/05_experiments.tex`）
2. 编辑保存
3. 编译: `pdflatex s2_main_modular.tex`

### 查找内容
不再需要在569行文件中搜索！直接打开对应章节：
- 修改引言贡献 → `01_introduction.tex`
- 添加实验参数 → `05_experiments.tex`
- 修改结论 → `07_conclusion.tex`

---

## 🎯 常用章节内容

| 章节文件 | 主要内容 | 行数 |
|---------|---------|------|
| `01_introduction.tex` | 研究背景、本文贡献 | ~20 |
| `02_related_work.tex` | 四条主线、三方面缺口 | ~5 |
| `03_modeling.tex` | 节点模型、AOEI、博弈论 | ~120 |
| `04_mechanism_design.tex` | E1-M1到E6-M6、算法伪代码 | ~120 |
| `05_experiments.tex` | 参数表、基线表、消融设计 | ~100 |
| `06_discussion.tex` | 可解释性、复杂度、推广边界 | ~15 |
| `07_conclusion.tex` | 总结、未来工作 | ~15 |

---

## 💡 提示

### ✅ 推荐做法
- 日常使用模块化版本 `s2_main_modular.tex`
- 每次修改后运行两次pdflatex（解析交叉引用）
- Git提交时逐章节commit

### ⚠️ 注意事项
- 章节文件不包含 `\begin{document}` 等导言区内容
- 修改表格/公式后建议重新编译两次
- 保持 `s2_main.tex` 作为备份（如期刊要求单文件）

---

## 📞 帮助

详细文档：
- **使用说明**: `sections/README.md`
- **完整指南**: `MODULAR_STRUCTURE_GUIDE.md`
- **改进建议**: `S2_IMPROVEMENT_RECOMMENDATIONS.md`

---

**最后更新**: 2025-11-10  
**状态**: ✅ 可用

