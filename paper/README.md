# IEEE LaTeX Project

This directory contains the LaTeX source files for the IEEE paper on energy-aware scheduling and routing in wireless sensor networks.

## Project Structure

```
paper/
├── main.tex           # Main LaTeX document
├── references.bib     # Bibliography file
├── figures/          # Directory for figures and images
├── sections/         # Directory for separate section files (optional)
└── README.md         # This file
```

## Requirements

To compile this document, you need:
- A LaTeX distribution (e.g., TeX Live, MiKTeX, or MacTeX)
- The IEEEtran document class (usually included in standard distributions)

### Required LaTeX Packages
- cite
- amsmath, amssymb, amsfonts
- algorithmic
- graphicx
- textcomp
- xcolor
- hyperref

## Compilation Instructions

### Using pdflatex (recommended)

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (automated)

```bash
cd paper
latexmk -pdf main.tex
```

### Using Make (if you have a Makefile)

```bash
cd paper
make
```

## Document Structure

The main document (`main.tex`) includes:
1. **Abstract**: Brief overview of the work
2. **Introduction**: Motivation and contributions
3. **Related Work**: Literature review
4. **System Model**: Problem formulation and network architecture
5. **Proposed Approach**: DQN and DDPG-based algorithms
6. **Experimental Results**: Performance evaluation
7. **Conclusion**: Summary and future work

## Adding Figures

Place your figure files in the `figures/` directory and reference them in the document:

```latex
\begin{figure}[htbp]
\centerline{\includegraphics[width=0.48\textwidth]{figures/your_figure.pdf}}
\caption{Your caption here.}
\label{fig:your_label}
\end{figure}
```

## Bibliography Management

Edit `references.bib` to add your references. Use standard BibTeX format:

```bibtex
@article{key,
  author = {Author Name},
  title = {Paper Title},
  journal = {Journal Name},
  year = {2024},
  volume = {1},
  pages = {1--10}
}
```

Cite references in the text using `\cite{key}`.

## IEEE Formatting Guidelines

This template follows IEEE conference paper format:
- Two-column layout
- 10pt font
- Letter size (8.5" × 11")
- Standard IEEE margins

For journal papers, change the document class option:
```latex
\documentclass[journal]{IEEEtran}
```

## Tips

1. **Clean build**: Remove auxiliary files with `latexmk -c` or manually delete `.aux`, `.log`, `.bbl`, etc.
2. **Draft mode**: Add `draft` option to document class for faster compilation during editing
3. **Line numbers**: Add `\usepackage{lineno}` and `\linenumbers` for review versions
4. **Comments**: Use `\usepackage{todonotes}` for inline comments during writing

## Common Issues

- **Bibliography not showing**: Make sure to run `bibtex` after the first `pdflatex` run
- **Figures not found**: Check that figure paths are correct relative to `main.tex`
- **Missing packages**: Install missing packages using your LaTeX distribution's package manager

## Resources

- [IEEEtran Homepage](http://www.michaelshell.org/tex/ieeetran/)
- [IEEE Author Center](https://ieeeauthorcenter.ieee.org/)
- [LaTeX Project](https://www.latex-project.org/)

## License

Please follow IEEE copyright policies when submitting papers.

