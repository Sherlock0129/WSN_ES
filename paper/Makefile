# Makefile for LaTeX document compilation

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Compilation flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Default target
all: $(MAIN).pdf

# Compile PDF (full build with bibliography)
$(MAIN).pdf: $(MAIN).tex references.bib
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN)
	$(LATEX) $(LATEX_FLAGS) $(MAIN)

# Quick compile (no bibliography update)
quick:
	$(LATEX) $(LATEX_FLAGS) $(MAIN)

# Clean auxiliary files
clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.lof *.lot
	rm -f *.fdb_latexmk *.fls *.synctex.gz
	rm -f *.nav *.snm *.vrb

# Clean all generated files including PDF
cleanall: clean
	rm -f $(MAIN).pdf

# View PDF (platform-specific, adjust as needed)
view: $(MAIN).pdf
ifeq ($(OS),Windows_NT)
	start $(MAIN).pdf
else
	open $(MAIN).pdf || xdg-open $(MAIN).pdf || evince $(MAIN).pdf
endif

# Watch for changes and recompile (requires latexmk)
watch:
	latexmk -pvc -pdf $(MAIN)

# Count words (approximate)
count:
	@texcount -inc -incbib $(MAIN).tex

# Check for common LaTeX errors
check:
	@echo "Checking for common issues..."
	@grep -n "\\cite{}" $(MAIN).tex || echo "No empty citations found"
	@grep -n "\\ref{}" $(MAIN).tex || echo "No empty references found"
	@grep -n "TODO" $(MAIN).tex || echo "No TODOs found"

.PHONY: all quick clean cleanall view watch count check

