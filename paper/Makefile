LATEX=pdflatex
BIBTEX=bibtex

all:
	$(LATEX) main.tex
	$(BIBTEX) main
	$(LATEX) main.tex
	$(LATEX) main.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc
