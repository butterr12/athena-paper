#
LATEX=latex
BIBLIO=bibtex
DVIPS=dvips
PS2PDF=ps2pdf

shortrep:
	$(LATEX) shortrep
	$(LATEX) shortrep
	$(DVIPS) -f shortrep.dvi > shortrep.ps


thesis:	
	$(LATEX) thesis
	$(LATEX) thesis
	$(DVIPS) -f thesis.dvi > thesis.ps

users:	
	$(LATEX) users
	$(LATEX) users
	$(DVIPS) -f users.dvi > users.ps

biblio:
	$(LATEX) thesis
	$(BIBLIO) thesis

psandpdf: thesis.dvi users.dvi
	$(DVIPS) -f thesis.dvi > thesis.ps
	$(PS2PDF) thesis.ps thesis.pdf
	$(DVIPS) -f users.dvi > users.ps
	$(PS2PDF) users.ps users.pdf
	$(DVIPS) -f shortrep.dvi > shortrep.ps
	$(PS2PDF) shortrep.ps shortrep.pdf

real-clean:
	rm -f *~
	rm -f *.dvi
	rm -f *.aux
	rm -f *.toc
	rm -f *.tof
	rm -f *.loc
	rm -f *.lof
	rm -f *.log
	rm -f *.lot
	rm -f *.bbl
	rm -f *.blg



