#########   Main Makefile for the sctk  #############


####  Creation Date:  Aug 26, 2004
MAKE=make
### Set the installation prefix directory to the default.  (If
### relative paths are used, the are relative to two directories below
### 'src'.)
PREFIX=`cd ../..; pwd`

all clean distclean install:
	(cd asclite; $(MAKE) $@)
	(cd sclite; $(MAKE) $@)
	(cd rfilter1; $(MAKE) $@)
	(cd csrfilt; $(MAKE) $@)
	(cd chfilt; $(MAKE) $@)
	(cd hamzaNorm; $(MAKE) $@)
	(cd tanweenFilt; $(MAKE) $@)
	(cd acomp; $(MAKE) $@)	
	(cd def_art; $(MAKE) $@)
	(cd hubscr; $(MAKE) $@)
	(cd utf_filt; $(MAKE) $@)
	(cd rttmSort; $(MAKE) $@)
	(cd rttm2ctm; $(MAKE) $@)
	(cd rttmSmooth; $(MAKE) $@)
	(cd spkr2sad; $(MAKE) $@)
	(cd mergectm2rttm; $(MAKE) $@)
	(cd align2html; $(MAKE) $@)
	(cd md-eval; $(MAKE) $@)
	(cd stm2rttm; $(MAKE) $@)
	(cd stmValidator; $(MAKE) $@)
	(cd rttmValidator; $(MAKE) $@)
	(cd ctmValidator; $(MAKE) $@)
	(cd slatreport; $(MAKE) $@)

config:
	(cd asclite; $(MAKE) $@)
	(cd sclite; sh config.sh -prefix=${PREFIX})
	(cd rfilter1; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)
	(cd csrfilt; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd chfilt; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd hamzaNorm; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd tanweenFilt; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd acomp; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd def_art; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd hubscr; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd utf_filt; sh config.sh -prefix=${PREFIX})
	(cd rttmSort; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd rttmSmooth; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd rttm2ctm; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd spkr2sad; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd mergectm2rttm; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd align2html; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd md-eval; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)	
	(cd stm2rttm; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)
	(cd stmValidator; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)
	(cd rttmValidator; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)
	(cd ctmValidator; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)
	(cd slatreport; sed -e "s:PREFIX=.*:PREFIX=${PREFIX}:" < makefile.in > makefile)

checkFast:
	(cd asclite; $(MAKE) check)
	(cd rfilter1; $(MAKE) check)
	(cd csrfilt; $(MAKE) check)
	(cd chfilt; $(MAKE) check)
	(cd hamzaNorm; $(MAKE) check)
	(cd tanweenFilt; $(MAKE) check)
	(cd acomp; $(MAKE) check)
	(cd def_art; $(MAKE) check)
	(cd hubscr; $(MAKE) check)
	(cd utf_filt; $(MAKE) check)
	(cd rttmSort; $(MAKE) check)
	(cd rttm2ctm; $(MAKE) check)
	(cd mergectm2rttm; $(MAKE) check)
	(cd align2html; $(MAKE) check)
	(cd md-eval; $(MAKE) check)
	(cd stmValidator; $(MAKE) check)
	(cd rttmValidator; $(MAKE) check)
	(cd ctmValidator; $(MAKE) check)
	(cd slatreport; $(MAKE) check)

check: all checkFast
	(cd sclite; $(MAKE) $@)
