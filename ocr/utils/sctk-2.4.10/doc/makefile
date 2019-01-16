# Makefile for SCTK
# Creation Date: May 18th, 2007
MAKE=make

POD2MAN = pod2man
POD2HTML = pod2html

PODS =  align2html \
	asclite \
	ctmValidator \
	mergectm2rttm

all:
	for p in $(PODS); do \
		$(POD2MAN) $$p.pod -o $$p.1; \
		$(POD2HTML) $$p.pod > $$p.html; \
	done 2> /dev/null

man:
	for p in $(PODS); do \
		$(POD2MAN) $$p.pod -o $$p.1; \
	done 2> /dev/null

html:
	for p in $(PODS); do \
		$(POD2HTML) $$p.pod > $$p.html; \
	done 2> /dev/null

clean:
	for p in $(PODS); do \
		rm -f $$p.1 $$p.html pod2htm?.tmp; \
	done 2> /dev/null

