# makefile for SCLITE

#### Start of system configuration section. ####

srcdir = .


CC = g++
INSTALL = /usr/bin/install

CPPFLAGS = 
DEFS = -DPACKAGE_NAME=\"\" -DPACKAGE_TARNAME=\"\" -DPACKAGE_VERSION=\"\" -DPACKAGE_STRING=\"\" -DPACKAGE_BUGREPORT=\"\" -DHAVE_LIBM=1 -DSTDC_HEADERS=1 -DHAVE_SYS_TYPES_H=1 -DHAVE_SYS_STAT_H=1 -DHAVE_STDLIB_H=1 -DHAVE_STRING_H=1 -DHAVE_MEMORY_H=1 -DHAVE_STRINGS_H=1 -DHAVE_INTTYPES_H=1 -DHAVE_STDINT_H=1 -DHAVE_UNISTD_H=1 -DHAVE_UNISTD_H=1 
CFLAGS = -Os  -Wall -Wconversion
LDFLAGS = 
LIBS = -lm 

# Some System V machines do not come with libPW.
# If this is true for you, use the GNU alloca.o here.

prefix = /home/ubuntu/sctk-2.4.10
exec_prefix = ${prefix}
edit_program_name = sed 's,x,x,'

bindir = $(exec_prefix)/bin
mandir =$(prefix)/man
man1dir =$(mandir)/man1
man5dir = $(mandir)/man5

infodir = $(prefix)/info

DEFAULT_EDITOR_PROGRAM = ed
NULL_DEVICE = /dev/null
PR_PROGRAM = /bin/pr

#### End of system configuration section. ####

SHELL = /bin/sh

######################################################################
# The source files for all of the programs.
src_c = $(wildcard *.cpp)  $(filter-out ../core/main.cpp, $(wildcard ../core/*.cpp ../core/*.c ../core/lzma/*.c))

######################################################################
# The Object files for all of the library code.
src_o := $(src_c:%.cc=%.o)
src_o := $(src_o:%.cpp=%.o)
src_o := $(src_o:%.c=%.o)

PROGRAMS = asclite_test
######################################################################


all: makefile $(PROGRAMS)


COMPILE = $(CC) -c $(OFLAGS) $(DEFS) -I. -I../core $(CFLAGS)

.cpp.o:
	$(COMPILE) $<

asclite_test: $(src_o)
	$(CXX) -o $@ $(CFLAGS) $(LDFLAGS) $(src_o) $(LIBS) 

clean:
	rm -f *.o $(PROGRAMS) core.*

install:

mostlyclean: clean

distclean: clean
	rm -rf *~ config.status config.log config.cache config.h makefile

realclean: distclean
	rm -f TAGS *.info*

check:
	./asclite_test -d ../testfiles

config.sh: config.in
	autoconf config.in | sed 's/Makefile/makefile/g' > config.sh

# autoheader might not change config.hin.
config.hin: stamp-h.in
stamp-h.in: config.in
	cd $(srcdir) && autoheader
	date > $(srcdir)/stamp-h.in

config.status: config.sh
	./config.status --recheck

# config.status might not change config.h, but it changes stamp-h.
makefile: makefile.in config.status
	./config.status

# Prevent GNU make v3 from overflowing arg limit on SysV.
.NOEXPORT:
