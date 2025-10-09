# Fortran compiler & flags (override on the command line if needed)
FC     ?= gfortran
FFLAGS ?= -O3
LIBS   ?=

.PHONY: all clean benchmark

all: dc_dgemm do_dgemm

dc_dgemm: dc_dgemm.f90
	$(FC) $(FFLAGS) $< -o $@ $(LIBS)

do_dgemm: do_dgemm.f90
	$(FC) $(FFLAGS) $< -o $@ $(LIBS)

benchmark: all
	./do_dgemm 512 512 512 10 2
	./dc_dgemm 512 512 512 10 2

clean:
	rm -f dc_dgemm do_dgemm *.o *.mod

