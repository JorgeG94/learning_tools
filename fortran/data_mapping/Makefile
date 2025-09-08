# Simple Makefile for cartographer.f90 with three build variants

NVFORTRAN ?= nvfortran
SRC := cartographer.f90

CPU_FLAGS      := -mp=multicore
NOMAP_FLAGS    := -stdpar=gpu -mp=multicore -gpu=mem:separate
WITHMAP_FLAGS  := -stdpar=gpu -mp=gpu      -gpu=mem:separate

.PHONY: all clean

all: cpu_only no_omp_mapping with_omp_mapping

cpu_only: $(SRC)
	$(NVFORTRAN) $(CPU_FLAGS) $< -o $@

no_omp_mapping: $(SRC)
	$(NVFORTRAN) $(NOMAP_FLAGS) $< -o $@

with_omp_mapping: $(SRC)
	$(NVFORTRAN) $(WITHMAP_FLAGS) $< -o $@

clean:
	rm -f cpu_only no_omp_mapping with_omp_mapping *.o *.mod

