# Makefile
FC      := nvfortran
COMMON  := -Minfo=all -gpu=mem:separate
OMP     := -mp=gpu
STDPAR  := -stdpar=gpu

PROGS := asynch_omp_dc dc_omp

all: $(PROGS)

# Link
asynch_omp_dc: asynch_omp_dc.o
	$(FC) $(COMMON) $(OMP) -o $@ $^

dc_omp: dc_omp.o
	$(FC) $(COMMON) $(OMP) -o $@ $^

# Compile (per-file flags)
asynch_omp_dc.o: asynch_omp_dc.f90
	$(FC) $(COMMON) $(OMP) -c $< -o $@

dc_omp.o: dc_omp.f90
	$(FC) $(COMMON) $(OMP) $(STDPAR) -c $< -o $@

clean:
	rm -f $(PROGS) *.o *.mod

.PHONY: all clean

