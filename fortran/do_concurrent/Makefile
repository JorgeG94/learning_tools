# Compiler
FC = nvfortran

# Base flags
FFLAGS = -O3 -Minfo

# Files
SRC_HELLO = hello.F90
SRC_DC = dc_pure_fortran.f90
SRC_SANITY = sanity_check_dc.f90
SRC_OMP = sanity_check_openmp.f90

# Targets
ALL_TARGETS = \
	dc_multicore dc_gpu hello \
	sanity_multicore sanity_gpu sanity_omp_gpu

# Default target
all: $(ALL_TARGETS)

hello: $(SRC_HELLO)
	$(FC) $(FFLAGS) -mp=gpu -o $@ $<

# dc_pure_fortran builds
dc_multicore: $(SRC_DC)
	$(FC) $(FFLAGS) -stdpar=multicore -o $@ $<

dc_gpu: $(SRC_DC)
	$(FC) $(FFLAGS) -stdpar=gpu -o $@ $<

# sanity_check builds (with OpenMP)
sanity_multicore: $(SRC_SANITY)
	$(FC) $(FFLAGS) -stdpar=multicore -mp -o $@ $<

sanity_gpu: $(SRC_SANITY)
	$(FC) $(FFLAGS) -stdpar=gpu -mp -o $@ $<

sanity_omp_gpu: $(SRC_OMP)
	$(FC) $(FFLAGS) -mp=gpu -o $@ $<

# Clean rule
clean:
	rm -f $(ALL_TARGETS) *.o *.mod

# Check rule
check: sanity_multicore sanity_gpu
	@echo "Running sanity_multicore..."
	./sanity_multicore 1024 1024 1024
	@echo "Running sanity_gpu..."
	./sanity_gpu 1024 1024 1024
	@echo "running sanity omp" 
	./sanity_omp_gpu 1024 1024 1024
