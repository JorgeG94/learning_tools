# Compiler
FC = nvfortran

# Base flags
FFLAGS = -O3 -Minfo

# Files
SRC_DC = dc_pure_fortran.f90
SRC_SANITY = sanity_check.f90

# Targets
ALL_TARGETS = \
	dc_multicore dc_gpu \
	sanity_multicore sanity_gpu

# Default target
all: $(ALL_TARGETS)

# dc_pure_fortran builds
dc_multicore: $(SRC_DC)
	$(FC) $(FFLAGS) -stdpar=multicore -o $@ $<

dc_gpu: $(SRC_DC)
	$(FC) $(FFLAGS) -stdpar=gpu -o $@ $<

# sanity_check builds (with OpenMP)
sanity_multicore: $(SRC_SANITY)
	$(FC) $(FFLAGS) -stdpar=multicore -fopenmp -o $@ $<

sanity_gpu: $(SRC_SANITY)
	$(FC) $(FFLAGS) -stdpar=gpu -fopenmp -o $@ $<

# Clean rule
clean:
	rm -f $(ALL_TARGETS) *.o *.mod

# Check rule
check: sanity_multicore sanity_gpu
	@echo "Running sanity_multicore..."
	./sanity_multicore 1024 1024 1024
	@echo "Running sanity_gpu..."
	./sanity_gpu 1024 1024 1024
