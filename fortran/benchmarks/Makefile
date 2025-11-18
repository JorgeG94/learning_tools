# ============================================================
#  NVFORTRAN benchmark build script
# ============================================================

FC       = nvfortran
FFLAGS   = -O3 -Minfo=all
GPUFLAGS = -stdpar=gpu -mp=gpu -gpu=mem:separate
MCFLAGS  = -stdpar=multicore

SRC_REAL    = real.f90

# Default target
all: benchmark_tridiagonal_gpu benchmark_tridiagonal_cpu

# ------------------------------------------------------------
# DO CONCURRENT - multicore CPU
# ------------------------------------------------------------
benchmark_tridiagonal_cpu: $(SRC_REAL)
	$(FC) $(FFLAGS) $(MCFLAGS) $< -o $@

# ------------------------------------------------------------
# DO CONCURRENT - GPU offload
# ------------------------------------------------------------
benchmark_tridiagonal_gpu: $(SRC_REAL)
	$(FC) $(FFLAGS) $(GPUFLAGS) $< -o $@

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
clean:
	rm -f benchmark_tridiagonal_gpu benchmark_tridiagonal_cpu

