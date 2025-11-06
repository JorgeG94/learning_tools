# ============================================================
#  NVFORTRAN benchmark build script
# ============================================================

FC       ?= nvfortran
FFLAGS   ?= -O3
GPUFLAGS ?= -stdpar=gpu -mp=gpu -gpu=mem:separate
MCFLAGS  ?= -stdpar=multicore

SRC_SERIAL  = serial_buffer_benchmark.f90
SRC_DC      = do_concurrent_v_benchmark.f90
SRC_REAL    = real.f90
SRC_REAL_KFIRST = real-kfirst.f90

# Default target
all: benchmark_serial benchmark_serial_dc benchmark_multicore_dc benchmark_gpu_dc benchmark_tridiagonal_gpu benchmark_tridiagonal_cpu benchmark_tridiagonal_kfirst_cpu benchmark_tridiagonal_kfirst_gpu

# ------------------------------------------------------------
# Serial baseline (DO-DO-DO)
# ------------------------------------------------------------
benchmark_serial: $(SRC_SERIAL)
	$(FC) $(FFLAGS) $< -o $@

# ------------------------------------------------------------
# DO CONCURRENT - serial (CPU only)
# ------------------------------------------------------------
benchmark_serial_dc: $(SRC_DC)
	$(FC) $(FFLAGS) $< -o $@

# ------------------------------------------------------------
# DO CONCURRENT - multicore CPU
# ------------------------------------------------------------
benchmark_tridiagonal_cpu: $(SRC_REAL)
	$(FC) $(FFLAGS) $(MCFLAGS) $< -o $@

benchmark_tridiagonal_kfirst_cpu: $(SRC_REAL_KFIRST)
	$(FC) $(FFLAGS) $(MCFLAGS) $< -o $@

benchmark_multicore_dc: $(SRC_DC)
	$(FC) $(FFLAGS) $(MCFLAGS) $< -o $@
# ------------------------------------------------------------
# DO CONCURRENT - GPU offload
# ------------------------------------------------------------
benchmark_tridiagonal_gpu: $(SRC_REAL)
	$(FC) $(FFLAGS) $(GPUFLAGS) $< -o $@

benchmark_tridiagonal_kfirst_gpu: $(SRC_REAL_KFIRST)
	$(FC) $(FFLAGS) $(GPUFLAGS) $< -o $@

benchmark_gpu_dc: $(SRC_DC)
	$(FC) $(FFLAGS) $(GPUFLAGS) $< -o $@

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
clean:
	rm -f benchmark_serial benchmark_serial_dc benchmark_multicore_dc benchmark_gpu_dc benchmark_tridiagonal_gpu benchmark_tridiagonal_cpu

