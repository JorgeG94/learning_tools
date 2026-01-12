CC = nvc
FC = nvfortran
NVCC = nvcc
CFLAGS = -mp=gpu,multicore -O3
FFLAGS = -mp=gpu,multicore -O3
NVFLAGS = -O3 -arch=native

# Default parameters
GRID ?= 500
NITER ?= 1000
YIELDSTEP ?= 100

# MPI compiler
MPICC = mpicc

# Targets
all: sw_transfer sw_resident sw_yieldstep sw_verbose sw_verbose_f90 sw_cfl sw_cfl2 sw_cfl2_f90 sw_cfl2_mpi sw_async test_async test_async_cuda

sw_transfer: sw_transfer.c
	$(CC) $(CFLAGS) -o sw_transfer sw_transfer.c

sw_resident: sw_resident.c
	$(CC) $(CFLAGS) -o sw_resident sw_resident.c

sw_yieldstep: sw_yieldstep.c
	$(CC) $(CFLAGS) -o sw_yieldstep sw_yieldstep.c

sw_verbose: sw_verbose.c
	$(CC) $(CFLAGS) -o sw_verbose sw_verbose.c

sw_async: sw_async.c
	$(CC) $(CFLAGS) -o sw_async sw_async.c

test_async: test_async.c
	$(CC) $(CFLAGS) -o test_async test_async.c

test_async_cuda: test_async_cuda.cu
	$(NVCC) $(NVFLAGS) -o test_async_cuda test_async_cuda.cu

sw_cuda_async: sw_cuda_async.cu
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp -o sw_cuda_async sw_cuda_async.cu

sw_verbose_f90: sw_verbose.f90
	$(FC) $(FFLAGS) -o sw_verbose_f90 sw_verbose.f90

sw_cfl: sw_cfl.f90
	$(FC) $(FFLAGS) -o sw_cfl sw_cfl.f90

sw_cfl2: sw_cfl2.c
	$(CC) $(CFLAGS) -o sw_cfl2 sw_cfl2.c

sw_cfl2_f90: sw_cfl2.f90
	$(FC) $(FFLAGS) -o sw_cfl2_f90 sw_cfl2.f90

sw_cfl2_mpi: sw_cfl2_mpi.c
	$(MPICC) -mp=gpu -O3 -o sw_cfl2_mpi sw_cfl2_mpi.c

# Quick test of async behavior
test-async: test_async
	OMP_TARGET_OFFLOAD=mandatory ./test_async

test-async-cuda: test_async_cuda
	./test_async_cuda

# CUDA shallow water with async
run-cuda-sync: sw_cuda_async
	./sw_cuda_async $(GRID) $(NITER) $(YIELDSTEP) 0

run-cuda-async: sw_cuda_async
	./sw_cuda_async $(GRID) $(NITER) $(YIELDSTEP) 1

compare-cuda: sw_cuda_async
	@echo "=== CUDA SYNC ==="
	./sw_cuda_async $(GRID) $(NITER) $(YIELDSTEP) 0
	@echo ""
	@echo "=== CUDA ASYNC ==="
	./sw_cuda_async $(GRID) $(NITER) $(YIELDSTEP) 1

# Run shallow water - transfer version (slow due to per-kernel transfers)
run-transfer-cpu: sw_transfer
	OMP_TARGET_OFFLOAD=disabled ./sw_transfer $(GRID) $(NITER)

run-transfer-gpu: sw_transfer
	OMP_TARGET_OFFLOAD=mandatory ./sw_transfer $(GRID) $(NITER)

# Run shallow water - resident version (fast, data stays on GPU)
run-resident-cpu: sw_resident
	OMP_TARGET_OFFLOAD=disabled ./sw_resident $(GRID) $(NITER)

run-resident-gpu: sw_resident
	OMP_TARGET_OFFLOAD=mandatory ./sw_resident $(GRID) $(NITER)

# Run shallow water - yieldstep version (periodic transfers for output)
run-yieldstep-cpu: sw_yieldstep
	OMP_TARGET_OFFLOAD=disabled ./sw_yieldstep $(GRID) $(NITER) $(YIELDSTEP)

run-yieldstep-gpu: sw_yieldstep
	OMP_TARGET_OFFLOAD=mandatory ./sw_yieldstep $(GRID) $(NITER) $(YIELDSTEP)

# Run shallow water - verbose version (heavy output transfers)
run-verbose-cpu: sw_verbose
	OMP_TARGET_OFFLOAD=disabled ./sw_verbose $(GRID) $(NITER) $(YIELDSTEP)

run-verbose-gpu: sw_verbose
	OMP_TARGET_OFFLOAD=mandatory ./sw_verbose $(GRID) $(NITER) $(YIELDSTEP)

# Run shallow water - Fortran verbose version
run-verbose-f90-cpu: sw_verbose_f90
	OMP_TARGET_OFFLOAD=disabled ./sw_verbose_f90 $(GRID) $(NITER) $(YIELDSTEP)

run-verbose-f90-gpu: sw_verbose_f90
	OMP_TARGET_OFFLOAD=mandatory ./sw_verbose_f90 $(GRID) $(NITER) $(YIELDSTEP)

# Compare C and Fortran verbose versions
compare-c-fortran: sw_verbose sw_verbose_f90
	@echo "=== C VERSION ==="
	OMP_TARGET_OFFLOAD=mandatory ./sw_verbose $(GRID) $(NITER) $(YIELDSTEP)
	@echo ""
	@echo "=== FORTRAN VERSION ==="
	OMP_TARGET_OFFLOAD=mandatory ./sw_verbose_f90 $(GRID) $(NITER) $(YIELDSTEP)

# CFL timing estimator (estimates wall-clock for 2-day simulation)
# Usage: make run-cfl GRID=10000 NITER=1000 DOMAIN_KM=100 DEPTH_M=10
DOMAIN_KM ?= 100
DEPTH_M ?= 10

run-cfl: sw_cfl
	OMP_TARGET_OFFLOAD=mandatory ./sw_cfl $(GRID) $(NITER) $(DOMAIN_KM) $(DEPTH_M)

# Second-order CFL timing (C and Fortran)
run-cfl2: sw_cfl2
	OMP_TARGET_OFFLOAD=mandatory ./sw_cfl2 $(GRID) $(NITER) $(YIELDSTEP) $(DOMAIN_KM) $(DEPTH_M)

run-cfl2-f90: sw_cfl2_f90
	OMP_TARGET_OFFLOAD=mandatory ./sw_cfl2_f90 $(GRID) $(NITER) $(DOMAIN_KM) $(DEPTH_M)

# MPI version (set NPROCS for number of ranks)
NPROCS ?= 2
GATHER ?= 0

run-cfl2-mpi: sw_cfl2_mpi
	OMP_TARGET_OFFLOAD=mandatory mpirun -np $(NPROCS) ./sw_cfl2_mpi $(GRID) $(NITER) $(YIELDSTEP) $(GATHER) $(DOMAIN_KM) $(DEPTH_M)

# Compare distributed vs gather modes
compare-io-modes: sw_cfl2_mpi
	@echo "=== DISTRIBUTED I/O (scalable) ==="
	OMP_TARGET_OFFLOAD=mandatory mpirun -np $(NPROCS) ./sw_cfl2_mpi $(GRID) $(NITER) $(YIELDSTEP) 0 $(DOMAIN_KM) $(DEPTH_M)
	@echo ""
	@echo "=== GATHER I/O (rank 0) ==="
	OMP_TARGET_OFFLOAD=mandatory mpirun -np $(NPROCS) ./sw_cfl2_mpi $(GRID) $(NITER) $(YIELDSTEP) 1 $(DOMAIN_KM) $(DEPTH_M)

# Compare first vs second order
compare-order: sw_cfl sw_cfl2
	@echo "=== FIRST ORDER ==="
	OMP_TARGET_OFFLOAD=mandatory ./sw_cfl $(GRID) $(NITER) $(DOMAIN_KM) $(DEPTH_M)
	@echo ""
	@echo "=== SECOND ORDER ==="
	OMP_TARGET_OFFLOAD=mandatory ./sw_cfl2 $(GRID) $(NITER) $(DOMAIN_KM) $(DEPTH_M)

# Run shallow water - async version (compare sync vs async)
run-async-sync: sw_async
	OMP_TARGET_OFFLOAD=mandatory ./sw_async $(GRID) $(NITER) $(YIELDSTEP) 0

run-async-async: sw_async
	OMP_TARGET_OFFLOAD=mandatory ./sw_async $(GRID) $(NITER) $(YIELDSTEP) 1

# Compare sync vs async
compare-async: sw_async
	@echo "=== SYNC transfers ==="
	OMP_TARGET_OFFLOAD=mandatory ./sw_async $(GRID) $(NITER) $(YIELDSTEP) 0
	@echo ""
	@echo "=== ASYNC transfers ==="
	OMP_TARGET_OFFLOAD=mandatory ./sw_async $(GRID) $(NITER) $(YIELDSTEP) 1

# Compare all four versions on GPU
compare: sw_transfer sw_resident sw_yieldstep sw_verbose
	@echo "=============================================="
	@echo "=== 1. TRANSFER: Data moved EVERY kernel ==="
	@echo "=============================================="
	OMP_TARGET_OFFLOAD=mandatory ./sw_transfer $(GRID) $(NITER)
	@echo ""
	@echo "=============================================="
	@echo "=== 2. RESIDENT: Data stays on GPU        ==="
	@echo "=============================================="
	OMP_TARGET_OFFLOAD=mandatory ./sw_resident $(GRID) $(NITER)
	@echo ""
	@echo "=============================================="
	@echo "=== 3. YIELDSTEP: Light output (every $(YIELDSTEP)) ==="
	@echo "=============================================="
	OMP_TARGET_OFFLOAD=mandatory ./sw_yieldstep $(GRID) $(NITER) $(YIELDSTEP)
	@echo ""
	@echo "=============================================="
	@echo "=== 4. VERBOSE: Heavy output (every $(YIELDSTEP))  ==="
	@echo "=============================================="
	OMP_TARGET_OFFLOAD=mandatory ./sw_verbose $(GRID) $(NITER) $(YIELDSTEP)

clean:
	rm -f sw_transfer sw_resident sw_yieldstep sw_verbose sw_async test_async test_async_cuda sw_cuda_async sw_mini sw_verbose_f90 sw_cfl sw_cfl2 sw_cfl2_f90 sw_cfl2_mpi *.mod

.PHONY: all clean compare compare-async compare-c-fortran \
	run-transfer-cpu run-transfer-gpu \
	run-resident-cpu run-resident-gpu \
	run-yieldstep-cpu run-yieldstep-gpu \
	run-verbose-cpu run-verbose-gpu \
	run-verbose-f90-cpu run-verbose-f90-gpu \
	run-async-sync run-async-async
