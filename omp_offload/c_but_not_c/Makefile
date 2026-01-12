CC = nvc
CFLAGS = -mp=gpu,multicore -O3

# Default parameters
GRID ?= 500
NITER ?= 1000
YIELDSTEP ?= 100

# Targets
all: sw_transfer sw_resident sw_yieldstep sw_verbose

sw_transfer: sw_transfer.c
	$(CC) $(CFLAGS) -o sw_transfer sw_transfer.c

sw_resident: sw_resident.c
	$(CC) $(CFLAGS) -o sw_resident sw_resident.c

sw_yieldstep: sw_yieldstep.c
	$(CC) $(CFLAGS) -o sw_yieldstep sw_yieldstep.c

sw_verbose: sw_verbose.c
	$(CC) $(CFLAGS) -o sw_verbose sw_verbose.c

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
	rm -f sw_transfer sw_resident sw_yieldstep sw_verbose sw_mini

.PHONY: all clean compare \
	run-transfer-cpu run-transfer-gpu \
	run-resident-cpu run-resident-gpu \
	run-yieldstep-cpu run-yieldstep-gpu \
	run-verbose-cpu run-verbose-gpu
