# Makefile for CUDA MPI examples

# Compiler
NVCC = nvcc
MPICC = mpicc

# Find MPI installation
MPI_HOME ?= $(shell mpicc --showme:compile | sed 's/-I\([^[:space:]]*\).*/\1/' | sed 's|/include||')
ifeq ($(MPI_HOME),)
    MPI_HOME = /usr/lib/x86_64-linux-gnu/openmpi
endif

# Compiler flags
NVCC_FLAGS = -O3 -arch=sm_60 --compiler-options -Wall
MPI_INCLUDE = $(shell mpicc --showme:compile)
MPI_LIBS = -lmpi

# Targets
TARGETS = aware unaware
SOURCES = aware.cu unaware.cu

# Default target
all: $(TARGETS)

# Build CUDA-aware MPI version
aware: aware.cu
	@echo "Compiling CUDA-aware MPI version..."
	$(NVCC) $(NVCC_FLAGS) $(MPI_INCLUDE) -o $@ $< $(MPI_LIBS)

# Build non-CUDA-aware MPI version
unaware: unaware.cu
	@echo "Compiling non-CUDA-aware MPI version..."
	$(NVCC) $(NVCC_FLAGS) $(MPI_INCLUDE) -o $@ $< $(MPI_LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGETS)

# Run targets
run-aware: aware
	@echo "Running CUDA-aware MPI example..."
	mpirun -np 2 ./aware

run-unaware: unaware
	@echo "Running non-CUDA-aware MPI example..."
	mpirun -np 2 ./unaware

run-both: aware unaware
	@echo "=== Running CUDA-aware version ==="
	mpirun -np 2 ./aware
	@echo ""
	@echo "=== Running non-CUDA-aware version ==="
	mpirun -np 2 ./unaware

# Profile targets (requires nsys)
profile-aware: aware
	@echo "Profiling CUDA-aware MPI..."
	nsys profile --trace=cuda,mpi --output=aware_profile mpirun -np 2 ./aware

profile-unaware: unaware
	@echo "Profiling non-CUDA-aware MPI..."
	nsys profile --trace=cuda,mpi --output=unaware_profile mpirun -np 2 ./unaware

# Debug build
debug: NVCC_FLAGS += -g -G 
debug: $(TARGETS)

# Check MPI installation
check-mpi:
	@echo "MPI Compiler: $(shell which mpicc)"
	@echo "MPI Home: $(MPI_HOME)"
	@echo "MPI Include: $(MPI_INCLUDE)"
	@echo "MPI Libs: $(MPI_LIBS)"
	@echo "CUDA Compiler: $(shell which nvcc)"
	@nvcc --version

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build both versions (default)"
	@echo "  aware        - Build CUDA-aware version only"
	@echo "  unaware      - Build non-CUDA-aware version only"
	@echo "  clean        - Remove compiled binaries"
	@echo "  run-aware    - Build and run CUDA-aware version"
	@echo "  run-unaware  - Build and run non-CUDA-aware version"
	@echo "  run-both     - Build and run both versions"
	@echo "  debug        - Build with debug flags"
	@echo "  profile-*    - Profile with nsys (requires NVIDIA Nsight Systems)"
	@echo "  check-mpi    - Show MPI and CUDA configuration"
	@echo "  help         - Show this help message"

.PHONY: all clean run-aware run-unaware run-both profile-aware profile-unaware debug check-mpi help
