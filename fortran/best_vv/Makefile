# ============================================================
#  NVFORTRAN benchmark build script
# ============================================================

FC       ?= nvfortran
FFLAGS   ?= -O3
GPUFLAGS ?= -stdpar=gpu -mp=gpu -gpu=mem:separate -Minfo=all
MCFLAGS  ?= -stdpar=multicore

SRC_GPU    = jik.f90 
SRC_CPU    = jki.f90

# Default target
all: gpu_vv cpu_vv 

# ------------------------------------------------------------
# DO CONCURRENT - multicore CPU
# ------------------------------------------------------------
cpu_vv: $(SRC_CPU)
	$(FC) $(FFLAGS) $(MCFLAGS) $< -o $@

# ------------------------------------------------------------
# DO CONCURRENT - GPU offload
# ------------------------------------------------------------
gpu_vv: $(SRC_GPU)
	$(FC) $(FFLAGS) $(GPUFLAGS) $< -o $@

# ------------------------------------------------------------
# Cleanup
# ------------------------------------------------------------
clean:
	rm -f gpu_vv cpu_vv 

