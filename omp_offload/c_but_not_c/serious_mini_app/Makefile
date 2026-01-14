
# MPI compilers
MPICC  = mpicc
MPICXX = mpicxx

# Caliper
#CALIPER_ROOT ?= $(CALIPER_ROOT)
#CALIPER_INC  = -I$(CALIPER_ROOT)/include
#CALIPER_LIB  = -L$(CALIPER_ROOT)/lib
#CALIPER_LDLIBS = -lcaliper

CFLAGS = #-mp=gpu,multicore -O3 -gpu=cc70 #$(CALIPER_INC)
LDFLAGS = $(CALIPER_LIB)
LDLIBS  = $(CALIPER_LDLIBS)

# Targets
all: sw_cfl2_mpi 

sw_cfl2_mpi: miniapp_mpi.c
	$(MPICC) $(CFLAGS) -o $@ $< $(LDFLAGS) $(LDLIBS)

clean:
	rm -f sw_cfl2_mpi 

.PHONY: all clean

