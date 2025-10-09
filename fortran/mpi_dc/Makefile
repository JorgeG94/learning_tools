# Minimal build-only Makefile for gemm.f90
# Usage:
#   make         # builds gemm.exe
#   make clean   # removes build artifacts

MPIFC  = mpifort
FFLAGS = -O3 -stdpar=gpu -mp=gpu
SRC    = gemm.f90
EXE    = gemm.exe

.PHONY: all clean

all: $(EXE)

$(EXE): $(SRC)
	$(MPIFC) $(FFLAGS) -o $@ $<

clean:
	rm -f $(EXE) *.o *.mod

