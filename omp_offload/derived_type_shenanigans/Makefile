# NVHPC / NVFORTRAN build
NVFORTRAN ?= nvfortran
SRC        = memory_mappings.F90

# Common compile flags (yours)
FFLAGS    = -gpu=mem:separate -mp=gpu -stdpar=gpu -Minfo=all

# Macro for the OMP-enabled build (a.F90 is preprocessed because of .F90)
OMP_DEF   = -DOMP

# Executable names
EXE_NOOMP = dc_mappings 
EXE_OMP   = omp_mappins

.PHONY: all clean

all: $(EXE_NOOMP) $(EXE_OMP)

$(EXE_NOOMP): $(SRC)
	$(NVFORTRAN) $(FFLAGS) $< -o $@

$(EXE_OMP): $(SRC)
	$(NVFORTRAN) $(FFLAGS) $(OMP_DEF) $< -o $@

clean:
	rm -f $(EXE_NOOMP) $(EXE_OMP) *.o *.mod

