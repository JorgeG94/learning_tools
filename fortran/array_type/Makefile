# make                     nvfortran, BACKEND=acc: -stdpar=gpu + OpenACC data
# make BACKEND=omp         nvfortran, -stdpar=gpu + OpenMP target data
# make BACKEND=none        nvfortran, CPU only
# make COMPILER=gnu        gfortran >= 15 (DO CONCURRENT REDUCE is F2023), CPU only
# make COMPILER=intel      ifx, CPU only
# make NO_FINAL=1          build devarray without finalizers
# make run / make test / make generate / make check-generated
# (source mom6_env.sh first for nvfortran)

COMPILER ?= nv
GPU_CC ?= cc70

ifeq ($(COMPILER),nv)
  BACKEND ?= acc
  FC := nvfortran
  FFLAGS := -O2 -Mpreprocess -Minform=warn
  MODFLAG := -module
  # do concurrent -> device kernels (-stdpar=gpu); the data lifecycle stays
  # with the directives inside devarray (mem:separate, as in roundabout/terco).
  ifeq ($(BACKEND),acc)
    FFLAGS += -stdpar=gpu -acc -gpu=$(GPU_CC),mem:separate -Minfo=accel,stdpar \
              -DDEVARRAY_ACC -DDEVARRAY_DEVICE
  else ifeq ($(BACKEND),omp)
    # OpenMP has no free-memory query: on NVIDIA use cudaMemGetInfo (-cuda links cudart)
    FFLAGS += -stdpar=gpu -mp=gpu -gpu=$(GPU_CC),mem:separate -cuda -Minfo=mp,stdpar \
              -DDEVARRAY_OMP -DDEVARRAY_DEVICE -DDEVARRAY_CUDA_MEMINFO
  else ifneq ($(BACKEND),none)
    $(error BACKEND must be acc, omp or none)
  endif
else ifeq ($(COMPILER),intel)
  BACKEND := none
  FC := ifx
  # 5423: locality specs are ignored in a serial build, which is fine here
  FFLAGS := -O2 -fpp -stand f23 -warn all -check bounds -diag-disable=5423
  MODFLAG := -module
else
  BACKEND := none
  FC := gfortran
  # -Wno-external-argument-mismatch: gfortran 15.1 bug, see mem_collective.F90
  FFLAGS := -O2 -cpp -std=f2023 -Wall -Wextra -Wno-compare-reals -Wno-external-argument-mismatch -fcheck=all
  MODFLAG := -J
endif

ifeq ($(NO_FINAL),1)
  FFLAGS += -DDEVARRAY_NO_FINAL
  FINAL_TAG := -nofinal
endif

BUILD := build/$(COMPILER)-$(BACKEND)$(FINAL_TAG)
LIB_SRC := src/mem_ledger_core.F90 src/mem_ledger.F90 src/mem_ledger_backend.F90 \
           src/mem_device.F90 src/devarray.F90 src/mem_collective.F90
LIB_OBJ := $(patsubst src/%.F90,$(BUILD)/%.o,$(LIB_SRC))
TESTS := test_ledger test_collective test_must_fail

all: $(BUILD)/swe $(addprefix $(BUILD)/,$(TESTS))

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.F90 | $(BUILD)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) -c $< -o $@

# module dependencies
$(BUILD)/mem_ledger.o $(BUILD)/mem_ledger_backend.o $(BUILD)/mem_device.o: $(BUILD)/mem_ledger_core.o
$(BUILD)/devarray.o: $(BUILD)/mem_ledger.o $(BUILD)/mem_ledger_backend.o $(BUILD)/mem_device.o
$(BUILD)/mem_collective.o: $(BUILD)/mem_ledger.o $(BUILD)/mem_ledger_backend.o

$(BUILD)/swe_model.o: app/swe_model.F90 $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) -c $< -o $@

$(BUILD)/swe: app/swe.F90 $(BUILD)/swe_model.o $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) $^ -o $@

$(BUILD)/test_%: test/test_%.F90 $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) $^ -o $@

run: $(BUILD)/swe
	$(BUILD)/swe $(ARGS)

test: $(addprefix $(BUILD)/,$(TESTS))
	$(BUILD)/test_ledger
	$(BUILD)/test_collective
	test/must_fail.sh $(BUILD)/test_must_fail

generate:
	tools/autogen/autogen.sh

check-generated: | $(BUILD)
	@fypp tools/autogen/devarray.fypp $(BUILD)/devarray.check.F90 && \
	  diff -u src/devarray.F90 $(BUILD)/devarray.check.F90 && echo "src/devarray.F90 up to date"

clean:
	rm -rf build

.PHONY: all run test generate check-generated clean
