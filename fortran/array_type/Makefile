# make                     nvfortran + OpenACC (source mom6_env.sh first)
# make ACC=0               nvfortran, CPU only
# make COMPILER=gnu        gfortran, CPU only
# make COMPILER=intel      ifx, CPU only
# make NO_FINAL=1          build devarray without finalizers
# make run / make test / make generate / make check-generated

COMPILER ?= nv
ACC ?= 1
GPU_CC ?= cc70

ifeq ($(COMPILER),nv)
  FC := nvfortran
  FFLAGS := -O2 -Mpreprocess -Minform=warn
  MODFLAG := -module
  ifeq ($(ACC),1)
    FFLAGS += -acc -gpu=$(GPU_CC) -Minfo=accel
  endif
else ifeq ($(COMPILER),intel)
  FC := ifx
  FFLAGS := -O2 -fpp -stand f18 -warn all -check bounds
  MODFLAG := -module
  ACC := 0
else
  FC := gfortran
  FFLAGS := -O2 -cpp -std=f2018 -Wall -Wextra -Wno-compare-reals -fcheck=all
  MODFLAG := -J
  ACC := 0
endif

ifeq ($(NO_FINAL),1)
  FFLAGS += -DDEVARRAY_NO_FINAL
  FINAL_TAG := -nofinal
endif

BUILD := build/$(COMPILER)-acc$(ACC)$(FINAL_TAG)
LIB_SRC := src/mem_ledger.F90 src/mem_device.F90 src/devarray.F90
LIB_OBJ := $(patsubst src/%.F90,$(BUILD)/%.o,$(LIB_SRC))

all: $(BUILD)/swe $(BUILD)/test_ledger $(BUILD)/test_copy_forbidden

$(BUILD):
	mkdir -p $@

$(BUILD)/%.o: src/%.F90 | $(BUILD)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) -c $< -o $@

$(BUILD)/swe_model.o: app/swe_model.F90 $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) -c $< -o $@

$(BUILD)/swe: app/swe.F90 $(BUILD)/swe_model.o $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) $^ -o $@

$(BUILD)/test_ledger: test/test_ledger.F90 $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) $^ -o $@

$(BUILD)/test_copy_forbidden: test/test_copy_forbidden.F90 $(LIB_OBJ)
	$(FC) $(FFLAGS) $(MODFLAG) $(BUILD) $^ -o $@

# module dependencies
$(BUILD)/devarray.o: $(BUILD)/mem_ledger.o

run: $(BUILD)/swe
	$(BUILD)/swe $(ARGS)

test: $(BUILD)/test_ledger $(BUILD)/test_copy_forbidden
	$(BUILD)/test_ledger
	@if $(BUILD)/test_copy_forbidden >/dev/null 2>&1; then \
	  echo "test_copy_forbidden: FAIL (copy was allowed)"; exit 1; \
	else echo "test_copy_forbidden: ok (copy rejected)"; fi

generate:
	tools/autogen/autogen.sh

check-generated:
	@tmp=$$(mktemp); fypp tools/autogen/devarray.fypp $$tmp && \
	  diff -u src/devarray.F90 $$tmp && echo "src/devarray.F90 up to date"; \
	  rc=$$?; rm -f $$tmp; exit $$rc

clean:
	rm -rf build

.PHONY: all run test generate check-generated clean
