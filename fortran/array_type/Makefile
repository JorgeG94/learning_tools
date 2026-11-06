# make                     nvfortran + OpenACC (source mom6_env.sh first)
# make ACC=0               nvfortran, CPU only
# make COMPILER=gnu        gfortran, CPU only
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
else
  FC := gfortran
  FFLAGS := -O2 -cpp -std=f2018 -Wall -Wextra -Wno-compare-reals -fcheck=all
  MODFLAG := -J
  ACC := 0
endif

BUILD := build/$(COMPILER)-acc$(ACC)
LIB_SRC := src/mem_ledger.F90 src/mem_device.F90 src/devarray.F90
LIB_OBJ := $(patsubst src/%.F90,$(BUILD)/%.o,$(LIB_SRC))

all: $(BUILD)/swe $(BUILD)/test_ledger

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

# module dependencies
$(BUILD)/devarray.o: $(BUILD)/mem_ledger.o

run: $(BUILD)/swe
	$(BUILD)/swe $(ARGS)

test: $(BUILD)/test_ledger
	$(BUILD)/test_ledger

generate:
	tools/autogen/autogen.sh

check-generated:
	@tmp=$$(mktemp); fypp tools/autogen/devarray.fypp $$tmp && \
	  diff -u src/devarray.F90 $$tmp && echo "src/devarray.F90 up to date"; \
	  rc=$$?; rm -f $$tmp; exit $$rc

clean:
	rm -rf build

.PHONY: all run test generate check-generated clean
