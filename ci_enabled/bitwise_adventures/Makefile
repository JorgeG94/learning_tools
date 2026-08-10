# bit_repro_adventure: bit-reproducible transcendentals, accuracy + speed harness.
#
#   make ifx        - build with compiler-intel-llvm (the accuracy/speed reference)
#                     run:  module load compiler-intel-llvm/2026.0.0 first
#   make nv         - build CPU-only with nvfortran (source nvhpc_env.sh first)
#   make nvgpu      - nvfortran with -stdpar=gpu -mp=gpu; harness loops still run on
#                     host but the module compiles for device (offloadability gate)
#
# -fp-model=precise / default nv flags: no FMA contraction surprises in the
# candidates; the intrinsic side is measured however the compiler ships it.

#: the candidates follow MOM6's default-real convention (promoted to double).
FC_IFX   = ifx
FL_IFX   = -O3 -xHost -ipo -fp-model=precise -no-fma
FC_NV    = mpifort
FL_NV    = -O2 -Mnofma -Mnoflushz -Mnodaz
FL_NVGPU = -O2 -Mnofma -Mnoflushz -Mnodaz -stdpar=gpu -mp=gpu -gpu=cc70,nofma,mem:separate,lto

MOD = src/bit_repro.f90 src/bit_repro_helpers.f90
SRC = $(MOD) test/harness.f90

ifx: ; $(FC_IFX) $(FL_IFX) $(SRC) -o harness_ifx
nv:  ; $(FC_NV) $(FL_NV) $(SRC) -o harness_nv
nvgpu: ; $(FC_NV) $(FL_NVGPU) $(SRC) -o harness_nvgpu

clean: ; rm -f harness_ifx harness_nv harness_nvgpu *.mod
.PHONY: ifx nv nvgpu clean gpubit gpuperf crossbit_ifx crossbit_nv crossbit_gf crossbit_fl
gpubit: ; $(FC_NV) $(FL_NVGPU) $(MOD) test/gpu_bitwise.f90 -o gpu_bitwise
gpuperf: ; $(FC_NV) $(FL_NVGPU) $(MOD) test/gpu_perf.f90 -o gpu_perf
crossbit_ifx: ; $(FC_IFX) $(FL_IFX) $(MOD) test/crossbit.f90 -o crossbit_ifx
crossbit_nv:  ; $(FC_NV) $(FL_NV) $(MOD) test/crossbit.f90 -o crossbit_nv
crossbit_gf: ; gfortran -O2 -ffp-contract=off $(MOD) test/crossbit.f90 -o crossbit_gf
FLANG ?= /home/jorge/install/llvm/21.1.3/bin/flang
crossbit_fl: ; $(FLANG) -O2 -ffp-contract=off $(MOD) test/crossbit.f90 -o crossbit_fl

# Generic CI leg: compiler and flags injected by the workflow (see
# ../../.github/workflows/bitwise_repro.yml).  Verify with:
#   python3 -c "import hashlib;print(hashlib.sha256(open('crossbit_ci.bin','rb').read()).hexdigest())"
# against GOLDEN.sha256.
crossbit_ci: ; $(FC) $(CI_FCFLAGS) $(MOD) test/crossbit.f90 -o crossbit_ci
.PHONY: crossbit_ci
