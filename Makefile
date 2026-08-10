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

# -r8: the candidates follow MOM6's default-real convention (promoted to double).
FC_IFX   = ifx
FL_IFX   = -O2 -r8 -fp-model=precise -no-fma
FC_NV    = mpifort
FL_NV    = -O2 -r8 -Mnofma -Mnoflushz -Mnodaz
FL_NVGPU = -O2 -r8 -Mnofma -Mnoflushz -Mnodaz -stdpar=gpu -mp=gpu -gpu=cc70,nofma,mem:separate

SRC = src/bit_repro.f90 test/harness.f90

ifx: ; $(FC_IFX) $(FL_IFX) $(SRC) -o harness_ifx
nv:  ; $(FC_NV) $(FL_NV) $(SRC) -o harness_nv
nvgpu: ; $(FC_NV) $(FL_NVGPU) $(SRC) -o harness_nvgpu

clean: ; rm -f harness_ifx harness_nv harness_nvgpu *.mod
.PHONY: ifx nv nvgpu clean
gpubit: ; $(FC_NV) $(FL_NVGPU) src/bit_repro.f90 test/gpu_bitwise.f90 -o gpu_bitwise
