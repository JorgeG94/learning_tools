# bitwise_adventures

Bit-reproducible, GPU-offloadable transcendental functions for MOM6:
`pow`, `exp`, `log`, `erfc`, `cuberoot`.  Built only from operations IEEE-754
requires to be correctly rounded (`+ - * / sqrt`) plus integer bit
manipulation, in a frozen evaluation order — so every conforming platform
produces the identical bits **by construction**, and CI proves it on every
push.

## Status

**One golden bit pattern (`GOLDEN.sha256`), 6M values across five functions,
reproduced bit-for-bit by:** gcc 13/14, Intel ifx (Linux & Windows), Intel
classic ifort, nvfortran, LLVM flang, AOCC flang-classic · Linux + macOS +
Windows · x86_64 + arm64 · and an NVIDIA V100 via `do concurrent` — including
vectorized builds, denormals, and the full IEEE specials matrix.

### Accuracy (vs a correctly-rounded real128 oracle; enforced in CI)

| Function | max ulp | notes |
|---|---|---|
| exp | 1, 100% faithful, full range | |
| log | 1, ~100% correctly rounded | dd core |
| pow | ≤2, ≥99.9% faithful | `sqrt(pow(x,2)) == x` exact |
| cuberoot | 1, 100% faithful | full IEEE incl. denormals |
| erfc | ≤5 mid-band, 1 near 0 | budget-gated; polish on roadmap |

### Performance

| Function | GPU (V100, ns/elem) | CPU scalar (ifx -xHost, ns) | note |
|---|---|---|---|
| exp | 3.5 | 23 | |
| log | 3.6 | 47 | |
| erfc | 3.6 | 23–118 branchy / ~50 flat (`_bf`) | distribution-dependent; see variants |
| cuberoot | 3.5 | 30 | |
| pow (±0.25) | 5.4 | 5.6 | matches 8-lane SVML *scalar* |
| pow (general) | 5.3 | 87 | |

Per-lane the scalar code matches or beats the vendor intrinsics; their
remaining vector edge is FMA + hand-masking, both unavailable under the
cross-vendor reproducibility contract (in any language).  On the GPU — the
platform this library exists for — everything costs 3.5–5.5 ns/element.

### erfc variants (all bit-identical, lockstep-gated)

- `erfc_reprod` — elemental, default; fastest on band-clustered arguments and
  on the GPU.
- `erfc_reprod_bf` — branch-free; flat ~50 ns, 2× faster when arguments mix
  bands (branch mispredicts).
- `erfc_reprod_v` — hand-eight-wide array kernel; technique demonstrator
  (vector gathers, bit-built scaling), chain-latency-bound like `_bf`.

## Ground rules

1. Bitwise identical on CPU, GPU, and every compiler in the matrix.
2. IEEE-compliant specials (0, −0, ±Inf, NaN, denormals, gradual underflow).
3. `elemental` + `!$omp declare target`; scalar arguments only from
   `do concurrent`.
4. No FMA contraction, no FTZ/DAZ — the flag contract per compiler lives in
   the Makefile and workflow; see `LANDMINES.md` for why every flag is there.
5. Kind-explicit (`wp = real64`, `_wp` literals): no promotion flags needed.
6. Coefficients are generated, never transcribed: `tools/gen_erfc.py` runs
   stdlib-`Decimal` at 100 digits with in-generator fit verification.

## Layout

```
src/bit_repro.f90          public functions, full bodies (reads as the algorithms)
src/bit_repro_helpers.f90  submodule: two_sum/two_prod, dd log2/exp2 cores, rescalers
test/harness.f90           ULP-vs-oracle accuracy CONTRACT (budgets; nonzero exit on regression)
test/crossbit.f90          deterministic 6M-value bit dump for cross-platform comparison
test/fingerprint.f90       mechanism canaries (fma, nint ties, denormals, per-band values)
test/gpu_bitwise.f90       CPU-vs-GPU bitwise gate (do concurrent)
test/gpu_perf.f90          GPU throughput
tools/gen_erfc.py          coefficient generator (pure stdlib, 100-digit Decimal)
GOLDEN.sha256 / GOLDEN.md5 the reproducibility contract
REFERENCE_*.txt            expected canaries / per-set hashes for CI-log diffing
LANDMINES.md               the portability findings catalogue — read before porting
```

## Gates

Local: `make ifx` (harness = accuracy contract) · `make crossbit_<ifx|nv|gf|fl>`
then compare to `GOLDEN.sha256` · `make gpubit` / `make gpuperf` (needs a GPU).

CI (`.github/workflows/bitwise_repro.yml`): stage 1 = ten-leg bitwise matrix
against the golden hash; stage 2 (only if stage 1 is green) = the accuracy
contract under gcc and ifx.  Value-changing commits regenerate the golden in
the same commit; value-neutral commits prove the hash unchanged.

## Roadmap

MOM6 integration (MEKE `pow`, EPBL `exp`, wave_interface `erfc`) · erfc
mid-band accuracy polish · upstream the cuberoot IEEE fixes (NaN guard is
live-wrong on arm64 gcc) · lfortran ICE report (repo is the reproducer).

Origin: extracted from the MOM6 dev/gpu port work (`exp_reprod` from the
tidal-mixing port; `log/pow` from the reverted MEKE pow branch; `cuberoot`
upstream) and developed here independently, in parallel with Marshall's
effort.
