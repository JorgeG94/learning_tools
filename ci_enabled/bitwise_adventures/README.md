# bit_repro_adventure

Independent development of **bit-reproducible, GPU-offloadable transcendental
functions** for MOM6, targeting the functions MEKE and EPBL need: `exp`, `log`,
`pow` (and `cuberoot`, already solid upstream, kept as the reference for the
integer-manipulation style). Parallel effort to Marshall's — the point is to
develop independently and compare.

## Rules of the game

1. **Bit-reproducible**: identical results CPU vs GPU, any rank count, and across
   the compilers we use (nvfortran, ifx). No `-Kieee`-dependent behavior.
2. **IEEE compliant**: correct specials (`0`, `-0`, `Inf`, `NaN`, denormals,
   overflow to `Inf`, underflow), no FTZ assumptions.
3. **As fast as possible** subject to 1–2. Integer/bit manipulation encouraged
   (see `rescale_cbrt` for the house style) — anything goes if we can prove it.
4. **Offloadable**: `elemental` + `!$omp declare target`, callable from
   `do concurrent` (scalar args only — see the neutral-diffusion lesson).
5. **Reference implementation**: `compiler-intel-llvm/2026.0.0` (`ifx`) intrinsics
   — arguably excellent; we measure both accuracy and speed against it.
6. **Accuracy oracle**: `real128` evaluation rounded to `real64` = the correctly
   rounded result. ULP distances are measured against *that*, for candidates AND
   intrinsics, so the comparison is symmetric.
7. **pow target property**: `sqrt(pow_reprod(x, 2.0)) == x` exactly. This falls
   out of IEEE sqrt/multiply iff `pow_reprod(x, 2.0) == RN(x*x)`, so it forces an
   exact integer-exponent path (the current exp∘log composition does NOT have it —
   measured by the harness round-trip test).

## Starting material (extracted from the MOM6 port branches)

| Function | Origin | Method | Known state |
|---|---|---|---|
| `exp_reprod` | tidal port (`c32ea99ba`) | Cody-Waite reduction + degree-12 Taylor, `scale()` | in production use, tidal gates bitwise |
| `log_reprod` | reverted MEKE pow work (`b98035c6f`, reflog) | exponent/fraction split + atanh series | unmerged |
| `pow_reprod` | same | `exp_reprod(y*log_reprod(x))` | REVERTED — accuracy insufficient, error amplified by `y·log x`; no integral-exponent path |
| `cuberoot` | upstream MOM6 (Adcroft/Ward) | integer rescale by 8^k + division-free Halley + Newton polish | upstream-blessed; the style guide |

## Layout

- `src/bit_repro.f90` — the candidates (standalone, no MOM dependencies)
- `test/harness.f90` — ULP-vs-oracle accuracy, IEEE specials, pow round-trip,
  throughput microbenchmark
- `make ifx` / `make nv` / `make nvgpu`

## Roadmap

1. Baseline numbers: current candidates vs ifx intrinsics (accuracy + speed).
2. `pow`: integral-exponent exact path (`y == nint(y)` → repeated squaring),
   half-integer path via IEEE `sqrt`, then extended-precision (double-double)
   `y*log2(x)` core for the general case.
3. `exp`/`log`: table-free vs small-table trade study; the GPU wants branch-free.
4. GPU leg: same harness arrays through `do concurrent`, diff vs CPU run bitwise.
5. Cross-compiler leg: ifx vs nvfortran binaries must agree bitwise on the
   candidates (they will NOT on the intrinsics — that is the point).
