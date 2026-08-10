# The portability landmine catalogue

Every entry was discovered by a gate failure in this repo, root-caused, and
routed around.  If you are writing reproducible floating-point Fortran, read
this before your first cross-platform diff — each item cost between an hour
and a day the first time and is free the second time.

## Compiler / flag behavior

1. **ifx's FMA-contraction flag is `-no-fma`; `-fno-fma` is silently
   ignored** (a command-line warning is the only tell).  Contraction changes
   Veltkamp/Dekker arithmetic bit-for-bit between compilers.

2. **ifx on Windows accepts dash-style FP flags and silently does not honor
   them.**  `-no-fma -no-ftz` built, linked, ran — with `/fp:fast` semantics
   (denormals flushed, fused ulp drift).  Use the native spellings
   (`/fp:precise /Qfma- /Qftz-`) and disable git-bash path conversion
   (`MSYS_NO_PATHCONV=1`, `MSYS2_ARG_CONV_EXCL='*'`) so the slashes survive.

3. **nvfortran defaults the HOST to FTZ/DAZ** (`-Mnoflushz -Mnodaz` to
   disable): denormal inputs read as zero on the CPU but not on the GPU, so
   host-vs-device results differ by O(1) wherever denormals occur.  MOM6's
   production GPU builds do not pass these flags.

4. **`-qopenmp-simd` licenses reduction reassociation even under
   `-fp-model=precise`.**  Function bits unchanged, but a `s = s + f(x)` loop
   summed in a different order.  Never add it to a reproducible build without
   auditing every reduction in the compilation unit.

5. **gfortran constant-folds intrinsic edge cases into compile errors**:
   `huge(x)*2.0` in one expression, `exp(750.0)`, `(-2.0)**3.0`.  Split into
   two statements or route arguments through `volatile` locals.

## Intrinsic / runtime behavior

6. **`scale()` is processor-dependent at BOTH exponent-range edges.**
   nvfortran flushes denormal results to zero AND returns Inf for n > 1023
   even when the mantissa keeps the result representable (seen at
   `exp(709.774)`).  Fix: do the outer 64 doublings/halvings with one IEEE
   multiply by an exact power of two (`scale_reprod`).

7. **`exponent()`/`fraction()` disagree between compilers on denormal
   arguments** (and nvfortran's answer is wrong).  Pre-scale denormals by an
   exact 2**54 first (`norm_split`), or for cube roots by the exact cube
   2**54 = (2**18)**3.

8. **`sign(1.0, -0.0)` is processor-dependent** (ifx and nvfortran disagree).
   Read a zero's sign at the bit level via `transfer`.

9. **NaN guards must be `x /= x`.**  The idiom `(x>=0.).eqv.(x<=0.)` relies
   on both compares being false for NaN; on gcc/arm64 the unordered compare
   leaks through the LE condition code, the guard fails, and NaN's bit
   pattern flows into exponent arithmetic (cuberoot(NaN) returned 6.7e102 on
   Apple Silicon; every x86 compiler agreed with each other).

10. **`min`/`max` with a NaN operand is processor-dependent** (gfortran
    disagrees with ifx/nvfortran).  Clamp NaN-explicitly:
    `if (v == v) v = min(max(v, lo), hi)`.

11. **`ieee_fma` is not a portable primitive**: ifx computes it UNFUSED under
    `-no-fma` (residual probe returns 0 instead of −2⁻⁶⁰) and nvfortran does
    not export it from `ieee_arithmetic` at all.

## Test-harness discipline

12. **Impure function references may be duplicated or elided by optimizers**
    (ifx `-O3 -ipo` called a stateful PRNG a different number of times than
    the program text says, desynchronizing the sequence between compilers).
    Advance stateful generators with SUBROUTINE calls — statements cannot be
    duplicated.  Corollaries: two impure calls in one expression have
    unspecified evaluation order (differs ifx vs nvfortran); never construct
    denormal test inputs with `scale()` (see #6); compare results as raw bits
    (`transfer` to int64, or binary dumps + `cmp`), never as printed decimals;
    and give xor-fold diagnostics a salt that is never zero
    (`v XOR ishft(v,0) = 0` blinds the fold to every eighth element).

## Design notes that follow from the above

- Kind-explicit code (`wp = real64`, `_wp` literals) removes the per-compiler
  promotion-flag zoo (`-r8` / `-fdefault-real-8` / `/real-size:64` /
  lfortran: none) entirely.
- Branch-free variants built from `merge()` + clamped per-band arguments +
  unconditional two-step scaling are bit-identical to their branchy
  originals — a reusable, gateable branch-elimination recipe.
- SIMD width does not shorten serial dependency chains (Clenshaw), and the
  no-FMA rule doubles every chain link: some functions are chain-bound and no
  language choice changes that.
