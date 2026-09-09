# dc_portability_hell

Source-to-source translators that turn one Fortran tree written in
**`do concurrent` + OpenACC** into the OpenMP variants you need to get the same
code through NVHPC, Cray, LLVM flang, ifx, amdflang and GNU.

The premise: you maintain **one** source tree — `do concurrent` for the
data-parallel loops, `!$acc` only for data residency and reductions — and CI
mechanically derives the OpenMP flavours from it. Nobody hand-maintains a
parallel port, so the ports cannot drift.

## The three derived variants

| Variant | `do concurrent` | `!$acc` | Target |
|---|---|---|---|
| `dc-openmp-target` | kept verbatim | → `!$omp target ...` | compilers that map DC to the device themselves (NVHPC `-stdpar=gpu`, LLVM flang) |
| `openmp` | rewritten as `!$omp target teams distribute parallel do` | → `!$omp target ...` | any OpenMP-5 offload compiler |
| `openmp-cpu` | rewritten as `!$omp parallel do` | neutralised to inert comments | host multicore, no offload |

`openmp-cpu` exists because `target teams distribute` on a **host fallback** is
flaky on several compilers (notably GNU) — for CPU you want a plain worksharing
region, not an offload region that happens to land on the host.

## Tools

| Script | Does |
|---|---|
| `tools/dc_audit.py` | Diagnostic/lint pass over every `do concurrent` loop: index count, clauses (`local`/`local_init`/`shared`/`default`/`reduce`), what directive sits above it. `--strict` fails on loops the translators cannot handle. Run this first. |
| `tools/acc_to_omp.py` | `!$acc` → `!$omp`. Directive-block aware (joins `&` continuations, transforms head + clauses as one unit). `--target gpu\|cpu`. |
| `tools/dc_to_omp.py` | `do concurrent` → `!$omp` worksharing loop nest. Collapses the header, maps `local()`→`private()`, and runs the whole-program **`pure`-strip cascade**. `--target gpu\|cpu`. |
| `tools/acc_reduce_to_dc.py` | The *inward* migration: `!$acc parallel loop reduction(op:v)` → F2023 `do concurrent (...) reduce(op:v)`. Run this on your **canonical** tree to shrink the surface that needs translating at all. |
| `tools/omp_to_doconcurrent.py` | The reverse translator (`!$omp target teams distribute parallel do` → `do concurrent`), for importing an OpenMP-first tree into the DC convention. |
| `tools/regen_dc_openmp.sh` | The pipeline: audit → `acc_to_omp` → `dc_to_omp` → overlay patches → project post-hook. |
| `tools/_locality_macro.py` | Unwraps CPP-hidden locality specifiers (`DO_LOCALITY(local(a,b))`) before clause parsing. See below. |
| `tools/test_acc_to_omp.py`, `tools/test_locality_macro.py`, `tools/test_dc_to_omp.py` | pytest regression guards for clause handling, macro unwrapping, and block-finder structure. |
| `tools/_parallel.py`, `tools/_progress.py` | Multi-process fan-out (order-preserving) and a TTY progress bar. |

Python 3.9+, **standard library only**.

## Usage

```bash
# lint first — --strict fails on loops the translators can't handle
python tools/dc_audit.py --strict src/

# dry-run, then apply
python tools/acc_to_omp.py --target gpu src/
python tools/acc_to_omp.py --target gpu --write src/
python tools/dc_to_omp.py  --target gpu --write src/

# or the whole pipeline in one go, in place, from the repo root
bash tools/regen_dc_openmp.sh dc-openmp-target
bash tools/regen_dc_openmp.sh openmp
bash tools/regen_dc_openmp.sh openmp-cpu 0     # 0 = one worker per core
```

## Wiring it into a project

`regen_dc_openmp.sh` is configured entirely by environment:

| Variable | Default | Meaning |
|---|---|---|
| `DCPORT_ROOTS` | whichever of `src app benchmarks tests` exist | source roots to translate |
| `DCPORT_PATCH_DIR` | `patches/openmp` | overlay patches applied after translation |
| `DCPORT_JOBS` | `1` | worker processes; `0` = one per core |
| `DCPORT_LOCALITY_MACROS` | `DO_LOCALITY` | comma-separated CPP locality-macro names to unwrap |
| `DCPORT_NO_PROGRESS` | unset | set to silence the progress bar |

**List every directive-bearing root.** A root left out keeps its `!$acc`
(inert on an OpenMP build) and — for the full `openmp` variant — is excluded
from the whole-program `pure` cascade, so you get `pure` procedures containing
`!$omp target` regions, which every compiler rejects.

Two escape hatches for what a translator can't express:

* **`patches/openmp/**/*.patch`** — overlay patches `git apply`'d after
  translation. This is where an OpenMP-only workaround lives.
* **`tools/regen_post_hook.sh`** — optional, run last with `$variant` and
  `$omp_target` in the environment. Put build-system edits here: flipping a
  CMake backend default, swapping a toolchain file, selecting a preset.

## Portability macros: `DO_LOCALITY`

Not every compiler accepts F2018 locality specifiers on a `do concurrent`
header, so portable codebases hide them behind a CPP macro. MOM6 does this in
`src/framework/do_concurrent_compat.h`:

```c
#ifdef HAVE_FC_DO_CONCURRENT_LOCAL
#define DO_LOCALITY(X) X
#else
#define DO_LOCALITY(X) ;
#endif
```

```fortran
do concurrent (I=ish-1:ieh, do_I(I)) DO_LOCALITY(local(u_new, duhdu))
```

**These translators read unpreprocessed source**, so they see the macro call,
not the specifier. A naive clause parser finds no `local(...)`, and the emitted
OpenMP loop silently loses its `private(...)` — the variables become **shared
across threads**. Wrong answers, no diagnostic.

So `_locality_macro.py` unwraps the macro (balanced-paren, to a fixpoint)
before any clause parsing, in both `dc_audit.py` and `dc_to_omp.py`. It also
drops the bare `;` the false branch expands to, so a header preprocessed down
the *unsupported* path parses identically.

Measured on MOM6 (`MOM6-examples/src/MOM6/src`, 687 `do concurrent` loops):
without unwrapping the audit reports **zero** locality clauses; with it, 67
`local`, 2 `local_init`, 25 `reduce` — 94 `DO_LOCALITY` call sites that would
otherwise have been dropped from every generated OpenMP tree.

Set `DCPORT_LOCALITY_MACROS` if your project names the macro something else.

## Known gaps

* **Masked headers are not converted.** `do concurrent (I=is:ie, do_I(I))` —
  a mask expression in place of a range — is reported by `dc_audit.py` and
  skipped by `dc_to_omp.py`. The OpenMP form needs the mask lowered into an
  `if` inside the body, which restructures the block rather than just its
  header. MOM6 has 45 such loops. On a `do concurrent`-native backend they are
  fine, so the gap only bites the full `openmp` variant.
* **`;`-compound loops are refused, not converted.** MOM6 writes
  `do concurrent (i=is:ie) ; press(i,j) = 0.0 ; enddo` and
  `do concurrent (j=js:je, I=is-1:ie) ; if (mask(I,j) > 0.0) then`. The emitter
  replaces whole lines, so converting these would delete the trailing
  statements. They are skipped with a message naming the offending text.
* `acc_reduce_to_dc.py` deliberately leaves `async(...)` directives alone —
  those are CUDA-graph batching sites with no portable equivalent.

Both gaps are handled with an overlay patch under `patches/openmp/` for now.

## `;`-compound source: why the parsers are statement-aware

MOM6 packs several statements per line. That breaks line-anchored parsing in
ways that are **silent**, which is the worst property a code generator can
have. Three bugs found by running these tools over it, all now fixed and
covered by `tools/test_dc_to_omp.py`:

1. **A skipped loop ended the scan.** `find_dc_block` returned `None` both for
   "no more loops in this file" and "this loop is unconvertible", and the
   caller read `None` as end-of-file. One masked loop silenced every loop below
   it. On MOM6 this reported 5 skips where there were 57, and abandoned 246
   convertible loops; on the origin project it hid 8.
2. **The `end do` matcher stole outer loops.** Depth tracking matched `^\s*do`
   and `^\s*end\s*do`, so in `endif ; enddo` the close was invisible (depth
   never fell) and in `if (m==1) then ; do k=1,nk` the open was invisible
   (depth fell too early). The second is the dangerous one: the matcher returns
   an `end do` belonging to an **outer** construct and the emitted footer
   overwrites that line. On MOM6, 37 headers matched a different `end do` than
   the statement-aware matcher, and running the old one over
   `MOM_set_viscosity.F90` silently deleted an `else`, an `endif`, an
   `if (...) then` and a whole nested loop header — output that will not even
   compile. Depth is now tracked per `;`-separated statement.
3. **A string literal was parsed as a loop.** `call check(error, ok,
   "do concurrent reduce(+) must be exact")` — an assertion message — matched
   the construct regex, and (via bug 1) took the rest of the file with it.
   Detection now runs on string-stripped text.

Diagnostics carry the file name and an ORIGINAL-source line number: reported
lines are corrected for the drift each preceding rewrite introduces (a
converted one-line header becomes 2+N lines). A skip message is the only
handle on what needs a manual patch, so one naming a line the loop is not on
is worse than none.

## CI

`.github/workflows/sync-dc-openmp.yml` regenerates all three variants on every
push to `main` and force-pushes them to `auto/dc-openmp`, `auto/openmp` and
`auto/openmp-cpu`. Adjust the `paths:` filter to match your `DCPORT_ROOTS`.

The generated branches are **machine-owned** — never edit them by hand. Every
OpenMP-side fix belongs in an overlay patch or in the translator.

Build verification is deliberately not in the workflow: installing an offload
toolchain (Intel oneAPI, ROCm) costs ~12 minutes of runner budget per job.
Check out a generated branch and build it locally when you want validation.

## Things learned the hard way

These are the reasons the translators are more than a `sed` script.

* **`pure` + `!$omp target` is illegal.** An executable target region launches a
  kernel and moves data — side effects a `pure` procedure may not have. But
  `do concurrent` *is* side-effect-free, so the canonical OpenACC/`-stdpar`
  source legitimately keeps `pure`. So `dc_to_omp.py` has to strip `pure` from
  every procedure that ends up holding a target region **and propagate that
  impurity up the call graph to a fixpoint, across module and file boundaries**
  (a `pure` procedure may only call `pure` procedures). This is a
  whole-program analysis, not a per-file rewrite. Declarative directives like
  `!$omp declare target` are `pure`-compatible and are left alone.
* **OpenACC loop-mapping clauses are invalid on an OpenMP loop construct.**
  `gang`/`vector`/`worker`/`num_gangs(n)`/`vector_length(n)`/`tile(...)` must be
  dropped — in OpenMP the parallelism is the construct, not a clause. A naive
  head swap emits `!$omp target teams distribute parallel do gang vector` and
  ifx rejects it (#5082). Dropping them needs paren-balanced matching so a
  *variable* named `gang` inside `reduction(...)` survives.
* **`default(present)` has no OpenMP equivalent** — it is not `defaultmap(present)`,
  which means something else. ifx and amdflang both reject it. It needs its own
  strip rule, because a `present`-followed-by-`(` matcher can't see it (here
  `present` is the *argument*).
* **Express reductions as `do concurrent ... reduce`, not `!$acc`.** An OpenACC
  reduction forces a compute directive into the enclosing procedure, which then
  (a) becomes an `!$omp target` region — illegal in a `pure` procedure and fatal
  for any `do concurrent` that calls it — and (b) needs a translator at all.
  F2023 `reduce` is plain Fortran: portable to nvfortran / gfortran / ifx /
  flang, `pure`-safe, zero translation. `acc_reduce_to_dc.py` does that
  migration. It deliberately leaves `async(...)` directives alone — those are
  CUDA-graph batching sites with no portable equivalent.
* **Host fallback ≠ host code.** See `openmp-cpu` above.
* **Directives are logical blocks, not lines.** A `&`-continued directive has to
  be joined, transformed as one unit, and re-emitted, or clause rewriting
  silently misses everything past the first line.
* **Processes, not threads, for the fan-out.** These passes are pure-Python
  `re` work and CPython's `re` does not release the GIL. `pmap` preserves input
  order regardless of completion order — a source-to-source translator whose
  output depends on scheduling would be untrustworthy.

## Provenance

Extracted from a production Fortran GPU solver's `tools/`, with the
project-specific bits removed. The lints that lived beside these
(assumed-shape-dummy detection in `do concurrent`, intrinsic shadowing,
device-transfer auditing, an OpenMP-portability rules linter) were left behind
— they are checks, not transforms.
