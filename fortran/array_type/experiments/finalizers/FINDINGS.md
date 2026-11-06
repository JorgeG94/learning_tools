# Finalizers for ledger-tracked arrays: what works where

`probe.F90` models the ledger: every `init` issues a fresh id, the final
procedure retires the id of an object holding data. Per scenario it counts
**leaked** ids (finalizer never ran) and **doubles** (an id retired twice,
i.e. a copy shared the id). Run `./run_probe.sh`; it builds two variants,
intrinsic assignment and `-DDEFINED_ASSIGN` (copy gets a new id).

Tested 2026-11-06: nvfortran 26.5 (with and without `-acc`), gfortran
12.2/13.2/14.2/15.1, ifx 2025.3.2.

## Intrinsic assignment + finalizer: broken everywhere

`a = b` copies the id, both objects get finalized, the id is retired twice
(S6, S8, S10, S12 on every compiler). A finalizer therefore *requires* a
defined assignment.

## Defined assignment + finalizer

| scenario | ifx | gfortran 13-15 | nvfortran 26.5 | gfortran 12 |
|---|---|---|---|---|
| S1-S5 locals, BLOCK, deallocate, components, arrays | ok | ok | ok | ok |
| S6 `a = b` | ok | ok | ok | ok |
| S7 `intent(out)` dummy | ok | ok | ok | ok |
| S8 `a = make_res()` | ok | ok | **leak** (result not finalized) | **leak** |
| S12 `p = q`, component with defined assignment | ok | **double** (bitwise component copy) | **double** | **double** |
| S13 `call f(make_res())` | ok | ok | **leak** | **leak** |
| S14 `intent(out)` with SAVEd actual | ok | ok | **leak** | ok |

Notes:

* S12: the standard says intrinsic assignment of a parent invokes the
  type-bound defined assignment of a nonallocatable component. gfortran
  and nvfortran copy it bitwise instead.
* S14 also covers main-program variables (implicitly SAVE). Module
  variables are fine on nvfortran.
* Defined assignment never auto-allocates an unallocated allocatable LHS
  (intrinsic assignment does). `a = b` with unallocated `a` is invalid and
  segfaults on every compiler.
* Standard behaviour, not a bug: main-program and module variables are
  **never** finalized at program end.

## What devarray does with this

* `final` calls `destroy` (idempotent): a safety net for locals,
  components, deallocated objects and `intent(out)`.
* `assignment(=)` is **fatal**: copying a tracked/mapped array is almost
  always a bug. Where gfortran/nvfortran bypass it (S12), the copy shares
  the id and the ledger stops on the second destroy, naming the object.
* Don't return these types from functions (nvfortran leaks the result).
* Explicit `destroy` stays the primary API; `mem_check_leaks` at shutdown
  catches whatever the finalizer can't (main-program/module variables,
  nvfortran S8/S13/S14).
* `-DDEVARRAY_NO_FINAL` (`make NO_FINAL=1`) builds without finalizers.

No measurable cost: SWE 4096^2 x 2000 steps on V100 is 6.56 s with
finalizers vs 6.73 s without (noise).
