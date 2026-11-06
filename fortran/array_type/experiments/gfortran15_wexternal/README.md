# gfortran 15.1: -Wexternal-argument-mismatch breaks dummy procedures

`-Wexternal-argument-mismatch` (new in GCC 15, enabled by `-Wall`) treats a
dummy procedure declared with an explicit abstract interface
(`procedure(visit_i) :: visit`) as an external procedure, and infers an
interface from its call site. Valid code is then rejected:

    $ gfortran -Wall repro.f90
    Error: Interface mismatch in dummy procedure 'visit' at (1):
           Shape mismatch in argument '_formal_1'

    $ gfortran repro.f90 && ./a.out        # fine
    $ gfortran -Wall -Wno-external-argument-mismatch repro.f90   # fine

In `mem_collective` (where the code did compile) the same warning pass
silently miscompiled the call through the dummy procedure: segfault on the
first access of the array argument inside an internal-procedure callback.
A warning flag must never change code generation.

Workaround: `-Wno-external-argument-mismatch`. Seen with GCC 15.1.0 on
Gadi, 2026-11-06; not checked against 15.2+.
