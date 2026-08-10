#!/usr/bin/env python3
import itertools
from textwrap import dedent

loop_types = ["do", "do_concurrent", "openmp_do"]
dims = [2, 3]

def make_loops(dim, order, kind):
    """Generate nested loops for the given order and kind."""
    indices = order
    idx_decl = ",".join(indices)

    if kind == "do":
        open_loops = "\n".join([f"  do {i}=1,n{i}" for i in indices])
        close_loops = "\n".join([f"  end do" for _ in indices])
        body = f"     c({idx_decl}) = alpha * a({idx_decl}) + b({idx_decl})"
        return open_loops + "\n" + body + "\n" + close_loops

    elif kind == "do_concurrent":
        return f"  do concurrent({', '.join(f'{i}=1:n{i}' for i in indices)})\n" \
               f"     c({idx_decl}) = alpha * a({idx_decl}) + b({idx_decl})\n" \
               f"  end do"

    elif kind == "openmp_do":
        open_loops = "\n".join([f"  do {i}=1,n{i}" for i in indices])
        close_loops = "\n".join([f"  end do" for _ in indices])
        body = f"     c({idx_decl}) = alpha * a({idx_decl}) + b({idx_decl})"
        return f"  !$omp parallel do collapse({dim})\n{open_loops}\n{body}\n{close_loops}\n  !$omp end parallel do"

def generate_program(dim, order, kind):
    indices = ["i", "j", "k", "l", "m"][:dim]
    args = " ".join(f"n{i}" for i in indices)
    alloc_dims = ",".join(f"n{i}" for i in indices)
    idx_decl = ",".join(indices)
    loop_code = make_loops(dim, indices, kind)

    prog = f"""
program saxpy_{dim}d_{kind}_{''.join(indices)}
  use iso_fortran_env, only: real64
  use omp_lib, only: omp_get_wtime
  implicit none

  real(real64), parameter :: alpha = 2.0
  integer :: run
  integer :: {', '.join(f'n{i}' for i in indices)}
  real(real64) :: t1, t2, timings, total
  integer :: {', '.join(indices)}
  real, allocatable :: a({','.join(':' for _ in indices)}), b({','.join(':' for _ in indices)}), c({','.join(':' for _ in indices)})

  {''.join(f'n{i} = 100\n  ' for i in indices)}
  allocate(a({alloc_dims}), b({alloc_dims}), c({alloc_dims}))

  ! initialize
  do concurrent({', '.join(f'{i}=1:n{i}' for i in indices)})
     a({idx_decl}) = 13.0
     b({idx_decl}) = 67.0
     c({idx_decl}) = 0.0
  end do

  total = 0.0
  do run = 1,10
     t1 = omp_get_wtime()
{loop_code}
     t2 = omp_get_wtime()
     total = total + (t2 - t1)
  end do

  print '(A,F10.4," s")', "avg elapsed ({kind}, {''.join(indices)}): ", total / 10.0
  print *, "sample: ", c(1{''.join(',1' for _ in range(dim-1))})

end program
"""
    return dedent(prog)

def main():
    for dim in dims:
        indices = ["i", "j", "k", "l", "m"][:dim]
        for kind in loop_types:
            for order in itertools.permutations(indices):
                fname = f"{kind}_saxpy_{dim}d_{''.join(order)}.f90"
                with open(fname, "w") as f:
                    f.write(generate_program(dim, order, kind))
    print("✅ Generated all benchmarks.")

if __name__ == "__main__":
    main()

