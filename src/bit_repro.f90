!> Bit-reproducible, GPU-offloadable transcendental functions for MOM6, developed
!! standalone in bit_repro_adventure.  This module is the PUBLIC INTERFACE only --
!! every implementation and helper lives in the bit_repro_impl submodule, so the
!! API surface is exactly the five functions below and nothing else is reachable.
!!
!! Ground rules (see README.md):
!!  - bitwise identical results on CPU and GPU, any rank count, any compiler we use
!!  - IEEE-compliant handling of specials (0, -0, Inf, NaN, denormals)
!!  - elemental + `!$omp declare target` so everything drops into `do concurrent`
!!  - as fast as possible subject to all of the above
module bit_repro

implicit none ; private

public :: exp_reprod, log_reprod, pow_reprod, pow_reprod_explog, cuberoot

interface

  !> Bit-reproducible exp(x)
  elemental module function exp_reprod(x) result(ex)
    !$omp declare target
    real, intent(in) :: x  !< The argument of the exponential
    real :: ex             !< The reproducible exponential of x
  end function exp_reprod

  !> Bit-reproducible natural log, x > 0
  elemental module function log_reprod(x) result(lx)
    !$omp declare target
    real, intent(in) :: x  !< The argument of the logarithm, x > 0
    real :: lx             !< The reproducible natural logarithm of x
  end function log_reprod

  !> Bit-reproducible x**y with the full IEEE-754 pow special matrix
  elemental module function pow_reprod(x, y) result(p)
    !$omp declare target
    real, intent(in) :: x  !< The base
    real, intent(in) :: y  !< The exponent
    real :: p              !< The reproducible x**y
  end function pow_reprod

  !> The original exp(y*log(x)) form, kept for comparison only
  elemental module function pow_reprod_explog(x, y) result(p)
    !$omp declare target
    real, intent(in) :: x  !< The base, x > 0
    real, intent(in) :: y  !< The exponent
    real :: p              !< The reproducible x**y
  end function pow_reprod_explog

  !> Cube root at roundoff accuracy (upstream MOM6, Adcroft/Ward)
  elemental module function cuberoot(x) result(root)
    !$omp declare target
    real, intent(in) :: x !< The argument of cuberoot
    real :: root !< The real cube root of x
  end function cuberoot

end interface

end module bit_repro
