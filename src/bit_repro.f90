!> Bit-reproducible, GPU-offloadable transcendental functions, extracted from the
!! MOM6 dev/gpu port work (MOM_intrinsic_functions.F90 on the tidal and meke branches)
!! into a standalone module for independent development.
!!
!! Ground rules for everything in here:
!!  - bitwise identical results on CPU and GPU, any rank count, any compiler we use
!!  - IEEE-compliant handling of specials (0, -0, Inf, NaN, denormals)
!!  - elemental + `!$omp declare target` so it drops into `do concurrent` kernels
!!  - integer/bit manipulation welcome when provably correct
!!  - as fast as possible subject to all of the above
module bit_repro

use, intrinsic :: iso_fortran_env, only : int64
implicit none ; private

public :: exp_reprod, log_reprod, pow_reprod, cuberoot

! IEEE-754 binary64 layout constants
integer, parameter :: fraclen = 52  !< Bits in the fraction (mantissa)
integer, parameter :: explen  = 11  !< Bits in the exponent
integer, parameter :: signbit = 63  !< Position of the sign bit
integer, parameter :: expbit  = fraclen !< Position of the lowest exponent bit
integer(kind=int64), parameter :: bias = 1023_int64 !< Exponent bias

contains

!> Bit-reproducible exp(x): Cody-Waite range reduction to r in [-ln2/2, ln2/2],
!! degree-12 Taylor polynomial in Horner form, exact scale() by 2**k.
elemental function exp_reprod(x) result(ex)
  !$omp declare target
  real, intent(in) :: x  !< The argument of the exponential
  real :: ex             !< The reproducible exponential of x

  ! Cody-Waite split of ln(2): ln2 = ln2_hi + ln2_lo, with ln2_hi chosen so k*ln2_hi is ~exact.
  real, parameter :: invln2 = 1.44269504088896338700  ! 1/ln(2)
  real, parameter :: ln2_hi = 0.693147180369123816490 ! High part of ln(2)
  real, parameter :: ln2_lo = 1.90821492927058770002e-10 ! Low part of ln(2)
  ! Reciprocal factorials 1/2! .. 1/12! (compile-time constant-folded -> identical host/device).
  real, parameter :: c2 = 1.0/2.0,        c3 = 1.0/6.0,         c4 = 1.0/24.0
  real, parameter :: c5 = 1.0/120.0,      c6 = 1.0/720.0,       c7 = 1.0/5040.0
  real, parameter :: c8 = 1.0/40320.0,    c9 = 1.0/362880.0,    c10 = 1.0/3628800.0
  real, parameter :: c11 = 1.0/39916800.0, c12 = 1.0/479001600.0
  real :: r  ! The reduced argument, x - k*ln2, in [-ln2/2, ln2/2]
  real :: p  ! The polynomial estimate of exp(r)
  integer :: k ! The integer number of factors of 2 in exp(x)

  k = nint(x*invln2)
  r = (x - real(k)*ln2_hi) - real(k)*ln2_lo
  ! Horner form of 1 + r + r^2/2! + ... + r^12/12!
  p = 1.0 + r*(1.0 + r*(c2 + r*(c3 + r*(c4 + r*(c5 + r*(c6 + r*(c7 + &
      r*(c8 + r*(c9 + r*(c10 + r*(c11 + r*c12)))))))))))
  ex = scale(p, k)
end function exp_reprod

!> Bit-reproducible natural log: exponent/fraction split, mantissa recentered to
!! [sqrt(1/2), sqrt(2)), atanh series in s = (m-1)/(m+1).
elemental function log_reprod(x) result(lx)
  !$omp declare target
  real, intent(in) :: x  !< The argument of the logarithm, x > 0
  real :: lx             !< The reproducible natural logarithm of x

  real, parameter :: ln2 = 0.69314718055994530942     ! ln(2)
  real, parameter :: sqrt2_2 = 0.70710678118654752440 ! sqrt(1/2), the mantissa reduction threshold
  ! Reciprocal odd integers 1/3 .. 1/21 for the atanh series (compile-folded -> identical host/device).
  real, parameter :: a3=1.0/3.0,   a5=1.0/5.0,   a7=1.0/7.0,   a9=1.0/9.0
  real, parameter :: a11=1.0/11.0, a13=1.0/13.0, a15=1.0/15.0, a17=1.0/17.0
  real, parameter :: a19=1.0/19.0, a21=1.0/21.0
  real :: m  ! The mantissa of x, reduced to [sqrt(1/2), sqrt(2))
  real :: s  ! (m-1)/(m+1), the atanh-series argument, |s| <= 0.172
  real :: s2 ! s*s
  real :: poly ! The polynomial estimate of log(m)
  integer :: k ! The binary exponent of x

  k = exponent(x) ; m = fraction(x)                    ! x = m * 2**k, m in [0.5, 1)
  if (m < sqrt2_2) then ; m = m + m ; k = k - 1 ; endif ! recenter m to [sqrt(1/2), sqrt(2))
  s = (m - 1.0) / (m + 1.0) ; s2 = s*s
  poly = 2.0*s*(1.0 + s2*(a3 + s2*(a5 + s2*(a7 + s2*(a9 + s2*(a11 + s2*(a13 + &
         s2*(a15 + s2*(a17 + s2*(a19 + s2*a21))))))))))
  lx = poly + real(k)*ln2
end function log_reprod

!> Bit-reproducible x**y for x > 0 via exp_reprod(y * log_reprod(x)).
!! KNOWN WEAKNESS (the starting point of this project): the error of y*log(x) is
!! amplified by exp, so accuracy degrades as |y*log(x)| grows, and integral y is
!! not special-cased, so pow_reprod(x,2.0) /= x*x in the last bits.
elemental function pow_reprod(x, y) result(p)
  !$omp declare target
  real, intent(in) :: x  !< The base, x > 0
  real, intent(in) :: y  !< The exponent
  real :: p              !< The reproducible x**y

  p = exp_reprod(y * log_reprod(x))
end function pow_reprod

!> Returns the cube root of a real argument at roundoff accuracy, in a form that
!! works properly with rescaling of the argument by integer powers of 8 (upstream
!! MOM6, Adcroft/Ward).  Bit-reproducible by construction: integer exponent
!! manipulation plus division-free Halley iterations and one Newton polish.
elemental function cuberoot(x) result(root)
  !$omp declare target
  real, intent(in) :: x !< The argument of cuberoot
  real :: root !< The real cube root of x

  real :: asx ! |x| rescaled by an integer power of 8 into 0.125 < asx <= 1.0
  real :: root_asx ! The cube root of asx
  real :: ra_3 ! root_asx cubed
  real :: num, den ! Evolving estimate numerator/denominator
  real :: num_prev, den_prev ! Previous iteration values
  real :: np_3, dp_3 ! Cubes of the previous values
  real :: r0, r0_3 ! Initial estimate and its cube
  integer :: itt
  integer(kind=int64) :: e_x, s_x

  if ((x >= 0.0) .eqv. (x <= 0.0)) then
    ! Return 0 for an input of 0, or NaN for a NaN input.
    root = x
  else
    call rescale_cbrt(x, asx, e_x, s_x)
    ! Halley's method in fractional form (no divisions inside the iterations).
    r0 = 0.707106
    r0_3 = r0 * r0 * r0
    num = r0 * (r0_3 + 2.0 * asx)
    den = 2.0 * r0_3 + asx
    do itt=1,2
      num_prev = num ; den_prev = den
      np_3 = num_prev * num_prev * num_prev
      dp_3 = den_prev * den_prev * den_prev
      num = num_prev * (np_3 + 2.0 * asx * dp_3)
      den = den_prev * (2.0 * np_3 + asx * dp_3)
    enddo
    root_asx = num / den
    ! One Newton polish to within the last bit.
    ra_3 = root_asx * root_asx * root_asx
    root_asx = root_asx - (ra_3 - asx) / (3.0 * (root_asx * root_asx))
    root = descale(root_asx, e_x, s_x)
  endif
end function cuberoot

!> Rescale `a` to the range [0.125, 1) while computing its cube-root exponent and
!! sign, by direct manipulation of the IEEE-754 bit representation.
pure subroutine rescale_cbrt(a, x, e_r, s_a)
  !$omp declare target
  real, intent(in) :: a  !< The number to be rescaled for cube-root computation
  real, intent(out) :: x !< The rescaled value of `a` in the range [0.125, 1)
  integer(kind=int64), intent(out) :: e_r !< The integral component of the cube-root exponent of `a`
  integer(kind=int64), intent(out) :: s_a !< Sign bit of `a`; nonzero indicates negative

  integer(kind=int64) :: xb  ! Bit representation of `a`
  integer(kind=int64) :: e_a ! Exponent of `a`
  integer(kind=int64) :: e_x ! Exponent of `x`

  xb = transfer(a, 1_int64)
  s_a = ibits(xb, signbit, 1)
  e_a = ibits(xb, expbit, explen) - bias
  ! e = 3*(floor(e/3)+1) + (modulo(e,3) - 3); the last term is the exponent of x.
  e_r = (e_a + sign(1_int64, e_a) + 2) / 3
  e_x = e_a - e_r * 3
  call mvbits(e_x + bias, 0, explen + 1, xb, fraclen)
  x = transfer(xb, 1.)
end subroutine rescale_cbrt

!> Undo the rescaling of a real number back to its original base.
pure function descale(x, e_a, s_a) result(a)
  !$omp declare target
  real, intent(in) :: x !< The rescaled value which is to be restored
  integer(kind=int64), intent(in) :: e_a !< Exponent of the unscaled value
  integer(kind=int64), intent(in) :: s_a !< Sign bit of the unscaled value
  real :: a !< Restored value with the corrected exponent and sign

  integer(kind=int64) :: xb  ! Bit representation
  integer(kind=int64) :: e_x ! Biased exponent of x

  xb = transfer(x, 1_int64)
  e_x = ibits(xb, expbit, explen)
  call mvbits(e_a + e_x, 0, explen, xb, expbit)
  call mvbits(s_a, 0, 1, xb, signbit)
  a = transfer(xb, 1.)
end function descale

end module bit_repro
