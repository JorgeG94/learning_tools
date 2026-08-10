!> Bit-reproducible, GPU-offloadable transcendental functions for MOM6, developed
!! standalone in bit_repro_adventure.  The five public functions live HERE, in
!! full, so this file reads as the algorithms; the arithmetic machinery they use
!! (exact-rounding primitives, the dd log2/exp2 core, binary exponentiation, the
!! cuberoot rescalers) is implemented in the bit_repro_helpers submodule and is
!! unreachable from outside.
!!
!! Ground rules (see README.md):
!!  - bitwise identical results on CPU and GPU, any rank count, any compiler we use
!!  - IEEE-compliant handling of specials (0, -0, Inf, NaN, denormals)
!!  - elemental + `!$omp declare target` so everything drops into `do concurrent`
!!  - as fast as possible subject to all of the above
module bit_repro

use, intrinsic :: iso_fortran_env, only : int64
implicit none ; private

public :: exp_reprod, log_reprod, pow_reprod, pow_reprod_explog, cuberoot

!> Private machinery, implemented in the bit_repro_helpers submodule.
interface

  !> x**n by binary exponentiation (n<0 via one reciprocal); ipow(x,2) == RN(x*x)
  elemental module function ipow(x, n) result(p)
    !$omp declare target
    real, intent(in) :: x
    integer, intent(in) :: n
    real :: p
  end function ipow

  !> ipow alias used by the half-integral path
  elemental module function ipow_signed(x, n) result(p)
    !$omp declare target
    real, intent(in) :: x
    integer, intent(in) :: n
    real :: p
  end function ipow_signed

  !> Knuth two_sum: s + e == a + b exactly, branch-free
  elemental module subroutine two_sum(a, b, s, e)
    !$omp declare target
    real, intent(in)  :: a, b
    real, intent(out) :: s, e
  end subroutine two_sum

  !> Dekker/Veltkamp two_prod (no FMA): p + e == a*b exactly
  elemental module subroutine two_prod(a, b, p, e)
    !$omp declare target
    real, intent(in)  :: a, b
    real, intent(out) :: p, e
  end subroutine two_prod

  !> log2(x) as a double-double pair, ~2**-85 relative, finite x > 0
  elemental module subroutine log2_dd(x, h, l)
    !$omp declare target
    real, intent(in)  :: x
    real, intent(out) :: h, l
  end subroutine log2_dd

  !> 2**(h+l) for a double-double exponent: table + short polynomial
  elemental module function exp2_pair(h, l) result(e2)
    !$omp declare target
    real, intent(in) :: h, l
    real :: e2
  end function exp2_pair

  !> Rescale for cuberoot: |a| -> [0.125,1) plus integral cube-root exponent + sign
  pure module subroutine rescale_cbrt(a, x, e_r, s_a)
    !$omp declare target
    real, intent(in) :: a
    real, intent(out) :: x
    integer(kind=int64), intent(out) :: e_r, s_a
  end subroutine rescale_cbrt

  !> Undo rescale_cbrt
  pure module function descale(x, e_a, s_a) result(a)
    !$omp declare target
    real, intent(in) :: x
    integer(kind=int64), intent(in) :: e_a, s_a
    real :: a
  end function descale

end interface

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

!> The ORIGINAL (reverted-MEKE) x**y via exp_reprod(y*log_reprod(x)), kept for
!! comparison.  Known weak: the error of y*log(x) is amplified by exp, and
!! integral y is not special-cased, so pow_reprod_explog(x,2.0) /= x*x.
elemental function pow_reprod_explog(x, y) result(p)
  !$omp declare target
  real, intent(in) :: x  !< The base, x > 0
  real, intent(in) :: y  !< The exponent
  real :: p              !< The reproducible x**y

  p = exp_reprod(y * log_reprod(x))
end function pow_reprod_explog

!> Bit-reproducible x**y, v2.  Built only from IEEE-exact operations (+,-,*,/,
!! sqrt, integer bit manipulation), so it is reproducible across CPU/GPU and
!! compilers by construction (compile WITHOUT fma contraction).
!!
!! Paths:
!!  - full IEEE-754 pow special-value matrix (0, -0, +-Inf, NaN, x<0)
!!  - integral y, |y| <= 16: binary exponentiation.  pow(x,2.0) == RN(x*x)
!!    exactly, so sqrt(pow_reprod(x,2.0)) == x by the IEEE sqrt round-trip
!!    theorem.
!!  - y = +-0.5, +-0.25: IEEE sqrt / sqrt(sqrt) -- the MEKE exponents become
!!    one or two hardware square roots.
!!  - half-integral y, |y| <= 16.5: ipow * sqrt.
!!  - general y: double-double log2(x) (Dekker/Veltkamp two_prod, no FMA),
!!    dd multiply by y, then a 2**r polynomial.  Faithful (<=1 ulp) for the
!!    moderate |y| MOM6 uses; error grows slowly as ~|y|*2**-60 beyond that.
elemental function pow_reprod(x, y) result(p)
  !$omp declare target
  real, intent(in) :: x  !< The base
  real, intent(in) :: y  !< The exponent
  real :: p              !< The reproducible x**y

  real :: ax     ! |x|
  real :: sgn    ! Sign to apply to the result (negative base, odd integral y)
  real :: h, l   ! y*log2(|x|) as a double-double pair
  real :: lh, ll ! log2(|x|) as a double-double pair
  real :: t      ! Scratch for the dd product y*(lh,ll)
  logical :: y_integral, y_odd

  ! y integral / odd tests that are safe for any finite y.
  y_integral = (y == aint(y))
  y_odd = .false.
  if (y_integral .and. abs(y) < 9.0e15) y_odd = (mod(abs(y), 2.0) == 1.0)

  ! --- IEEE special-value matrix (mirrors C99/IEEE-754 pow) ------------------
  if (y == 0.0) then                    ! pow(x, 0) = 1 for every x, even NaN
    p = 1.0
  elseif (x == 1.0) then                ! pow(1, y) = 1 for every y, even NaN
    p = 1.0
  elseif (x /= x .or. y /= y) then      ! any other NaN in -> NaN out
    p = x + y
  elseif (y == 1.0) then
    p = x
  elseif (abs(y) > huge(y)) then        ! y = +-Inf
    ax = abs(x)
    if (ax == 1.0) then ; p = 1.0
    elseif ((ax > 1.0) .eqv. (y > 0.0)) then ; p = abs(y)  ! +Inf
    else ; p = 0.0
    endif
  elseif (x == 0.0) then                ! +-0 base
    if (y > 0.0) then
      p = merge(x, 0.0, y_odd)          ! preserves -0 for odd integral y
    else
      p = merge(sign(1.0, x), 1.0, y_odd) / abs(x)  ! +-Inf
    endif
  elseif (abs(x) > huge(x)) then        ! x = +-Inf
    if (x > 0.0) then
      p = merge(abs(x), 0.0, y > 0.0)
    else
      sgn = merge(-1.0, 1.0, y_odd)
      p = merge(sgn*abs(x), sgn*0.0, y > 0.0)
    endif
  elseif (x < 0.0 .and. .not. y_integral) then
    p = (x - x) / (x - x)               ! NaN: negative base, non-integral power
  else
    ! --- computational paths, on ax = |x| with the sign restored at the end --
    ax = abs(x)
    sgn = 1.0
    if (x < 0.0 .and. y_odd) sgn = -1.0

    if (y_integral .and. abs(y) <= 16.0) then
      p = sgn * ipow(ax, nint(y))
    elseif (y == 0.5) then
      p = sqrt(ax)
    elseif (y == -0.5) then
      p = 1.0 / sqrt(ax)
    elseif (y == 0.25) then
      p = sqrt(sqrt(ax))
    elseif (y == -0.25) then
      p = 1.0 / sqrt(sqrt(ax))
    elseif ((2.0*y == aint(2.0*y)) .and. abs(y) <= 16.5) then
      ! half-integral: x**(n+0.5) = ipow(x,n) * sqrt(x)
      p = ipow_signed(ax, nint(y - 0.5)) * sqrt(ax)
    else
      call log2_dd(ax, lh, ll)
      h = y * lh
      if (abs(h) > 1200.0) then
        ! Far past over/underflow; also protects the Veltkamp splitter below from
        ! |y| ~ 1e292 overflow (for x /= 1, |lh| >= ~1.6e-16, so any |y| big
        ! enough to overflow the splitter lands here first).
        p = sgn * exp2_pair(sign(1201.0, h), 0.0)
      else
        ! (h,l) = y * (lh,ll), double-double product without FMA
        call two_prod(y, lh, h, t)
        l = t + y*ll
        p = sgn * exp2_pair(h, l)
      endif
    endif
  endif
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

end module bit_repro
