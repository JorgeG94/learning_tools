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

public :: exp_reprod, log_reprod, pow_reprod, pow_reprod_explog, cuberoot

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

!> x**n for n >= 0 by binary exponentiation: pure IEEE multiplies, so it is
!! reproducible everywhere, and ipow(x,2) == RN(x*x) exactly.
elemental function ipow(x, n) result(p)
  !$omp declare target
  real, intent(in) :: x    !< The base
  integer, intent(in) :: n !< The non-negative exponent
  real :: p                !< x**|n|, or its reciprocal for n < 0

  real :: b    ! Running square
  integer :: m ! Remaining exponent bits

  m = abs(n)
  if (m == 0) then
    p = 1.0
  else
    p = 1.0 ; b = x
    do while (m > 1)
      if (btest(m, 0)) p = p * b
      b = b * b
      m = shiftr(m, 1)
    enddo
    p = p * b
  endif
  if (n < 0) p = 1.0 / p
end function ipow

!> ipow with negative exponents allowed (wrapper name kept separate so the hot
!! integral path in pow_reprod stays branch-light).
elemental function ipow_signed(x, n) result(p)
  !$omp declare target
  real, intent(in) :: x
  integer, intent(in) :: n
  real :: p
  p = ipow(x, n)
end function ipow_signed

!> Knuth two_sum: s + e == a + b exactly, branch-free.
elemental subroutine two_sum(a, b, s, e)
  !$omp declare target
  real, intent(in)  :: a, b
  real, intent(out) :: s, e
  real :: bb
  s = a + b
  bb = s - a
  e = (a - (s - bb)) + (b - bb)
end subroutine two_sum

!> Dekker/Veltkamp two_prod WITHOUT fma: p + e == a*b exactly.
!! Requires the compiler not to contract these expressions into fma (build with
!! -fno-fma / -Mnofma / -gpu=nofma) -- contraction would still be exact here but
!! would change the split arithmetic bit-for-bit between compilers.
elemental subroutine two_prod(a, b, p, e)
  !$omp declare target
  real, intent(in)  :: a, b
  real, intent(out) :: p, e
  real, parameter :: splitter = 134217729.0  ! 2**27 + 1
  real :: a1, a2, b1, b2, ta, tb
  p = a * b
  ta = splitter * a
  a1 = ta - (ta - a) ; a2 = a - a1
  tb = splitter * b
  b1 = tb - (tb - b) ; b2 = b - b1
  e = ((a1*b1 - p) + a1*b2 + a2*b1) + a2*b2
end subroutine two_prod

!> log2(x) for finite normal/denormal x > 0 as a double-double pair (h, l),
!! accurate to ~2**-85 relative: exponent/fraction split, mantissa recentered to
!! [sqrt(1/2), sqrt(2)), atanh series with the leading term carried in dd.
elemental subroutine log2_dd(x, h, l)
  !$omp declare target
  real, intent(in)  :: x  !< The argument, x > 0, finite
  real, intent(out) :: h  !< High part of log2(x)
  real, intent(out) :: l  !< Low part of log2(x)

  real, parameter :: sqrt2_2 = 0.70710678118654752440 ! sqrt(1/2)
  ! 2/ln(2) as a double-double constant
  real, parameter :: two_invln2_hi = 2.8853900817779268
  real, parameter :: two_invln2_lo = 4.0710547481862066e-17
  ! Reciprocal odd integers for the atanh series
  real, parameter :: a3=1.0/3.0,   a5=1.0/5.0,   a7=1.0/7.0,   a9=1.0/9.0
  real, parameter :: a11=1.0/11.0, a13=1.0/13.0, a15=1.0/15.0, a17=1.0/17.0
  real, parameter :: a19=1.0/19.0, a21=1.0/21.0
  real :: m          ! Mantissa recentered to [sqrt(1/2), sqrt(2))
  real :: u          ! m - 1, exact by Sterbenz
  real :: vh, vl     ! m + 1 as a dd pair (exact via two_sum)
  real :: s, sl      ! (m-1)/(m+1) as a dd pair
  real :: ph, pe     ! two_prod scratch
  real :: lh2, ll2   ! Leading term (2/ln2)*s as a dd pair
  real :: rv         ! 1/vh, so the routine needs only one division
  real :: s2, s4, s8 ! Powers of s for the Estrin evaluation
  real :: tail       ! The series tail (double precision suffices)
  real :: h1, l1     ! Intermediate dd sum
  integer :: k       ! Binary exponent of x

  k = exponent(x) ; m = fraction(x)
  if (m < sqrt2_2) then ; m = m + m ; k = k - 1 ; endif

  u = m - 1.0                       ! exact: m in [0.70, 1.42)
  call two_sum(m, 1.0, vh, vl)      ! v = m + 1 exactly as (vh, vl)
  rv = 1.0 / vh                     ! one division for the whole routine
  s = u * rv                        ! quotient to ~1 ulp ...
  call two_prod(s, vh, ph, pe)
  sl = ((u - ph) - pe - s*vl) * rv  ! ... refined: (s, sl) = u/v to ~2**-105

  ! Leading term L = (2/ln2) * s in dd
  call two_prod(s, two_invln2_hi, lh2, ll2)
  ll2 = ll2 + (s*two_invln2_lo + sl*two_invln2_hi)

  ! Series tail: L * s2*P(s2), |tail| <= 0.005, double is ample.  P is evaluated
  ! in a FIXED Estrin grouping (explicit parentheses; this order is part of the
  ! reproducibility contract) to shorten the dependency chain.
  s2 = s*s ; s4 = s2*s2 ; s8 = s4*s4
  tail = lh2 * (s2 * ( ((a3 + s2*a5) + s4*(a7 + s2*a9)) + &
                       s8*(((a11 + s2*a13) + s4*(a15 + s2*a17)) + s8*(a19 + s2*a21)) ))

  call two_sum(lh2, tail, h1, l1)
  l1 = l1 + ll2
  call two_sum(real(k), h1, h, l)   ! add the exponent, keeping the residual
  l = l + l1
end subroutine log2_dd

!> 2**(h+l) for a double-double exponent, |l| << 1.  Table-based reduction:
!! h = n + i/32 + r with |r| <= 1/64, so 2**h = 2**n * T(i) * 2**r with T(i) a
!! 32-entry double-double table and 2**r a short degree-6 polynomial.  The
!! T(i)*poly product is carried through two_prod, keeping the assembly error
!! near half an ulp.  Overflow -> Inf, underflow -> 0/denormal via scale().
elemental function exp2_pair(h, l) result(e2)
  !$omp declare target
  real, intent(in) :: h  !< High part of the exponent
  real, intent(in) :: l  !< Low part of the exponent
  real :: e2             !< 2**(h+l)

  ! T(i) = 2**(i/32) as double-double (generated at 60-digit precision)
  real, parameter :: t2hi(0:31) = [ &
    1.0, 1.0218971486541166, 1.0442737824274138, 1.0671404006768237, &
    1.0905077326652577, 1.1143867425958924, 1.1387886347566916, 1.1637248587775775, &
    1.189207115002721, 1.215247359980469, 1.241857812073484, 1.2690509571917332, &
    1.2968395546510096, 1.3252366431597413, 1.3542555469368927, 1.383909881963832, &
    1.4142135623730951, 1.4451808069770467, 1.4768261459394993, 1.5091644275934228, &
    1.5422108254079407, 1.5759808451078865, 1.6104903319492543, 1.645755478153965, &
    1.681792830507429, 1.718619298122478, 1.7562521603732995, 1.7947090750031072, &
    1.8340080864093424, 1.8741676341103, 1.9152065613971474, 1.9571441241754002 ]
  real, parameter :: t2lo(0:31) = [ &
    0.0, 5.109225028973444e-17, 8.551889705537965e-17, -7.899853966841582e-17, &
    -3.046782079812471e-17, 1.0410278456845571e-16, 8.912812676025408e-17, 3.8292048369240935e-17, &
    3.982015231465646e-17, -7.712630692681488e-17, 4.658027591836937e-17, 2.667932131342186e-18, &
    2.5382502794888315e-17, -2.8587312100388614e-17, 7.70094837980299e-17, -6.770511658794786e-17, &
    -9.667293313452913e-17, -3.0237581349939873e-17, -3.483994556892796e-17, -1.016455327754295e-16, &
    7.949834809697621e-17, -1.0136916471278304e-17, 2.4707192569797888e-17, -1.0125679913674773e-16, &
    8.199010020581497e-17, -1.851380418263111e-17, 2.960140695448873e-17, 1.8227458427912087e-17, &
    3.283107224245627e-17, -6.122763413004143e-17, -1.0619946056195963e-16, 8.960767791036668e-17 ]
  ! Coefficients ln2**j / j! for 2**r over |r| <= 1/64
  real, parameter :: e2c1 = 0.6931471805599453,    e2c2 = 0.24022650695910072
  real, parameter :: e2c3 = 0.05550410866482158,   e2c4 = 0.009618129107628477
  real, parameter :: e2c5 = 0.0013333558146428443, e2c6 = 0.0001540353039338161
  real :: w    ! h*32
  real :: r    ! Fractional remainder of the exponent, |r| <= 1/64
  real :: pp   ! Polynomial value of 2**r
  real :: vh, vl ! two_prod parts of T(i)*pp
  real :: corr ! Low-order correction: table low part and the l input
  integer :: n32, idx, n

  if (h > 1100.0) then
    e2 = huge(h) * 2.0               ! +Inf, reproducibly
  elseif (h < -1130.0) then
    e2 = 0.0
  else
    w = h * 32.0
    n32 = nint(w)
    r = h - real(n32) * 0.03125      ! exact: n32/32 is an exact multiple of 2**-5
    idx = iand(n32, 31)              ! n32 = 32*n + idx with 0 <= idx < 32 for any
    n = (n32 - idx) / 32             ! sign; the division is exact (nvfortran has no shifta)
    pp = 1.0 + r*(e2c1 + r*(e2c2 + r*(e2c3 + r*(e2c4 + r*(e2c5 + r*e2c6)))))
    corr = t2lo(idx) + t2hi(idx)*(l*e2c1)   ! 2**l ~ 1 + l*ln2, l ~ 2**-50
    call two_prod(t2hi(idx), pp, vh, vl)
    e2 = scale(vh + (vl + corr), n)
  endif
end function exp2_pair

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
