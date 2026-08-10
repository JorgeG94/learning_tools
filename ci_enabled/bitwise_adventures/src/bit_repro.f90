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

use, intrinsic :: iso_fortran_env, only : int64, real64
implicit none ; private

!> Working precision: the library is kind-explicit (every declaration and
!! literal carries wp), so NO default-real promotion flag (-r8,
!! -fdefault-real-8) is needed on any compiler.
integer, parameter :: wp = real64

public :: exp_reprod, log_reprod, pow_reprod, pow_reprod_explog, cuberoot, erfc_reprod

!> Private machinery, implemented in the bit_repro_helpers submodule.
interface

  !> x**n by binary exponentiation (n<0 via one reciprocal); ipow(x,2) == RN(x*x)
  elemental module function ipow(x, n) result(p)
    !$omp declare target
    real(wp), intent(in) :: x
    integer, intent(in) :: n
    real(wp) :: p
  end function ipow


  !> Knuth two_sum: s + e == a + b exactly, branch-free
  elemental module subroutine two_sum(a, b, s, e)
    !$omp declare target
    real(wp), intent(in)  :: a, b
    real(wp), intent(out) :: s, e
  end subroutine two_sum

  !> Dekker/Veltkamp two_prod (no FMA): p + e == a*b exactly
  elemental module subroutine two_prod(a, b, p, e)
    !$omp declare target
    real(wp), intent(in)  :: a, b
    real(wp), intent(out) :: p, e
  end subroutine two_prod

  !> Denormal-safe exponent()/fraction() split (bit-identical to the intrinsics
  !! for normal x; see the helper for why the intrinsics cannot be used raw)
  elemental module subroutine norm_split(x, m, k)
    !$omp declare target
    real(wp), intent(in)  :: x
    real(wp), intent(out) :: m
    integer, intent(out) :: k
  end subroutine norm_split

  !> log2(x) as a double-double pair, ~2**-85 relative, finite x > 0
  elemental module subroutine log2_dd(x, h, l)
    !$omp declare target
    real(wp), intent(in)  :: x
    real(wp), intent(out) :: h, l
  end subroutine log2_dd

  !> 2**(h+l) for a double-double exponent: table + short polynomial
  elemental module function exp2_pair(h, l) result(e2)
    !$omp declare target
    real(wp), intent(in) :: h, l
    real(wp) :: e2
  end function exp2_pair

  !> Rescale for cuberoot: |a| -> [0.125,1) plus integral cube-root exponent + sign
  pure module subroutine rescale_cbrt(a, x, e_r, s_a)
    !$omp declare target
    real(wp), intent(in) :: a
    real(wp), intent(out) :: x
    integer(kind=int64), intent(out) :: e_r, s_a
  end subroutine rescale_cbrt

  !> Undo rescale_cbrt
  pure module function descale(x, e_a, s_a) result(a)
    !$omp declare target
    real(wp), intent(in) :: x
    integer(kind=int64), intent(in) :: e_a, s_a
    real(wp) :: a
  end function descale

end interface

contains

!> Bit-reproducible exp(x): dd multiply by log2(e), then the table-based 2**r
!! core shared with pow_reprod.  Full IEEE specials: NaN passes through,
!! exp(+Inf)=Inf, exp(-Inf)=0, saturating clamps protect the integer reduction.
!! NOTE: this DEPARTS bitwise from the in-tree tidal-port exp_reprod (Cody-Waite
!! + Taylor); adopting it into MOM6 requires re-gating the tidal baseline.
elemental function exp_reprod(x) result(ex)
  !$omp declare target
  real(wp), intent(in) :: x  !< The argument of the exponential
  real(wp) :: ex             !< The reproducible exponential of x

  ! log2(e) = 1/ln(2) as a double-double constant (exact halves of 2/ln2)
  real(wp), parameter :: log2e_hi = 1.4426950408889634_wp
  real(wp), parameter :: log2e_lo = 2.0355273740931033e-17_wp
  real(wp) :: h, l, t  ! x*log2(e) as a double-double pair, and two_prod scratch

  if (x /= x) then
    ex = x
  elseif (x > 710.0_wp) then       ! exp overflows past 709.78...; covers +Inf
    ex = huge(x)
    ex = ex * 2.0_wp                ! +Inf via runtime IEEE overflow (a constant
                                 ! expression here is a compile error on gfortran)
  elseif (x < -746.0_wp) then      ! exp underflows to zero past -745.2; covers -Inf
    ex = 0.0_wp
  else
    call two_prod(x, log2e_hi, h, t)
    l = t + x*log2e_lo
    ex = exp2_pair(h, l)
  endif
end function exp_reprod

!> Bit-reproducible natural log, rebuilt on the double-double log2 core:
!! ln(x) = (h+l)*ln2 with the leading product carried through two_prod.  This
!! replaces the earlier single-double atanh evaluation, whose rounding shows
!! at ~3 ulp (52% correctly rounded near x=1); the dd core is the same one the
!! pow general path uses, which measures ~1 ulp there.  IEEE specials:
!! log(0) = -Inf raising divideByZero, log(negative) = NaN raising invalid,
!! log(+Inf) = +Inf, NaN passes through; denormals via norm_split.
elemental function log_reprod(x) result(lx)
  !$omp declare target
  real(wp), intent(in) :: x  !< The argument of the logarithm
  real(wp) :: lx             !< The reproducible natural logarithm of x

  ! ln(2) as a double-double constant
  real(wp), parameter :: ln2_hi = 0.6931471805599453_wp
  real(wp), parameter :: ln2_lo = 2.3190468138462996e-17_wp
  real(wp) :: h, l   ! log2(x) as a double-double pair
  real(wp) :: ph, pe ! two_prod parts of h*ln2_hi

  if (.not. (x > 0.0_wp)) then          ! zero, negative, or NaN
    if (x /= x) then ; lx = x                     ! NaN passes through
    elseif (x == 0.0_wp) then ; lx = -1.0_wp / abs(x)  ! -Inf, raising divideByZero
    else ; lx = (x - x) / (x - x)                 ! negative: NaN, raising invalid
    endif
  elseif (x > huge(x)) then
    lx = x                                        ! log(+Inf) = +Inf
  else
    call log2_dd(x, h, l)
    call two_prod(h, ln2_hi, ph, pe)
    lx = ph + (pe + (h*ln2_lo + l*ln2_hi))
  endif
end function log_reprod

!> The ORIGINAL (reverted-MEKE) x**y via exp_reprod(y*log_reprod(x)), kept for
!! comparison.  Known weak: the error of y*log(x) is amplified by exp, and
!! integral y is not special-cased, so pow_reprod_explog(x,2.0) /= x*x.
elemental function pow_reprod_explog(x, y) result(p)
  !$omp declare target
  real(wp), intent(in) :: x  !< The base, x > 0
  real(wp), intent(in) :: y  !< The exponent
  real(wp) :: p              !< The reproducible x**y

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
  real(wp), intent(in) :: x  !< The base
  real(wp), intent(in) :: y  !< The exponent
  real(wp) :: p              !< The reproducible x**y

  real(wp) :: ax     ! |x|
  real(wp) :: sgn    ! Sign to apply to the result (negative base, odd integral y)
  real(wp) :: h, l   ! y*log2(|x|) as a double-double pair
  real(wp) :: lh, ll ! log2(|x|) as a double-double pair
  real(wp) :: t      ! Scratch for the dd product y*(lh,ll)
  logical :: y_integral, y_odd

  ! y integral / odd tests that are safe for any finite y.
  y_integral = (y == aint(y))
  y_odd = .false.
  if (y_integral .and. abs(y) < 9007199254740992.0_wp) y_odd = (mod(abs(y), 2.0_wp) == 1.0_wp)  ! 2**53

  ! --- IEEE special-value matrix (mirrors C99/IEEE-754 pow) ------------------
  if (y == 0.0_wp) then                    ! pow(x, 0) = 1 for every x, even NaN
    p = 1.0_wp
  elseif (x == 1.0_wp) then                ! pow(1, y) = 1 for every y, even NaN
    p = 1.0_wp
  elseif (x /= x .or. y /= y) then      ! any other NaN in -> NaN out
    p = x + y
  elseif (y == 1.0_wp) then
    p = x
  elseif (abs(y) > huge(y)) then        ! y = +-Inf
    ax = abs(x)
    if (ax == 1.0_wp) then ; p = 1.0_wp
    elseif ((ax > 1.0_wp) .eqv. (y > 0.0_wp)) then ; p = abs(y)  ! +Inf
    else ; p = 0.0_wp
    endif
  elseif (x == 0.0_wp) then                ! +-0 base
    if (y > 0.0_wp) then
      p = merge(x, 0.0_wp, y_odd)          ! preserves -0 for odd integral y
    else
      ! The zero's sign must be read at the BIT level: sign(1.0, -0.0) is
      ! processor-dependent (ifx and nvfortran disagree), transfer() is not.
      if (y_odd .and. transfer(x, 1_int64) /= 0_int64) then
        p = -1.0_wp / abs(x)               ! pow(-0, negative odd int) = -Inf
      else
        p = 1.0_wp / abs(x)                ! +Inf
      endif
    endif
  elseif (abs(x) > huge(x)) then        ! x = +-Inf
    if (x > 0.0_wp) then
      p = merge(abs(x), 0.0_wp, y > 0.0_wp)
    else
      sgn = merge(-1.0_wp, 1.0_wp, y_odd)
      p = merge(sgn*abs(x), sgn*0.0_wp, y > 0.0_wp)
    endif
  elseif (x < 0.0_wp .and. .not. y_integral) then
    p = (x - x) / (x - x)               ! NaN: negative base, non-integral power
  else
    ! --- computational paths, on ax = |x| with the sign restored at the end --
    ax = abs(x)
    sgn = 1.0_wp
    if (x < 0.0_wp .and. y_odd) sgn = -1.0_wp

    ! Fast paths are restricted to small POSITIVE exponents (plus the exact
    ! single-operation reciprocals): repeated squaring loses ~1 ulp per multiply
    ! (y=16 measured 12-15 ulp vs 3 through the general path), and for negative
    ! y the intermediate x**|n| can over/underflow where the true result is
    ! representable.  y=2 stays a single multiply, preserving the round-trip
    ! property sqrt(pow(x,2)) == x.
    if (y_integral .and. y >= 0.0_wp .and. y <= 4.0_wp) then
      p = sgn * ipow(ax, nint(y))
    elseif (y == -1.0_wp) then
      p = sgn * (1.0_wp / ax)
    elseif (y == 0.5_wp) then
      p = sqrt(ax)
    elseif (y == -0.5_wp) then
      p = 1.0_wp / sqrt(ax)
    elseif (y == 0.25_wp) then
      p = sqrt(sqrt(ax))
    elseif (y == -0.25_wp) then
      p = 1.0_wp / sqrt(sqrt(ax))
    elseif ((2.0_wp*y == aint(2.0_wp*y)) .and. y >= 0.5_wp .and. y <= 4.5_wp) then
      ! positive half-integral: x**(n+0.5) = ipow(x,n) * sqrt(x)
      p = ipow(ax, nint(y - 0.5_wp)) * sqrt(ax)
    else
      call log2_dd(ax, lh, ll)
      h = y * lh
      if (abs(h) > 1200.0_wp) then
        ! Far past over/underflow; also protects the Veltkamp splitter below from
        ! |y| ~ 1e292 overflow (for x /= 1, |lh| >= ~1.6e-16, so any |y| big
        ! enough to overflow the splitter lands here first).
        p = sgn * exp2_pair(sign(1201.0_wp, h), 0.0_wp)
      else
        ! (h,l) = y * (lh,ll), double-double product without FMA
        call two_prod(y, lh, h, t)
        l = t + y*ll
        p = sgn * exp2_pair(h, l)
      endif
    endif
  endif
end function pow_reprod

!> Bit-reproducible complementary error function.  Representation:
!!  - |x| <= 0.46875: erfc = 1 - x*P(x**2) with the exact erf Maclaurin
!!    coefficients (1-erf amplification is bounded by ~2 here; beyond it the
!!    subtraction is the classic catastrophic cancellation)
!!  - 0.46875 < |x| <= 2 and 2 < |x| <= 7: Chebyshev fits (Clenshaw, fixed
!!    order) of the scaled function erfcx(x) = erfc(x)*exp(x**2), times
!!    exp_reprod(-x**2) with the argument squared EXACTLY via two_prod and the
!!    low part applied as a (1-e) correction (dropping it costs ~1e3 ulp near
!!    the underflow end)
!!  - 7 < |x| < 28: asymptotic series (exact (2n-1)!!/(-2)**n coefficients)
!!    times the same exp core; gradual underflow comes out of the IEEE
!!    arithmetic naturally
!!  - erfc(x) = 2 - erfc(-x) for negative x (addition, no cancellation);
!!    clamps to 2 / 0 past +-28 cover +-Inf; NaN passes through.
!! Coefficients generated by tools/gen_erfc.py (stdlib Decimal, 100 digits,
!! fit residuals ~1e-21, verified in-generator).
elemental function erfc_reprod(x) result(ec)
  !$omp declare target
  real(wp), intent(in) :: x  !< The argument
  real(wp) :: ec             !< The reproducible erfc(x)

  real(wp), parameter :: erf_mac(0:15) = [ &
    1.1283791670955126_wp, -0.37612638903183754_wp, 0.11283791670955126_wp, &
    -0.026866170645131252_wp, 0.005223977625442188_wp, -0.0008548327023450853_wp, &
    0.00012055332981789664_wp, -1.492565035840625e-05_wp, 1.6462114365889248e-06_wp, &
    -1.6365844691234924e-07_wp, 1.4807192815879218e-08_wp, -1.2290555301717928e-09_wp, &
    9.422759064650411e-11_wp, -6.7113668551641105e-12_wp, 4.4632242632864775e-13_wp, &
    -2.7835162072109215e-14_wp ]
  real(wp), parameter :: cxB(0:23) = [ &
    0.4062917348653128_wp, -0.18151089830658812_wp, 0.036282977131463424_wp, &
    -0.0066414453256516195_wp, 0.0011300743535714113_wp, -0.0001806468989853791_wp, &
    2.7342880285563356e-05_wp, -3.942598196325111e-06_wp, 5.441792670106758e-07_wp, &
    -7.218251070104982e-08_wp, 9.231522167631464e-09_wp, -1.1414743168430211e-09_wp, &
    1.367848537001346e-10_wp, -1.591764656641901e-11_wp, 1.802051255167746e-12_wp, &
    -1.9878778583973728e-13_wp, 2.1397210370699588e-14_wp, -2.2501737469358202e-15_wp, &
    2.3145069843923556e-16_wp, -2.3309339983129675e-17_wp, 2.300571581031184e-18_wp, &
    -2.227132705200476e-19_wp, 2.1164143036953425e-20_wp, -1.9756603241087112e-21_wp ]
  real(wp), parameter :: cxC(0:31) = [ &
    0.14343982965978574_wp, -0.08130645231062354_wp, 0.022452024071075903_wp, &
    -0.006052234353050054_wp, 0.0015951792668183191_wp, -0.0004116660363681922_wp, &
    0.00010414866949375157_wp, -2.585854449250302e-05_wp, 6.3068506378351416e-06_wp, &
    -1.5123488236862958e-06_wp, 3.5682745315339563e-07_wp, -8.289641543220519e-08_wp, &
    1.8974118029549293e-08_wp, -4.281434919086646e-09_wp, 9.529100726995677e-10_wp, &
    -2.0929752414574342e-10_wp, 4.5386282067980504e-11_wp, -9.72115691336315e-12_wp, &
    2.057386474938545e-12_wp, -4.304064020968324e-13_wp, 8.903413177574481e-14_wp, &
    -1.8217553228327414e-14_wp, 3.688175647160491e-15_wp, -7.390023395686382e-16_wp, &
    1.4659202020335926e-16_wp, -2.879495205656065e-17_wp, 5.60233695521311e-18_wp, &
    -1.0798634228268147e-18_wp, 2.0625765454436175e-19_wp, -3.9046600929478305e-20_wp, &
    7.327811665218642e-21_wp, -1.3635318363407182e-21_wp ]
  real(wp), parameter :: asyD(0:20) = [ &
    1.0_wp, -0.5_wp, 0.75_wp, &
    -1.875_wp, 6.5625_wp, -29.53125_wp, &
    162.421875_wp, -1055.7421875_wp, 7918.06640625_wp, &
    -67303.564453125_wp, 639383.8623046875_wp, -6713530.554199219_wp, &
    77205601.37329102_wp, -965070017.1661377_wp, 13028445231.742859_wp, &
    -188912455860.27145_wp, 2928143065834.2075_wp, -48314360586264.42_wp, &
    845501310259627.4_wp, -1.5641774239803108e+16_wp, 3.050145976761606e+17_wp ]
  real(wp), parameter :: one_over_sqrtpi = 0.5641895835477563_wp

  real(wp) :: ax     ! |x|
  real(wp) :: u      ! x**2 for the Maclaurin band
  real(wp) :: p, e   ! x**2 as an exact two_prod pair
  real(wp) :: ex2    ! exp(-x**2), dd-corrected
  real(wp) :: t      ! polynomial/Clenshaw value
  real(wp) :: tt     ! Clenshaw abscissa in [-1, 1]
  real(wp) :: b1, b2, bt ! Clenshaw recurrence carries
  real(wp) :: s      ! 1/x**2 for the asymptotic band
  integer :: j

  if (x /= x) then
    ec = x
  elseif (x >= 28.0_wp) then
    ec = 0.0_wp
  elseif (x <= -28.0_wp) then
    ec = 2.0_wp
  else
    ax = abs(x)
    if (ax <= 0.46875_wp) then
      u = x*x
      t = erf_mac(15)
      do j = 14, 0, -1
        t = erf_mac(j) + u*t
      enddo
      ec = 1.0_wp - x*t
    else
      call two_prod(ax, ax, p, e)
      ex2 = exp_reprod(-p) * (1.0_wp - e)
      if (ax <= 2.0_wp) then
        tt = (2.0_wp*ax - 2.46875_wp) / 1.53125_wp   ! both constants exact in binary
        b1 = 0.0_wp ; b2 = 0.0_wp
        do j = 23, 1, -1
          bt = 2.0_wp*tt*b1 - b2 + cxB(j)
          b2 = b1 ; b1 = bt
        enddo
        t = tt*b1 - b2 + cxB(0)
        ec = ex2 * t
      elseif (ax <= 7.0_wp) then
        tt = (2.0_wp*ax - 9.0_wp) / 5.0_wp
        b1 = 0.0_wp ; b2 = 0.0_wp
        do j = 31, 1, -1
          bt = 2.0_wp*tt*b1 - b2 + cxC(j)
          b2 = b1 ; b1 = bt
        enddo
        t = tt*b1 - b2 + cxC(0)
        ec = ex2 * t
      else
        s = 1.0_wp / p
        t = asyD(20)
        do j = 19, 0, -1
          t = asyD(j) + s*t
        enddo
        ec = ex2 * (one_over_sqrtpi * t) / ax
      endif
      if (x < 0.0_wp) ec = 2.0_wp - ec
    endif
  endif
end function erfc_reprod

!> Returns the cube root of a real argument at roundoff accuracy, in a form that
!! works properly with rescaling of the argument by integer powers of 8 (upstream
!! MOM6, Adcroft/Ward).  Bit-reproducible by construction: integer exponent
!! manipulation plus division-free Halley iterations and one Newton polish.
elemental function cuberoot(x) result(root)
  !$omp declare target
  real(wp), intent(in) :: x !< The argument of cuberoot
  real(wp) :: root !< The real cube root of x

  real(wp) :: asx ! |x| rescaled by an integer power of 8 into 0.125 < asx <= 1.0
  real(wp) :: root_asx ! The cube root of asx
  real(wp) :: ra_3 ! root_asx cubed
  real(wp) :: num, den ! Evolving estimate numerator/denominator
  real(wp) :: num_prev, den_prev ! Previous iteration values
  real(wp) :: np_3, dp_3 ! Cubes of the previous values
  real(wp) :: r0, r0_3 ! Initial estimate and its cube
  integer :: itt
  integer(kind=int64) :: e_x, s_x
  real(wp) :: xs        ! x, pre-scaled to normal range if x is denormal
  logical :: denorm ! True for denormal x
  real(wp), parameter :: two_p54 = 18014398509481984.0_wp  ! 2**54 = (2**18)**3, exact

  if (x /= x) then
    ! NaN passes through.  x /= x is the one NaN predicate every backend gets
    ! right: the former combined guard ((x>=0.0) .eqv. (x<=0.0)) relies on both
    ! comparisons being false for NaN, but on gcc/arm64 the unordered compare
    ! leaks through the LE condition code and the guard evaluates false, sending
    ! NaN's bit pattern into the exponent arithmetic below (measured: a finite
    ! ~6.7e102 on macOS arm64).
    root = x
  elseif (x == 0.0_wp) then
    ! +-0, sign preserved.
    root = x
  elseif (abs(x) > huge(x)) then
    ! The cube root of +-Inf is +-Inf.  Without this guard the exponent-field
    ! arithmetic below reads Inf's bit pattern as 2**1024 and returns a finite
    ! ~5.6e102 -- and that ill-defined path is where x86 and arm64 disagreed.
    root = x
  else
    xs = x
    denorm = (abs(x) < tiny(x))
    if (denorm) xs = x * two_p54
    ! Denormal inputs: the exponent field is zero, so rescale_cbrt's bit
    ! arithmetic reads a nonsense exponent.  Pre-scaling by the exact cube
    ! 2**54 = (2**18)**3 makes the argument normal; the exact scale(-18)
    ! afterwards undoes it (the result, >= ~2**-358, is comfortably normal).
    call rescale_cbrt(xs, asx, e_x, s_x)
    ! Halley's method in fractional form (no divisions inside the iterations).
    r0 = 0.707106_wp
    r0_3 = r0 * r0 * r0
    num = r0 * (r0_3 + 2.0_wp * asx)
    den = 2.0_wp * r0_3 + asx
    do itt=1,2
      num_prev = num ; den_prev = den
      np_3 = num_prev * num_prev * num_prev
      dp_3 = den_prev * den_prev * den_prev
      num = num_prev * (np_3 + 2.0_wp * asx * dp_3)
      den = den_prev * (2.0_wp * np_3 + asx * dp_3)
    enddo
    root_asx = num / den
    ! One Newton polish to within the last bit.
    ra_3 = root_asx * root_asx * root_asx
    root_asx = root_asx - (ra_3 - asx) / (3.0_wp * (root_asx * root_asx))
    root = descale(root_asx, e_x, s_x)
    if (denorm) root = scale(root, -18)
  endif
end function cuberoot

end module bit_repro
