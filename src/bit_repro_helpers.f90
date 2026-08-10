!> The arithmetic machinery behind the bit_repro public functions: exact-rounding
!! primitives, the double-double log2/exp2 core, binary exponentiation, and the
!! cuberoot rescalers.  Bodies are verbatim from the pre-submodule layout; the
!! split is bit-neutral and gated by the harness checksums.
submodule (bit_repro) bit_repro_helpers

implicit none

! IEEE-754 binary64 layout constants
integer, parameter :: fraclen = 52  !< Bits in the fraction (mantissa)
integer, parameter :: explen  = 11  !< Bits in the exponent
integer, parameter :: signbit = 63  !< Position of the sign bit
integer, parameter :: expbit  = fraclen !< Position of the lowest exponent bit
integer(kind=int64), parameter :: bias = 1023_int64 !< Exponent bias

contains

!> x**n for n >= 0 by binary exponentiation: pure IEEE multiplies, so it is
!! reproducible everywhere, and ipow(x,2) == RN(x*x) exactly.
elemental module function ipow(x, n) result(p)
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
elemental module function ipow_signed(x, n) result(p)
  !$omp declare target
  real, intent(in) :: x
  integer, intent(in) :: n
  real :: p
  p = ipow(x, n)
end function ipow_signed

!> Knuth two_sum: s + e == a + b exactly, branch-free.
elemental module subroutine two_sum(a, b, s, e)
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
elemental module subroutine two_prod(a, b, p, e)
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
elemental module subroutine log2_dd(x, h, l)
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
elemental module function exp2_pair(h, l) result(e2)
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

!> Rescale `a` to the range [0.125, 1) while computing its cube-root exponent and
!! sign, by direct manipulation of the IEEE-754 bit representation.
pure module subroutine rescale_cbrt(a, x, e_r, s_a)
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
pure module function descale(x, e_a, s_a) result(a)
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

end submodule bit_repro_helpers
