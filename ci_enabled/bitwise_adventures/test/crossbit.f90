!> Cross-compiler bitwise gate: writes the raw result bits of every bit_repro
!! function over deterministic input sets to a binary file.  Build and run this
!! under two compilers, then `cmp` the two files -- any differing byte is a
!! reproducibility break.
!!
!! Inputs are constructed ONLY from IEEE-exact operations (integer xorshift PRNG,
!! exact real() of small integers, exact power-of-two division, scale()), so both
!! binaries see bit-identical inputs without trusting any libm.  Sets include
!! normals over the full exponent range, DENORMALS (the set that catches the
!! nvfortran scale()/exponent() behaviors), near-1 bases, integral/half/quarter
!! and general exponents, and the IEEE specials.  NaN results are canonicalized
!! to one payload before writing (payloads are not part of the contract).
program crossbit

use, intrinsic :: iso_fortran_env, only : int64, real64
use, intrinsic :: ieee_arithmetic, only : ieee_value, ieee_positive_inf, &
                                          ieee_negative_inf, ieee_quiet_nan
use bit_repro, only : exp_reprod, log_reprod, pow_reprod, cuberoot
implicit none

integer, parameter :: n = 200000
real(real64), allocatable :: x(:), y(:)
integer(int64), allocatable :: bits(:)
integer(int64) :: seed_state
character(len=64) :: tag
character(len=4), parameter :: fnames(4) = ['pow ','exp ','log ','cbrt']
integer :: i, u, set, ee
real(real64) :: mm
integer(int64) :: rr

allocate(x(n), y(n), bits(4*n))
seed_state = 88172645463325252_int64
call get_command_argument(1, tag)
if (len_trim(tag) == 0) tag = 'out'

open(newunit=u, file='crossbit_'//trim(tag)//'.bin', access='stream', &
     form='unformatted', status='replace')

do set = 1, 6
  select case (set)
  case (1)  ! normal bases, wide exponent range; general y
    do i = 1, n
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, -320, 320)
      x(i) = mk(mm, ee)
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, -2, 2)
      y(i) = mk(mm, ee)
      call next(rr) ; if (btest(rr, 0)) y(i) = -y(i)
    enddo
  case (2)  ! DENORMAL bases; moderate y
    do i = 1, n
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = -1074 + int(modulo(rr, 50_int64))
      x(i) = mk(mm, ee)
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, -3, 0)
      y(i) = mk(mm, ee)
      call next(rr) ; if (btest(rr, 0)) y(i) = -y(i)
    enddo
  case (3)  ! near-1 bases; large y (the amplification regime)
    do i = 1, n
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, -30, -3)
      x(i) = 1.0_real64 + mk(mm, ee)
      call next(rr) ; if (btest(rr, 0)) x(i) = 1.0_real64 / x(i)
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, 2, 9)
      y(i) = mk(mm, ee)
      call next(rr) ; if (btest(rr, 0)) y(i) = -y(i)
    enddo
  case (4)  ! integral and half-integral y (fast paths + boundaries), +- bases
    do i = 1, n
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, -40, 40)
      x(i) = mk(mm, ee)
      call next(rr) ; if (btest(rr, 0)) x(i) = -x(i)
      call next(rr) ; y(i) = real(int(modulo(rr, 41_int64)) - 20, real64)
      call next(rr) ; if (btest(rr, 1)) y(i) = y(i) + 0.5_real64
    enddo
  case (5)  ! quarter exponents (MEKE) incl. exact +-0.25
    do i = 1, n
      call next(rr) ; mm = mant(rr) ; call next(rr) ; ee = expo(rr, -300, 300)
      x(i) = mk(mm, ee)
      call next(rr) ; y(i) = 0.25_real64 * real(int(modulo(rr, 9_int64)) - 4, real64)
    enddo
  case (6)  ! IEEE specials cross product
    call specials_fill()
  end select

  do i = 1, n
    ! NaN-explicit clamp: min/max with a NaN operand is processor-dependent
    ! (gfortran disagrees with ifx/nvfortran), so never clamp through them.
    mm = y(i)
    if (mm == mm) mm = min(max(mm, -800.0_real64), 800.0_real64)
    bits(i)     = canon( pow_reprod(x(i), y(i)) )
    bits(n+i)   = canon( exp_reprod(mm) )
    bits(2*n+i) = canon( log_reprod(x(i)) )
    bits(3*n+i) = canon( cuberoot(x(i)) )
  enddo
  ! per-set, per-function xor-fold: localizes a cross-platform divergence to a
  ! (set, function) cell straight from the CI log
  do i = 0, 3
    print '(2x,a,i0,2x,a,2x,z16.16)', 'set', set, fnames(i+1), &
      xorfold(bits(i*n+1 : (i+1)*n))
  enddo
  write(u) bits
enddo
close(u)
print '(a)', 'wrote crossbit_'//trim(tag)//'.bin'

contains

!> Advance the xorshift64 state and return the draw.  A SUBROUTINE on purpose:
!! impure function references may be duplicated or elided by optimizers (ifx
!! -O3 -ipo was caught doing exactly that), which silently desynchronizes the
!! sequence between compilers; call statements cannot be.
subroutine next(r)
  integer(int64), intent(out) :: r
  seed_state = ieor(seed_state, ishft(seed_state, 13))
  seed_state = ieor(seed_state, ishft(seed_state, -7))
  seed_state = ieor(seed_state, ishft(seed_state, 17))
  r = abs(seed_state)
end subroutine next

!> Mantissa in [1, 2) from a draw: 1 + (low 52 bits)/2**52 -- every step exact.
pure function mant(r) result(m)
  integer(int64), intent(in) :: r
  real(real64) :: m
  m = 1.0_real64 + real(ibits(r, 0, 52), real64) * 2.0_real64**(-52)
end function mant

!> m * 2**e, denormal-safe everywhere: nvfortran's scale() flushes denormal
!! results regardless of -Mnoflushz/-Mnodaz, so the last 64 halvings are one
!! IEEE multiply (the library's own R1 fix, applied to input construction too).
pure function mk(m, e) result(v)
  real(real64), intent(in) :: m
  integer, intent(in) :: e
  real(real64) :: v
  if (e >= -1000) then
    v = scale(m, e)
  else
    v = scale(m, e + 64) * 2.0_real64**(-64)
  endif
end function mk

!> Exponent in [lo, hi] from a draw.
pure function expo(r, lo, hi) result(e)
  integer(int64), intent(in) :: r
  integer, intent(in) :: lo, hi
  integer :: e
  e = lo + int(modulo(r, int(hi - lo + 1, int64)))
end function expo

subroutine specials_fill()
  real(real64) :: vals(12)
  integer :: k
  vals = [ 0.0_real64, -0.0_real64, ieee_value(1.0_real64, ieee_positive_inf), &
           ieee_value(1.0_real64, ieee_negative_inf), &
           ieee_value(1.0_real64, ieee_quiet_nan), tiny(1.0_real64), &
           mk(1.0_real64, -1074), mk(1.5_real64, -1050), huge(1.0_real64), &
           1.0_real64, -1.0_real64, 2.0_real64 ]
  do k = 1, n
    x(k) = vals(1 + mod(k, 12))
    y(k) = vals(1 + mod(k/12, 12))
  enddo
end subroutine specials_fill

!> XOR-fold of a bits array, position-salted so permutations also show.  The
!! salt is 1+mod(k,7): never zero (v xor ishft(v,0) = 0 would blind the fold to
!! every element at that residue -- a real blind spot that hid the macOS set-6
!! divergence), and coprime to the 144-period specials pattern.
pure function xorfold(bb) result(f)
  integer(int64), intent(in) :: bb(:)
  integer(int64) :: f
  integer :: k
  f = 0_int64
  do k = 1, size(bb)
    f = ieor(ieor(f, bb(k)), ishft(bb(k), 1 + mod(k, 7)))
  enddo
end function xorfold

!> Raw bits with NaN canonicalized (payloads are not part of the contract).
function canon(v) result(b)
  real(real64), intent(in) :: v
  integer(int64) :: b
  if (v /= v) then
    b = int(z'7FF8000000000000', int64)
  else
    b = transfer(v, 1_int64)
  endif
end function canon

end program crossbit
