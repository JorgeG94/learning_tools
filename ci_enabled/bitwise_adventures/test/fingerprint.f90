!> Mechanism fingerprint: a dozen hex canaries, each chosen so that a specific
!! portability mechanism flips it.  Run on every CI leg (output is printed, not
!! gated) -- when the golden gate fails on a platform, diff its fingerprint
!! against a passing leg to identify the mechanism instead of bisecting 4.8M
!! values.
!!
!!  contract   : 0000000000000000 if a*b+c is NOT contracted; 3C90000000000000 if fused
!!  contractn  : mirror test for the fused negate-multiply-add pattern
!!  nint_tie   : 4008000000000000 (3.0) for Fortran half-away; 4000000000000000 (2.0)
!!               if the target rounds ties-to-even
!!  two_prod   : residual of a canonical Veltkamp split (BC90000000000000 expected)
!!  div, sqrt_ : correctly-rounded division / sqrt spot checks
!!  denorm_*   : gradual-underflow round-trips (flush-to-zero shows as 0)
!!  exp/log/pow: fixed-argument library calls covering poly, dd-log2, exp2 table
program fingerprint

use, intrinsic :: iso_fortran_env, only : int64, real64
use bit_repro, only : exp_reprod, log_reprod, pow_reprod, cuberoot
implicit none

real(real64) :: a, b, c, z, p, e
real(real64), volatile :: va, vb, vc   ! volatile defeats constant folding, so
                                       ! these exercise the RUNTIME instruction
integer(int64) :: i64

! --- fma contraction discriminators -----------------------------------------
va = 1.0_real64 + 2.0_real64**(-27)
vb = 1.0_real64 + 2.0_real64**(-27)
vc = -(1.0_real64 + 2.0_real64**(-26))
z = va*vb + vc
call pr('contract ', z)
z = -(va*vb) + (-vc)
call pr('contractn', z)

! --- nint tie rounding (Fortran: half away from zero) ------------------------
va = 2.5_real64
z = real(nint(va), real64)
call pr('nint_tie ', z)
va = -2.5_real64
z = real(nint(va), real64)
call pr('nint_tien', z)

! --- two_prod-style residual via the split (inline copy of the algebra) ------
a = 1.0_real64 + 2.0_real64**(-30)
b = 1.0_real64 - 2.0_real64**(-30)
p = a*b
block
  real(real64), parameter :: splitter = 134217729.0_real64
  real(real64) :: a1, a2, b1, b2, ta, tb
  ta = splitter*a ; a1 = ta - (ta - a) ; a2 = a - a1
  tb = splitter*b ; b1 = tb - (tb - b) ; b2 = b - b1
  e = ((a1*b1 - p) + a1*b2 + a2*b1) + a2*b2
end block
call pr('two_prod ', e)

! --- correctly-rounded spot checks -------------------------------------------
va = 1.0_real64 ; vb = 3.0_real64
call pr('div      ', va/vb)
va = 2.0_real64
call pr('sqrt_    ', sqrt(va))

! --- gradual underflow -------------------------------------------------------
va = tiny(1.0_real64)
vb = va * 2.0_real64**(-20)          ! denormal via IEEE multiply
call pr('denorm_mk', vb)
call pr('denorm_x2', vb * 2.0_real64)
call pr('denormsqr', sqrt(vb))

! --- library calls, fixed args covering each internal mechanism --------------
call pr('exp_third', exp_reprod(0.3125_real64))
call pr('exp_deep ', exp_reprod(-745.0_real64))          ! denormal result
call pr('log_near1', log_reprod(1.0_real64 + 2.0_real64**(-10)))
call pr('log_denrm', log_reprod(vb))                     ! denormal argument
call pr('pow_gen  ', pow_reprod(3.0_real64, 0.3_real64))
call pr('pow_meke ', pow_reprod(3.0_real64, 0.25_real64))
call pr('pow_big  ', pow_reprod(1.5_real64, 800.0_real64))
call pr('cbrt     ', cuberoot(7.0_real64))

contains

subroutine pr(name, v)
  character(*), intent(in) :: name
  real(real64), intent(in) :: v
  print '(2x,a,2x,z16.16)', name, transfer(v, 1_int64)
end subroutine pr

end program fingerprint
