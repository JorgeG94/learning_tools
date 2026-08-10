!> CPU-vs-GPU bitwise reproducibility gate for the bit_repro functions.
!!
!! The same input arrays go through a plain host do-loop and a `do concurrent`
!! loop (offloaded under -stdpar=gpu).  Results are compared bit-for-bit via
!! integer transfer; any single differing bit is a FAIL.  NaN results compare
!! as equal-if-both-NaN (payloads are not part of the contract).
program gpu_bitwise

use, intrinsic :: iso_fortran_env, only : int64, real64
use bit_repro, only : exp_reprod, log_reprod, pow_reprod, cuberoot, erfc_reprod
implicit none

integer, parameter :: n = 1000000
real(real64), allocatable :: x(:), y(:), r_cpu(:), r_gpu(:)
integer(int64) :: seed_state
integer :: i, nbad_total

allocate(x(n), y(n), r_cpu(n), r_gpu(n))
seed_state = 88172645463325252_int64
nbad_total = 0

print '(a)', 'CPU vs GPU bitwise gate (any differing bit = FAIL)'
print '(a)', repeat('-', 64)

! --- pow, general path -------------------------------------------------------
do i = 1, n
  x(i) = exp( -30.0_real64 + 60.0_real64 * rand01() )
  y(i) = -3.0_real64 + 6.0_real64 * rand01()
enddo
do i = 1, n
  r_cpu(i) = pow_reprod(x(i), y(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = pow_reprod(x(i), y(i))
enddo
call check('pow  general y in [-3,3]')

! --- pow, MEKE fast path -----------------------------------------------------
do i = 1, n
  y(i) = merge( 0.25_real64, -0.25_real64, mod(i,2)==0 )
enddo
do i = 1, n
  r_cpu(i) = pow_reprod(x(i), y(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = pow_reprod(x(i), y(i))
enddo
call check('pow  y = +-0.25 (MEKE)')

! --- pow, integral + half-integral + negative bases --------------------------
do i = 1, n
  x(i) = sign( x(i), rand01() - 0.3_real64 )
  y(i) = real( -4 + mod(i, 9), real64 ) + merge(0.5_real64, 0.0_real64, mod(i,3)==0)
enddo
do i = 1, n
  r_cpu(i) = pow_reprod(x(i), y(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = pow_reprod(x(i), y(i))
enddo
call check('pow  integral/half, +-x')

! --- exp over EPBL range -----------------------------------------------------
do i = 1, n
  x(i) = -60.0_real64 + 63.0_real64 * rand01()
enddo
do i = 1, n
  r_cpu(i) = exp_reprod(x(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = exp_reprod(x(i))
enddo
call check('exp  [-60,3]')

! --- log ---------------------------------------------------------------------
do i = 1, n
  x(i) = exp( -690.0_real64 + 1380.0_real64 * rand01() )
enddo
do i = 1, n
  r_cpu(i) = log_reprod(x(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = log_reprod(x(i))
enddo
call check('log  log-uniform')

! --- cuberoot ----------------------------------------------------------------
do i = 1, n
  x(i) = sign( exp( -300.0_real64 + 600.0_real64 * rand01() ), rand01() - 0.5_real64 )
enddo
do i = 1, n
  r_cpu(i) = cuberoot(x(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = cuberoot(x(i))
enddo
call check('cuberoot  +-')

! --- erfc over the full working range ----------------------------------------
do i = 1, n
  x(i) = -7.0_real64 + 36.0_real64 * rand01()
enddo
do i = 1, n
  r_cpu(i) = erfc_reprod(x(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = erfc_reprod(x(i))
enddo
call check('erfc  [-7, 29]')

! --- specials through pow ----------------------------------------------------
call specials_set()
do i = 1, n
  r_cpu(i) = pow_reprod(x(i), y(i))
enddo
do concurrent (i = 1:n)
  r_gpu(i) = pow_reprod(x(i), y(i))
enddo
call check('pow  specials/denormals')

print '(a)', repeat('-', 64)
if (nbad_total == 0) then
  print '(a)', 'ALL SETS BITWISE'
else
  print '(a,i0,a)', 'FAIL: ', nbad_total, ' differing elements in total'
endif

contains

function rand01() result(r)
  real(real64) :: r
  seed_state = ieor(seed_state, ishft(seed_state, 13))
  seed_state = ieor(seed_state, ishft(seed_state, -7))
  seed_state = ieor(seed_state, ishft(seed_state, 17))
  r = 0.5_real64 + 0.25_real64*real(seed_state, real64)/real(huge(seed_state), real64)
end function rand01

subroutine specials_set()
  use, intrinsic :: ieee_arithmetic
  real(real64) :: vals(10)
  integer :: k
  vals = [ 0.0_real64, -0.0_real64, ieee_value(1.0_real64, ieee_positive_inf), &
           ieee_value(1.0_real64, ieee_negative_inf), ieee_value(1.0_real64, ieee_quiet_nan), &
           tiny(1.0_real64), tiny(1.0_real64)/2.0_real64**20, huge(1.0_real64), &
           1.0_real64, -1.0_real64 ]
  do k = 1, n
    x(k) = vals(1 + mod(k, 10))
    y(k) = vals(1 + mod(k/10, 10))
  enddo
end subroutine specials_set

subroutine check(name)
  character(*), intent(in) :: name
  integer :: k, nbad
  integer(int64) :: ic, ig
  nbad = 0
  do k = 1, n
    ic = transfer(r_cpu(k), 1_int64)
    ig = transfer(r_gpu(k), 1_int64)
    if (ic /= ig) then
      ! both-NaN compares equal (payloads are not part of the contract)
      if (.not. (r_cpu(k) /= r_cpu(k) .and. r_gpu(k) /= r_gpu(k))) then
        if (nbad < 3) print '(4x,a,i8,3es24.16)', 'diff at ', k, x(k), r_cpu(k), r_gpu(k)
        nbad = nbad + 1
      endif
    endif
  enddo
  nbad_total = nbad_total + nbad
  if (nbad == 0) then
    print '(2x,a,2x,a)', name, 'BITWISE'
  else
    print '(2x,a,2x,a,i0,a)', name, 'FAIL (', nbad, ' diffs)'
  endif
end subroutine check

end program gpu_bitwise
