!> GPU kernel throughput for the bit_repro functions: times do concurrent kernels
!! and prints ns/element plus an order-sensitive checksum (also used to compare
!! layouts bitwise).  Kernels are synchronous under -stdpar=gpu, so host
!! system_clock brackets are valid; one warmup launch precedes each timing.
program gpu_perf

use, intrinsic :: iso_fortran_env, only : int64, real64
use bit_repro, only : exp_reprod, pow_reprod, erfc_reprod
implicit none

integer, parameter :: n = 4000000
integer, parameter :: reps = 30
real(real64), allocatable :: x(:), y(:), r(:)
integer(int64) :: seed_state
integer :: i

allocate(x(n), y(n), r(n))
seed_state = 88172645463325252_int64

! general-path pow: y avoids every fast path
do i = 1, n
  x(i) = exp( -30.0_real64 + 60.0_real64 * rand01() )
  y(i) = -2.9_real64 + 5.8_real64 * rand01()
enddo
call bench_pow('pow general ')

! MEKE fast path
do i = 1, n
  y(i) = merge( 0.25_real64, -0.25_real64, mod(i,2)==0 )
enddo
call bench_pow('pow +-0.25  ')

! exp
do i = 1, n
  x(i) = -60.0_real64 + 63.0_real64 * rand01()
enddo
call bench_exp('exp         ')

! erfc over the MOM6-ish band mix
do i = 1, n
  x(i) = 8.0_real64 * rand01()
enddo
call bench_erfc('erfc        ')

contains

function rand01() result(rr)
  real(real64) :: rr
  seed_state = ieor(seed_state, ishft(seed_state, 13))
  seed_state = ieor(seed_state, ishft(seed_state, -7))
  seed_state = ieor(seed_state, ishft(seed_state, 17))
  rr = 0.5_real64 + 0.25_real64*real(seed_state, real64)/real(huge(seed_state), real64)
end function rand01

subroutine bench_pow(name)
  character(*), intent(in) :: name
  integer(int64) :: t0, t1, rate
  real(real64) :: chk
  integer :: k, rep
  do concurrent (k = 1:n)                    ! warmup (also first-touch transfers)
    r(k) = pow_reprod(x(k), y(k))
  enddo
  call system_clock(t0, rate)
  do rep = 1, reps
    do concurrent (k = 1:n)
      r(k) = pow_reprod(x(k), y(k))
    enddo
  enddo
  call system_clock(t1)
  chk = 0.0_real64
  do k = 1, n
    chk = chk + r(k)
  enddo
  print '(2x,a,f8.3,a,es22.15,a)', name//' ns/elem: ', &
    1.0e9_real64*real(t1-t0,real64)/real(rate,real64)/real(n,real64)/real(reps,real64), &
    '   (chk ', chk, ')'
end subroutine bench_pow

subroutine bench_erfc(name)
  character(*), intent(in) :: name
  integer(int64) :: t0, t1, rate
  real(real64) :: chk
  integer :: k, rep
  do concurrent (k = 1:n)
    r(k) = erfc_reprod(x(k))
  enddo
  call system_clock(t0, rate)
  do rep = 1, reps
    do concurrent (k = 1:n)
      r(k) = erfc_reprod(x(k))
    enddo
  enddo
  call system_clock(t1)
  chk = 0.0_real64
  do k = 1, n
    chk = chk + r(k)
  enddo
  print '(2x,a,f8.3,a,es22.15,a)', name//' ns/elem: ', &
    1.0e9_real64*real(t1-t0,real64)/real(rate,real64)/real(n,real64)/real(reps,real64), &
    '   (chk ', chk, ')'
end subroutine bench_erfc

subroutine bench_exp(name)
  character(*), intent(in) :: name
  integer(int64) :: t0, t1, rate
  real(real64) :: chk
  integer :: k, rep
  do concurrent (k = 1:n)
    r(k) = exp_reprod(x(k))
  enddo
  call system_clock(t0, rate)
  do rep = 1, reps
    do concurrent (k = 1:n)
      r(k) = exp_reprod(x(k))
    enddo
  enddo
  call system_clock(t1)
  chk = 0.0_real64
  do k = 1, n
    chk = chk + r(k)
  enddo
  print '(2x,a,f8.3,a,es22.15,a)', name//' ns/elem: ', &
    1.0e9_real64*real(t1-t0,real64)/real(rate,real64)/real(n,real64)/real(reps,real64), &
    '   (chk ', chk, ')'
end subroutine bench_exp

end program gpu_perf
