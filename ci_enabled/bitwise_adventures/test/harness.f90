!> Accuracy + speed harness for the bit_repro functions.
!!
!! Accuracy: candidate and the compiler intrinsic are both measured in ULPs against
!! a quad-precision (real128) oracle rounded to double -- i.e. against the
!! correctly-rounded result.  Reports max ULP, mean ULP, % correctly rounded
!! (0 ulp) and % faithful (<=1 ulp).
!!
!! Speed: wall-clock throughput of candidate vs intrinsic over the same array,
!! with a checksum printed so the loops cannot be dead-code eliminated.
!!
!! Also: IEEE special-value table and the pow round-trip property
!! sqrt(pow(x,2)) == x.
program harness

use, intrinsic :: iso_fortran_env, only : int64, real64, real128
use bit_repro, only : exp_reprod, log_reprod, pow_reprod, pow_reprod_explog, cuberoot, erfc_reprod, erfc_reprod_bf, erfc_reprod_v
implicit none

integer, parameter :: n = 2000000
real(real64), allocatable :: xs(:), ys(:), out1(:), out2(:)
integer :: i
integer :: n_violations = 0  ! budget violations; nonzero exit makes this a CI gate
integer(int64) :: seed_state

allocate(xs(n), ys(n), out1(n), out2(n))
seed_state = 88172645463325252_int64

print '(a)', repeat('=', 78)
print '(a)', 'bit_repro_adventure harness  (oracle: real128, correctly rounded to real64)'
print '(a)', repeat('=', 78)

! ---------------- exp: EPBL-like range, decay arguments in [-60, 3] ------------
do i = 1, n
  xs(i) = -60.0_real64 + 63.0_real64 * rand01()
enddo
call accuracy_exp('exp  [-60,3] (EPBL decay range)', xs)

! exp: full finite range
do i = 1, n
  xs(i) = -700.0_real64 + 1400.0_real64 * rand01()
enddo
call accuracy_exp('exp  [-700,700] (full range)', xs)

! ---------------- log: mantissa-dense + wide dynamic range ---------------------
do i = 1, n
  xs(i) = exp( -690.0_real64 + 1380.0_real64 * rand01() )   ! log-uniform over ~[1e-300,1e300]
enddo
call accuracy_log('log  log-uniform [1e-300,1e300]', xs)

do i = 1, n
  xs(i) = 0.5_real64 + 1.5_real64 * rand01()                 ! the hard cancellation zone near 1
enddo
call accuracy_log('log  [0.5,2] (near-1 cancellation)', xs)

! ---------------- pow: MEKE-like exponents on positive bases -------------------
do i = 1, n
  xs(i) = exp( -30.0_real64 + 60.0_real64 * rand01() )       ! bases over ~[1e-13,1e13]
  ys(i) = merge( 0.25_real64, -0.25_real64, mod(i,2)==0 )    ! MEKE lengthscale exponents
enddo
call accuracy_pow('pow  x**(+-0.25), x in [1e-13,1e13] (MEKE)', xs, ys)

do i = 1, n
  xs(i) = exp( -30.0_real64 + 60.0_real64 * rand01() )
  ys(i) = -3.0_real64 + 6.0_real64 * rand01()                ! general small exponents
enddo
call accuracy_pow('pow  x**y, y in [-3,3]', xs, ys)

do i = 1, n
  xs(i) = exp( -30.0_real64 + 60.0_real64 * rand01() )
  ys(i) = real( 2 + mod(i, 7), real64 )                     ! integral y in [2,8]
enddo
call accuracy_pow('pow  x**n, n integral 2..8', xs, ys)

do i = 1, n
  xs(i) = exp( -30.0_real64 + 60.0_real64 * rand01() )
  ys(i) = 1.5_real64 * merge(1.0_real64, -1.0_real64, mod(i,2)==0)  ! y = +-1.5
enddo
call accuracy_pow('pow  x**(+-1.5) (half-integral)', xs, ys)

! ---------------- cuberoot ------------------------------------------------------
do i = 1, n
  xs(i) = sign( exp( -690.0_real64 + 1380.0_real64 * rand01() ), rand01() - 0.5_real64 )
enddo
call accuracy_cbrt('cuberoot  +- log-uniform', xs)

! ---------------- erfc ----------------------------------------------------------
do i = 1, n
  xs(i) = 6.5_real64 * rand01()                       ! MOM6 range: erfc(sqrt(...)) >= 0
enddo
call accuracy_erfc('erfc [0, 6.5] (MOM6/wave range)', xs)

do i = 1, n
  xs(i) = -6.5_real64 + 34.0_real64 * rand01()        ! full working range incl. tail
enddo
call accuracy_erfc('erfc [-6.5, 27.5] (full range)', xs)

do i = 1, n
  xs(i) = sign( 0.47_real64 * rand01(), rand01() - 0.5_real64 )  ! Maclaurin band, +-
enddo
call accuracy_erfc('erfc [-0.47, 0.47] (Maclaurin band)', xs)

! ---------------- erfc branch-free variant: enforced lockstep -------------------
call erfc_bf_lockstep()

! ---------------- pow round-trip property --------------------------------------
call roundtrip_pow2()

! ---------------- IEEE specials -------------------------------------------------
call specials()

! ---------------- throughput ----------------------------------------------------
print '(/,a)', repeat('-', 78)
print '(a)', 'Throughput (ns/call, lower is better; checksums prevent DCE)'
print '(a)', repeat('-', 78)
do i = 1, n
  xs(i) = -60.0_real64 + 63.0_real64 * rand01()
enddo
call bench_exp(xs)
do i = 1, n
  xs(i) = exp( -30.0_real64 + 60.0_real64 * rand01() ) ; ys(i) = -3.0_real64 + 6.0_real64*rand01()
enddo
call bench_log_pow_cbrt(xs, ys)

print '(/,a)', repeat('=', 78)
if (n_violations == 0) then
  print '(a)', 'ACCURACY CONTRACT: PASS (all candidates within budget)'
else
  print '(a,i0,a)', 'ACCURACY CONTRACT: FAIL (', n_violations, ' budget violations)'
  error stop 1
endif

contains

!> xorshift64 PRNG -> uniform in (0,1); reproducible across compilers.
function rand01() result(r)
  real(real64) :: r
  seed_state = ieor(seed_state, ishft(seed_state, 13))
  seed_state = ieor(seed_state, ishft(seed_state, -7))
  seed_state = ieor(seed_state, ishft(seed_state, 17))
  r = 0.5_real64 + 0.25_real64*real(seed_state, real64)/real(huge(seed_state), real64)
end function rand01

!> Monotone integer mapping of an IEEE double, so ULP distance is a subtraction.
elemental function ulpmap(x) result(m)
  real(real64), intent(in) :: x
  integer(int64) :: m
  m = transfer(x, 1_int64)
  if (m < 0_int64) m = ibset(0_int64, 63) - m
end function ulpmap

function ulpdist(a, b) result(d)
  real(real64), intent(in) :: a, b
  integer(int64) :: d
  d = abs(ulpmap(a) - ulpmap(b))
end function ulpdist

subroutine report(name, impl, dmax, dsum, ncr, nfaith, nn)
  character(*), intent(in) :: name, impl
  integer(int64), intent(in) :: dmax, dsum, ncr, nfaith
  integer, intent(in) :: nn
  print '(2x,a,2x,a12,2x,a,i6,2x,a,f8.4,2x,a,f7.2,a,2x,a,f7.2,a)', &
    name, impl, 'max_ulp=', dmax, 'mean_ulp=', real(dsum)/real(nn), &
    'CR=', 100.0*real(ncr)/real(nn), '%', 'faithful<=1=', 100.0*real(nfaith)/real(nn), '%'
end subroutine report

subroutine tally(name, impl, ref, got, nn, budget)
  character(*), intent(in) :: name, impl
  real(real64), intent(in) :: ref(:), got(:)
  integer, intent(in) :: nn
  integer, intent(in), optional :: budget !< max acceptable ulp (the accuracy contract)
  integer(int64) :: dmax, dsum, ncr, nfaith, d
  integer :: k
  dmax = 0 ; dsum = 0 ; ncr = 0 ; nfaith = 0
  do k = 1, nn
    d = ulpdist(ref(k), got(k))
    dmax = max(dmax, d) ; dsum = dsum + min(d, 1000000_int64)
    if (d == 0) ncr = ncr + 1
    if (d <= 1) nfaith = nfaith + 1
  enddo
  call report(name, impl, dmax, dsum, ncr, nfaith, nn)
  if (present(budget)) then
    if (dmax > int(budget, int64)) then
      print '(4x,a,i0,a,i0)', 'ACCURACY REGRESSION: max_ulp ', dmax, ' > budget ', budget
      n_violations = n_violations + 1
    endif
  endif
end subroutine tally

subroutine accuracy_exp(name, x)
  character(*), intent(in) :: name
  real(real64), intent(in) :: x(:)
  integer :: k
  do k = 1, size(x)
    out1(k) = real( exp(real(x(k), real128)), real64 )  ! correctly-rounded oracle
    out2(k) = exp(x(k))
  enddo
  print '(/,a)', trim(name)
  call tally(name, 'intrinsic', out1, out2, size(x))
  do k = 1, size(x)
    out2(k) = exp_reprod(x(k))
  enddo
  call tally(name, 'reprod', out1, out2, size(x), budget=1)
end subroutine accuracy_exp

subroutine accuracy_log(name, x)
  character(*), intent(in) :: name
  real(real64), intent(in) :: x(:)
  integer :: k
  do k = 1, size(x)
    out1(k) = real( log(real(x(k), real128)), real64 )
    out2(k) = log(x(k))
  enddo
  print '(/,a)', trim(name)
  call tally(name, 'intrinsic', out1, out2, size(x))
  do k = 1, size(x)
    out2(k) = log_reprod(x(k))
  enddo
  call tally(name, 'reprod', out1, out2, size(x), budget=1)
end subroutine accuracy_log

subroutine accuracy_pow(name, x, y)
  character(*), intent(in) :: name
  real(real64), intent(in) :: x(:), y(:)
  integer :: k
  do k = 1, size(x)
    out1(k) = real( real(x(k), real128)**real(y(k), real128), real64 )
    out2(k) = x(k)**y(k)
  enddo
  print '(/,a)', trim(name)
  call tally(name, 'intrinsic', out1, out2, size(x))
  do k = 1, size(x)
    out2(k) = pow_reprod_explog(x(k), y(k))
  enddo
  call tally(name, 'explog(old)', out1, out2, size(x))
  do k = 1, size(x)
    out2(k) = pow_reprod(x(k), y(k))
  enddo
  call tally(name, 'reprod(v2)', out1, out2, size(x), budget=2)
end subroutine accuracy_pow

subroutine accuracy_erfc(name, x)
  character(*), intent(in) :: name
  real(real64), intent(in) :: x(:)
  integer :: k
  do k = 1, size(x)
    out1(k) = real( erfc(real(x(k), real128)), real64 )
    out2(k) = erfc(x(k))
  enddo
  print '(/,a)', trim(name)
  call tally(name, 'intrinsic', out1, out2, size(x))
  do k = 1, size(x)
    out2(k) = erfc_reprod(x(k))
  enddo
  call tally(name, 'reprod', out1, out2, size(x), budget=5)
end subroutine accuracy_erfc

subroutine accuracy_cbrt(name, x)
  character(*), intent(in) :: name
  real(real64), intent(in) :: x(:)
  integer :: k
  do k = 1, size(x)
    out1(k) = real( sign( abs(real(x(k), real128))**(1.0_real128/3.0_real128), &
                          real(x(k), real128)), real64 )
    out2(k) = sign( abs(x(k))**(1.0_real64/3.0_real64), x(k) )
  enddo
  print '(/,a)', trim(name)
  call tally(name, 'intrinsic**', out1, out2, size(x))
  do k = 1, size(x)
    out2(k) = cuberoot(x(k))
  enddo
  call tally(name, 'cuberoot', out1, out2, size(x), budget=1)
end subroutine accuracy_cbrt

!> The branch-free erfc variant must remain BIT-IDENTICAL to the branchy one
!! (same selected-arm operation sequences by construction); any drift between
!! them is a contract violation.
subroutine erfc_bf_lockstep()
  integer :: k, nbad
  real(real64) :: xv
  nbad = 0
  do k = 1, n
    xv = -30.0_real64 + 60.0_real64 * rand01()
    if (transfer(erfc_reprod(xv), 1_int64) /= transfer(erfc_reprod_bf(xv), 1_int64)) nbad = nbad + 1
  enddo
  print '(/,a,i0)', 'erfc branchy-vs-branch-free lockstep diffs: ', nbad
  if (nbad /= 0) then
    print '(4x,a)', 'ACCURACY REGRESSION: erfc_bf drifted from erfc_reprod'
    n_violations = n_violations + 1
  endif
  ! the 8-wide kernel too (odd count exercises its scalar tail)
  block
    real(real64) :: xv8(100001), rv8(100001)
    integer :: kk
    do kk = 1, size(xv8)
      xv8(kk) = -30.0_real64 + 60.0_real64 * rand01()
    enddo
    call erfc_reprod_v(xv8, rv8)
    nbad = 0
    do kk = 1, size(xv8)
      if (transfer(erfc_reprod(xv8(kk)), 1_int64) /= transfer(rv8(kk), 1_int64)) nbad = nbad + 1
    enddo
    print '(a,i0)', 'erfc branchy-vs-8-wide lockstep diffs: ', nbad
    if (nbad /= 0) then
      print '(4x,a)', 'ACCURACY REGRESSION: erfc_v drifted from erfc_reprod'
      n_violations = n_violations + 1
    endif
  end block
end subroutine erfc_bf_lockstep

subroutine roundtrip_pow2()
  integer :: k, bad_rep, bad_int
  real(real64) :: x, r1, r2
  bad_rep = 0 ; bad_int = 0
  do k = 1, n
    x = exp( -150.0_real64 + 300.0_real64 * rand01() )
    r1 = sqrt( pow_reprod(x, 2.0_real64) )
    r2 = sqrt( x**2.0_real64 )
    if (r1 /= x) bad_rep = bad_rep + 1
    if (r2 /= x) bad_int = bad_int + 1
  enddo
  print '(/,a)', 'Round-trip property: sqrt(pow(x,2)) == x   (n = 2e6 log-uniform x)'
  print '(2x,a,i9,a,f8.4,a)', 'intrinsic **2.0  failures: ', bad_int, '  (', 100.0*real(bad_int)/real(n), ' %)'
  print '(2x,a,i9,a,f8.4,a)', 'pow_reprod       failures: ', bad_rep, '  (', 100.0*real(bad_rep)/real(n), ' %)'
  if (bad_rep /= 0) then
    print '(4x,a)', 'ACCURACY REGRESSION: round-trip property violated'
    n_violations = n_violations + 1
  endif
end subroutine roundtrip_pow2

subroutine specials()
  use, intrinsic :: ieee_arithmetic
  real(real64) :: pinf, ninf, qnan, pzero, nzero, denorm
  ! volatile: keeps gfortran from constant-folding the intrinsic reference
  ! calls, which is a compile ERROR for overflowing/negative-base constants
  real(real64), volatile :: v750, vm750, vz, vm2, v3, vh
  v750 = 750.0_real64 ; vm750 = -750.0_real64 ; vz = 0.0_real64
  vm2 = -2.0_real64 ; v3 = 3.0_real64 ; vh = 0.5_real64
  pinf = ieee_value(1.0_real64, ieee_positive_inf)
  ninf = ieee_value(1.0_real64, ieee_negative_inf)
  qnan = ieee_value(1.0_real64, ieee_quiet_nan)
  pzero = 0.0_real64 ; nzero = -0.0_real64 ; denorm = tiny(1.0_real64)/2.0_real64**10
  print '(/,a)', 'IEEE specials (candidate | intrinsic)'
  print '(2x,a,es13.5,a,es13.5)', 'exp(+Inf):  ', exp_reprod(pinf), ' | ', exp(pinf)
  print '(2x,a,es13.5,a,es13.5)', 'exp(-Inf):  ', exp_reprod(ninf), ' | ', exp(ninf)
  print '(2x,a,es13.5,a,es13.5)', 'exp(NaN):   ', exp_reprod(qnan), ' | ', exp(qnan)
  print '(2x,a,es13.5,a,es13.5)', 'exp(750):   ', exp_reprod(750.0_real64), ' | ', exp(v750)
  print '(2x,a,es13.5,a,es13.5)', 'exp(-750):  ', exp_reprod(-750.0_real64), ' | ', exp(vm750)
  print '(2x,a,es13.5,a,es13.5)', 'log(0):     ', log_reprod(pzero), ' | ', log(pzero)
  print '(2x,a,es13.5,a,es13.5)', 'log(+Inf):  ', log_reprod(pinf), ' | ', log(pinf)
  print '(2x,a,es13.5,a,es13.5)', 'log(denorm):', log_reprod(denorm), ' | ', log(denorm)
  print '(2x,a,es13.5,a,es13.5)', 'pow(x,0):   ', pow_reprod(qnan, 0.0_real64), ' | ', qnan**0.0_real64
  print '(2x,a,es13.5,a,es13.5)', 'pow(0,-2):  ', pow_reprod(0.0_real64, -2.0_real64), ' | ', vz**(-2.0_real64)
  print '(2x,a,es13.5,a,es13.5)', 'pow(-0,3):  ', pow_reprod(nzero, 3.0_real64), ' | ', nzero**3.0_real64
  print '(2x,a,es13.5,a,es13.5)', 'pow(-2,3):  ', pow_reprod(-2.0_real64, 3.0_real64), ' | ', vm2**v3
  print '(2x,a,es13.5,a,es13.5)', 'pow(-2,.5): ', pow_reprod(-2.0_real64, 0.5_real64), ' | ', sqrt(qnan)
  print '(2x,a,es13.5,a,es13.5)', 'pow(-1,Inf):', pow_reprod(-1.0_real64, pinf), ' | ', 1.0_real64
  print '(2x,a,es13.5,a,es13.5)', 'pow(.5,-If):', pow_reprod(0.5_real64, ninf), ' | ', pinf
  print '(2x,a,es13.5,a,es13.5)', 'pow(2,1e3): ', pow_reprod(2.0_real64, 1.0e3_real64), ' | ', 2.0_real64**1.0e3_real64
  print '(2x,a,es13.5,a,es13.5)', 'pow(2,-1e9):', pow_reprod(2.0_real64, -1.0e9_real64), ' | ', 2.0_real64**(-1.0e9_real64)
  print '(2x,a,es13.5,a,es13.5)', 'cbrt(-8):   ', cuberoot(-8.0_real64), ' | ', -2.0_real64
  print '(2x,a,es13.5,a,es13.5)', 'cbrt(-0):   ', cuberoot(nzero), ' | ', nzero
  print '(2x,a,es13.5,a,es13.5)', 'cbrt(NaN):  ', cuberoot(qnan), ' | ', qnan
end subroutine specials

subroutine bench_exp(x)
  real(real64), intent(in) :: x(:)
  real(real64) :: s
  integer(int64) :: t0, t1, rate
  integer :: k, rep
  integer, parameter :: reps = 20
  s = 0.0
  call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x)
    s = s + exp(x(k))
  enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'exp   intrinsic: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, &
        '  (chk ', s, ')'
  s = 0.0
  call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x)
    s = s + exp_reprod(x(k))
  enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'exp   reprod:    ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, &
        '  (chk ', s, ')'
end subroutine bench_exp

subroutine bench_log_pow_cbrt(x, y)
  real(real64), intent(in) :: x(:), y(:)
  real(real64) :: s
  integer(int64) :: t0, t1, rate
  integer :: k, rep
  integer, parameter :: reps = 20
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + log(x(k)) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'log   intrinsic: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + log_reprod(x(k)) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'log   reprod:    ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + x(k)**y(k) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'pow   intrinsic: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + pow_reprod_explog(x(k), y(k)) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'pow   explog:    ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + pow_reprod(x(k), y(k)) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'pow   reprod v2: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + x(k)**0.25_real64 ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'p.25  intrinsic: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + pow_reprod(x(k), 0.25_real64) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'p.25  reprod v2: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + sign(abs(x(k))**(1.0_real64/3.0_real64), x(k)) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'cbrt  intrinsic: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + cuberoot(x(k)) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'cbrt  reprod:    ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + erfc(abs(x(k))**0.1_real64) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'erfc  intrinsic: ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
  s = 0.0 ; call system_clock(t0, rate)
  do rep = 1, reps ; do k = 1, size(x) ; s = s + erfc_reprod(abs(x(k))**0.1_real64) ; enddo ; enddo
  call system_clock(t1)
  print '(2x,a,f8.2,a,es22.15,a)', 'erfc  reprod:    ', 1.0e9*real(t1-t0)/real(rate)/real(size(x))/reps, '  (chk ', s, ')'
end subroutine bench_log_pow_cbrt

end program harness
