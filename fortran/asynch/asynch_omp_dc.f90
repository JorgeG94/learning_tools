program main
use iso_fortran_env, only: int64, real64
implicit none

integer(int64) :: t0, t1, rate, tmax
real(real64)   :: dt
integer :: i, j, k, ii
integer :: ai,aj,ak
integer :: bi,bj,bk
integer :: ci,cj,ck
real, allocatable :: A(:,:,:)
real, allocatable :: B(:,:,:)
real, allocatable :: C(:,:,:)
real, allocatable :: d(:)


allocate(A(1500,250,150))
allocate(B(1500,250,150))
allocate(C(1500,250,150))
allocate(d(1500))

!$omp target enter data map(alloc: A,B,C,d)

call system_clock(count_rate=rate, count_max=tmax)
call system_clock(t0)

!$omp target nowait

!$omp teams loop
do ii = 1,1500
  d(ii) = 0.0
end do
!$omp end teams loop

!$omp end target

!$omp target nowait
!$omp teams loop collapse(3)
do i = 1, 1500
  do j = 1, 250
    do k = 1, 150
      a(i,j,k) = 1.0
      b(i,j,k) = 1.0
      c(i,j,k) = 1.0
    end do
  end do
end do
!$omp end teams loop
!$omp end target

!$omp taskwait

!$omp target nowait
!$omp teams loop collapse(3)
do i = 1, 1500
  do j = 1, 250
    do k = 1, 150
      a(i,j,k) = real(i*j*k)
    end do
  end do
end do
!$omp end teams loop
!$omp end target

!$omp target nowait
!$omp teams loop collapse(3)
do i = 1, 1500
  do j = 1, 250
    do k = 1, 150
      b(i,j,k) = real(i*j*k)
    end do
  end do
end do
!$omp end teams loop
!$omp end target

!$omp target nowait
!$omp teams loop collapse(3)
do i = 1, 1500
  do j = 1, 250
    do k = 1, 150
      c(i,j,k) = real(i*j*k)
    end do
  end do
end do
!$omp end teams loop
!$omp end target

!$omp taskwait

call system_clock(t1)

if (t1 >= t0) then
   dt = real(t1 - t0, real64) / real(rate, real64)
else
   ! handle wrap-around: counter goes 0..tmax then wraps
   dt = real((tmax - t0) + t1 + 1_int64, real64) / real(rate, real64)
end if

print '(A,F10.6,A)', 'Elapsed: ', dt, ' s'


!$omp target exit data map(delete:A,B,C,d)
!!$omp target exit data map(from: d)

!print *, d(1)
end program main
