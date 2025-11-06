  program loop_order_sweep_do_concurrent_tridiag_reuse
  use iso_fortran_env, only: real64
  implicit none

  integer, parameter :: nz_values(*) = [400]
  !integer, parameter :: nz_values(*) = [10, 25, 50, 100, 200, 400]
  integer, parameter :: ntests = size(nz_values)

  integer :: nx, ny, nz, narg
  integer :: i, j, k, idx
  character(len=32) :: arg

  ! 3D coefficient / state fields
  real(real64), allocatable :: a3d(:,:,:), h3d(:,:,:)
  real(real64), allocatable :: u(:,:,:),   unew(:,:,:), c1(:,:,:)

  ! 2D fields
  real(real64), allocatable :: ray(:,:), mask(:,:), b1(:,:), d1(:,:)
  real(real64),allocatable :: sum_b1(:,:), sum_d1(:,:)
  real(real64) :: d1_1, d1_2, d1_3, d1_4, d1_5

  real(real64) :: dt, t1, t2
  real(real64) :: timings(ntests,5)
  real(real64) :: val
  real(real64) :: a_loc, h_loc, ray_loc, b1_loc, d1_loc, b_denom, u_prev, u_loc
  real(real64), parameter :: one = 1.0_real64, zero = 0.0_real64

  !----------------------------------
  ! Command-line argument parsing
  !----------------------------------
  allocate(sum_b1(ntests,5), sum_d1(ntests,5))
  narg = command_argument_count()
  nx = 256
  ny = 256
  if (narg >= 1) then
     call get_command_argument(1, arg)
     read(arg, *) nx
  end if
  if (narg >= 2) then
     call get_command_argument(2, arg)
     read(arg, *) ny
  end if

  dt = 0.01_real64

  print *, "===================================================="
  print *, " DO CONCURRENT tridiag-like benchmark (reused arrays)"
  write(*,'(" Grid config (nx, ny) = (",I0,", ",I0,")")') nx, ny
  print *, "===================================================="
     sum_b1 = 0.0_real64
     sum_d1 = 0.0_real64

  do idx = 1, ntests
     nz = nz_values(idx)
     print *
     print '(A,I5)', ">>> Testing Nz = ", nz
     print *, "---------------------------------------------"

     ! Allocate fields
     allocate(a3d(nx,ny,nz), h3d(nx,ny,nz))
     allocate(u(nx,ny,nz),   unew(nx,ny,nz), c1(nx,ny,nz))
     allocate(ray(nx,ny), mask(nx,ny), b1(nx,ny), d1(nx,ny))

     call random_number(val)

     !------------------------------------------
     ! Move arrays to device (no copy) via OpenMP
     !------------------------------------------
     !$omp target enter data map(alloc: a3d,h3d,u,unew,c1,ray,mask,b1,d1)

     !------------------------------------------
     ! Initial coefficients and initial state
     ! (done on device through do concurrent)
     !------------------------------------------
     do concurrent (i=1:nx, j=1:ny, k=1:nz)
        ! 3D coefficients
        a3d(i,j,k)  = 0.1_real64
        h3d(i,j,k)  = 1.0_real64

        ! Initial u field
        u(i,j,k)    = val
        unew(i,j,k) = u(i,j,k)

        c1(i,j,k)   = 0.0_real64

        ! 2D fields, once per (i,j)
        if (k == 1) then
           ray(i,j)  = 0.01_real64
           mask(i,j) = 1.0_real64   ! all active; structural mask
           b1(i,j)   = 0.0_real64
           d1(i,j)   = 0.0_real64
        end if
     end do


     !=========================================================
     ! 1. vertical -> i -> j
     !    k outer (serial), inside parallel over (i,j)
     !=========================================================
     call cpu_time(t1)
     do k = 2, nz-1
        do concurrent (j=1:ny)
           do concurrent (i=1:nx)
              if (mask(i,j) <= 0.0_real64) cycle

              a_loc   = a3d(i,j,k)
              h_loc   = h3d(i,j,k)
              ray_loc = ray(i,j)
              b1_loc  = b1(i,j)
              d1_loc  = d1(i,j)

              c1(i,j,k) = dt * a_loc * b1_loc

              b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
              b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
              d1_loc  = b_denom * b1_loc

              u_prev  = unew(i,j,k-1)
              u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
              unew(i,j,k) = u_loc

              b1(i,j) = b1_loc
              d1(i,j) = d1_loc
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,1) = t2 - t1
     !$omp target update from(d1)
     d1_1 = d1(1,17)
     print '(A,F10.4," s")', " vertical->i->j elapsed:       ", timings(idx,1)

     !=========================================================
     ! 2. i -> j -> vertical
     !    parallel over (i,j), inner serial k
     !=========================================================
     call reset_state(nx,ny,nz,u,unew,c1,b1,d1)
     call cpu_time(t1)
     do concurrent (j=1:ny, i=1:nx)
        if (mask(i,j) <= 0.0_real64) cycle
        do k = 2, nz-1
           a_loc   = a3d(i,j,k)
           h_loc   = h3d(i,j,k)
           ray_loc = ray(i,j)
           b1_loc  = b1(i,j)
           d1_loc  = d1(i,j)

           c1(i,j,k) = dt * a_loc * b1_loc

           b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
           b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
           d1_loc  = b_denom * b1_loc

           u_prev  = unew(i,j,k-1)
           u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
           unew(i,j,k) = u_loc

           b1(i,j) = b1_loc
           d1(i,j) = d1_loc
        end do
     end do
     !$omp target update from(d1)
     d1_2 = d1(1,17)
     call cpu_time(t2)
     timings(idx,2) = t2 - t1
     print '(A,F10.4," s")', " i->j->vertical elapsed:        ", timings(idx,2)

     !=========================================================
     ! 3. j -> vertical -> i
     !    parallel over j, inner serial k, inner concurrent i
     !=========================================================
     call reset_state(nx,ny,nz,u,unew,c1,b1,d1)
     call cpu_time(t1)
     do concurrent (j=1:ny)
        do k = 2, nz-1
           do concurrent (i=1:nx)
              if (mask(i,j) <= 0.0_real64) cycle

              a_loc   = a3d(i,j,k)
              h_loc   = h3d(i,j,k)
              ray_loc = ray(i,j)
              b1_loc  = b1(i,j)
              d1_loc  = d1(i,j)

              c1(i,j,k) = dt * a_loc * b1_loc

              b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
              b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
              d1_loc  = b_denom * b1_loc

              u_prev  = unew(i,j,k-1)
              u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
              unew(i,j,k) = u_loc

              b1(i,j) = b1_loc
              d1(i,j) = d1_loc
           end do
        end do
     end do
     call cpu_time(t2)
     !$omp target update from(d1)
     d1_3 = d1(1,17)
     timings(idx,3) = t2 - t1
     print '(A,F10.4," s")', " j->vertical->i elapsed:        ", timings(idx,3)

     !=========================================================
     ! 4. vertical -> j -> i
     !    k outer, then concurrent j, then concurrent i
     !=========================================================
     call reset_state(nx,ny,nz,u,unew,c1,b1,d1)
     call cpu_time(t1)
     do k = 2, nz-1
        do concurrent (j=1:ny)
           do concurrent (i=1:nx)
              if (mask(i,j) <= 0.0_real64) cycle

              a_loc   = a3d(i,j,k)
              h_loc   = h3d(i,j,k)
              ray_loc = ray(i,j)
              b1_loc  = b1(i,j)
              d1_loc  = d1(i,j)

              c1(i,j,k) = dt * a_loc * b1_loc

              b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
              b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
              d1_loc  = b_denom * b1_loc

              u_prev  = unew(i,j,k-1)
              u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
              unew(i,j,k) = u_loc

              b1(i,j) = b1_loc
              d1(i,j) = d1_loc
           end do
        end do
     end do
     call cpu_time(t2)
     !$omp target update from(d1)
     d1_4 = d1(1,17)
     timings(idx,4) = t2 - t1
     print '(A,F10.4," s")', " vertical->j->i elapsed:        ", timings(idx,4)

     !=========================================================
     ! 5. j -> i -> vertical
     !    concurrent over j,i, then serial k
     !=========================================================
     call reset_state(nx,ny,nz,u,unew,c1,b1,d1)
     call cpu_time(t1)
     do concurrent (j=1:ny, i=1:nx)
        if (mask(i,j) <= 0.0_real64) cycle
        do k = 2, nz-1
           a_loc   = a3d(i,j,k)
           h_loc   = h3d(i,j,k)
           ray_loc = ray(i,j)
           b1_loc  = b1(i,j)
           d1_loc  = d1(i,j)

           c1(i,j,k) = dt * a_loc * b1_loc

           b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
           b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
           d1_loc  = b_denom * b1_loc

           u_prev  = unew(i,j,k-1)
           u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
           unew(i,j,k) = u_loc

           b1(i,j) = b1_loc
           d1(i,j) = d1_loc
        end do
     end do
     call cpu_time(t2)
     !$omp target update from(d1)
     d1_5 = d1(1,17)
     timings(idx,5) = t2 - t1
     print '(A,F10.4," s")', " j->i->vertical elapsed:        ", timings(idx,5)

     !------------------------------------------
     ! Free device and host memory
     !------------------------------------------
     !$omp target exit data map(delete: a3d,h3d,u,unew,c1,ray,mask,b1,d1)

     deallocate(a3d,h3d,u,unew,c1,ray,mask,b1,d1)
  end do


  sum_d1(1,1) = d1_1
  sum_d1(1,2) = d1_2
  sum_d1(1,3) = d1_3
  sum_d1(1,4) = d1_4
  sum_d1(1,5) = d1_5

  print *
  print *, "===================================================="
  print *, " Benchmark complete."
  print *, "===================================================="

  print *
  print *, "Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i,j->i->vertical"
  do idx = 1, ntests
     write(*,'(I5,5(",",F12.6))') nz_values(idx), timings(idx,1), timings(idx,2), &
                                   timings(idx,3), timings(idx,4), timings(idx,5)
  end do
! Compact validation
block
! Validate that all loop orderings give the same results
logical :: b1_valid, d1_valid
real(real64) :: b1_ref, d1_ref, b1_diff, d1_diff
real(real64), parameter :: tol = 1.0e-9_real64

print *
print *, "===================================================="
print *, " Validation Results"
print *, "===================================================="

d1_valid = .true.
print *, sum_d1(idx,1)

do idx = 1, ntests
  ! Use first loop ordering as reference
  d1_ref = sum_d1(idx, 1)

  ! Check all other orderings against reference
  do i = 2, 5
    d1_diff = abs(sum_d1(idx, i) - d1_ref)


    if (d1_diff > tol) then
      d1_valid = .false.
      write(*,'(A,I5,A,I1,A,ES12.3)') " FAIL: Nz=", nz_values(idx), &
        " d1 ordering ", i, " diff = ", d1_diff
    end if
  end do
end do

if (d1_valid) then
  print *, "✓ Validation PASSED! All loop orderings agree within tolerance."
else
  print *, "✗ Validation FAILED! See differences above."
end if
print *, "===================================================="
end block
contains

  subroutine reset_state(nx,ny,nz,u,unew,c1,b1,d1)
    integer, intent(in) :: nx, ny, nz
    real(real64), intent(in)    :: u(nx,ny,nz)
    real(real64), intent(inout) :: unew(nx,ny,nz), c1(nx,ny,nz)
    real(real64), intent(inout) :: b1(nx,ny), d1(nx,ny)
    integer :: ii, jj, kk

    do concurrent (ii=1:nx, jj=1:ny, kk=1:nz)
       unew(ii,jj,kk) = u(ii,jj,kk)
       c1(ii,jj,kk)   = 0.0_real64
       if (kk == 1) then
          b1(ii,jj) = 0.0_real64
          d1(ii,jj) = 0.0_real64
       end if
    end do
  end subroutine reset_state

end program loop_order_sweep_do_concurrent_tridiag_reuse
