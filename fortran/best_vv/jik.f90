module aa
implicit none 

contains 

pure subroutine tridiag_forward_sweep(i, j, nz, dt, a3d, h3d, u, unew, &
                                  ray, b1, d1, c1)
   
   ! Arguments
   integer, intent(in) :: i, j, nz
   real(kind=8), intent(in) :: dt
   real(kind=8), intent(in) :: a3d(:,:,:)
   real(kind=8), intent(in) :: h3d(:,:,:)
   real(kind=8), intent(in) :: u(:,:,:)
   real(kind=8), intent(inout) :: unew(:,:,:)
   real(kind=8), intent(in) :: ray(:,:)
   real(kind=8), intent(inout) :: b1(:,:)
   real(kind=8), intent(inout) :: d1(:,:)
   real(kind=8), intent(inout) :: c1(:,:,:)
   
   ! Local variables
   integer :: k
   real(kind=8) :: a_loc, h_loc, ray_loc, b1_loc, d1_loc
   real(kind=8) :: b_denom, u_prev, u_loc
   real(kind=8), parameter :: one = 1.0d0
   
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
   
end subroutine tridiag_forward_sweep

end module aa

  program loop_order_sweep_do_concurrent_tridiag_reuse
  use iso_fortran_env, only: real64
  use aa
  use omp_lib, only: omp_get_wtime
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

  integer, parameter :: n_iters = 200
  integer :: ii
  real(real64) :: dt, t1, t2
  real(real64) :: total_t1, total_t2
  real(real64) :: timings(ntests,1)
  real(real64) :: flops(ntests)
  real(real64) :: val
  real(real64) :: a_loc, h_loc, ray_loc, b1_loc, d1_loc, b_denom, u_prev, u_loc
  real(real64), parameter :: one = 1.0_real64, zero = 0.0_real64
  logical :: validate 

  validate = .false.
  !----------------------------------
  ! Command-line argument parsing
  !----------------------------------
  allocate(sum_b1(ntests,1), sum_d1(ntests,1))
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

  total_t1 = omp_get_wtime()
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

     flops(idx) = real(n_iters) * real((nz - 2),real64) * real(ny,real64) * real(nx,real64) * real(15,real64)/real(1e9,real64)
     print *, flops(idx)

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
     ! 5. j -> i -> vertical
     !    concurrent over j,i, then serial k
     !=========================================================
     call reset_state(nx,ny,nz,u,unew,c1,b1,d1)
     t1 = omp_get_wtime()
     do ii = 1, n_iters
     !!$omp target teams distribute parallel do collapse(2)
     !!$omp target teams loop collapse(2)
     !do j = 1, ny 
     !do i = 1, nx
     do concurrent (j=1:ny, i=1:nx, mask(i,j)>=0.0_real64) local(b1)

     call tridiag_forward_sweep(i, j, nz, dt, a3d, h3d, u, unew, &
                           ray, b1, d1, c1)
!        do k = 2, nz-1
!           a_loc   = a3d(i,j,k)
!           h_loc   = h3d(i,j,k)
!           ray_loc = ray(i,j)
!           b1_loc  = b1(i,j)
!           d1_loc  = d1(i,j)
!
!           c1(i,j,k) = dt * a_loc * b1_loc
!
!           b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
!           b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
!           d1_loc  = b_denom * b1_loc
!
!           u_prev  = unew(i,j,k-1)
!           u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
!           unew(i,j,k) = u_loc
!
!           b1(i,j) = b1_loc
!           d1(i,j) = d1_loc
!        end do
     !end do
     end do
     end do
     t2 = omp_get_wtime()
     timings(idx,1) = t2 - t1
     print '(A,F10.4," s")', " j->i->vertical elapsed:        ", timings(idx,1)
     print '(A,F10.4," GFLOP/s")', " j->i->vertical flop rate:        ", flops(idx)/timings(idx,1)

     !------------------------------------------
     ! Free device and host memory
     !------------------------------------------
     !$omp target exit data map(delete: a3d,h3d,u,unew,c1,ray,mask,b1,d1)

     deallocate(a3d,h3d,u,unew,c1,ray,mask,b1,d1)
  end do

  total_t2 = omp_get_wtime()

  print '(A , F10.4, " s")', "Total time elapsed: ", total_t2 - total_t1



  print *
  print *, "===================================================="
  print *, " Benchmark complete."
  print *, "===================================================="

  print *
  print *, "j->i->vertical"
  do idx = 1, ntests
     write(*,'(I5,1(",",F12.6))') nz_values(idx), timings(idx,1)
  end do
print *, "You are confident in your resulst, we didn't check for correctness, just S P E E D"

contains

!pure subroutine tridiag_forward_sweep(i, j, nz, dt, a3d, h3d, u, unew, &
!                                  ray, b1, d1, c1)
!   
!   ! Arguments
!   integer, intent(in) :: i, j, nz
!   real(kind=8), intent(in) :: dt
!   real(kind=8), intent(in) :: a3d(:,:,:)
!   real(kind=8), intent(in) :: h3d(:,:,:)
!   real(kind=8), intent(in) :: u(:,:,:)
!   real(kind=8), intent(inout) :: unew(:,:,:)
!   real(kind=8), intent(in) :: ray(:,:)
!   real(kind=8), intent(inout) :: b1(:,:)
!   real(kind=8), intent(inout) :: d1(:,:)
!   real(kind=8), intent(inout) :: c1(:,:,:)
!   
!   ! Local variables
!   integer :: k
!   real(kind=8) :: a_loc, h_loc, ray_loc, b1_loc, d1_loc
!   real(kind=8) :: b_denom, u_prev, u_loc
!   real(kind=8), parameter :: one = 1.0d0
!   
!   do k = 2, nz-1
!      a_loc   = a3d(i,j,k)
!      h_loc   = h3d(i,j,k)
!      ray_loc = ray(i,j)
!      b1_loc  = b1(i,j)
!      d1_loc  = d1(i,j)
!      c1(i,j,k) = dt * a_loc * b1_loc
!      b_denom = h_loc + dt * (ray_loc + a_loc * d1_loc)
!      b1_loc  = one / (b_denom + dt * a3d(i,j,k+1))
!      d1_loc  = b_denom * b1_loc
!      u_prev  = unew(i,j,k-1)
!      u_loc   = (h_loc * u(i,j,k) + dt * a_loc * u_prev) * b1_loc
!      unew(i,j,k) = u_loc
!      b1(i,j) = b1_loc
!      d1(i,j) = d1_loc
!   end do
!   
!end subroutine tridiag_forward_sweep

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
