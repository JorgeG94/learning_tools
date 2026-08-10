program flux_demo
  use iso_fortran_env, only : real64
  use omp_lib,          only : omp_get_wtime
  implicit none

  integer, parameter :: nx = 8096, nz = 8096
  real(real64), allocatable :: u(:,:), h(:,:), hL(:,:), hR(:,:), uh(:,:), duhdu(:,:)
  real(real64) :: t1, t2, dt
  integer :: i, k, ii
  integer :: niters = 100

  ! Allocate fields
  allocate(u(nx,nz), h(nx,nz), hL(nx,nz), hR(nx,nz), uh(nx,nz), duhdu(nx,nz))

  !$omp target enter data map(alloc:u,h,hL,hR,uh,duhdu)
  ! Initialize
  do concurrent (k=1:nz, i=1:nx)
     u(i,k)  = 0.1_real64 * i
     h(i,k)  = 5.0_real64
     hL(i,k) = 4.5_real64
     hR(i,k) = 5.5_real64
  end do

  dt = 1.0e-2_real64

  t1 = omp_get_wtime()
  do ii = 1, niters
  !$omp target teams loop collapse(2)
  do k = 1, nz ; do i = 1, nx
     call flux_elem(u(i,k), h(i,k), hL(i,k), hR(i,k), uh(i,k), duhdu(i,k), dt)
  end do ; end do
  end do
  t2 = omp_get_wtime()
  !$omp target update from(duhdu, uh)

  print '(A,F10.6," s")', 'Elapsed: ', t2 - t1
  print '(A,1X,F10.6,1X,F10.6)', 'Sample uh,duhdu =', uh(10,10), duhdu(10,10)
  !$omp target exit data map(delete: u,h,hL,hR,uh,duhdu)

contains

  elemental subroutine flux_elem(u, h, hL, hR, uh, duhdu, dt)
    real(real64), intent(in)  :: u, h, hL, hR, dt
    real(real64), intent(out) :: uh, duhdu
    real(real64) :: dh, CFL, curv3

    if (u > 0.0_real64) then
       dh    = hL - hR
       curv3 = (hL + hR) - 2.0_real64*h
       CFL   = u * dt
       uh    = u * (hR + CFL * (0.5*dh + curv3*(CFL - 1.5_real64)))
    elseif (u < 0.0_real64) then
       dh    = hR - hL
       curv3 = (hR + hL) - 2.0_real64*h
       CFL   = -u * dt
       uh    = u * (hL + CFL * (0.5*dh + curv3*(CFL - 1.5_real64)))
    else
       uh = 0.0_real64
    end if

    duhdu = h * 0.1_real64  
  end subroutine flux_elem

end program flux_demo

