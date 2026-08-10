
program main
  use iso_fortran_env, only: real64
  use omp_lib, only: omp_get_wtime
  use density_mod
  implicit none

  real(real64), allocatable :: T(:,:,:), S(:,:,:), p(:,:,:), rho(:,:,:)
  integer :: i,j,k
  integer :: nx, ny, nz
  real(real64) :: t1, t2

  nx = 2024; ny = 2024; nz = 200
  allocate(T(nx,ny,nz), S(nx,ny,nz), p(nx,ny,nz))

  ! Initialize dummy data
  do concurrent (k=1:nz, j=1:ny, i=1:nx)
    T(i,j,k) = 10.0
    S(i,j,k) = 35.0
    p(i,j,k) = 1.0e5
  end do

  t1 = omp_get_wtime()
  rho = calculate_density(T, S, p)
  t2 = omp_get_wtime()

  print *, rho(1,1,1)
  print '(A,F10.4," s")', 'elapsed: ', t2 - t1

end program main

