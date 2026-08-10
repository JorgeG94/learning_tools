program main
  use iso_fortran_env, only: real64
  use omp_lib, only: omp_get_wtime
  implicit none

  real, allocatable :: a(:,:,:), b(:,:,:), c(:,:,:)
  real, parameter :: alpha = 2.0
  real(real64) :: t1, t2, timings
  integer :: nx, ny, nz
  integer :: i, j, k
  integer :: narg
  character(len=32) :: arg

  ! Default dimensions
  nx = 100
  ny = 100
  nz = 100

  narg = command_argument_count()
  if (narg >= 3) then
     call get_command_argument(1, arg); read(arg, *) nx
     call get_command_argument(2, arg); read(arg, *) ny
     call get_command_argument(3, arg); read(arg, *) nz
  end if

  print *, " grid size: ", nx, "x", ny, "x", nz

  allocate(a(nx, ny, nz), b(nx, ny, nz), c(nx, ny, nz))

  t1 = omp_get_wtime()

  ! Initialize
  do concurrent(i=1:nx, j=1:ny, k=1:nz)
     a(i,j,k) = 13.0
     b(i,j,k) = 67.0
     c(i,j,k) = 0.0
  end do

  ! 3D SAXPY: c = alpha * a + b
  do concurrent(i=1:nx, j=1:ny, k=1:nz)
     c(i,j,k) = alpha * a(i,j,k) + b(i,j,k)
  end do

  t2 = omp_get_wtime()
  timings = t2 - t1

  print '(A,F10.4," s")', " elapsed:       ", timings
  print *, " sample value: ", c(4,4,4)

end program main

