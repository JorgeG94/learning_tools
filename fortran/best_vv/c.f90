program main
  use iso_fortran_env, only: real64
  use omp_lib, only: omp_get_wtime
  implicit none

  real, allocatable :: a(:,:), b(:,:), c(:,:)
  real, parameter :: alpha = 2.0
  real(real64) :: t1, t2, timings
  integer :: nx, ny
  integer :: i, j
  integer :: narg
  character(len=32) :: arg

  ! Default dimensions
  nx = 1000
  ny = 1000

  narg = command_argument_count()
  if (narg >= 2) then
     call get_command_argument(1, arg); read(arg, *) nx
     call get_command_argument(2, arg); read(arg, *) ny
  end if

  print *, " grid size: ", nx, "x", ny

  allocate(a(nx, ny), b(nx, ny), c(nx, ny))

  t1 = omp_get_wtime()

  ! Initialize
  do concurrent(i=1:nx, j=1:ny)
     a(i,j) = 13.0
     b(i,j) = 67.0
     c(i,j) = 0.0
  end do

  ! 2D SAXPY: c = alpha * a + b
  do concurrent(i=1:nx, j=1:ny)
     c(i,j) = alpha * a(i,j) + b(i,j)
  end do

  t2 = omp_get_wtime()
  timings = t2 - t1

  print '(A,F10.4," s")', " elapsed:       ", timings
  print *, " sample value: ", c(4,4)

end program main

