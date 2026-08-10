program main 
  use iso_fortran_env, only: real64
use omp_lib, only: omp_get_wtime
implicit none 
real, allocatable :: a(:), b(:), c(:)
real, parameter :: alpha = 2.0
real(real64) :: t1, t2, timings
integer :: m 
integer :: i
integer :: narg
character(len=32) :: arg

m = 100000
  narg = command_argument_count()
if (narg >= 1) then
   call get_command_argument(1, arg)
   read(arg, *) m
end if
print *, " m is ", m
allocate(a(m),b(m), c(m))
t1 = omp_get_wtime()
do concurrent(i=1:m)
a(i) = 13.0
b(i) = 67.0
c(i) = 0.0
end do
do concurrent(i=1:m)
c(i) = alpha * a(i) + b(i)
end do
t2 = omp_get_wtime()
timings = t2 - t1
print '(A,F10.4," s")', " elapsed:       ", timings
print *, c(4)

end program main
