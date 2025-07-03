program sequential_pi
use pic_types, only: dp
use pic_timers
use pic_string_utils, only: to_string
  !! so we're using my library for some custom cool things like timers and stuff!
implicit none 

integer, parameter :: num_steps = 1024 * 1024 * 1024
real(dp) :: pi, step, sum 
type(pic_timer) :: pi_timer
integer :: numthreads
real(dp) :: x
print *, ""
print *, "Starting the loop based parallelism pi caclualtion program"
pi = 0.0_dp
step = 0.0_dp
sum = 0.0_dp
x = 0.0_dp

!! this is how you cast things to a certain variable in Fortran
step = 1.0_dp / real(num_steps,dp) 

call pi_timer%start()

!$omp parallel
block 
  integer :: i


  !$omp single nowait 
    numthreads = omp_get_num_threads() 
  !$omp end single
    !! single implies barrier, the code above is only executed by a single thread
    !! numthreads is _shared_ since it was declared outside the parallel block 
  !$omp do private(x) reduction(+:sum)
  do i = 1, num_steps
    x = (i + 0.5_dp) * step
    sum = sum + 4.0_dp / (1.0_dp + x*x)
  end do 
  !$omp end do


end block 
!$omp end parallel

call pi_timer%stop()
pi = step * sum
print *, " Pi = " // to_string(pi) // " using "// to_string(num_steps) // " steps and took " // to_string(pi_timer%get_elapsed_time()) // " seconds"





end program sequential_pi
