program sequential_pi
use pic_types, only: dp
use pic_timers
use pic_string_utils, only: to_string
use omp_lib
  !! so we're using my library for some custom cool things like timers and stuff!
implicit none 

integer, parameter :: num_steps = 1024 * 1024 * 1024
real(dp) :: pi, step, sum 
type(pic_timer) :: pi_timer
integer :: numthreads
print *, ""
print *, "Starting SPMD parallelized PI program "
pi = 0.0_dp
step = 0.0_dp
sum = 0.0_dp

!! this is how you cast things to a certain variable in Fortran
step = 1.0_dp / real(num_steps,dp) 

call pi_timer%start()

!$omp parallel
block 
  integer :: i, id
  real(dp) :: x, partial_sum

  id = omp_get_thread_num()
  x = 0.0_dp
  partial_sum = 0.0_dp

  !$omp single 
    numthreads = omp_get_num_threads() 
  !$omp end single
    !! single implies barrier, the code above is only executed by a single thread
    !! numthreads is _shared_ since it was declared outside the parallel block 

  do i = id, num_steps, numthreads
    x = (i + 0.5_dp) * step
    partial_sum = partial_sum + 4.0_dp / (1.0_dp + x*x)
  end do 

  !$omp critical 
  sum = sum + partial_sum
  !$omp end critical
    !! critical ensures that only one thread at a time writes to sum
    !! this seems to cause some floating point issues, as the answers is a bit different between serial, this one, and parallel do

end block 
!$omp end parallel

call pi_timer%stop()
pi = step * sum
print *, " Pi = " // to_string(pi) // " using "// to_string(num_steps) // " steps and took " // to_string(pi_timer%get_elapsed_time()) // " seconds"





end program sequential_pi
