program sequential_pi
use pic_types, only: dp
use pic_timers
use pic_string_utils, only: to_string
  !! so we're using my library for some custom cool things like timers and stuff!
implicit none 

integer, parameter :: num_steps = 1024 * 1024 * 1024
real(dp) :: pi, step, sum 
print *, ""
print *, "Starting sequential PI program"
pi = 0.0_dp
step = 0.0_dp
sum = 0.0_dp

!! this is how you cast things to a certain variable in Fortran
step = 1.0_dp / real(num_steps,dp) 

block 

  integer :: i
  real(dp) :: x, time
  type(pic_timer) :: pi_timer
  x = 0.0_dp
  call pi_timer%start()
  do i = 1, num_steps 
    x = (i + 0.5_dp) * step
    sum = sum + 4.0_dp / (1.0_dp + x*x)
  end do 
  call pi_timer%end()

  pi = step * sum
  print *, " Pi = " // to_string(pi) // " using "// to_string(num_steps) // " steps and took " // to_string(pi_timer%get_elapsed_time()) // " seconds"

end block 





end program sequential_pi
