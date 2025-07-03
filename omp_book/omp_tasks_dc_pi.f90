program task_dc_pi
use pic_types
use pic_timers
implicit none 
type(pic_timer) :: my_timer
integer(int64), parameter :: num_steps = 1024 * 1024 * 1024
real(dp) :: step, pi, sum 


call my_timer%start()
step = 1.0_dp / real(num_steps,dp)
!$omp parallel 
!$omp single
sum = compute_sum(1_int64, num_steps, step)
!$omp end single
!$omp end parallel
pi = step * sum 
call my_timer%stop()


print *, "Pi = ", pi, " and took ", my_timer%get_elapsed_time(), " seconds "

contains 

function compute_sum(nstart, num_steps, step) result(sum)
  implicit none 
  integer(int64), parameter :: min_bulk = 1024 * 256
  integer(int64), intent(in) :: nstart 
  integer(int64), intent(in) :: num_steps
  real(dp), intent(in) :: step 
  real(dp) :: sum,x, sum_1, sum_2

  x = 0.0_dp
  sum = 0.0_dp
  sum_1 = 0.0_dp
  sum_2 = 0.0_dp

  if( (num_steps - nstart) < min_bulk) then 
    block 
      integer(default_int) :: i
      do i = nstart, num_steps
        x = (i +0.5_dp) * step
        sum = sum + 4.0_dp / (1.0_dp + x*x)
      end do
    end block 
  else 
    block 
      integer(int64) :: iblk 
      iblk = num_steps - nstart 
      !$omp task shared(sum_1)
      sum_1 = compute_sum(nstart, num_steps - iblk/2,step)
      !$omp end task  
      !$omp task shared(sum_2)
      sum_2 = compute_sum(num_steps - iblk/2,num_steps, step)
      !$omp end task
      !$omp taskwait 
      sum = sum_1 + sum_2
    end block 
  end if
end function compute_sum

end program task_dc_pi
