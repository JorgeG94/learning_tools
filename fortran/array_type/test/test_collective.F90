!! Collective operations over the ledger: per-tag update, checksum, visit.
program test_collective
   use, intrinsic :: iso_fortran_env, only: dp => real64, int64, error_unit
   use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
   use mem_ledger
   use mem_collective
   use devarray, only: dp1_t, dp2_t, dp3_t
   implicit none

   type(dp2_t) :: a
   type(dp3_t) :: b
   type(dp1_t) :: c
   type(mem_checksum_t) :: cs, cs2, cs_all
   integer :: ts, to, fails, nvisit
   real(dp) :: visited_sum

   fails = 0
   ts = mem_register_tag("state")
   to = mem_register_tag("other")

   call a%create("a", [4, 5], tag=ts, lb=[0, 0], fill=1.0_dp)   ! 5*6 = 30 values
   call b%create("b", [2, 3, 4], tag=ts, fill=2.0_dp)            ! 24 values
   call c%create("c", [10], tag=to, fill=5.0_dp)                 ! 10 values

   ! ---- checksum: host data, per tag and global
   cs = mem_checksum(ts)
   call check(cs%nobj == 2 .and. cs%nelem == 54, "checksum covers exactly the tagged arrays")
   call check(cs%sum == 30*1.0_dp + 24*2.0_dp, "checksum sum")
   cs_all = mem_checksum()
   call check(cs_all%nobj == 3 .and. cs_all%sum == cs%sum + 50.0_dp, "untagged call covers everything")

   ! ---- hash detects a single changed bit, and comes back when restored
   a%data(3, 2) = nearest(1.0_dp, 2.0_dp)
   cs2 = mem_checksum(ts)
   call check(cs2%hash /= cs%hash, "hash sees a 1-ulp change")
   a%data(3, 2) = 1.0_dp
   cs2 = mem_checksum(ts)
   call check(cs2%hash == cs%hash, "hash identical after restore")

   ! ---- update: device copy is the truth, host gets clobbered, update restores
   call a%to_device()
   call b%to_device()
   call c%to_device()
   a%data = -1.0_dp
   b%data = -1.0_dp
   c%data = -1.0_dp
   call mem_update_host(ts)
#ifdef _OPENACC
   call check(all(a%data == 1.0_dp) .and. all(b%data == 2.0_dp), "update_host(tag) restored tagged arrays")
   call check(all(c%data == -1.0_dp), "update_host(tag) left other tags alone")
#else
   call check(all(a%data == -1.0_dp), "update_host is a no-op without OpenACC")
#endif

   ! ---- update_device then update_host round trip
   a%data = 7.0_dp
   b%data = 7.0_dp
   call mem_update_device(ts)
   a%data = 0.0_dp
   b%data = 0.0_dp
   call mem_update_host(ts)
#ifdef _OPENACC
   call check(all(a%data == 7.0_dp) .and. all(b%data == 7.0_dp), "update_device/update_host round trip")
#endif

   ! ---- for_each with an internal procedure capturing local state
   a%data = 1.0_dp
   b%data = 1.0_dp
   nvisit = 0
   visited_sum = 0.0_dp
   call mem_for_each_r64(accumulate, ts)
   call check(nvisit == 2 .and. visited_sum == 54.0_dp, "for_each visits tagged arrays with full extent")

   ! ---- for_each can also write: scale everything in the tag
   call mem_for_each_r64(double_it, ts)
   call check(all(a%data == 2.0_dp) .and. all(b%data == 2.0_dp) .and. all(c%data == -1.0_dp), &
              "for_each can modify through the view")

   ! ---- NaN hunting
   b%data(2, 2, 2) = ieee_value(1.0_dp, ieee_quiet_nan)
   call mem_report_values(ts)

   ! ---- destroyed arrays drop out
   call b%destroy()
   cs = mem_checksum(ts)
   call check(cs%nobj == 1 .and. cs%nelem == 30, "destroyed arrays are no longer visited")

   call a%destroy()
   call c%destroy()
   call check(mem_check_leaks() == 0, "no leaks")

   if (fails > 0) then
      write (error_unit, '(i0, a)') fails, ' check(s) FAILED'
      error stop 1
   end if
   print '(a)', 'test_collective: all checks passed'

contains

   subroutine accumulate(name, x)
      character(len=*), intent(in) :: name
      real(dp), intent(inout) :: x(:)
      nvisit = nvisit + 1
      visited_sum = visited_sum + sum(x)
      if (len(name) == 0) error stop
   end subroutine accumulate

   subroutine double_it(name, x)
      character(len=*), intent(in) :: name
      real(dp), intent(inout) :: x(:)
      x = 2.0_dp*x
      if (len(name) == 0) error stop
   end subroutine double_it

   subroutine check(ok, what)
      logical, intent(in) :: ok
      character(len=*), intent(in) :: what
      if (ok) then
         print '(a, a)', '  ok   ', what
      else
         print '(a, a)', '  FAIL ', what
         fails = fails + 1
      end if
   end subroutine check

end program test_collective
