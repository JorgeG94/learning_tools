!! Collective operations over the ledger: per-tag update, fill, checksum,
!! visit, validate (CPU and GPU builds).
program test_collective
   use, intrinsic :: iso_fortran_env, only: dp => real64, int32, int64, error_unit
   use, intrinsic :: ieee_arithmetic, only: ieee_value, ieee_quiet_nan
   use mem_ledger
   use mem_collective
   use devarray, only: dp_1d_t, dp_2d_t, dp_3d_t, i32_1d_t
   implicit none

   type(dp_2d_t) :: a
   type(dp_3d_t) :: b
   type(dp_1d_t) :: c
   type(i32_1d_t) :: m
   type(mem_checksum_t) :: cs, cs2, cs_all
   type(mem_tag_t) :: ts, to
   integer :: fails, nvisit
   real(dp) :: visited_sum

   fails = 0
   ts = mem_tag("state")
   to = mem_tag("other")

   call a%create("a", ts, ub=[4, 5], lb=[0, 0], fill=1.0_dp)   ! 30 values
   call b%create("b", ts, [2, 3, 4], fill=2.0_dp)                ! 24 values
   call c%create("c", to, [10], fill=5.0_dp)                     ! 10 values
   call m%create("m", ts, [6], fill=3_int32)                     ! integer: not in checksums

   ! ---- checksum covers the real(dp) arrays of the tag
   cs = mem_checksum(ts)
   call check(cs%nobj == 2 .and. cs%nelem == 54, "checksum covers exactly the tagged real(dp) arrays")
   call check(cs%sum == 30*1.0_dp + 24*2.0_dp, "checksum sum")
   cs_all = mem_checksum()
   call check(cs_all%nobj == 3 .and. cs_all%sum == cs%sum + 50.0_dp, "no tag means every array")

   a%data(3, 2) = nearest(1.0_dp, 2.0_dp)
   cs2 = mem_checksum(ts)
   call check(cs2%hash /= cs%hash, "hash sees a 1-ulp change")
   a%data(3, 2) = 1.0_dp
   cs2 = mem_checksum(ts)
   call check(cs2%hash == cs%hash, "hash identical after restore")

   ! ---- inspection syncs from the device by default
   call a%to_device()
   call b%to_device()
   call c%to_device()
   call m%to_device()
   a%data = -1.0_dp
   b%data = -1.0_dp
   c%data = -1.0_dp
   cs = mem_checksum(ts, sync=.false.)
   call check(cs%sum == -54.0_dp, "sync=.false. reads the host copies as they are")
   cs = mem_checksum(ts)
#ifdef _OPENACC
   call check(cs%sum == 78.0_dp .and. all(a%data == 1.0_dp), "default sync pulls the device copies first")
   call check(all(c%data == -1.0_dp), "sync only touches the requested tag")
#else
   call check(cs%sum == -54.0_dp, "without OpenACC the host copy is the only copy")
#endif

   ! ---- explicit transfers
   a%data = 7.0_dp
   b%data = 7.0_dp
   call mem_update_device(ts)
   a%data = 0.0_dp
   b%data = 0.0_dp
   call mem_update_host(ts)
#ifdef _OPENACC
   call check(all(a%data == 7.0_dp) .and. all(b%data == 7.0_dp), "update_device/update_host round trip")
#endif

   ! ---- fill where the data lives
   call mem_fill(0.5_dp, ts)
   call mem_fill(11_int32, ts)
#ifdef _OPENACC
   call check(all(a%data == 7.0_dp), "mem_fill on mapped arrays leaves the host copy alone")
   call mem_update_host(ts)
#endif
   call check(all(a%data == 0.5_dp) .and. all(b%data == 0.5_dp), "mem_fill(real) fills the tag's real(dp) arrays")
   call check(all(m%data == 11), "mem_fill(int32) fills the tag's integer arrays")

   ! ---- visitors: internal procedures, read and write
   nvisit = 0
   visited_sum = 0.0_dp
   call mem_for_each_r64(accumulate, ts)
   call check(nvisit == 2 .and. visited_sum == 27.0_dp, "for_each visits the tagged real(dp) arrays")
   call mem_for_each_r64(double_it, ts, sync_back=.true.)
   a%data = 0.0_dp
   b%data = 0.0_dp
   call mem_update_host(ts)
#ifdef _OPENACC
   call check(all(a%data == 1.0_dp) .and. all(b%data == 1.0_dp), "for_each sync_back pushes changes to the device")
#else
   call check(all(a%data == 0.0_dp), "without OpenACC update_host leaves the host copy alone")
#endif

   ! ---- validation and NaN hunting
   call mem_validate()
   call check(.true., "mem_validate passes on intact arrays")
   b%data(2, 2, 2) = ieee_value(1.0_dp, ieee_quiet_nan)
   call mem_report_values(ts, sync=.false.)

   ! ---- destroyed arrays drop out
   call b%destroy()
   cs = mem_checksum(ts, sync=.false.)
   call check(cs%nobj == 1 .and. cs%nelem == 30, "destroyed arrays are no longer visited")

   call a%destroy()
   call c%destroy()
   call m%destroy()
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
