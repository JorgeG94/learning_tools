!! CPU tests of the ledger accounting through dp1/dp2/dp3_t.
program test_ledger
   use, intrinsic :: iso_fortran_env, only: dp => real64, int64, error_unit
   use mem_ledger
   use devarray, only: dp1_t, dp2_t, dp3_t
   implicit none

   type(dp1_t) :: a1
   type(dp2_t) :: a2, copy2
   type(dp3_t) :: a3
   integer :: ta, tb, fails
   integer(int64) :: b1, b2, b3

   fails = 0

   ta = mem_register_tag("alpha")
   tb = mem_register_tag("beta")
   call check(mem_register_tag("alpha") == ta, "register_tag is idempotent")
   call check(ta /= tb, "distinct tags")

   call a1%create("a1", [10], tag=ta)
   call a2%create("a2", [11, 21], tag=ta, lb=[0, 0], fill=1.0_dp)
   call a3%create("a3", [4, 5, 6], tag=tb)
   b1 = 10*8
   b2 = 12*22*8
   b3 = 4*5*6*8

   call check(a1%bytes() == b1, "dp1 bytes")
   call check(a2%bytes() == b2, "dp2 bytes with lower bounds")
   call check(lbound(a2%data, 1) == 0 .and. ubound(a2%data, 2) == 21, "dp2 bounds")
   call check(all(a2%data == 1.0_dp), "fill")
   call check(mem_host_bytes(ta) == b1 + b2, "tag alpha host bytes")
   call check(mem_host_bytes(tb) == b3, "tag beta host bytes")
   call check(mem_host_bytes() == b1 + b2 + b3, "total host bytes")
   call check(mem_live_count() == 3, "live count")

   call a2%to_device()
   call a2%update_host()
#ifdef _OPENACC
   call check(mem_device_bytes(ta) == b2, "device bytes tracked under OpenACC")
#else
   call check(mem_device_bytes() == 0, "device bytes stay zero without OpenACC")
#endif
   call check(a2%is_on_device(), "state is HOST_DEVICE")

   call a1%destroy()
   call check(mem_host_bytes(ta) == b2, "destroy releases host bytes")
   call check(mem_host_peak(ta) == b1 + b2, "peak survives destroy")
   call a1%destroy()   ! idempotent on an empty object
   call check(mem_live_count() == 2, "second destroy is a no-op")

   ! Copy detection: a copy shares the ledger id; once the original is
   ! destroyed the copy's id is dead, so its destroy would be fatal.
   copy2 = a2
   call check(copy2%ledger_id() == a2%ledger_id(), "intrinsic copy duplicates the id")
   call a2%destroy()
   call check(.not. mem_is_live(copy2%ledger_id()), "copy's id is dead after original destroy")
   call check(mem_object_name(copy2%ledger_id()) == "a2", "dead id still resolves to its name")
   deallocate (copy2%data)   ! drop the copy's storage without touching the ledger

   call check(mem_check_leaks() == 1, "one leak before last destroy")
   call a3%destroy()
   call check(mem_check_leaks() == 0, "no leaks at the end")
   call check(mem_host_bytes() == 0 .and. mem_device_bytes() == 0, "ledger back to zero")

   call mem_report(title="test_ledger end")
   if (fails > 0) then
      write (error_unit, '(i0, a)') fails, ' check(s) FAILED'
      error stop 1
   end if
   print '(a)', 'test_ledger: all checks passed'

contains

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

end program test_ledger
