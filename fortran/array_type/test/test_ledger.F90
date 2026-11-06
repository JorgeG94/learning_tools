!! CPU tests of the ledger accounting through dp1/dp2/dp3_t.
program test_ledger
   use, intrinsic :: iso_fortran_env, only: dp => real64, int64, error_unit
   use mem_ledger
   use devarray, only: dp1_t, dp2_t, dp3_t
   implicit none

   type(dp1_t) :: a1
   type(dp2_t) :: a2
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

   call a2%destroy()

#ifndef DEVARRAY_NO_FINAL
   ! Finalizer safety net: none of these call destroy explicitly.
   call forget_local(tb)
   call check(mem_live_count(tb) == 1 .and. mem_device_bytes(tb) == 0, &
              "final: mapped local retired at end of subroutine")
   call forget_allocatable(tb)
   call check(mem_live_count(tb) == 1, "final: deallocate retires an allocatable object")
   call forget_component(tb)
   call check(mem_live_count(tb) == 1, "final: component of a local parent retired")
   ! The actual must not be SAVEd (main-program variables are implicitly
   ! SAVE): nvfortran 26.5 skips intent(out) finalization for SAVEd actuals.
   call intent_out_case(tb)
   call check(mem_live_count(tb) == 1, "final: intent(out) dummy retired on entry")
#endif

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

   subroutine forget_local(tag)
      integer, intent(in) :: tag
      type(dp2_t) :: x
      call x%create("local x", [100, 100], tag=tag, fill=0.0_dp)
      call x%to_device()
   end subroutine forget_local

   subroutine forget_allocatable(tag)
      integer, intent(in) :: tag
      type(dp2_t), allocatable :: x
      allocate (x)
      call x%create("allocatable x", [10, 10], tag=tag)
      deallocate (x)
      call check(mem_live_count(tag) == 1, "final: (inside) retired before scope end")
   end subroutine forget_allocatable

   subroutine forget_component(tag)
      integer, intent(in) :: tag
      type :: holder_t
         type(dp2_t) :: w
      end type holder_t
      type(holder_t) :: h
      call h%w%create("holder%w", [10, 10], tag=tag)
   end subroutine forget_component

   subroutine intent_out_case(tag)
      integer, intent(in) :: tag
      type(dp1_t) :: y
      call y%create("intent(out) y", [5], tag=tag)
      call reset_out(y)
      call check(.not. y%is_created() .and. mem_live_count(tag) == 1, &
                 "final: (inside) intent(out) retired before return")
   end subroutine intent_out_case

   subroutine reset_out(x)
      type(dp1_t), intent(out) :: x
      if (x%is_created()) continue
   end subroutine reset_out

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
