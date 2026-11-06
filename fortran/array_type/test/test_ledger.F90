!! Ledger accounting and the devarray lifecycle (CPU and GPU builds).
program test_ledger
   use, intrinsic :: iso_fortran_env, only: dp => real64, int32, int64, error_unit
   use mem_ledger
   use mem_ledger_backend, only: mem_num_slots
   use devarray, only: dp_1d_t, dp_2d_t, dp_3d_t, i32_2d_t
   implicit none

   type(dp_1d_t) :: a1
   type(dp_2d_t) :: a2, b2
   type(dp_3d_t) :: a3
   type(i32_2d_t) :: mask
   type(mem_tag_t) :: ta, tb
   type(mem_tag_stats_t) :: st
   real(dp), allocatable :: src(:, :)
   integer :: fails, k, nslots
   integer(int64) :: b1, bb2, b3, bm

   fails = 0

   ! ---- tags
   ta = mem_tag("alpha")
   tb = mem_tag("beta")
   call check(mem_tag("alpha") == ta, "mem_tag is idempotent")
   call check(mem_find_tag("beta") == tb, "mem_find_tag finds an existing tag")
   call check(.not. (ta == tb), "distinct tags compare unequal")
   call check(mem_tag_name(tb) == "beta", "mem_tag_name")

   ! ---- create, bytes, kinds
   call a1%create("a1", ta, [10])
   call a2%create("a2", ta, ub=[11, 21], lb=[0, 0], fill=1.0_dp)
   call a3%create("a3", tb, [4, 5, 6])
   call mask%create("mask", tb, [3, 3], fill=7_int32)
   b1 = 10*8
   bb2 = 12*22*8
   b3 = 4*5*6*8
   bm = 9*4
   call check(a1%bytes() == b1 .and. a2%bytes() == bb2 .and. a3%bytes() == b3, "real(dp) bytes")
   call check(mask%bytes() == bm .and. all(mask%data == 7), "integer(int32) kind, bytes and fill")
   call check(lbound(a2%data, 1) == 0 .and. ubound(a2%data, 2) == 21, "lower bounds")
   call check(all(a2%data == 1.0_dp), "fill on create")
   call check(mem_host_bytes(ta) == b1 + bb2 .and. mem_host_bytes(tb) == b3 + bm, "per-tag host bytes")
   call check(mem_host_bytes() == b1 + bb2 + b3 + bm, "total host bytes")
   st = mem_tag_stats(ta)
   call check(st%name == "alpha" .and. st%live == 2 .and. st%host == b1 + bb2, "mem_tag_stats")

   ! ---- create_like / create_from keep bounds
   call b2%create_like("b2 like a2", ta, a2)
   call check(all(lbound(b2%data) == [0, 0]) .and. all(ubound(b2%data) == [11, 21]), &
              "create_like(object) copies bounds")
   call b2%destroy()
   allocate (src(-1:2, 3:4))
   src = 5.0_dp
   call b2%create_like("b2 like src", ta, src)
   call check(all(lbound(b2%data) == [-1, 3]) .and. all(ubound(b2%data) == [2, 4]), &
              "create_like(allocatable array) copies bounds")
   call b2%destroy()
   call b2%create_from("b2 from src", ta, src)
   call check(all(lbound(b2%data) == [-1, 3]) .and. all(b2%data == 5.0_dp), "create_from copies bounds and values")
   call b2%destroy()
   call a2%assert_bounds([11, 21], lb=[0, 0])
   call check(.true., "assert_bounds passes on matching bounds")

   ! ---- fill, device mapping
   call a1%fill(3.0_dp)
   call check(all(a1%data == 3.0_dp), "fill on the host when not mapped")
   call a2%to_device()
   call a2%update_host()
   call check(a2%is_on_device(), "to_device sets the on-device state")
#ifdef DEVARRAY_DEVICE
   call check(mem_device_bytes(ta) == bb2, "device bytes tracked under OpenACC")
#else
   call check(mem_device_bytes() == 0, "device bytes stay zero without OpenACC")
#endif

   ! alloc_device honours `fill` from create, instead of leaving stale memory
   call a1%destroy()
   call a1%create("a1 filled", ta, [10], fill=4.0_dp)
   a1%data = -1.0_dp
   call a1%alloc_device()
   call a1%update_host()
#ifdef DEVARRAY_DEVICE
   call check(all(a1%data == 4.0_dp), "alloc_device puts the create fill on the device")
   call a1%fill(9.0_dp)
   call check(all(a1%data == 4.0_dp), "fill on a mapped array leaves the host copy alone")
   call a1%update_host()
   call check(all(a1%data == 9.0_dp), "fill on a mapped array writes the device")
#else
   call check(all(a1%data == 4.0_dp), "alloc_device without OpenACC fills the host copy")
#endif
   call a1%destroy()

   ! ---- destroy, peaks, idempotence
   call check(mem_host_bytes(ta) == bb2, "destroy releases host bytes")
   call check(mem_host_peak(ta) >= b1 + bb2, "peak survives destroy")
   call a1%destroy()
   call check(mem_live_count() == 3, "destroy on an empty object is a no-op")

   ! ---- the registry reuses slots
   nslots = mem_num_slots()
   do k = 1, 1000
      call b2%create("churn", ta, [4, 4])
      call b2%destroy()
   end do
   call check(mem_num_slots() <= nslots + 1, "1000 create/destroy cycles do not grow the registry")

   call mem_report(title="test_ledger (objects)", objects=.true.)

#ifndef DEVARRAY_NO_FINAL
   ! ---- finalizer safety net: none of these call destroy explicitly
   call forget_local(tb)
   call check(mem_live_count(tb) == 2 .and. mem_device_bytes(tb) == 0, &
              "final: mapped local retired at end of subroutine")
   call forget_array(tb)
   call check(mem_live_count(tb) == 2, "final: local array of wrappers retired (elemental final)")
   call forget_allocatable(tb)
   call check(mem_live_count(tb) == 2, "final: deallocate retires an allocatable object")
   call forget_component(tb)
   call check(mem_live_count(tb) == 2, "final: component of a local parent retired")
   ! The actual must not be SAVEd (main-program variables are implicitly
   ! SAVE): nvfortran 26.5 skips intent(out) finalization for SAVEd actuals.
   call intent_out_case(tb)
   call check(mem_live_count(tb) == 2, "final: intent(out) dummy retired on entry")
#endif

   ! ---- small utilities
   call check(mem_human_bytes(1536_int64) == "1.50 KiB" .and. mem_human_bytes(0_int64) == "0 B", &
              "mem_human_bytes is unpadded")

   ! ---- teardown and reset
   call a2%destroy()
   call a3%destroy()
   call mask%destroy()
   call check(mem_check_leaks() == 0, "no leaks at the end")
   call check(mem_host_bytes() == 0 .and. mem_device_bytes() == 0, "ledger back to zero")
   call mem_reset()
   ta = mem_tag("alpha")
   st = mem_tag_stats(ta)
   call check(st%live == 0 .and. st%host_peak == 0 .and. mem_live_count() == 0, "mem_reset starts over")

   if (fails > 0) then
      write (error_unit, '(i0, a)') fails, ' check(s) FAILED'
      error stop 1
   end if
   print '(a)', 'test_ledger: all checks passed'

contains

   subroutine forget_local(tag)
      type(mem_tag_t), intent(in) :: tag
      type(dp_2d_t) :: x
      call x%create("local x", tag, [100, 100], fill=0.0_dp)
      call x%to_device()
   end subroutine forget_local

   subroutine forget_array(tag)
      type(mem_tag_t), intent(in) :: tag
      type(dp_1d_t) :: xs(3)
      integer :: i
      do i = 1, 3
         call xs(i)%create("xs element", tag, [8])
      end do
   end subroutine forget_array

   subroutine forget_allocatable(tag)
      type(mem_tag_t), intent(in) :: tag
      type(dp_2d_t), allocatable :: x
      allocate (x)
      call x%create("allocatable x", tag, [10, 10])
      deallocate (x)
      call check(mem_live_count(tag) == 2, "final: (inside) retired before scope end")
   end subroutine forget_allocatable

   subroutine forget_component(tag)
      type(mem_tag_t), intent(in) :: tag
      type :: holder_t
         type(dp_2d_t) :: w
      end type holder_t
      type(holder_t) :: h
      call h%w%create("holder%w", tag, [10, 10])
   end subroutine forget_component

   subroutine intent_out_case(tag)
      type(mem_tag_t), intent(in) :: tag
      type(dp_1d_t) :: y
      call y%create("intent(out) y", tag, [5])
      call reset_out(y)
      call check(.not. y%is_created() .and. mem_live_count(tag) == 2, &
                 "final: (inside) intent(out) retired before return")
   end subroutine intent_out_case

   subroutine reset_out(x)
      type(dp_1d_t), intent(out) :: x
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
