!! Misuse cases that must stop the program with a named, readable error.
!! Usage: test_must_fail <case>. test/must_fail.sh runs every case and checks
!! both the nonzero exit and the message.
program test_must_fail
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use mem_ledger
   use mem_collective, only: mem_checksum_t, mem_checksum
   use devarray, only: dp_1d_t, dp_2d_t
   implicit none
   character(len=32) :: which
   type(mem_tag_t) :: t, unset
   type(dp_1d_t) :: x, xs(2), ys(2)
   type(dp_2d_t) :: a, b
   type(mem_checksum_t) :: cs

   call get_command_argument(1, which)
   t = mem_tag("tests")

   select case (trim(which))
   case ('copy_scalar')
      call a%create("original", t, [4, 4])
      b = a
   case ('copy_array')
      call xs(1)%create("xs(1)", t, [4])
      call xs(2)%create("xs(2)", t, [4])
      ys = xs
   case ('realloc_method')
      call x%create("x", t, [4])
      x%data = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp]   ! realloc-on-assignment
      call x%destroy()
   case ('realloc_collective')
      call x%create("x", t, [4])
      x%data = [1.0_dp, 2.0_dp, 3.0_dp, 4.0_dp, 5.0_dp, 6.0_dp]
      cs = mem_checksum(t)
   case ('unset_tag')
      call x%create("x", unset, [4])
   case ('unknown_tag')
      t = mem_find_tag("swe_sate")
   case ('stale_tag')
      call mem_reset()
      call x%create("x", t, [4])
   case ('long_tag')
      t = mem_tag(repeat('x', MEM_TAG_NAME_LEN + 1))
   case ('bounds')
      call a%create("a", t, ub=[4, 4], lb=[0, 0])
      call a%assert_bounds([4, 4])
   case ('double_map')
      call x%create("x", t, [4])
      call x%to_device()
      call x%to_device()
   case ('reset_live')
      call x%create("x", t, [4])
      call mem_reset()
   case default
      print '(a)', 'unknown case '//trim(which)
      stop 0      ! exit 0 so must_fail.sh reports it
   end select
   print '(a)', 'case '//trim(which)//' was NOT stopped'
end program test_must_fail
