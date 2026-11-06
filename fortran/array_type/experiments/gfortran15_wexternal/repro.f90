module m
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none
   private
   integer, parameter :: dp = real64
   public :: for_each, visit_i
   abstract interface
      subroutine visit_i(name, a)
         import :: dp
         character(len=*), intent(in) :: name
         real(dp), intent(inout) :: a(:)
      end subroutine visit_i
   end interface
contains
   subroutine for_each(visit)
      procedure(visit_i) :: visit
      real(dp), target, save :: buf(6) = 1.0_dp
      call visit("buf", buf)
   end subroutine for_each
end module m
program p
   use m
   implicit none
   call for_each(show)
contains
   subroutine show(name, a)
      character(len=*), intent(in) :: name
      real(8), intent(inout) :: a(:)
      print *, name, size(a), sum(a)
   end subroutine show
end program p
