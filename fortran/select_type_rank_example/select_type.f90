module shapes
   implicit none

   type :: shape
      real :: area
   end type shape

   type, extends(shape) :: circle
      real :: radius
   end type circle

   type, extends(shape) :: square
      real :: side
   end type square

contains
   subroutine print_shape(s)
      class(shape), intent(in) :: s
      select type(s)
      type is(circle)
         print *, "Circle with radius =", s%radius
      type is(square)
         print *, "Square with side =", s%side
      class default
         print *, "Unknown shape with area =", s%area
      end select
   end subroutine print_shape

end module shapes

program demo
    use shapes
    implicit none

    class(shape), allocatable :: s

    allocate(circle :: s)
    select type(s)
    type is(circle)
        s%radius = 5.0
    end select
    call print_shape(s)

    deallocate(s)
    allocate(square :: s)
    select type(s)
    type is(square)
        s%side = 3.0
    end select

    call print_shape(s)
end program demo

