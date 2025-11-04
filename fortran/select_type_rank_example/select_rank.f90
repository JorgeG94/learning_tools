module describe_mod
    implicit none
contains
    subroutine describe(x)
        implicit none
        real, intent(in) :: x(..)
        select rank(x)
        rank (0)
            print *, "Scalar:", x
        rank (1)
            print *, "1D array of size", size(x)
        rank (2)
            print *, "2D array of shape", shape(x)
        rank default
            print *, "Higher-rank array of rank", rank(x)
        end select
    end subroutine describe
end module describe_mod


program demo
    use describe_mod
    implicit none
    real :: a = 1.23
    real :: v(3) = [1.0, 2.0, 3.0]
    real :: m(2,2) = reshape([1,2,3,4],[2,2])

    call describe(a)
    call describe(v)
    call describe(m)
end program demo

