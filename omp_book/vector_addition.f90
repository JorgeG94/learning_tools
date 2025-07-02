program vector_addition_simple
use pic_types, only: dp
implicit none 

  !integer, parameter :: dp = SELECTED_REAL_KIND(15, 307)
    !! pro tip, use select real kind for better control for declaring types
  integer, parameter :: n = 400

  real(dp) :: A(n), B(n), C(n)
print *, ""
print *, "Starting the vector addition program"
    !! now you can declare real(dp) and ensure consistence between compilers, never do real(8) 

  ! in Fortran you can initialize entire arrays by doing this, no need to loop over them 
  A = 1.0_dp
    !! note the _dp, this tells the compiler that our number is of the dp type we've defined above
  B = 2.0_dp
  C = 0.0_dp

  ! option using arrays, for completeness
!  block 
!    integer :: i 
!    do i = 1, n 
!      A(i) = 1.0_dp
!      B(i) = 2.0_dp
!      C(i) = 0.0_dp
!    end do
!  end block 

  block 
    !! Fortran has blocks, think of them as what happens between  {} in C/C++, variables declared there are local to the scope of the block. 
    !! no need to declare everything at the beginning unless it has a program wide life
    integer :: i
    do i = 1, n 
      C(i) = A(i) + B(i)
    end do 
    ! additional print just for fun
    do i = 1, 5
      print *, C(i)
    end do
  end block 
print *, "***************************"
!! pro tip, in Fortran you don't need to loop 
  block 
    integer :: i 
    C = 0.0_dp
    C = A + B
    do i = 1, 5
      print *, C(i)
    end do 
  end block


end program vector_addition_simple


