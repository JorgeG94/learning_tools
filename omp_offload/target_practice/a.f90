module target_practice 
implicit none 

contains 

subroutine alloc_vector_sp(vector)
real, allocatable, intent(in) :: vector(:)

!$omp target enter data map(alloc: vector)

end subroutine alloc_vector_sp

subroutine free_vector_sp(vector)
real, allocatable, intent(in) :: vector(:)

!$omp target exit data map(delete: vector)

end subroutine free_vector_sp

subroutine target_fill(vector, val)
real, allocatable, intent(inout) :: vector(:)
real, intent(in) :: val
integer :: length, i 

length = size(vector,1)

!$omp target teams loop private(i)
do i = 1, length 
 vector(i) = val
end do 
!$omp end target teams loop

end subroutine target_fill

end module 

program main 
use target_practice 
implicit none 
real, allocatable :: A(:), B(:), C(:)
integer, parameter :: len = 100000

allocate(A(len), B(len), C(len))

call alloc_vector_sp(A)
call alloc_vector_sp(B)
call alloc_vector_sp(C)

call target_fill(A, 1.0)
call target_fill(B, 2.0)
call target_fill(C, 0.0)

block 
integer :: i 

!$omp target teams loop 
do i=1, size(A,1)
C(i) = A(i) + 2.0*B(i)
end do 
!$omp end target teams loop

!$omp target update from(C)
print *, C(1)
end block

call free_vector_sp(A)
call free_vector_sp(B)
call free_vector_sp(C)

end program main 
