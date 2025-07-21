module omp_offloading
use pic_types, only: dp
  implicit none
  private
  public :: dscal
  public :: dgemm
  public :: fill
contains

subroutine fill(A, number)
  real(dp), allocatable, intent(inout) :: A(:,:)
  real(dp), intent(in) :: number 
  integer :: i,j
  integer :: m, n 

  m = size(A,1)
  n = m 

  do i = 1, m 
    do j = 1, n 
      A(i,j) = number
    end do 
  end do 

end subroutine fill
 
 subroutine dscal(A, alpha) 
  real(dp), allocatable, intent(inout) :: A(:,:) 
  real(dp), intent(in) :: alpha
  integer :: i, j 
  integer :: m, n
  
  m = size(A,1)
  n = m
  do i = 1, m 
    do j = 1, n 
      A(i,j) = alpha * A(i,j)
    end do 
  end do 

 end subroutine dscal

 subroutine dgemm(A, B, C)
 real(dp), intent(in) :: A(:,:), B(:,:)
 real(dp), intent(inout) :: C(:,:)
 integer :: i,j,l 
 integer :: m,n,k 

! only symmetrical matrices
 m = size(C,1)
 n = m 
 k = m


 do i = 1, m 
  do j = 1, n 
    do l = 1, k 
      C(i,j) = C(i,j) + A(i,l) * B(l,j)
    end do 
  end do 
end do 

 end subroutine dgemm
end module omp_offloading
