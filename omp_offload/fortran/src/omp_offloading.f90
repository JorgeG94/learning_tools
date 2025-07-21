module matrix_packaging
use pic_types, only: dp 
implicit none 
  private 

  public :: matrix_packer_type
  public :: allocate_C
  type :: matrix_packer_type 
    real(dp), allocatable :: C(:,:)
  end type matrix_packer_type

contains 

  subroutine allocate_C(packer, m, n)
    type(matrix_packer_type), intent(inout) :: packer 
    integer, intent(in) :: m, n 

    allocate(packer%C(m,n))

  end subroutine allocate_C

end module matrix_packaging

module omp_offloading
use pic_types, only: dp
use matrix_packaging, only: matrix_packer_type
  implicit none
  private
  public :: dscal
  public :: dgemm
  public :: fill
  interface dgemm 
    module procedure :: dgemm_struct
    module procedure :: dgemm_normal
  end interface dgemm
contains

subroutine fill(A, number)
  real(dp), allocatable, intent(inout) :: A(:,:)
  real(dp), intent(in) :: number 
  integer :: i,j
  integer :: m, n 

  m = size(A,1)
  n = m 
 !$omp target teams distribute parallel do collapse(2) 
  do i = 1, m 
    do j = 1, n 
      A(i,j) = number
    end do 
  end do 
!$omp end target teams distribute parallel do

end subroutine fill
 
 subroutine dscal(A, alpha) 
  real(dp), allocatable, intent(inout) :: A(:,:) 
  real(dp), intent(in) :: alpha
  integer :: i, j 
  integer :: m, n
  
  m = size(A,1)
  n = m
 !$omp target teams distribute parallel do collapse(2) 
  do i = 1, m 
    do j = 1, n 
      A(i,j) = alpha * A(i,j)
    end do 
  end do 
!$omp end target teams distribute parallel do

 end subroutine dscal

 subroutine dgemm_struct(A, B, packer)
 real(dp), intent(in) :: A(:,:), B(:,:)
 type(matrix_packer_type), intent(inout) :: packer
 integer :: i,j,l 
 integer :: m,n,k 

! only symmetrical matrices
 m = size(packer%C,1)
 n = m 
 k = m


 !$omp target teams distribute parallel do collapse(2) 
 do i = 1, m 
  do j = 1, n 
    do l = 1, k 
      packer%C(i,j) = packer%C(i,j) + A(i,l) * B(l,j)
    end do 
  end do 
end do 
!$omp end target teams distribute parallel do

 end subroutine dgemm_struct

 subroutine dgemm_normal(A, B, C)
 real(dp), intent(in) :: A(:,:), B(:,:)
 real(dp), intent(out) :: C(:,:)
 integer :: i,j,l 
 integer :: m,n,k 

! only symmetrical matrices
 m = size(C,1)
 n = m 
 k = m


 !$omp target teams distribute parallel do collapse(2) 
 do i = 1, m 
  do j = 1, n 
    do l = 1, k 
      C(i,j) = C(i,j) + A(i,l) * B(l,j)
    end do 
  end do 
end do 
!$omp end target teams distribute parallel do

 end subroutine dgemm_normal
end module omp_offloading
