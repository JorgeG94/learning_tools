module help
use pic_types, only: dp 
use matrix_packaging, only: matrix_packer_type

public :: hello
contains 

subroutine hello(packer)
 type(matrix_packer_type), intent(inout) :: packer
 integer :: m, i, j 

 m = size(packer%C,1)

!$omp target teams distribute parallel do
 do i = 1, m 
  do j = 1, m 
    packer%C(i,j) = 2.0_dp * packer%C(i,j)
  end do 
end do
!$omp end target teams distribute parallel do

end subroutine hello 

end module help
