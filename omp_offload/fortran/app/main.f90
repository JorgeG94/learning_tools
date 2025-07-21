program main
  use pic_types, only: dp
  use omp_offloading, only: dscal, dgemm, fill
  implicit none
  real(dp), allocatable :: A(:,:), B(:,:), C(:,:)
  integer, parameter :: m = 512
  integer :: i,j 

  allocate(A(m,m),B(m,m), C(m,m)) 

  call fill(A, 1.0_dp)
  call fill(B, 1.0_dp)
  call fill(C, 0.0_dp)

  call dscal(A, 2.0_dp)

  call dgemm(A,B,C)


  do i = 1, 4
    do j = 1, 2
    print *, C(i,j)
    end do
  end do 
end program main
