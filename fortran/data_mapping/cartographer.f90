program main 
use iso_fortran_env, only: real64
use omp_lib
implicit none 

real, allocatable :: A(:,:,:), B(:,:,:)
real :: alpha 
real(real64) :: t0, t1
integer :: nz 
integer :: i,j,k
integer :: ni, nj 
nz = 1000
nj = nz
ni = nz 
alpha = 73.0

allocate(A(nz,nz,nz), B(nz,nz,nz))

!$omp target enter data map(to:i,j,k,ni,nj,nz)
!$omp target enter data map(alloc: A,B)

t0 = omp_get_wtime()
do concurrent (k=1:nz, j=1:nj, i=1:ni)
A(i,j,k) = 1.0
B(i,j,k) = 2.0
end do 
t1 = omp_get_wtime()
print '(A,F12.6)', 'k-loop wall time (s): ', t1 - t0

t0 = omp_get_wtime()
do k = 2, nz 
  do concurrent (j=1:nj, i=1:ni)
  A(i,j,k) = A(i,j,k-1) + alpha * B(i,j,k)
  end do 
end do 

t1 = omp_get_wtime()
print '(A,F12.6)', 'k-loop wall time (s): ', t1 - t0
!$omp target exit data map(release:i,j,k,ni,nj,nz)
!$omp target exit data map(from: A)
!$omp target exit data map(delete: A,B)


print *, A(4,5,6)

deallocate(A,B)

end program main
