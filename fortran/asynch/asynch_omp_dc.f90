program main 
implicit none 

integer :: i
integer :: j,k 
integer :: jj,kk
integer :: jjj,kkk
real, allocatable :: A(:,:,:)
real, allocatable :: B(:,:,:)
real, allocatable :: C(:,:,:)
real, allocatable :: d(:)


allocate(A(1500,250,150))
allocate(B(1500,250,150))
allocate(C(1500,250,150))
allocate(d(1500))


!$omp target enter data map(alloc: A,B,C,d)

!$omp target nowait 

do concurrent (i=1:1500)
  d(i) = 0.0
end do

!$omp end target 

!$omp target nowait 
do concurrent (i=1:1500,j=1:250,k=1:150)
      a(i,j,k) = 1.0
      b(i,j,k) = 1.0
      c(i,j,k) = 1.0
end do
!$omp end target 

!$omp taskwait
print *, "here?"

!$omp target nowait 
do concurrent (i=1:1500,j=1:250,k=1:150)
      a(i,j,k) = real(i*j*k) * a(i,j,k)
end do
!$omp end target 

!$omp target nowait 
do concurrent (i=1:1500,j=1:250,k=1:150)
      b(i,j,k) = real(i*j*k) * b(i,j,k)
end do
!$omp end target 

!$omp target nowait 
do concurrent (i=1:1500,j=1:250,k=1:150)
      c(i,j,k) = real(i*j*k) * c(i,j,k)
end do
!$omp end target 

print *, "there?"
!$omp taskwait

!$omp target exit data map(delete:A,B,C,d)

end program main
