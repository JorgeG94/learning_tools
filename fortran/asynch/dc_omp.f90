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
do concurrent (i=1:1500)
d(i) = 0.0
do concurrent (j=1:250,k=1:150)
a(i,j,k) = 1.0
b(i,j,k) = 1.0
c(i,j,k) = 1.0
end do 

end do



do concurrent (i=1:1500)

  do concurrent (j=1:250,k=1:150)
    A(i,j,k) = real(i*j*k)*A(i,j,k)
  end do 
  
  do concurrent (j=1:250,k=1:150)
    B(i,j,k) = real(i*j*k)*B(i,j,k)
  end do 
  
  do concurrent (j=1:250,k=1:150)
    C(i,j,k) = real(i*j*k)*C(i,j,k)
  end do


end do 

!!$omp target exit data map(from: d)

!print *, d(1)
end program main
