program naive_dgemm_omp
use pic_types
use pic_timers
use pic_string_utils, only: to_string
implicit none 

real(dp), allocatable :: A(:,:), B(:,:), C(:,:)
integer, parameter :: matrix_size = 1024
integer, parameter :: m = matrix_size
integer, parameter :: n = matrix_size
integer, parameter :: k = matrix_size

print *, ""
print *, "Beginning naive symmetric dgemm program with matrix size " // to_string(matrix_size) 
allocate(A(m,k), B(k,n), C(m,n))

A = 1.0_dp
B = 2.0_dp
C = 0.0_dp

block 
type(pic_timer) :: my_timer
integer :: i,j,l
integer :: num_threads
num_threads = omp_get_max_threads()

  call my_timer%start()
  do i = 1, m
    do j = 1, n
      do l = 1, k
        C(i,j) = C(i,j) + A(i,l) * B(l,j)
      end do
    end do
  end do 
  call my_timer%end()
  print *, " Loop order is i -> j -> l serial"
  call my_timer%print_time()

  C = 0.0_dp

  call my_timer%start()
  !$omp parallel do collapse(2) private(i,j,l) schedule(static)
  do i = 1, m
    do j = 1, n
      do l = 1, k
        C(i,j) = C(i,j) + A(i,l) * B(l,j)
      end do
    end do
  end do 
  !$omp end parallel do
  call my_timer%end()
  print *, " Loop order is i -> j -> l and using " // to_string(num_threads) // " threads"
  call my_timer%print_time()

  C = 0.0_dp

  call my_timer%start()
  !$omp parallel do collapse(2) private(i,j,l) schedule(static)
  do j = 1, n
    do i = 1, m
      do l = 1, k
        C(i,j) = C(i,j) + A(i,l) * B(l,j)
      end do
    end do
  end do
  !$omp end parallel do
  call my_timer%end()
  print *, " Loop order is j -> i -> l and using " // to_string(num_threads) // " threads"
  call my_timer%print_time()

end block

deallocate(A,B,C)

end program naive_dgemm_omp
