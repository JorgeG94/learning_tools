program main
  use pic_types, only: dp, int64
  use pic_timers, only: pic_timer_type
  use omp_offloading, only: dscal, dgemm, fill
  use matrix_packaging, only: matrix_packer_type
  implicit none
  type(pic_timer_type) :: my_timer
  real(dp) :: time
  real(dp), allocatable :: A(:,:), B(:,:), C(:,:)
  type(matrix_packer_type) :: packer
  integer, parameter :: m = 2048
  integer, parameter :: m_count = 3
  integer :: i,j 
! this is a very simple program that uses openmp to do a couple of operations
! on the GPU naively. First, we allocate the arrays, all symmetirc matrices 
! we then initialize them to a certain number, I do something like a 
! dscal (multiply a matrix by a double) and then a naive dgemm. 
! the purpose of this program is to show how good can thins get once 
! you do things right. 
! Here we will just rely on omp target teams distribute parallel do map(tofrom...) 
! to move our data around whenever it is needed. This is stupid, do not do it. 
! what will happen is that once we encounter a target region we will mvoe things around 
! i.e. each fill uses the GPU. Once we get to fill(A,1.0_dp) it will copy A to the device, 
! fill it, and copy it back. Same for B and C
! then at the dscal it will do the same, and subsequently for the dgemm. The result 
! when you profile this using `nsys profile --stats=true ./naive` is that you see
! Time (%)  Total Time (ns)  Count  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)           Operation
! --------  ---------------  -----  ---------  ---------  --------  --------  -----------  ----------------------------
!     56.8        125267484     13  9635960.3  6324375.0   5403865  22741699    6752185.4  [CUDA memcpy Device-to-Host]
!     43.2         95390278     17  5611192.8  6579352.0      1056   9852371    3438092.7  [CUDA memcpy Host-to-Device]
! a staggering 13 D->H and 17 H->D. This is absolute crap. Now look at the normal.f90 

  allocate(A(m,m),B(m,m), C(m,m)) 
  call my_timer%start()
  call fill(A, 1.0_dp)
  call fill(B, 1.0_dp)
  call fill(C, 0.0_dp)
  call my_timer%stop()
  time = my_timer%get_elapsed_time()
  print *, "Time to fill arrays was ", time, " seconds"

  call my_timer%start()
  call dscal(A, 2.0_dp)
  call my_timer%stop()
  time = my_timer%get_elapsed_time()
  print *, "Time for dscal ", time, " seconds"
  
  call my_timer%start()
  do i = 1, m_count
  call dgemm(A,B,C)
  end do 
  call my_timer%stop()
  time = my_timer%get_elapsed_time()
  print *, "Time for ", m_count, " dgemms ", time, " seconds"


  block 

  real(dp) :: total_flops
  real(dp) :: flop_rate
  integer(int64) :: m64 
  m64 = m 

  total_flops = real(2_int64 * m64 * m64 * m64 * m_count,dp)
  print *, "total flops ", total_flops
  print *, "time is ", time

  flop_rate = real(total_flops,dp) / time / 1e9_dp
  print *, "flop rate is ", flop_rate, " GFLOP/s"

  end block 


  do i = 1, 4
    do j = 1, 2
    print *, C(i,j)
    end do
  end do 
end program main
