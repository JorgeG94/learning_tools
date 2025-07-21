program main
  use pic_types, only: dp, int64
  use pic_timers, only: pic_timer_type
  use omp_offloading, only: dscal, dgemm, fill
  use matrix_packaging, only: matrix_packer_type, allocate_C
  implicit none
  type(pic_timer_type) :: my_timer
  real(dp) :: time
  real(dp), allocatable :: A(:,:), B(:,:), C(:,:)
  type(matrix_packer_type) :: packer
  integer, parameter :: m = 2048
  integer, parameter :: m_count = 3
  integer :: i,j 

  allocate(A(m,m),B(m,m)) 
  ! allocate the matrix in the struct
  call allocate_C(packer, m, m)

! there's six H->D copies, A,B,C, 1.0_dp, 1.0_dp, and 0.0_dp 
! there's one D->H copy, C
 !$omp target data map(tofrom: packer, packer%C) map(to: A, B)
  call my_timer%start()
  call fill(A, 1.0_dp)
  call fill(B, 1.0_dp)
  call fill(packer%C, 0.0_dp)
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
  call dgemm(A,B,packer)
  end do 
  call my_timer%stop()
  time = my_timer%get_elapsed_time()
  print *, "Time for ", m_count, " dgemms ", time, " seconds"

  !$omp end target data

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
    print *, packer%C(i,j)
    end do
  end do 
end program main
