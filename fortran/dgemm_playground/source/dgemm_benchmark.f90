program dgemm_random_benchmark
  use lib_gemm
  use omp_lib
  implicit none

  integer :: m, k, n, reps, reps_of_reps
  integer :: i, j, l, rep, outer_rep
  real(rk), allocatable :: A(:,:), B(:,:), C(:,:)
  real(rk) :: t_start, t_end, flops, gflops
  integer :: seed_size
  integer, allocatable :: seed(:)
  integer :: ios, n_args
  character(len=32) :: arg
  integer :: thread_id, nthreads

  ! Get number of arguments
  n_args = command_argument_count()

  if (n_args < 5) then
     print *, "Usage: ./dgemm_random_benchmark m k n reps reps_of_reps"
     stop 1
  end if

  ! Read command-line arguments
  call get_command_argument(1, arg)
  read(arg, *, iostat=ios) m
  if (ios /= 0) stop "Error reading m"

  call get_command_argument(2, arg)
  read(arg, *, iostat=ios) k
  if (ios /= 0) stop "Error reading k"

  call get_command_argument(3, arg)
  read(arg, *, iostat=ios) n
  if (ios /= 0) stop "Error reading n"

  call get_command_argument(4, arg)
  read(arg, *, iostat=ios) reps
  if (ios /= 0) stop "Error reading reps"

  call get_command_argument(5, arg)
  read(arg, *, iostat=ios) reps_of_reps
  if (ios /= 0) stop "Error reading reps_of_reps"

  nthreads = omp_get_max_threads()
  print *, "Using ", nthreads, " threads"
  print *, "Matrix sizes: m=", m, " k=", k, " n=", n
  print *, "Reps: ", reps, " Outer reps: ", reps_of_reps

  ! Allocate matrices
  allocate(A(m,k), B(k,n), C(m,n))

  ! Initialize RNG
  call random_seed(size=seed_size)
  allocate(seed(seed_size))
  seed = 123456 + 37 * [(j, j=1,seed_size)]
  call random_seed(put=seed)

  ! Fill A and B with random numbers
  ! Fill A(m,k) with independent random numbers
do i = 1, m
  do j = 1, k
    call random_number(A(i,j))
  end do
end do

! Fill B(k,n) with independent random numbers
do i = 1, k
  do j = 1, n
    call random_number(B(i,j))
  end do
end do


  ! Main benchmarking loop
  do outer_rep = 1, reps_of_reps

     ! Zero C
     C = 0.0d0

     t_start = omp_get_wtime()

     call naive_omp_dgemm(A,B,C,m,n,k,reps)
     !call blocked_dgemm(A,B,C,m,n,k,reps)
     !call simd_dgemm(A,B,C,m,n,k,reps)
     !call blas_dgemm(A,B,C,m,n,k,reps)
    
     t_end = omp_get_wtime()

     flops = 2.0d0 * m * n * k * reps
     gflops = flops / (t_end - t_start) / 1.0d9

     print '(A, F12.6, A, F12.6, A, I4)', "Time (s): ", t_end - t_start, "  GFLOPS: ", gflops, "  Rep: ", outer_rep
  end do

end program dgemm_random_benchmark

