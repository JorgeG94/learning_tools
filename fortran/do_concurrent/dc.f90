! benchmark for do concurrent
program dgemm_test
   use omp_lib
   implicit none

   integer, parameter :: rk = SELECTED_REAL_KIND(15, 307)
   integer :: m, k, n, rep, reps, reps_of_reps, outer_rep
   real(rk), allocatable :: A(:, :), B(:, :), C(:, :)
   real(rk), allocatable :: C_naive(:, :), C_dc(:, :)
   real(rk) :: t_start, t_end, flops, gflops
   real(rk) :: t_sanity_start, t_sanity_end
   integer :: i, j
   real(rk) :: tol
   logical :: passed
   integer :: seed_size, jseed
   integer, allocatable :: seed(:)
   integer :: ios, n_args
   character(len=32) :: arg

   ! Get number of arguments
   n_args = command_argument_count()

   if (n_args < 5) then
      print *, "Usage: ./dgemm_random_benchmark m k n reps reps_of_reps"
      stop 1
   end if

   ! Read command-line arguments
   call get_command_argument(1, arg)
   read (arg, *, iostat=ios) m
   if (ios /= 0) stop "Error reading m"

   call get_command_argument(2, arg)
   read (arg, *, iostat=ios) k
   if (ios /= 0) stop "Error reading k"

   call get_command_argument(3, arg)
   read (arg, *, iostat=ios) n
   if (ios /= 0) stop "Error reading n"

   call get_command_argument(4, arg)
   read (arg, *, iostat=ios) reps
   if (ios /= 0) stop "Error reading reps"

   call get_command_argument(5, arg)
   read (arg, *, iostat=ios) reps_of_reps
   if (ios /= 0) stop "Error reading reps_of_reps"

   ! Allocate matrices
   allocate (A(m, k), B(k, n), C(m, n))

   ! Seed RNG
   call random_seed(size=seed_size)
   allocate (seed(seed_size))
   seed = 123456 + 37*[(jseed, jseed=1, seed_size)]
   call random_seed(put=seed)

   ! Fill A and B with random numbers
   ! Fill A(m,k) with independent random numbers
   do i = 1, m
      do j = 1, k
         A(i, j) = 1.0D0
         !call random_number(A(i,j))
      end do
   end do

! Fill B(k,n) with independent random numbers
   do i = 1, k
      do j = 1, n
         B(i, j) = 1.0D0
         !call random_number(B(i,j))
      end do
   end do

   print *, "Begin sanity check "

   do outer_rep = 1, reps_of_reps

      ! Zero C
      t_start = omp_get_wtime()
      do rep = 1, reps
         C = 0.0d0
         call do_concurrent_dgemm(A, B, C, m, n, k)
      end do

      t_end = omp_get_wtime()

      flops = 2.0d0*m*n*k*reps
      gflops = flops/(t_end - t_start)/1.0d9

      print '(A, F12.6, A, F12.6, A, I4)', "Time (s): ", t_end - t_start, "  GFLOPS: ", gflops, "  Rep: ", outer_rep
   end do

contains

   subroutine do_concurrent_dgemm(A, B, C, m, n, k)
      implicit none
      integer, intent(in) :: m, n, k
      real(rk), intent(in) :: A(m, k)
      real(rk), intent(in) :: B(k, n)
      real(rk), intent(inout) :: C(m, n)
      real(rk) :: tmp
      integer :: i, j, l, rep
      !do concurrent (i = 1:m, j = 1:n)
      !C(i,j) = 0.0d0
      !do l = 1, k
      !  C(i,j) = C(i,j) + A(i,l) * B(l,j)
      !end do
      !end do
      do concurrent(i=1:m, j=1:n)
         tmp = 0.0d0
         do l = 1, k
            tmp = tmp + A(i, l)*B(l, j)
         end do
         C(i, j) = tmp
      end do

   end subroutine do_concurrent_dgemm

end program dgemm_test

