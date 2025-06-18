! this program is to test the correctness of do concurrent dgemm versus naive one
program dgemm_test
   use omp_lib
   implicit none

   integer, parameter :: rk = SELECTED_REAL_KIND(15, 307)
   integer :: m, k, n, rep, reps, reps_of_reps, outer_rep
   real(rk), allocatable :: A(:, :), B(:, :), C(:, :)
   real(rk), allocatable :: C_naive(:, :), C_dc(:, :)
   real(rk) :: t_start, t_end, flops, gflops
   real(rk) :: t_sanity_start, t_sanity_end
   real(rk) :: time_naive, time_dc
   integer :: i, j
   real(rk) :: tol
   logical :: passed
   integer :: seed_size, jseed
   integer, allocatable :: seed(:)
   integer :: ios, n_args
   character(len=32) :: arg

   ! Get number of arguments
   n_args = command_argument_count()

   if (n_args < 3) then
      print *, "Usage: ./dgemm_random_benchmark m k n"
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

   ! Allocate matrices
   allocate (A(m, k), B(k, n), C(m, n))
   allocate (C_naive(m, n), C_dc(m, n))

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

   C_naive = 0.0d0
   t_start = omp_get_wtime()
   call naive_omp_dgemm(A, B, C_naive, m, n, k)
   t_end = omp_get_wtime()
   time_naive = t_end - t_start

   C_dc = 0.0d0
   t_sanity_start = omp_get_wtime()
   call naive_omp_dgemm_offloaded(A, B, C_dc, m, n, k)
   t_sanity_end = omp_get_wtime()
   time_dc = t_sanity_end - t_sanity_start

   print *, "Time for naive ", time_naive, " s"
   print *, "Time for do concurrent ", time_dc, " s"

   tol = 1.0d-12
   passed = .true.
   do i = 1, m
      do j = 1, n
         if (abs(C_naive(i, j) - C_dc(i, j)) > tol) then
            print '(A, I4, A, I4, A, 3(ES24.16, 1X))', "Mismatch at (", i, ",", j, "): ", &
               C_naive(i, j), C_dc(i, j), C_naive(i, j) - C_dc(i, j)
            passed = .false.
         end if
      end do
   end do

   print *, "End sanity check"
   if (passed) then
      print *, "Passed! Now onto the meat of the taco"

   else
      print *, "Test failed."
   end if

contains

   subroutine naive_omp_dgemm(A, B, C, m, n, k)
      implicit none
      integer, intent(in) :: m, n, k
      real(rk), intent(in) :: A(m, k)
      real(rk), intent(in) :: B(k, n)
      real(rk), intent(inout) :: C(m, n)
      integer :: i, j, l
      do i = 1, m
         do j = 1, n
            do l = 1, k
               C(i, j) = C(i, j) + A(i, l)*B(l, j)
            end do
         end do
      end do
   end subroutine naive_omp_dgemm

   subroutine naive_omp_dgemm_offloaded(A, B, C, m, n, k)
      implicit none
      integer, intent(in) :: m, n, k
      real(rk), intent(in) :: A(m, k)
      real(rk), intent(in) :: B(k, n)
      real(rk), intent(inout) :: C(m, n)
      integer :: i, j, l
!$omp target teams distribute parallel do private(i,j,l)
      do i = 1, m
         do j = 1, n
            do l = 1, k
               C(i, j) = C(i, j) + A(i, l)*B(l, j)
            end do
         end do
      end do
!$omp end target teams distribute parallel do
   end subroutine naive_omp_dgemm_offloaded

end program dgemm_test

