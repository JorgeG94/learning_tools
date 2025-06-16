program monte_carlo_pi_openmp
  use omp_lib
  implicit none

  integer, parameter :: n_points = 1000000000
  integer :: i, count_in, tid, nthreads, j
  real(8) :: x, y, pi_estimate
  real(8) :: t_start, t_end
  integer, allocatable :: seed(:,:)
  integer :: seed_size, my_seed

  count_in = 0

  ! Get OpenMP thread info
  nthreads = omp_get_max_threads()
  print *, 'Using ', nthreads, ' threads'

  ! Get size of random seed
  call random_seed(size = seed_size)
  allocate(seed(seed_size, nthreads))

  ! Precompute seeds for each thread
  do tid = 0, nthreads-1
     my_seed = int(123456 + tid * 997)
     seed(:, tid+1) = my_seed + 37 * [(j, j=1,seed_size)]
  end do

  t_start = omp_get_wtime()

!$omp parallel default(none) private(i, x, y, tid) shared(seed, seed_size) reduction(+:count_in)
  tid = omp_get_thread_num()

  call random_seed(put=seed(:,tid+1))

  !$omp do
  do i = 1, n_points
     call random_number(x)
     call random_number(y)
     if (x*x + y*y <= 1.0d0) count_in = count_in + 1
  end do
  !$omp end do
!$omp end parallel

  pi_estimate = 4.0d0 * count_in / n_points

  t_end = omp_get_wtime()
  print *, "Estimated Pi:", pi_estimate
  print *, "Elapsed time (s):", t_end - t_start

end program monte_carlo_pi_openmp

