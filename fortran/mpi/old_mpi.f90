program mpi_daxpy_test
  use mpi
  implicit none
  
  integer :: ierr, rank, nprocs
  integer :: i, iter, n
  integer, parameter :: niter = 10000
  real(8) :: a
  real(8), allocatable :: x(:), y(:)
  real(8) :: t1, t2
  
  ! Initialize MPI
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
  
  ! Set problem size
  n = 1000000
  a = 2.5d0
  
  ! Allocate arrays
  allocate(x(n), y(n))
  
  ! Initialize arrays
  do concurrent (i = 1:n)
    x(i) = dble(i)
    y(i) = dble(i) * 0.5d0
  end do
  
  ! Barrier to synchronize before timing
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  
  if (rank == 0) then
    print *, 'Starting DAXPY iterations on ', nprocs, ' ranks'
  end if
  
  t1 = MPI_Wtime()
  
  ! Perform niter DAXPY operations
  do iter = 1, niter
    do concurrent (i = 1:n)
      y(i) = a * x(i) + y(i)
    end do
  end do
  
  t2 = MPI_Wtime()
  print *, y(1)
  
  ! Print timing from rank 0
  if (rank == 0) then
    print '(A,F10.6)', 'Time (s) = ', t2 - t1
    print '(A,ES12.4)', 'First y value = ', y(1)
  end if
  
  ! Clean up
  deallocate(x, y)
  
  call MPI_Finalize(ierr)
  
end program mpi_daxpy_test
