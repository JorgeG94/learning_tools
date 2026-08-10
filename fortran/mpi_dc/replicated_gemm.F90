! mpi_gemm_omp_target.F90
program mpi_gemm_omp_target
  use, intrinsic :: iso_fortran_env, only: real64, int32
  use mpi_f08
  use omp_lib
  implicit none

  integer               :: ierr, nranks, myrank
  type(MPI_Comm)        :: comm
  integer(int32)        :: n
  real(real64), allocatable :: A(:,:), B(:,:), C(:,:)
  real(real64)          :: t0, t1, secs, gflops
  integer               :: arg_len, iostat
  character(len=256)    :: arg
  real(real64) :: acc
  integer :: i,j,k

  call MPI_Init()
  comm = MPI_COMM_WORLD
  call MPI_Comm_size(comm, nranks)
  call MPI_Comm_rank(comm, myrank)
  call omp_set_default_device(myrank)
  !$acc set device_num(myrank)

  ! --- Parse N from argv(1), default 2048 ---
  n = 2048_int32
  if (command_argument_count() >= 1) then
     call get_command_argument(1, arg, length=arg_len, status=iostat)
     if (iostat == 0) read(arg(1:arg_len), *, iostat=iostat) n
     if (iostat /= 0) then
        if (myrank == 0) print *, 'Could not parse N, using default 2048'
        n = 2048_int32
     end if
  end if

  if (myrank == 0) then
     print '(a,i0)', 'Ranks      : ', nranks
     print '(a,i0)', 'Matrix size: ', n
  end if

  allocate(A(n,n), B(n,n), C(n,n))
  !$omp target enter data map(alloc: A,B,C)
  ! --- Initialize on device (optional, keeps data on GPU right away) ---
#ifdef USE_DC
  do concurrent (j=1:n, i=1:n)
     A(i,j) = 1.0d0 + real(i-1, real64) * 1.0d-6
     B(i,j) = 2.0d0 + real(j-1, real64) * 1.0d-6
     C(i,j) = 0.0d0
  end do
#else
  !$omp target teams distribute parallel do collapse(2) 
  do j=1,n
  do i=1,n
     A(i,j) = 1.0d0 + real(i-1, real64) * 1.0d-6
     B(i,j) = 2.0d0 + real(j-1, real64) * 1.0d-6
     C(i,j) = 0.0d0
  end do
  end do
#endif
  call MPI_Barrier(comm)  ! sync before timing
  t0 = MPI_Wtime()

  ! --- Naive DGEMM offloaded to GPU: C = A * B (alpha=1, beta=0) ---
#ifdef USE_DC
  do concurrent (j=1:n, i=1:n)
    acc = 0.0d0
    do k = 1, n
      acc = acc + A(i,k)*B(k,j)
    end do
    C(i,j) = acc
  end do 
#else
    !$omp target teams loop collapse(2)
    do j = 1, n
      do i = 1, n
        acc = 0.0d0
        do k = 1, n
          acc = acc + A(i,k) * B(k,j)
        end do
        C(i,j) = acc
      end do
    end do
#endif

  call MPI_Barrier(comm)
  t1 = MPI_Wtime()
  secs = t1 - t0
  !$omp target exit data map(release: A,B,C)

  ! --- Simple check (sum of a few elements) so the compiler can't elide work ---
  if (mod(myrank, max(1,nranks)) == 0) then
     print '(a,es14.6)', 'Rank 0 checksum: ', sum(C(1:min(n,8),1:min(n,8)))
  end if

  ! --- Report per-rank performance ---
  gflops = (2.0d0 * real(n,real64)**3) / (secs * 1.0d9)
  print '(a,i0,a,f8.3,a,f10.2)', 'Rank ', myrank, ': time = ', secs, ' s,  GFLOP/s = ', gflops

  call MPI_Finalize()
end program mpi_gemm_omp_target

