program mpi_saxpy_3d
  use, intrinsic :: iso_fortran_env, only : real64
  use mpi_f08
  use omp_lib
  use nvtx
  implicit none

  integer :: ierr, nranks, myrank
  integer :: dims, base, remainder
  integer :: nk_local, kstart, kend
  integer, allocatable :: sendcounts(:), displs(:)
  real(real64), allocatable :: a_global(:,:,:)
  real(real64), allocatable :: a(:,:,:), b(:,:,:), c(:,:,:)
  real(real64) :: alpha = 2.0_real64
  real(real64) :: local_sum, global_sum
  real(real64) :: t1, t2
  real(real64) :: t1_rank, t2_rank
  integer :: i, j, k, r
  integer :: niters, ii
  integer :: narg
  character(len=32) :: arg

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  ! this is for multi GPU things
  call omp_set_default_device(myrank)
  !$acc set device_num(myrank)

  narg = command_argument_count()
  dims = 128
  niters = 10
  if (narg >= 1) then
     call get_command_argument(1, arg)
     read(arg, *) dims
  end if
  if (narg >= 2) then
     call get_command_argument(2, arg)
     read(arg, *) niters
  end if

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t1 = omp_get_wtime()

  ! split the array along the k direction
  base      = dims / nranks
  remainder = mod(dims, nranks)

  if (myrank < remainder) then
     nk_local = base + 1
     kstart   = myrank * nk_local + 1
  else
     nk_local = base
     kstart   = remainder * (base + 1) + (myrank - remainder) * base + 1
  end if
  kend = kstart + nk_local - 1
call nvtxStartRange("Domain decompo")
  ! mpi crap
  if (myrank == 0) then
    t1_rank = omp_get_wtime()
     allocate(sendcounts(nranks), displs(nranks))
     do r = 0, nranks - 1
        if (r < remainder) then
           sendcounts(r+1) = (base+1) * dims * dims
           displs(r+1)     = r * (base+1) * dims * dims
        else
           sendcounts(r+1) = base * dims * dims
           displs(r+1)     = remainder*(base+1)*dims*dims + &
                              (r-remainder)*base*dims*dims
        end if
     end do

     allocate(a_global(dims, dims, dims))
     !$omp target enter data map(alloc:a_global)
     
     ! can this be initialized on the GPU? 
     do concurrent(k=1:dims, j=1:dims, i=1:dims)
       a_global(i,j,k) = 1.0_real64
     end do
     t2_rank = omp_get_wtime()
     print '(A,F18.8)', 'Time rank 0 (s) = ', t2_rank - t1_rank
  end if
  call nvtxEndRange

  call nvtxStartRange("allocs and scatter")
  ! --- allocate local arrays ---
  allocate(a(dims, dims, max(1, nk_local)))
  allocate(b(dims, dims, max(1, nk_local)))
  allocate(c(dims, dims, max(1, nk_local)))
  !$omp target enter data map(alloc: a,b,c)

  ! --- scatter subdomains ---
  !$omp target data use_device_addr(a_global, a)
  call MPI_Scatterv(a_global, sendcounts, displs, MPI_REAL8, &
                    a, dims*dims*nk_local, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)
  !$omp end target data

  if (myrank == 0) then
     deallocate(a_global, sendcounts, displs)
     !$omp target exit data map(delete:a_global)
  end if
  !!$omp target update to(a)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t2 = omp_get_wtime()
  call nvtxEndRange
  call nvtxStartRange("Main compiutational loop")

  if (myrank == 0) then
     print '(A,F18.8)', 'Time init (s) = ', t2 - t1
  end if
! initialize things, alloc on the GPU 
  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t1 = omp_get_wtime()

  do concurrent (k = 1:nk_local, j = 1:dims, i = 1:dims)
    b(i,j,k) = 2.0_real64
    c(i,j,k) = 0.0_real64
  end do

  ! perform JAXPY
  do ii = 1, niters
  do concurrent (k = 1:nk_local, j = 1:dims, i = 1:dims)
     c(i,j,k) = alpha * a(i,j,k) + b(i,j,k)
  end do
  end do

  local_sum = 0.0_real64
  ! needs gfortran rcent
  do concurrent (k = 1:nk_local, j = 1:dims, i = 1:dims) reduce(+:local_sum)
     local_sum = local_sum + c(i,j,k)
  end do
  !$omp target update from(local_sum)
  !$omp target exit data map(delete:b,c)
  call nvtxEndRange()
  call nvtxStartRange("final reduce and print")

  call MPI_Reduce(local_sum, global_sum, 1, MPI_REAL8, MPI_SUM, 0, MPI_COMM_WORLD, ierr)

  call MPI_Barrier(MPI_COMM_WORLD, ierr)
  t2 = omp_get_wtime()

  ! time is the global time, not the per rank time although might be a good ide ato report per rank timigns
  if (myrank == 0) then
     print '(A,ES18.8)', 'Global sum = ', global_sum
     print '(A,ES18.8)', 'Expected   = ', 4.0_real64 * dims**3
     print '(A,F18.8)', 'Time (s) = ', t2 - t1
  end if

  call nvtxEndRange()
  deallocate(a,b,c)
  call MPI_Finalize(ierr)
end program mpi_saxpy_3d

