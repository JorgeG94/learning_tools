program mpi_scatterv_test
  use, intrinsic :: iso_fortran_env, only : real64
  use mpi_f08
  implicit none

  integer :: ierr, nranks, myrank
  integer :: dims, base, remainder
  integer :: nk_local, kstart, kend
  integer, allocatable :: sendcounts(:), displs(:)
  real(real64), allocatable :: a_global(:,:,:)
  real(real64), allocatable :: a_local(:,:,:)
  integer :: i, j, k, r

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

  ! Problem size
  dims = 72

  ! 1D block decomposition along k dimension
  base = dims / nranks
  remainder = mod(dims, nranks)
  if (myrank < remainder) then
     nk_local = base + 1
     kstart = myrank * nk_local + 1
  else
     nk_local = base
     kstart = remainder * (base + 1) + (myrank - remainder) * base + 1
  end if
  kend = kstart + nk_local - 1

  if (myrank == 0) then
     allocate(a_global(dims, dims, dims))
     allocate(sendcounts(nranks), displs(nranks))
     ! fill global array with identifiable values
     do k = 1, dims
        do j = 1, dims
           do i = 1, dims
              a_global(i,j,k) = 1000*k + 100*j + i
           end do
        end do
     end do
     ! build sendcounts / displacements (in contiguous elements)
     do r = 0, nranks-1
        if (r < remainder) then
           sendcounts(r+1) = (base + 1) * dims * dims
           displs(r+1)     = r * (base + 1) * dims * dims
        else
           sendcounts(r+1) = base * dims * dims
           displs(r+1)     = remainder * (base + 1) * dims * dims + &
                              (r - remainder) * base * dims * dims
        end if
     end do
  end if

  allocate(a_local(dims, dims, nk_local))

  call MPI_Scatterv(a_global, sendcounts, displs, MPI_REAL8, &
                    a_local, dims*dims*nk_local, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)

  if (myrank == 0) then
     deallocate(a_global, sendcounts, displs)
  end if

  ! Print a few values per rank to verify correct slices
  print '(A,I0,A,I0,A,I0,A,3(1X,F8.1))', 'Rank ', myrank, ': k=', kstart, '→', kend, ' sample=', &
        a_local(1,1,1), a_local(1,1,nk_local), a_local(dims,dims,1)

  call MPI_Finalize(ierr)
end program mpi_scatterv_test

