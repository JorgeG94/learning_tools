program mpi_scatter_hello
  use, intrinsic :: iso_fortran_env, only : real64
  use mpi_f08
  implicit none

  integer :: ierr, nranks, myrank
  integer :: dims, base, remainder
  integer :: nk_local, kstart, kend
  integer, allocatable :: sendcounts(:), displs(:)
  real(real64), allocatable :: a_global(:,:,:)
  real(real64), allocatable :: a(:,:,:)
  integer :: i, j, k, r

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nranks, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

  dims = 20   ! small for testing

  ! --- compute decomposition along k dimension ---
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

  ! --- allocate bookkeeping arrays on root ---
  if (myrank == 0) then
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
     do k = 1, dims
        do j = 1, dims
           do i = 1, dims
              a_global(i,j,k) = 1000*k + 100*j + i
           end do
        end do
     end do
  end if

  ! --- local allocation for received subdomain ---
  allocate(a(dims, dims, max(1, nk_local)))

  ! --- distribute the global array along k dimension ---
  call MPI_Scatterv(a_global, sendcounts, displs, MPI_REAL8, &
                    a, dims*dims*nk_local, MPI_REAL8, 0, MPI_COMM_WORLD, ierr)

  if (myrank == 0) then
     deallocate(a_global, sendcounts, displs)
  end if

  ! --- print a simple message from each rank ---
  if (nk_local > 0) then
     write(*,'(A,I0,A,I0,A,I0,A,3(1X,F8.1))') 'Rank ', myrank, &
          ' handles k=', kstart, '→', kend, ' sample:', &
          a(1,1,1), a(1,1,nk_local), a(dims,dims,1)
  else
     write(*,'(A,I0,A)') 'Rank ', myrank, ' received no data (empty slice).'
  end if

  call MPI_Finalize(ierr)
end program mpi_scatter_hello

