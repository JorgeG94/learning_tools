program mpi_dgemm_dc
  use, intrinsic :: iso_fortran_env, only : real64, int32
  use omp_lib
  use mpi_f08
  implicit none

  ! MPI stuff
  integer :: nranks, myrank
  type(MPI_Comm) :: comm
  integer :: ierr

  ! Problem setup
  integer(int32) :: N, BS
  integer(int32) :: i, j, k
  integer(int32) :: n_local, extra, i0, i1
  integer(int32), allocatable :: counts(:), displs(:)

  ! Arrays
  real(real64), allocatable :: A(:,:), B(:,:), C(:,:)         ! root
  real(real64), allocatable :: A_local(:,:), C_local(:,:)      ! local blocks

  ! Timings
  real(real64) :: t0, t1

  ! ---- MPI init ----
  call MPI_Init()
  comm = MPI_COMM_WORLD
  call MPI_Comm_size(comm, nranks)
  call MPI_Comm_rank(comm, myrank)
  !$    call omp_set_default_device (myrank)
  !$acc set device_num(myrank)

  ! Args: N [BS]
  if (command_argument_count() >= 1) then
     call get_int_arg(1, N)
  else
     N = 1024_int32
  end if
  if (command_argument_count() >= 2) then
     call get_int_arg(2, BS)
  else
     BS = 64_int32
  end if
  if (myrank == 0) then
     write(*,'(a,i0,a,i0)') "N=", N, "  BS=", BS
  end if

  ! Row-block sizes (contiguous in Fortran since leftmost index varies fastest)
  allocate(counts(nranks), displs(nranks))
  n_local = N / nranks
  extra   = mod(N, nranks)

  do i = 0, nranks-1
     counts(i+1) = (n_local + merge(1, 0, i < extra)) * N   ! elements (not rows)
  end do
  displs(1) = 0
  do i = 2, nranks
     displs(i) = displs(i-1) + counts(i-1)
  end do

  ! Local rows for this rank
  i0 = (displs(myrank+1))/N + 1                 ! starting global row (1-based)
  i1 = i0 + counts(myrank+1)/N - 1              ! ending global row
  n_local = i1 - i0 + 1

  ! Allocate local pieces
  allocate(A_local(n_local, N), C_local(n_local, N))
  allocate(B(N, N))       ! every rank needs B

  ! Root alloc & init whole matrices (deterministic so we can verify)
  if (myrank == 0) then
     allocate(A(N, N), C(N, N))
     call fill_A(A)       ! e.g., A(i,j) = i + 0.01*j
     call fill_B(B)       ! e.g., B(i,j) = (i==j) ? 1 : tiny value
  end if

  ! Broadcast B to all ranks
  call MPI_Bcast(B, size(B), MPI_DOUBLE_PRECISION, 0, comm)

  ! Scatter rows of A to A_local (contiguous blocks)
  call MPI_Scatterv( A, counts, displs, MPI_DOUBLE_PRECISION, &
                     A_local, size(A_local), MPI_DOUBLE_PRECISION, 0, comm )

  ! Compute C_local = A_local * B   (blocked naïve DGEMM with DO CONCURRENT)
  C_local = 0.0_real64
  call barrier_and_time(comm, t0)

  call dgemm_blocked_dc(A_local, B, C_local, n_local, N, BS)

  call barrier_and_time(comm, t1)
  if (myrank == 0) then
     write(*,'(a,f8.3,a)') "Compute time: ", (t1-t0), " s"
     ! Rough GFLOP/s (2*N^3 flops overall; each rank does its share)
     write(*,'(a,f8.1,a)') "Approx total GFLOP/s (sum ranks): ", &
          (2.0_real64*N*N*N)/(t1-t0)/1.0e9_real64, ""
  end if

  ! Gather C_local -> C at root
  if (myrank == 0) C = 0.0_real64
  call MPI_Gatherv( C_local, size(C_local), MPI_DOUBLE_PRECISION, &
                    C, counts, displs, MPI_DOUBLE_PRECISION, 0, comm )

  ! (Optional) Check a simple checksum on root
  if (myrank == 0) then
     write(*,'(a,1pe14.6)') "Checksum(C) = ", sum(C)
  end if

  call MPI_Finalize()

contains

  subroutine get_int_arg(pos, val)
    integer, intent(in) :: pos
    integer(int32), intent(out) :: val
    character(len=32) :: buf
    integer :: stat
    call get_command_argument(pos, buf, status=stat)
    if (stat == 0) read(buf,*) val
  end subroutine get_int_arg

  subroutine fill_A(A)
    real(real64), intent(out) :: A(:, :)
    integer :: i, j
    do concurrent (j = 1:size(A,2), i = 1:size(A,1))
       A(i,j) = real(i,real64) + 1.0e-2_real64*real(j,real64)
    end do
  end subroutine fill_A

  subroutine fill_B(B)
    real(real64), intent(out) :: B(:, :)
    integer :: i, j
    do concurrent (j = 1:size(B,2), i = 1:size(B,1))
       if (i == j) then
          B(i,j) = 1.0_real64
       else
          B(i,j) = 1.0e-3_real64
       end if
    end do
  end subroutine fill_B

  subroutine barrier_and_time(comm, t)
    type(MPI_Comm), intent(in) :: comm
    real(real64), intent(out)  :: t
    call MPI_Barrier(comm)
    t = MPI_Wtime()
  end subroutine barrier_and_time

  ! --- Blocked naïve DGEMM using DO CONCURRENT over (i,j) ---
  subroutine dgemm_blocked_dc(Ablk, B, Cblk, m, n, bs)
    real(real64), intent(in)  :: Ablk(m, n)    ! local rows of A
    real(real64), intent(in)  :: B(n, n)
    real(real64), intent(inout) :: Cblk(m, n)  ! local rows of C
    integer, intent(in) :: m, n, bs
    integer :: kk, kend
    integer :: i, j, k
    real(real64) :: s

    do kk = 1, n, bs
       kend = min(kk + bs - 1, n)
       ! Independent (i,j) pairs → good fit for DO CONCURRENT
       do concurrent (j = 1:n, i = 1:m)
          s = 0.0_real64
          do k = kk, kend
             s = s + Ablk(i,k) * B(k,j)
          end do
          Cblk(i,j) = Cblk(i,j) + s
       end do
    end do
  end subroutine dgemm_blocked_dc

end program mpi_dgemm_dc

