program blocked_dgemm
  use, intrinsic :: iso_fortran_env, only : real64, int32
  use omp_lib
  implicit none
  integer(int32), parameter :: Bsize = 32
  integer(int32) :: N
  integer(int32) :: Nblk, Mblk, Pblk
  integer(int32) :: ib, jb, kb, i, j, k
  integer(int32) :: iloc, jloc, kloc
  real(real64), allocatable :: A(:,:), B(:,:), C(:,:)
  real(real64) :: Awrk(Bsize, Bsize)
  real(real64) :: Bwrk(Bsize, Bsize)
  real(real64) :: t0, t1, dt
  real(real64) :: flops, gflops
  
  ! Get N from command line or use default
  if (command_argument_count() >= 1) then
     call get_int_arg(1, N)
  else
     N = 1024_int32
  end if
  write(*,'(a,i0)') "N = ", N
  
  ! Calculate number of blocks
  Nblk = N / Bsize
  Mblk = N / Bsize
  Pblk = N / Bsize
  
  ! Allocate matrices
  allocate(A(N, N), B(N, N), C(N, N))
  
  ! Initialize
  t0 = omp_get_wtime()
  A = 1.0_real64
  B = 1.0_real64
  C = 0.0_real64
  t1 = omp_get_wtime()
  dt = t1 - t0
  print *, "Alloc time was ", dt, " s"
  
  ! Time the computation
  t0 = omp_get_wtime()
  
  !$omp target data map(to:A,B) map(tofrom:C)

  !$omp target teams distribute collapse(2) num_teams(Nblk*Mblk) &
  !$omp thread_limit(Bsize*Bsize) &
  !$omp private(Awrk, Bwrk, kb, i, j, k, iloc, jloc, kloc)
  do ib = 0, Nblk-1
    do jb = 0, Mblk-1
      do kb = 0, Pblk-1
        
        !$omp parallel num_threads(Bsize*Bsize)
        
        ! Copy block of A into pteam memory
        !$omp do collapse(2) nowait
        do i = ib*Bsize, (ib+1)*Bsize-1
          do k = kb*Bsize, (kb+1)*Bsize-1
            iloc = mod(i, Bsize) + 1
            kloc = mod(k, Bsize) + 1
            Awrk(iloc, kloc) = A(k+1, i+1)
          end do
        end do
        !$omp end do nowait
        
        ! Copy block of B into pteam memory
        !$omp do collapse(2)
        do j = jb*Bsize, (jb+1)*Bsize-1
          do k = kb*Bsize, (kb+1)*Bsize-1
            kloc = mod(k, Bsize) + 1
            jloc = mod(j, Bsize) + 1
            Bwrk(kloc, jloc) = B(j+1, k+1)
          end do
        end do
        !$omp end do
        
        ! Matrix multiply block
        !$omp do collapse(2)
        do i = ib*Bsize, (ib+1)*Bsize-1
          do j = jb*Bsize, (jb+1)*Bsize-1
            iloc = mod(i, Bsize) + 1
            jloc = mod(j, Bsize) + 1
            do k = kb*Bsize, (kb+1)*Bsize-1
              kloc = mod(k, Bsize) + 1
              C(j+1, i+1) = C(j+1, i+1) + Awrk(iloc, kloc) * Bwrk(kloc, jloc)
            end do
          end do
        end do
        !$omp end do
        
        !$omp end parallel
        
      end do
    end do
  end do
  !$omp end target teams distribute
  !$omp end target data
  
  t1 = omp_get_wtime()
  dt = t1 - t0
  flops = 2.0_real64 * real(N, kind=real64)**3
  gflops = flops / dt / 1.0e9_real64
  
  write(*,'(a,1pe14.6)') "Checksum(C) = ", sum(C)
  write(*,'(a,f10.4,a)') "Time = ", dt, " s"
  write(*,'(a,f10.2,a)') "Performance = ", gflops, " GFLOP/s"
  
  deallocate(A, B, C)

contains

  subroutine get_int_arg(pos, val)
    integer(int32), intent(in) :: pos
    integer(int32), intent(out) :: val
    character(len=32) :: arg
    call get_command_argument(pos, arg)
    read(arg, *) val
  end subroutine get_int_arg

end program blocked_dgemm
