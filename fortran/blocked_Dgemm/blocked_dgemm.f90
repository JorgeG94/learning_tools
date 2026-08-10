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
  real(real64), allocatable :: Awrk(:, :)
  real(real64), allocatable :: Bwrk(:, :)
  !real(real64) :: Awrk(bsize, bsize)
  !real(real64) :: Bwrk(bsize, bsize)
  real(real64) :: t0, t1, dt
  real(real64) :: flops, gflops
  
  ! Get N from command line or use default
  if (command_argument_count() >= 1) then
     call get_int_arg(1, N)
  else
     N = 1024_int32
  end if
  write(*,'(a,i0)') "N = ", N
  
  ! Allocate matrices
  allocate(A(N, N), B(N, N), C(N, N))
  allocate(Awrk(bsize, bsize), Bwrk(bsize, bsize))
  
  ! Initialize
  t0 = omp_get_wtime()
  A = 1.0_real64
  B = 1.0_real64
  C = 0.0_real64
  t1 = omp_get_wtime()
  dt = t1 - t0
  print *, "Alloc time was ", dt, " s"
  
  ! Calculate number of blocks
  Nblk = N / Bsize
  Mblk = N / Bsize
  Pblk = N / Bsize
  
  ! Time the computation
  t0 = omp_get_wtime()
 !$omp target enter data map(alloc: A,B,C, awrk, bwrk) 
 !$omp target
  ! Loop over blocks of C
  do ib = 0, Nblk-1
    do jb = 0, Mblk-1
      do kb = 0, Pblk-1
     !do concurrent (ib=1:nblk, jb = 1:mblk, kb = 1:pblk)
        
        ! Copy block of A into workspace
        do concurrent (i=ib*bsize:(ib+1)*bsize-1, k = kb*bsize:(kb+1)*bsize-1)
        !do i = ib*Bsize, (ib+1)*Bsize-1
        !  do k = kb*Bsize, (kb+1)*Bsize-1
            iloc = mod(i, Bsize) + 1
            kloc = mod(k, Bsize) + 1
            Awrk(iloc, kloc) = A(k+1, i+1)
        !  end do
        end do
        
        ! Copy block of B into workspace
        do concurrent (j=jb*bsize:(jb+1)*bsize-1, k = kb*bsize:(kb+1)*bsize-1)
        !do j = jb*Bsize, (jb+1)*Bsize-1
        !  do k = kb*Bsize, (kb+1)*Bsize-1
            kloc = mod(k, Bsize) + 1
            jloc = mod(j, Bsize) + 1
            Bwrk(kloc, jloc) = B(j+1, k+1)
        !  end do
        end do
        
        ! Matrix multiply block
         do concurrent (i=ib*bsize:(ib+1)*bsize-1, j=jb*bsize:(jb+1)*bsize-1)
        !do i = ib*Bsize, (ib+1)*Bsize-1
        !  do j = jb*Bsize, (jb+1)*Bsize-1
            iloc = mod(i, Bsize) + 1
            jloc = mod(j, Bsize) + 1
            do k = kb*Bsize, (kb+1)*Bsize-1
              kloc = mod(k, Bsize) + 1
              C(j+1, i+1) = C(j+1, i+1) + Awrk(iloc, kloc) * Bwrk(kloc, jloc)
            end do
        !  end do
        end do
        
      end do
    end do
  end do
  !$omp end target
  
  t1 = omp_get_wtime()
  dt = t1 - t0
  flops = 2.0_real64 * real(N, kind=real64)**3
  gflops = flops / dt / 1.0e9_real64
  !$omp target update from(C)
  
  write(*,'(a,1pe14.6)') "Checksum(C) = ", sum(C)
  write(*,'(a,f10.4,a)') "Time = ", dt, " s"
  write(*,'(a,f10.2,a)') "Performance = ", gflops, " GFLOP/s"
 !$omp target exit data map(delete: A,B,C,Awrk, Bwrk) 
  
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
