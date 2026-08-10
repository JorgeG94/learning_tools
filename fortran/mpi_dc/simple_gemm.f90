program simple_dgemm_dc
  use, intrinsic :: iso_fortran_env, only : real64, int32
  use omp_lib
  implicit none

  integer(int32) :: N
  integer(int32) :: i, j, k
  real(real64), allocatable :: A(:,:), B(:,:), C(:,:)
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

  ! Initialize
  call fill_A(A)
  call fill_B(B)
  C = 0.0_real64

  ! Time the computation
  t0 = omp_get_wtime()
  
  call dgemm_dc(A, B, C, N)
  
  t1 = omp_get_wtime()

  dt = t1 - t0
  flops = 2.0_real64 * real(N, kind=real64)**3
  gflops = flops / dt / 1.0e9_real64

  write(*,'(a,1pe14.6)') "Checksum(C) = ", sum(C)
  write(*,'(a,f10.4,a)') "Time = ", dt, " s"
  write(*,'(a,f10.2,a)') "Performance = ", gflops, " GFLOP/s"

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

  ! Simple unblocked DGEMM: C = A * B
  subroutine dgemm_dc(A, B, C, n)
    real(real64), intent(in)    :: A(n, n)
    real(real64), intent(in)    :: B(n, n)
    real(real64), intent(inout) :: C(n, n)
    integer, intent(in) :: n
    integer :: i, j, k
    real(real64) :: s

    do concurrent (j = 1:n, i = 1:n)
       s = 0.0_real64
       do k = 1, n
          s = s + A(i,k) * B(k,j)
       end do
       C(i,j) = s
    end do
  end subroutine dgemm_dc

  ! C = C + A*B   (square N×N for simplicity; easy to generalize)
  subroutine dgemm_blocked(n, A, B, C)
    integer, intent(in)              :: n
    real(real64), intent(in)         :: A(n,n), B(n,n)
    real(real64), intent(inout)      :: C(n,n)

    integer :: i0, j0, k0, ib, jb, kb
    integer :: i, j, k
    integer, parameter :: BS = 32
    real(real64) :: AT(BS,BS), BT(BS,BS), CT(BS,BS)

    do j0 = 1, n, BS
      jb = min(BS, n - j0 + 1)
      do i0 = 1, n, BS
        ib = min(BS, n - i0 + 1)

        CT(1:ib,1:jb) = 0.0_real64

        do k0 = 1, n, BS
          kb = min(BS, n - k0 + 1)

          ! Pack tiles into contiguous locals
          AT(1:ib,1:kb) = A(i0:i0+ib-1, k0:k0+kb-1)
          BT(1:kb,1:jb) = B(k0:k0+kb-1, j0:j0+jb-1)

          ! Branchless micro-kernel over the packed tiles
          do concurrent (j = 1:jb, i = 1:ib) local(acc)
            real(real64) :: acc
            acc = 0.0_real64
            do k = 1, kb
              acc = acc + AT(i,k) * BT(k,j)
            end do
            CT(i,j) = CT(i,j) + acc
          end do

        end do

        ! Write back the finished C tile
        C(i0:i0+ib-1, j0:j0+jb-1) = C(i0:i0+ib-1, j0:j0+jb-1) + CT(1:ib,1:jb)
      end do
    end do
  end subroutine dgemm_blocked

end program simple_dgemm_dc
