program loop_order_sweep_cli
  use iso_fortran_env, only: real64
  implicit none

  integer :: nx, ny, narg
  integer, parameter :: nz_values(*) = [10, 25, 50, 100, 200, 400]
  integer, parameter :: ntests = size(nz_values)
  real(real64), allocatable :: A(:,:,:), B(:,:,:), C(:,:,:), D(:,:,:)
  real(real64) :: coeff = 0.05_real64, t1, t2, diff
  real(real64) :: timings(ntests,4)
  integer :: nz, i, j, k, idx
  character(len=32) :: arg

  !-------------------------------
  ! Parse command line arguments
  !-------------------------------
  narg = command_argument_count()
  nx = 256
  ny = 256

  if (narg >= 1) then
     call get_command_argument(1, arg)
     read(arg, *) nx
  end if
  if (narg >= 2) then
     call get_command_argument(2, arg)
     read(arg, *) ny
  end if

  print *, "===================================================="
  print *, " Loop order benchmark for vertical friction kernel"
  print '(A,I6,A,I6)', " Grid: ", nx, " x ", ny
  print *, "===================================================="

  do idx = 1, ntests
     nz = nz_values(idx)
     print *
     print '(A,I5)', ">>> Testing Nz = ", nz
     print *, "---------------------------------------------"

     allocate(A(nx,ny,nz), B(nx,ny,nz), C(nx,ny,nz), D(nx,ny,nz))
     call random_number(A)
     B = A
     C = A
     D = A

     !---------------------------------------------------------
     ! 1. vertical -> i -> j
     call cpu_time(t1)
     do k = 2, nz-1
        do i = 1, nx
           do j = 1, ny
              if (A(i,j,k) > 0.5d0) then
                 A(i,j,k) = A(i,j,k) + coeff * (A(i,j,k-1) - 2*A(i,j,k) + A(i,j,k+1))
              else
                 A(i,j,k) = A(i,j,k) - coeff * (A(i,j,k-1) + A(i,j,k+1))
              end if
              if (A(i,j,k) > 1.0d0) A(i,j,k) = 1.0d0
              if (A(i,j,k) < 0.0d0) A(i,j,k) = 0.0d0
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,1) = t2 - t1
     print '(A,F10.4," s")', " vertical->i->j elapsed: ", timings(idx,1)

     !---------------------------------------------------------
     ! 2. i -> j -> vertical
     call cpu_time(t1)
     do i = 1, nx
        do j = 1, ny
           do k = 2, nz-1
              if (B(i,j,k) > 0.5d0) then
                 B(i,j,k) = B(i,j,k) + coeff * (B(i,j,k-1) - 2*B(i,j,k) + B(i,j,k+1))
              else
                 B(i,j,k) = B(i,j,k) - coeff * (B(i,j,k-1) + B(i,j,k+1))
              end if
              if (B(i,j,k) > 1.0d0) B(i,j,k) = 1.0d0
              if (B(i,j,k) < 0.0d0) B(i,j,k) = 0.0d0
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,2) = t2 - t1
     print '(A,F10.4," s")', " i->j->vertical elapsed:  ", timings(idx,2)

     !---------------------------------------------------------
     ! 3. j -> vertical -> i
     call cpu_time(t1)
     do j = 1, ny
        do k = 2, nz-1
           do i = 1, nx
              if (C(i,j,k) > 0.5d0) then
                 C(i,j,k) = C(i,j,k) + coeff * (C(i,j,k-1) - 2*C(i,j,k) + C(i,j,k+1))
              else
                 C(i,j,k) = C(i,j,k) - coeff * (C(i,j,k-1) + C(i,j,k+1))
              end if
              if (C(i,j,k) > 1.0d0) C(i,j,k) = 1.0d0
              if (C(i,j,k) < 0.0d0) C(i,j,k) = 0.0d0
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,3) = t2 - t1
     print '(A,F10.4," s")', " j->vertical->i elapsed:  ", timings(idx,3)

     !---------------------------------------------------------
     ! 4. vertical -> j -> i
     call cpu_time(t1)
     do k = 2, nz-1
        do j = 1, ny
           do i = 1, nx
              if (D(i,j,k) > 0.5d0) then
                 D(i,j,k) = D(i,j,k) + coeff * (D(i,j,k-1) - 2*D(i,j,k) + D(i,j,k+1))
              else
                 D(i,j,k) = D(i,j,k) - coeff * (D(i,j,k-1) + D(i,j,k+1))
              end if
              if (D(i,j,k) > 1.0d0) D(i,j,k) = 1.0d0
              if (D(i,j,k) < 0.0d0) D(i,j,k) = 0.0d0
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,4) = t2 - t1
     print '(A,F10.4," s")', " vertical->j->i elapsed:  ", timings(idx,4)

     !---------------------------------------------------------
     diff = maxval(abs(A - B))
     diff = max(diff, maxval(abs(A - C)))
     diff = max(diff, maxval(abs(A - D)))
     print '(A,E12.4)', " Max difference between methods: ", diff

     deallocate(A, B, C, D)
  end do

  print *
  print *, "===================================================="
  print *, " Benchmark complete."
  print *, "===================================================="

  !---------------------------------------------------------
  write(*,'("Grid config (nx, ny) = (",I6,", ",I6,")")') nx, ny
  print *
  print *, "Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i"
  do idx = 1, ntests
     write(*,'(I5,4(",",F12.6))') nz_values(idx), timings(idx,1), timings(idx,2), &
                                   timings(idx,3), timings(idx,4)
  end do

end program loop_order_sweep_cli

