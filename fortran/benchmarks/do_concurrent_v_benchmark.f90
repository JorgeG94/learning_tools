program loop_order_sweep_do_concurrent_safe
  use iso_fortran_env, only: real64
  implicit none

  integer :: nx, ny, narg
  !integer, parameter :: nz_values(*) = [400]
  integer, parameter :: nz_values(*) = [10, 25, 50, 100, 200, 400]
  integer, parameter :: ntests = size(nz_values)
  real(real64), allocatable :: A(:,:,:), Anew(:,:,:)
  real(real64), allocatable :: B(:,:,:), Bnew(:,:,:)
  real(real64), allocatable :: C(:,:,:), Cnew(:,:,:)
  real(real64), allocatable :: D(:,:,:), Dnew(:,:,:)
  real(real64) :: coeff = 0.05_real64, t1, t2, diff
  real(real64) :: timings(ntests,4)
  real(real64) :: val
  integer :: i,j,k,nz, idx
  character(len=32) :: arg

  !----------------------------------
  ! Command-line argument parsing
  !----------------------------------
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
  print *, " Safe DO CONCURRENT benchmark (double-buffered)"
  write(*,'(" Grid config (nx, ny) = (",I0,", ",I0,")")') nx, ny
  print *, "===================================================="

  do idx = 1, ntests
     nz = nz_values(idx)
     print *
     print '(A,I5)', ">>> Testing Nz = ", nz
     print *, "---------------------------------------------"

     allocate(A(nx,ny,nz), Anew(nx,ny,nz))
     allocate(B(nx,ny,nz), Bnew(nx,ny,nz))
     allocate(C(nx,ny,nz), Cnew(nx,ny,nz))
     allocate(D(nx,ny,nz), Dnew(nx,ny,nz))

     call random_number(val)
     !B = A; C = A; D = A
     !Anew = A; Bnew = B; Cnew = C; Dnew = D
     !$omp target enter data map(alloc: A,B,C,D,Anew, Bnew, Cnew, Dnew)
    do concurrent (i=1:nx, j=1:ny, k=1:nz)
     A(i,j,k) = val 
     B(i,j,k) = val 
     C(i,j,k) = val 
     D(i,j,k) = val 
     Anew(i,j,k) = val
     Bnew(i,j,k) = val
     Cnew(i,j,k) = val
     Dnew(i,j,k) = val
    end do

     !---------------------------------------------------------
     ! 1. vertical -> i -> j
     call cpu_time(t1)
     do concurrent (k=2:nz-1, i=1:nx, j=1:ny)
        if (A(i,j,k) > 0.5d0) then
           Anew(i,j,k) = A(i,j,k) + coeff * (A(i,j,k-1) - 2*A(i,j,k) + A(i,j,k+1))
        else
           Anew(i,j,k) = A(i,j,k) - coeff * (A(i,j,k-1) + A(i,j,k+1))
        end if
        if (Anew(i,j,k) > 1.0d0) Anew(i,j,k) = 1.0d0
        if (Anew(i,j,k) < 0.0d0) Anew(i,j,k) = 0.0d0
     end do
     call cpu_time(t2)
     timings(idx,1) = t2 - t1
     print '(A,F10.4," s")', " vertical->i->j elapsed: ", timings(idx,1)

     !---------------------------------------------------------
     ! 2. i -> j -> vertical
     call cpu_time(t1)
     do concurrent (i=1:nx, j=1:ny, k=2:nz-1)
        if (B(i,j,k) > 0.5d0) then
           Bnew(i,j,k) = B(i,j,k) + coeff * (B(i,j,k-1) - 2*B(i,j,k) + B(i,j,k+1))
        else
           Bnew(i,j,k) = B(i,j,k) - coeff * (B(i,j,k-1) + B(i,j,k+1))
        end if
        if (Bnew(i,j,k) > 1.0d0) Bnew(i,j,k) = 1.0d0
        if (Bnew(i,j,k) < 0.0d0) Bnew(i,j,k) = 0.0d0
     end do
     call cpu_time(t2)
     timings(idx,2) = t2 - t1
     print '(A,F10.4," s")', " i->j->vertical elapsed:  ", timings(idx,2)

     !---------------------------------------------------------
     ! 3. j -> vertical -> i
     call cpu_time(t1)
     do concurrent (j=1:ny, k=2:nz-1, i=1:nx)
        if (C(i,j,k) > 0.5d0) then
           Cnew(i,j,k) = C(i,j,k) + coeff * (C(i,j,k-1) - 2*C(i,j,k) + C(i,j,k+1))
        else
           Cnew(i,j,k) = C(i,j,k) - coeff * (C(i,j,k-1) + C(i,j,k+1))
        end if
        if (Cnew(i,j,k) > 1.0d0) Cnew(i,j,k) = 1.0d0
        if (Cnew(i,j,k) < 0.0d0) Cnew(i,j,k) = 0.0d0
     end do
     call cpu_time(t2)
     timings(idx,3) = t2 - t1
     print '(A,F10.4," s")', " j->vertical->i elapsed:  ", timings(idx,3)

     !---------------------------------------------------------
     ! 4. vertical -> j -> i
     call cpu_time(t1)
     do concurrent (k=2:nz-1, j=1:ny, i=1:nx)
        if (D(i,j,k) > 0.5d0) then
           Dnew(i,j,k) = D(i,j,k) + coeff * (D(i,j,k-1) - 2*D(i,j,k) + D(i,j,k+1))
        else
           Dnew(i,j,k) = D(i,j,k) - coeff * (D(i,j,k-1) + D(i,j,k+1))
        end if
        if (Dnew(i,j,k) > 1.0d0) Dnew(i,j,k) = 1.0d0
        if (Dnew(i,j,k) < 0.0d0) Dnew(i,j,k) = 0.0d0
     end do
     call cpu_time(t2)
     timings(idx,4) = t2 - t1
     print '(A,F10.4," s")', " vertical->j->i elapsed:  ", timings(idx,4)

     !---------------------------------------------------------
     ! ✅ Verify correctness
     diff = 0.0_real64
    do concurrent (i=1:nx, j=1:ny, k=1:nz) reduce(max: diff)
       diff = max(diff, abs(Anew(i,j,k) - Bnew(i,j,k)))
       diff = max(diff, abs(Anew(i,j,k) - Cnew(i,j,k)))
       diff = max(diff, abs(Anew(i,j,k) - Dnew(i,j,k)))
    end do
     !$omp target exit data map(delete: A,B,C,D,Anew, Bnew, Cnew, Dnew)
     !$omp target update from(diff)

     print '(A,E12.4)', " Max difference between methods: ", diff

     deallocate(A, Anew, B, Bnew, C, Cnew, D, Dnew)
  end do

  print *
  print *, "===================================================="
  print *, " Benchmark complete."
  print *, "===================================================="

  !---------------------------------------------------------
  ! 🧾 CSV summary
  print *
  print *, "Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i"
  do idx = 1, ntests
     write(*,'(I5,4(",",F12.6))') nz_values(idx), timings(idx,1), timings(idx,2), &
                                   timings(idx,3), timings(idx,4)
  end do

end program loop_order_sweep_do_concurrent_safe

