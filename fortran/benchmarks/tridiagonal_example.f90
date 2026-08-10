program loop_order_sweep_do_concurrent_tridiag
  use iso_fortran_env, only: real64
  implicit none

  integer :: nx, ny, narg
  integer, parameter :: nz_values(*) = [10, 25, 50, 100, 200, 400]
  integer, parameter :: ntests = size(nz_values)
  real(real64), allocatable :: A(:,:,:), Anew(:,:,:)
  real(real64), allocatable :: B(:,:,:), Bnew(:,:,:)
  real(real64), allocatable :: C(:,:,:), Cnew(:,:,:)
  real(real64), allocatable :: D(:,:,:), Dnew(:,:,:)
  real(real64), allocatable :: E(:,:,:), Enew(:,:,:)
  real(real64) :: cdoeff = 0.05_real64, t1, t2, diff
  real(real64), allocatable :: coeff(:,:)
  real(real64) :: timings(ntests,5)
  real(real64) :: val, tmp
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
  print *, " DO CONCURRENT benchmark with vertical dependency"
  write(*,'(" Grid config (nx, ny) = (",I0,", ",I0,")")') nx, ny
  print *, "===================================================="
  allocate(coeff(nx,ny))
  !$omp target enter data map(alloc: coeff)

  do concurrent (i=1:nx, j=1:ny)
    coeff(i,j) = real(i,real64)
  end do

  do idx = 1, ntests
     nz = nz_values(idx)
     print *
     print '(A,I5)', ">>> Testing Nz = ", nz
     print *, "---------------------------------------------"

     allocate(A(nx,ny,nz), Anew(nx,ny,nz))
     allocate(B(nx,ny,nz), Bnew(nx,ny,nz))
     allocate(C(nx,ny,nz), Cnew(nx,ny,nz))
     allocate(D(nx,ny,nz), Dnew(nx,ny,nz))
     allocate(E(nx,ny,nz), Enew(nx,ny,nz))

     !------------------------------------------
     ! Initialization on device
     !------------------------------------------
     call random_number(val)

     !$omp target enter data map(alloc: A,B,C,D,E, Anew,Bnew,Cnew,Dnew,Enew)

     ! Initialize all arrays on the device in parallel
     do concurrent (i=1:nx, j=1:ny, k=1:nz)
        A(i,j,k)    = val
        B(i,j,k)    = val
        C(i,j,k)    = val
        D(i,j,k)    = val
        E(i,j,k)    = val
        Anew(i,j,k) = val
        Bnew(i,j,k) = val
        Cnew(i,j,k) = val
        Dnew(i,j,k) = val
        Enew(i,j,k) = val
     end do

     !---------------------------------------------------------
     ! 1. "vertical->i->j"
     !    k outer (serial), (i,j) inside do concurrent
     !---------------------------------------------------------
     call cpu_time(t1)
     do k = 2, nz
        do concurrent (i=1:nx, j=1:ny)
           tmp = A(i,j,k) - coeff(i,j) * Anew(i,j,k-1)
           if (tmp > 1.0d0) tmp = 1.0d0
           if (tmp < 0.0d0) tmp = 0.0d0
           Anew(i,j,k) = tmp
        end do
     end do
     call cpu_time(t2)
     timings(idx,1) = t2 - t1
     print '(A,F10.4," s")', " vertical->i->j elapsed:       ", timings(idx,1)

     !---------------------------------------------------------
     ! 2. "i->j->vertical"
     !    do concurrent over (i,j) outer, serial k inner
     !---------------------------------------------------------
     call cpu_time(t1)
     do concurrent (i=1:nx, j=1:ny)
        do k = 2, nz
           tmp = B(i,j,k) - coeff(i,j) * Bnew(i,j,k-1)
           if (tmp > 1.0d0) tmp = 1.0d0
           if (tmp < 0.0d0) tmp = 0.0d0
           Bnew(i,j,k) = tmp
        end do
     end do
     call cpu_time(t2)
     timings(idx,2) = t2 - t1
     print '(A,F10.4," s")', " i->j->vertical elapsed:        ", timings(idx,2)

     !---------------------------------------------------------
     ! 3. "j->vertical->i"
     !    do concurrent over j, then serial k, then do concurrent over i
     !    (still only (i,j) parallel; k serial)
     !---------------------------------------------------------
     call cpu_time(t1)
     do concurrent (j=1:ny)
        do k = 2, nz
           do concurrent (i=1:nx)
              tmp = C(i,j,k) - coeff(i,j) * Cnew(i,j,k-1)
              if (tmp > 1.0d0) tmp = 1.0d0
              if (tmp < 0.0d0) tmp = 0.0d0
              Cnew(i,j,k) = tmp
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,3) = t2 - t1
     print '(A,F10.4," s")', " j->vertical->i elapsed:        ", timings(idx,3)

     !---------------------------------------------------------
     ! 4. "vertical->j->i"
     !    k outer (serial), then do concurrent over j, then i
     !---------------------------------------------------------
     call cpu_time(t1)
     do k = 2, nz
        do concurrent (j=1:ny)
           do concurrent (i=1:nx)
              tmp = D(i,j,k) - coeff(i,j) * Dnew(i,j,k-1)
              if (tmp > 1.0d0) tmp = 1.0d0
              if (tmp < 0.0d0) tmp = 0.0d0
              Dnew(i,j,k) = tmp
           end do
        end do
     end do
     call cpu_time(t2)
     timings(idx,4) = t2 - t1
     print '(A,F10.4," s")', " vertical->j->i elapsed:        ", timings(idx,4)

     !---------------------------------------------------------
     ! 5. "j->i->vertical"
     !    do concurrent over j and i, then serial k
     !---------------------------------------------------------
     call cpu_time(t1)
     do concurrent (j=1:ny, i=1:nx)
        do k = 2, nz
           tmp = E(i,j,k) - coeff(i,j) * Enew(i,j,k-1)
           if (tmp > 1.0d0) tmp = 1.0d0
           if (tmp < 0.0d0) tmp = 0.0d0
           Enew(i,j,k) = tmp
        end do
     end do
     call cpu_time(t2)
     timings(idx,5) = t2 - t1
     print '(A,F10.4," s")', " j->i->vertical elapsed:        ", timings(idx,5)

     !---------------------------------------------------------
     ! ✅ Verify correctness across methods
     !---------------------------------------------------------
     diff = 0.0_real64
     do concurrent (i=1:nx, j=1:ny, k=1:nz) reduce(max: diff)
        diff = max(diff, abs(Anew(i,j,k) - Bnew(i,j,k)))
        diff = max(diff, abs(Anew(i,j,k) - Cnew(i,j,k)))
        diff = max(diff, abs(Anew(i,j,k) - Dnew(i,j,k)))
        diff = max(diff, abs(Anew(i,j,k) - Enew(i,j,k)))
     end do

     !$omp target update from(diff)
     !$omp target exit data map(delete: A,B,C,D,E, Anew,Bnew,Cnew,Dnew,Enew)

     print '(A,E12.4)', " Max difference between methods: ", diff

     deallocate(A, Anew, B, Bnew, C, Cnew, D, Dnew, E, Enew)
  end do

  print *
  print *, "===================================================="
  print *, " Benchmark complete."
  print *, "===================================================="

  !---------------------------------------------------------
  ! 🧾 CSV summary
  print *
  print *, "Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i,j->i->vertical"
  do idx = 1, ntests
     write(*,'(I5,5(",",F12.6))') nz_values(idx), timings(idx,1), timings(idx,2), &
                                   timings(idx,3), timings(idx,4), timings(idx,5)
  end do

end program loop_order_sweep_do_concurrent_tridiag

