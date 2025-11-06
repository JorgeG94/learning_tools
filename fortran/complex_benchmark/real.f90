program loop_order_sweep_do_concurrent_tridiag_reuse
   use iso_fortran_env, only: real64
   implicit none

   !integer, parameter :: nz_values(*) = [400]
   !integer, parameter :: nz_values(*) = [10, 25, 50, 100, 200, 400]
   integer, allocatable :: nz_values(:)
   integer, parameter :: ntests = 200
   !integer, parameter :: ntests = size(nz_values)
   integer, parameter :: n_iters = 20
   integer :: ii
   integer, parameter :: bar_width = 40
   integer :: filled
   real(real64) :: frac
   character(len=bar_width) :: bar

   integer :: nx, ny, nz, narg
   integer :: i, j, k, idx
   character(len=32) :: arg

   ! 3D coefficient / state fields
   real(real64), allocatable :: a3d(:, :, :), h3d(:, :, :)
   real(real64), allocatable :: u(:, :, :), unew(:, :, :), c1(:, :, :)

   ! 2D fields
   real(real64), allocatable :: ray(:, :), mask(:, :), b1(:, :), d1(:, :)

   real(real64) :: dt, t1, t2
   real(real64) :: timings(ntests, 5)
   real(real64) :: val
   real(real64) :: a_loc, h_loc, ray_loc, b1_loc, d1_loc, b_denom, u_prev, u_loc
   real(real64), parameter :: one = 1.0_real64, zero = 0.0_real64

   allocate (nz_values(ntests))
   do ii = 1, ntests
      nz_values(ii) = ii + 2
   end do

   !----------------------------------
   ! Command-line argument parsing
   !----------------------------------
   narg = command_argument_count()
   nx = 256
   ny = 256
   if (narg >= 1) then
      call get_command_argument(1, arg)
      read (arg, *) nx
   end if
   if (narg >= 2) then
      call get_command_argument(2, arg)
      read (arg, *) ny
   end if

   dt = 0.01_real64

   print *, "===================================================="
   print *, " DO CONCURRENT tridiag-like benchmark               "
   write (*, '(" Grid config (nx, ny) = (",I0,", ",I0,")")') nx, ny
   print *, "===================================================="

   do idx = 1, ntests

      nz = nz_values(idx)
      frac = real(idx, real64)/real(ntests, real64)
      filled = int(frac*bar_width)

      if (filled < 0) filled = 0
      if (filled > bar_width) filled = bar_width

      bar = repeat('#', filled)//repeat('-', bar_width - filled)

      ! Carriage return to overwrite the same line
      write (*, '(A)', advance='no') achar(13)
      write (*, '("Progress [",A,"] ",I0,"/",I0)', advance='no') bar, idx, ntests
      !print *
      !print '(A,I5)', ">>> Testing Nz = ", nz
      !print *, "---------------------------------------------"

      ! Allocate fields
      allocate (a3d(nx, ny, nz), h3d(nx, ny, nz))
      allocate (u(nx, ny, nz), unew(nx, ny, nz), c1(nx, ny, nz))
      allocate (ray(nx, ny), mask(nx, ny), b1(nx, ny), d1(nx, ny))

      call random_number(val)

      !------------------------------------------
      ! Move arrays to device (no copy) via OpenMP
      !------------------------------------------
      !$omp target enter data map(alloc: a3d,h3d,u,unew,c1,ray,mask,b1,d1)

      !------------------------------------------
      ! Initial coefficients and initial state
      ! (done on device through do concurrent)
      !------------------------------------------
      do concurrent(i=1:nx, j=1:ny, k=1:nz)
         ! 3D coefficients
         a3d(i, j, k) = 0.1_real64
         h3d(i, j, k) = 1.0_real64

         ! Initial u field
         u(i, j, k) = val
         unew(i, j, k) = u(i, j, k)

         c1(i, j, k) = 0.0_real64

         ! 2D fields, once per (i,j)
         if (k == 1) then
            ray(i, j) = 0.01_real64
            mask(i, j) = 1.0_real64   ! all active; structural mask
            b1(i, j) = 0.0_real64
            d1(i, j) = 0.0_real64
         end if
      end do

      !=========================================================
      ! 1. vertical -> i -> j
      !    k outer (serial), inside parallel over (i,j)
      !=========================================================
      call reset_state(nx, ny, nz, u, unew, c1, b1, d1)
      call cpu_time(t1)
      do ii = 1, n_iters
      do k = 2, nz - 1
         do concurrent(j=1:ny)
            do concurrent(i=1:nx)
               if (mask(i, j) <= 0.0_real64) cycle

               a_loc = a3d(i, j, k)
               h_loc = h3d(i, j, k)
               ray_loc = ray(i, j)
               b1_loc = b1(i, j)
               d1_loc = d1(i, j)

               c1(i, j, k) = dt*a_loc*b1_loc

               b_denom = h_loc + dt*(ray_loc + a_loc*d1_loc)
               b1_loc = one/(b_denom + dt*a3d(i, j, k + 1))
               d1_loc = b_denom*b1_loc

               u_prev = unew(i, j, k - 1)
               u_loc = (h_loc*u(i, j, k) + dt*a_loc*u_prev)*b1_loc
               unew(i, j, k) = u_loc

               b1(i, j) = b1_loc
               d1(i, j) = d1_loc
            end do
         end do
      end do
      end do
      call cpu_time(t2)
      timings(idx, 1) = t2 - t1
      !print '(A,F10.4," s")', " vertical->i->j elapsed:       ", timings(idx,1)

      !=========================================================
      ! 2. i -> j -> vertical
      !    parallel over (i,j), inner serial k
      !=========================================================
      call reset_state(nx, ny, nz, u, unew, c1, b1, d1)
      call cpu_time(t1)
      do ii = 1, n_iters
      do concurrent(j=1:ny, i=1:nx)
         if (mask(i, j) <= 0.0_real64) cycle
         do k = 2, nz - 1
            a_loc = a3d(i, j, k)
            h_loc = h3d(i, j, k)
            ray_loc = ray(i, j)
            b1_loc = b1(i, j)
            d1_loc = d1(i, j)

            c1(i, j, k) = dt*a_loc*b1_loc

            b_denom = h_loc + dt*(ray_loc + a_loc*d1_loc)
            b1_loc = one/(b_denom + dt*a3d(i, j, k + 1))
            d1_loc = b_denom*b1_loc

            u_prev = unew(i, j, k - 1)
            u_loc = (h_loc*u(i, j, k) + dt*a_loc*u_prev)*b1_loc
            unew(i, j, k) = u_loc

            b1(i, j) = b1_loc
            d1(i, j) = d1_loc
         end do
      end do
      end do
      call cpu_time(t2)
      timings(idx, 2) = t2 - t1
      !print '(A,F10.4," s")', " i->j->vertical elapsed:        ", timings(idx,2)

      !=========================================================
      ! 3. j -> vertical -> i
      !    parallel over j, inner serial k, inner concurrent i
      !=========================================================
      call reset_state(nx, ny, nz, u, unew, c1, b1, d1)
      call cpu_time(t1)
      do ii = 1, n_iters
      do concurrent(j=1:ny)
         do k = 2, nz - 1
            do concurrent(i=1:nx)
               if (mask(i, j) <= 0.0_real64) cycle

               a_loc = a3d(i, j, k)
               h_loc = h3d(i, j, k)
               ray_loc = ray(i, j)
               b1_loc = b1(i, j)
               d1_loc = d1(i, j)

               c1(i, j, k) = dt*a_loc*b1_loc

               b_denom = h_loc + dt*(ray_loc + a_loc*d1_loc)
               b1_loc = one/(b_denom + dt*a3d(i, j, k + 1))
               d1_loc = b_denom*b1_loc

               u_prev = unew(i, j, k - 1)
               u_loc = (h_loc*u(i, j, k) + dt*a_loc*u_prev)*b1_loc
               unew(i, j, k) = u_loc

               b1(i, j) = b1_loc
               d1(i, j) = d1_loc
            end do
         end do
      end do
      end do
      call cpu_time(t2)
      timings(idx, 3) = t2 - t1
      !print '(A,F10.4," s")', " j->vertical->i elapsed:        ", timings(idx,3)

      !=========================================================
      ! 4. vertical -> j -> i
      !    k outer, then concurrent j, then concurrent i
      !=========================================================
      call reset_state(nx, ny, nz, u, unew, c1, b1, d1)
      call cpu_time(t1)
      do ii = 1, n_iters
      do k = 2, nz - 1
         do concurrent(j=1:ny)
            do concurrent(i=1:nx)
               if (mask(i, j) <= 0.0_real64) cycle

               a_loc = a3d(i, j, k)
               h_loc = h3d(i, j, k)
               ray_loc = ray(i, j)
               b1_loc = b1(i, j)
               d1_loc = d1(i, j)

               c1(i, j, k) = dt*a_loc*b1_loc

               b_denom = h_loc + dt*(ray_loc + a_loc*d1_loc)
               b1_loc = one/(b_denom + dt*a3d(i, j, k + 1))
               d1_loc = b_denom*b1_loc

               u_prev = unew(i, j, k - 1)
               u_loc = (h_loc*u(i, j, k) + dt*a_loc*u_prev)*b1_loc
               unew(i, j, k) = u_loc

               b1(i, j) = b1_loc
               d1(i, j) = d1_loc
            end do
         end do
      end do
      end do
      call cpu_time(t2)
      timings(idx, 4) = t2 - t1
      !print '(A,F10.4," s")', " vertical->j->i elapsed:        ", timings(idx,4)

      !=========================================================
      ! 5. j -> i -> vertical
      !    concurrent over j,i, then serial k
      !=========================================================
      call reset_state(nx, ny, nz, u, unew, c1, b1, d1)
      call cpu_time(t1)
      do ii = 1, n_iters
      do concurrent(j=1:ny, i=1:nx)
         if (mask(i, j) <= 0.0_real64) cycle
         do k = 2, nz - 1
            a_loc = a3d(i, j, k)
            h_loc = h3d(i, j, k)
            ray_loc = ray(i, j)
            b1_loc = b1(i, j)
            d1_loc = d1(i, j)

            c1(i, j, k) = dt*a_loc*b1_loc

            b_denom = h_loc + dt*(ray_loc + a_loc*d1_loc)
            b1_loc = one/(b_denom + dt*a3d(i, j, k + 1))
            d1_loc = b_denom*b1_loc

            u_prev = unew(i, j, k - 1)
            u_loc = (h_loc*u(i, j, k) + dt*a_loc*u_prev)*b1_loc
            unew(i, j, k) = u_loc

            b1(i, j) = b1_loc
            d1(i, j) = d1_loc
         end do
      end do
      end do
      call cpu_time(t2)
      timings(idx, 5) = t2 - t1
      !print '(A,F10.4," s")', " j->i->vertical elapsed:        ", timings(idx,5)

      !------------------------------------------
      ! Free device and host memory
      !------------------------------------------
      !$omp target exit data map(delete: a3d,h3d,u,unew,c1,ray,mask,b1,d1)

      deallocate (a3d, h3d, u, unew, c1, ray, mask, b1, d1)
   end do

   print *
   print *, "===================================================="
   print *, " Benchmark complete."
   print *, "===================================================="
   print *

   !-------------------------------------------------------
   ! Compute average, variance, stddev per loop ordering
   !-------------------------------------------------------
   block
      real(real64) :: avg(5), var(5), std(5)
      real(real64) :: sum, sumsq, x
      integer :: m
      character(len=64) :: fname
      integer :: iu
      integer :: best_idx
      character(len=*), parameter :: labels(5) = [ &
                                     "k->i->j", "i->j->k", "j->k->i", "k->j->i", "j->i->k"]

      do m = 1, 5
         sum = 0.0_real64
         sumsq = 0.0_real64
         do idx = 1, ntests
            x = timings(idx, m)
            sum = sum + x
            sumsq = sumsq + x*x
         end do
         avg(m) = sum/real(ntests, real64)
         var(m) = sumsq/real(ntests, real64) - avg(m)*avg(m)   ! population variance
         if (var(m) < 0.0_real64) var(m) = 0.0_real64           ! guard tiny negatives
         std(m) = sqrt(var(m))
      end do

      ! Find fastest (smallest average)
      best_idx = 1
      do m = 2, 5
         if (avg(m) < avg(best_idx)) best_idx = m
      end do

      print *, "Average timings over ", ntests, " tests:"
      do m = 1, 5
         write (*, '(A10,": mean=",F12.6," s, var=",F12.6,", std=",F12.6)') &
            trim(labels(m)), avg(m), var(m), std(m)
      end do
      print *
      print *, "Fastest on average: ", trim(labels(best_idx))
      print *

      !-------------------------------------------------------
      ! Write detailed data to CSV file: nx_ny_ntests.csv
      !-------------------------------------------------------

      write (fname, '(I0,"_",I0,"_",I0,".csv")') nx, ny, ntests
      open (newunit=iu, file=fname, status="replace", action="write")

      ! Header
      write (iu, '(A)') "Nz,k->i->j,i->j->k,j->k->i,k->j->i,j->i->k"

      ! Data rows
      do idx = 1, ntests
         write (iu, '(I0,5(",",F12.6))') nz_values(idx), timings(idx, 1), timings(idx, 2), &
            timings(idx, 3), timings(idx, 4), timings(idx, 5)
      end do

      ! Summary rows (optional, but handy)
      write (iu, '(A)') "mean,"// &
         trim(adjustl(to_str(avg(1))))//","//trim(adjustl(to_str(avg(2))))//","// &
         trim(adjustl(to_str(avg(3))))//","//trim(adjustl(to_str(avg(4))))//","// &
         trim(adjustl(to_str(avg(5))))
      write (iu, '(A)') "var,"// &
         trim(adjustl(to_str(var(1))))//","//trim(adjustl(to_str(var(2))))//","// &
         trim(adjustl(to_str(var(3))))//","//trim(adjustl(to_str(var(4))))//","// &
         trim(adjustl(to_str(var(5))))
      write (iu, '(A)') "std,"// &
         trim(adjustl(to_str(std(1))))//","//trim(adjustl(to_str(std(2))))//","// &
         trim(adjustl(to_str(std(3))))//","//trim(adjustl(to_str(std(4))))//","// &
         trim(adjustl(to_str(std(5))))

      close (iu)

      print *, "CSV data written to file: ", trim(fname)

   end block
   ! print *
   ! print *, "Nz,k->i->j,i->j->k,j->k->i,k->j->i,j->i->k"
   ! do idx = 1, ntests
   !    write(*,'(I5,5(",",F12.6))') nz_values(idx), timings(idx,1), timings(idx,2), &
   !                                  timings(idx,3), timings(idx,4), timings(idx,5)
   ! end do

contains

   subroutine reset_state(nx, ny, nz, u, unew, c1, b1, d1)
      integer, intent(in) :: nx, ny, nz
      real(real64), intent(in)    :: u(nx, ny, nz)
      real(real64), intent(inout) :: unew(nx, ny, nz), c1(nx, ny, nz)
      real(real64), intent(inout) :: b1(nx, ny), d1(nx, ny)
      integer :: ii, jj, kk

      do concurrent(ii=1:nx, jj=1:ny, kk=1:nz)
         unew(ii, jj, kk) = u(ii, jj, kk)
         c1(ii, jj, kk) = 0.0_real64
         if (kk == 1) then
            b1(ii, jj) = 0.0_real64
            d1(ii, jj) = 0.0_real64
         end if
      end do
   end subroutine reset_state
   function to_str(x) result(s)
      use iso_fortran_env, only: real64
      real(real64), intent(in) :: x
      character(len=32) :: s
      write (s, '(F12.6)') x
   end function to_str
end program loop_order_sweep_do_concurrent_tridiag_reuse

