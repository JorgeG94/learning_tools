!! Nonlinear shallow-water model on a C-grid, doubly periodic f-plane.
module swe_model
   !! Written in the roundabout "pool" style: every component owns its
   !! arrays, allocates them in `init`, maps them to the device once, and
   !! frees them in `destroy`. The difference is that the arrays are
   !! `dp2_t`, so nobody has to hand-maintain a `bytes()` sum: each
   !! component just registers a tag and the ledger does the accounting.
   !!
   !! Grid (C-grid, one halo cell, all arrays dimensioned (0:nx+1, 0:ny+1)):
   !!   h(i,j)  cell centre         total fluid depth
   !!   u(i,j)  east face  (i+1/2)  x velocity
   !!   v(i,j)  north face (j+1/2)  y velocity
   !!
   !! Time stepping is forward-backward: continuity first, then u with the
   !! new h, then v with the new h and new u (which also makes the explicit
   !! Coriolis term neutrally stable).
   use, intrinsic :: iso_fortran_env, only: dp => real64
   use mem_ledger, only: mem_register_tag
   use devarray, only: dp2_t
   implicit none
   private

   public :: swe_params_t, swe_state_t, swe_work_t
   public :: swe_step, swe_total_mass, swe_total_energy

   type :: swe_params_t
      integer :: nx = 512, ny = 512
      real(dp) :: dx = 1.0e3_dp, dy = 1.0e3_dp
      real(dp) :: g = 9.81_dp
      real(dp) :: f = 1.0e-4_dp
      real(dp) :: H0 = 100.0_dp       !! resting depth [m]
      real(dp) :: bump = 1.0_dp       !! initial Gaussian amplitude [m]
      real(dp) :: cfl = 0.5_dp
      real(dp) :: dt = 0.0_dp         !! set by `swe_params_finalize`
   contains
      procedure :: finalize => swe_params_finalize
   end type swe_params_t

   type :: swe_state_t
      !! Prognostic variables.
      type(dp2_t) :: h, u, v
   contains
      procedure :: init => state_init
      procedure :: destroy => state_destroy
   end type swe_state_t

   type :: swe_work_t
      !! Scratch workspace: never needs to visit the host.
      type(dp2_t) :: fu, fv   !! mass fluxes h*u at u points and h*v at v points
   contains
      procedure :: init => work_init
      procedure :: destroy => work_destroy
   end type swe_work_t

contains

   subroutine swe_params_finalize(p)
      class(swe_params_t), intent(inout) :: p
      real(dp) :: c
      c = sqrt(p%g*(p%H0 + abs(p%bump)))
      p%dt = p%cfl*min(p%dx, p%dy)/c
   end subroutine swe_params_finalize

   ! ----------------------------------------------------------- lifecycle

   subroutine state_init(this, p)
      class(swe_state_t), intent(inout) :: this
      type(swe_params_t), intent(in) :: p
      integer :: tag, i, j
      integer :: lb(2), ub(2)
      real(dp) :: x, y, xc, yc, sig

      tag = mem_register_tag("swe_state")
      lb = [0, 0]
      ub = [p%nx + 1, p%ny + 1]
      call this%h%create("swe_state%h", ub, tag=tag, lb=lb)
      call this%u%create("swe_state%u", ub, tag=tag, lb=lb, fill=0.0_dp)
      call this%v%create("swe_state%v", ub, tag=tag, lb=lb, fill=0.0_dp)

      xc = 0.5_dp*p%nx*p%dx
      yc = 0.5_dp*p%ny*p%dy
      sig = 0.1_dp*min(p%nx*p%dx, p%ny*p%dy)
      do j = 0, p%ny + 1
         y = (j - 0.5_dp)*p%dy
         do i = 0, p%nx + 1
            x = (i - 0.5_dp)*p%dx
            this%h%data(i, j) = p%H0 + p%bump*exp(-((x - xc)**2 + (y - yc)**2)/(2.0_dp*sig**2))
         end do
      end do

      call this%h%to_device()
      call this%u%to_device()
      call this%v%to_device()
   end subroutine state_init

   subroutine state_destroy(this)
      class(swe_state_t), intent(inout) :: this
      call this%h%destroy()
      call this%u%destroy()
      call this%v%destroy()
   end subroutine state_destroy

   subroutine work_init(this, p)
      class(swe_work_t), intent(inout) :: this
      type(swe_params_t), intent(in) :: p
      integer :: tag
      tag = mem_register_tag("swe_workspace")
      call this%fu%create("swe_work%fu", [p%nx + 1, p%ny + 1], tag=tag, lb=[0, 0])
      call this%fv%create("swe_work%fv", [p%nx + 1, p%ny + 1], tag=tag, lb=[0, 0])
      call this%fu%to_device(copyin=.false.)
      call this%fv%to_device(copyin=.false.)
   end subroutine work_init

   subroutine work_destroy(this)
      class(swe_work_t), intent(inout) :: this
      call this%fu%destroy()
      call this%fv%destroy()
   end subroutine work_destroy

   ! ---------------------------------------------------------- time step
   ! The wrappers stop here: kernels receive plain explicit-shape arrays.

   subroutine swe_step(s, w, p)
      type(swe_state_t), intent(inout) :: s
      type(swe_work_t), intent(inout) :: w
      type(swe_params_t), intent(in) :: p
      integer :: nx, ny
      nx = p%nx
      ny = p%ny
      call fill_halo(s%h%data, nx, ny)
      call fill_halo(s%u%data, nx, ny)
      call fill_halo(s%v%data, nx, ny)
      call mass_flux(s%h%data, s%u%data, s%v%data, w%fu%data, w%fv%data, nx, ny)
      call continuity(s%h%data, w%fu%data, w%fv%data, nx, ny, p%dt, p%dx, p%dy)
      call fill_halo(s%h%data, nx, ny)
      call u_momentum(s%u%data, s%v%data, s%h%data, nx, ny, p%dt, p%dx, p%g, p%f)
      call fill_halo(s%u%data, nx, ny)
      call v_momentum(s%v%data, s%u%data, s%h%data, nx, ny, p%dt, p%dy, p%g, p%f)
   end subroutine swe_step

   subroutine fill_halo(a, nx, ny)
      integer, intent(in) :: nx, ny
      real(dp), intent(inout) :: a(0:nx + 1, 0:ny + 1)
      integer :: i, j
      !$acc parallel loop default(present)
      do j = 1, ny
         a(0, j) = a(nx, j)
         a(nx + 1, j) = a(1, j)
      end do
      !$acc parallel loop default(present)
      do i = 0, nx + 1
         a(i, 0) = a(i, ny)
         a(i, ny + 1) = a(i, 1)
      end do
   end subroutine fill_halo

   subroutine mass_flux(h, u, v, fu, fv, nx, ny)
      integer, intent(in) :: nx, ny
      real(dp), intent(in) :: h(0:nx + 1, 0:ny + 1), u(0:nx + 1, 0:ny + 1), v(0:nx + 1, 0:ny + 1)
      real(dp), intent(inout) :: fu(0:nx + 1, 0:ny + 1), fv(0:nx + 1, 0:ny + 1)
      integer :: i, j
      !$acc parallel loop collapse(2) default(present)
      do j = 0, ny
         do i = 0, nx
            fu(i, j) = 0.5_dp*(h(i, j) + h(i + 1, j))*u(i, j)
            fv(i, j) = 0.5_dp*(h(i, j) + h(i, j + 1))*v(i, j)
         end do
      end do
   end subroutine mass_flux

   subroutine continuity(h, fu, fv, nx, ny, dt, dx, dy)
      integer, intent(in) :: nx, ny
      real(dp), intent(inout) :: h(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: fu(0:nx + 1, 0:ny + 1), fv(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: dt, dx, dy
      integer :: i, j
      !$acc parallel loop collapse(2) default(present)
      do j = 1, ny
         do i = 1, nx
            h(i, j) = h(i, j) - dt*((fu(i, j) - fu(i - 1, j))/dx + (fv(i, j) - fv(i, j - 1))/dy)
         end do
      end do
   end subroutine continuity

   subroutine u_momentum(u, v, h, nx, ny, dt, dx, g, f)
      integer, intent(in) :: nx, ny
      real(dp), intent(inout) :: u(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: v(0:nx + 1, 0:ny + 1), h(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: dt, dx, g, f
      integer :: i, j
      real(dp) :: vbar
      !$acc parallel loop collapse(2) default(present) private(vbar)
      do j = 1, ny
         do i = 1, nx
            vbar = 0.25_dp*(v(i, j) + v(i + 1, j) + v(i, j - 1) + v(i + 1, j - 1))
            u(i, j) = u(i, j) + dt*(f*vbar - g*(h(i + 1, j) - h(i, j))/dx)
         end do
      end do
   end subroutine u_momentum

   subroutine v_momentum(v, u, h, nx, ny, dt, dy, g, f)
      integer, intent(in) :: nx, ny
      real(dp), intent(inout) :: v(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: u(0:nx + 1, 0:ny + 1), h(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: dt, dy, g, f
      integer :: i, j
      real(dp) :: ubar
      !$acc parallel loop collapse(2) default(present) private(ubar)
      do j = 1, ny
         do i = 1, nx
            ubar = 0.25_dp*(u(i, j) + u(i - 1, j) + u(i, j + 1) + u(i - 1, j + 1))
            v(i, j) = v(i, j) + dt*(-f*ubar - g*(h(i, j + 1) - h(i, j))/dy)
         end do
      end do
   end subroutine v_momentum

   ! -------------------------------------------------------- diagnostics

   real(dp) function swe_total_mass(s, p) result(m)
      type(swe_state_t), intent(in) :: s
      type(swe_params_t), intent(in) :: p
      m = mass_kernel(s%h%data, p%nx, p%ny)*p%dx*p%dy
   end function swe_total_mass

   real(dp) function mass_kernel(h, nx, ny) result(m)
      integer, intent(in) :: nx, ny
      real(dp), intent(in) :: h(0:nx + 1, 0:ny + 1)
      integer :: i, j
      m = 0.0_dp
      !$acc parallel loop collapse(2) default(present) reduction(+:m)
      do j = 1, ny
         do i = 1, nx
            m = m + h(i, j)
         end do
      end do
   end function mass_kernel

   real(dp) function swe_total_energy(s, p) result(e)
      !! Kinetic + available potential energy (approximate on the C-grid).
      type(swe_state_t), intent(in) :: s
      type(swe_params_t), intent(in) :: p
      e = energy_kernel(s%h%data, s%u%data, s%v%data, p%nx, p%ny, p%g, p%H0)*p%dx*p%dy
   end function swe_total_energy

   real(dp) function energy_kernel(h, u, v, nx, ny, g, H0) result(e)
      integer, intent(in) :: nx, ny
      real(dp), intent(in) :: h(0:nx + 1, 0:ny + 1), u(0:nx + 1, 0:ny + 1), v(0:nx + 1, 0:ny + 1)
      real(dp), intent(in) :: g, H0
      integer :: i, j
      e = 0.0_dp
      !$acc parallel loop collapse(2) default(present) reduction(+:e)
      do j = 1, ny
         do i = 1, nx
            e = e + 0.5_dp*h(i, j)*0.5_dp*(u(i, j)**2 + u(i - 1, j)**2 + v(i, j)**2 + v(i, j - 1)**2) &
                + 0.5_dp*g*(h(i, j) - H0)**2
         end do
      end do
   end function energy_kernel

end module swe_model
