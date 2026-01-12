! sw_cfl2.f90 - Shallow water mini app with SECOND-ORDER extrapolation
! Includes gradient reconstruction with minmod limiter
! This is much closer to real ANUGA computational patterns

program sw_cfl2
    use omp_lib
    implicit none
    integer, parameter :: dp = SELECTED_REAL_KIND(15, 307)

    ! Domain type with allocatable arrays
    type :: domain_t
        integer :: number_of_elements

        ! Centroid values (per triangle)
        real(dp), allocatable :: stage_centroid_values(:)
        real(dp), allocatable :: xmom_centroid_values(:)
        real(dp), allocatable :: ymom_centroid_values(:)
        real(dp), allocatable :: bed_centroid_values(:)
        real(dp), allocatable :: height_centroid_values(:)

        ! Explicit update arrays
        real(dp), allocatable :: stage_explicit_update(:)
        real(dp), allocatable :: xmom_explicit_update(:)
        real(dp), allocatable :: ymom_explicit_update(:)

        ! Edge values (3 per triangle) - now computed via 2nd order extrapolation
        real(dp), allocatable :: stage_edge_values(:)
        real(dp), allocatable :: height_edge_values(:)
        real(dp), allocatable :: xmom_edge_values(:)
        real(dp), allocatable :: ymom_edge_values(:)

        ! Gradients at centroids (for 2nd order reconstruction)
        real(dp), allocatable :: stage_x_gradient(:)
        real(dp), allocatable :: stage_y_gradient(:)
        real(dp), allocatable :: xmom_x_gradient(:)
        real(dp), allocatable :: xmom_y_gradient(:)
        real(dp), allocatable :: ymom_x_gradient(:)
        real(dp), allocatable :: ymom_y_gradient(:)
        real(dp), allocatable :: height_x_gradient(:)
        real(dp), allocatable :: height_y_gradient(:)

        ! Mesh geometry
        real(dp), allocatable :: centroid_x(:)        ! x-coord of centroid
        real(dp), allocatable :: centroid_y(:)        ! y-coord of centroid
        real(dp), allocatable :: edge_midpoint_x(:)   ! x-coord of edge midpoints (3 per tri)
        real(dp), allocatable :: edge_midpoint_y(:)   ! y-coord of edge midpoints (3 per tri)

        ! Mesh connectivity
        integer, allocatable :: neighbours(:)
        real(dp), allocatable :: edgelengths(:)
        real(dp), allocatable :: normals(:)
        real(dp), allocatable :: areas(:)
        real(dp), allocatable :: radii(:)
        real(dp), allocatable :: max_speed(:)

        ! Constants
        real(dp) :: g
        real(dp) :: epsilon
        real(dp) :: minimum_allowed_height

        ! CFL parameters
        real(dp) :: cfl
        real(dp) :: domain_length
        real(dp) :: char_length
    end type domain_t

    ! Local variables
    type(domain_t) :: D
    integer :: grid_size, niter, n
    integer :: iter, nargs
    real(dp) :: dt, sim_time
    real(dp) :: t0, t_init, t_to_gpu, t_compute
    real(dp) :: target_time, estimated_steps, estimated_wallclock
    real(dp) :: max_speed_global, dt_cfl
    real(dp) :: time_per_step, steps_per_second
    real(dp) :: initial_height, domain_length
    character(len=32) :: arg

    ! Parse command line arguments
    nargs = command_argument_count()
    if (nargs < 1 .or. nargs > 4) then
        print '(A)', 'Usage: sw_cfl2 N [niter] [domain_length_km] [initial_height_m]'
        print '(A)', '  N                = grid size (creates 2*(N-1)^2 triangles)'
        print '(A)', '  niter            = number of iterations to benchmark (default: 1000)'
        print '(A)', '  domain_length_km = physical domain size in km (default: 100)'
        print '(A)', '  initial_height_m = initial water depth in meters (default: 10)'
        print '(A)', ''
        print '(A)', 'This version uses SECOND-ORDER extrapolation with minmod limiter'
        stop 1
    end if

    call get_command_argument(1, arg)
    read(arg, *) grid_size

    niter = 1000
    domain_length = 100.0d3  ! 100 km default
    initial_height = 10.0d0  ! 10 m default

    if (nargs >= 2) then
        call get_command_argument(2, arg)
        read(arg, *) niter
    end if
    if (nargs >= 3) then
        call get_command_argument(3, arg)
        read(arg, *) domain_length
        domain_length = domain_length * 1000.0d0
    end if
    if (nargs >= 4) then
        call get_command_argument(4, arg)
        read(arg, *) initial_height
    end if

    if (grid_size < 3) then
        print '(A)', 'Error: Grid size must be at least 3'
        stop 1
    end if

    n = 2 * (grid_size - 1) * (grid_size - 1)
    target_time = 2.0d0 * 24.0d0 * 3600.0d0  ! 2 days

    print '(A)', '============================================================'
    print '(A)', '  SHALLOW WATER CFL TIMING - SECOND ORDER'
    print '(A)', '============================================================'
    print '(A)', ''
    print '(A)', 'Physical Setup:'
    print '(A,F10.2,A,F10.2,A)', '  Domain size:      ', domain_length/1000.0d0, ' km x ', &
          domain_length/1000.0d0, ' km'
    print '(A,F10.2,A)', '  Initial depth:    ', initial_height, ' m'
    print '(A,F10.2,A)', '  Wave speed:       ', sqrt(9.81d0 * initial_height), ' m/s'
    print '(A)', ''
    print '(A)', 'Mesh:'
    print '(A,I0,A,I0)', '  Grid:             ', grid_size, ' x ', grid_size
    print '(A,I0)', '  Triangles:        ', n
    print '(A)', ''
    print '(A)', 'Reconstruction:'
    print '(A)', '  Order:            SECOND (gradient + minmod limiter)'
    print '(A)', ''

    ! Allocate domain
    call allocate_domain(D, n)

    D%domain_length = domain_length
    D%cfl = 0.9d0
    D%char_length = domain_length / sqrt(real(n, dp) / 2.0d0)

    print '(A,F12.4,A)', '  Element size:     ', D%char_length, ' m'
    print '(A)', ''

    ! Initialize
    t0 = omp_get_wtime()
    call generate_mesh(D, grid_size, n)
    call init_quantities(D, n, initial_height)
    t_init = omp_get_wtime() - t0

    ! Transfer to GPU
    t0 = omp_get_wtime()
    call map_to_gpu(D, n)
    t_to_gpu = omp_get_wtime() - t0

    ! Run benchmark
    sim_time = 0.0d0
    dt = 0.0d0

    print '(A,I0,A)', 'Running ', niter, ' iterations (2nd order)...'
    print '(A)', ''

    t0 = omp_get_wtime()
    do iter = 1, niter
        ! 1. Compute gradients at centroids
        call compute_gradients_gpu(D, n)

        ! 2. Extrapolate to edges using gradients (2nd order with limiter)
        call extrapolate_second_order_gpu(D, n)

        ! 3. Compute fluxes
        call compute_fluxes_gpu(D, n, max_speed_global)

        ! CFL timestep
        if (max_speed_global > D%epsilon) then
            dt_cfl = D%cfl * D%char_length / max_speed_global
        else
            dt_cfl = D%cfl * D%char_length / sqrt(D%g * initial_height)
        end if
        dt = dt_cfl

        ! 4. Protect (wetting/drying)
        call protect_gpu(D, n)

        ! 5. Update
        call update_gpu(D, n, dt)
        sim_time = sim_time + dt

        if (mod(iter, 100) == 0) then
            call print_progress(iter, niter, omp_get_wtime() - t0, sim_time)
        end if
    end do
    t_compute = omp_get_wtime() - t0
    print '(A)', ''
    print '(A)', ''

    call unmap_from_gpu(D, n)

    ! Results
    time_per_step = t_compute / real(niter, dp)
    steps_per_second = 1.0d0 / time_per_step
    dt = sim_time / real(niter, dp)
    estimated_steps = target_time / dt
    estimated_wallclock = estimated_steps * time_per_step

    print '(A)', '============================================================'
    print '(A)', '  BENCHMARK RESULTS (SECOND ORDER)'
    print '(A)', '============================================================'
    print '(A)', ''
    print '(A,I0,A)', 'Ran ', niter, ' iterations:'
    print '(A,F12.4,A)', '  Wall-clock time:  ', t_compute, ' s'
    print '(A,F12.6,A)', '  Time per step:    ', time_per_step * 1000.0d0, ' ms'
    print '(A,F12.2)', '  Steps per second: ', steps_per_second
    print '(A)', ''
    print '(A)', 'CFL Timestep:'
    print '(A,ES12.4,A)', '  Average dt:       ', dt, ' s'
    print '(A,F12.4,A)', '  Simulated time:   ', sim_time, ' s'
    print '(A)', ''

    print '(A)', '============================================================'
    print '(A)', '  2-DAY SIMULATION ESTIMATE'
    print '(A)', '============================================================'
    print '(A)', ''
    print '(A,ES14.4)', '  Estimated steps:  ', estimated_steps
    print '(A)', ''
    print '(A)', '  Estimated wall-clock time:'

    if (estimated_wallclock < 60.0d0) then
        print '(A,F14.2,A)', '                    ', estimated_wallclock, ' seconds'
    else if (estimated_wallclock < 3600.0d0) then
        print '(A,F14.2,A)', '                    ', estimated_wallclock / 60.0d0, ' minutes'
    else if (estimated_wallclock < 86400.0d0) then
        print '(A,F14.2,A)', '                    ', estimated_wallclock / 3600.0d0, ' hours'
    else
        print '(A,F14.2,A)', '                    ', estimated_wallclock / 86400.0d0, ' days'
    end if
    print '(A)', ''

    if (estimated_wallclock < 3600.0d0) then
        print '(A)', '  EXCELLENT - Under 1 hour'
    else if (estimated_wallclock < 86400.0d0) then
        print '(A)', '  GOOD - Under 1 day'
    else if (estimated_wallclock < 7.0d0 * 86400.0d0) then
        print '(A)', '  ACCEPTABLE - Under 1 week'
    else
        print '(A)', '  CHALLENGING - More than 1 week'
    end if
    print '(A)', ''

    call deallocate_domain(D)

contains

    subroutine allocate_domain(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        D%number_of_elements = n
        D%g = 9.81d0
        D%epsilon = 1.0d-12
        D%minimum_allowed_height = 1.0d-6

        ! Centroid values
        allocate(D%stage_centroid_values(n))
        allocate(D%xmom_centroid_values(n))
        allocate(D%ymom_centroid_values(n))
        allocate(D%bed_centroid_values(n))
        allocate(D%height_centroid_values(n))

        ! Update arrays
        allocate(D%stage_explicit_update(n))
        allocate(D%xmom_explicit_update(n))
        allocate(D%ymom_explicit_update(n))

        ! Edge values (now including momentum)
        allocate(D%stage_edge_values(3*n))
        allocate(D%height_edge_values(3*n))
        allocate(D%xmom_edge_values(3*n))
        allocate(D%ymom_edge_values(3*n))

        ! Gradients
        allocate(D%stage_x_gradient(n))
        allocate(D%stage_y_gradient(n))
        allocate(D%xmom_x_gradient(n))
        allocate(D%xmom_y_gradient(n))
        allocate(D%ymom_x_gradient(n))
        allocate(D%ymom_y_gradient(n))
        allocate(D%height_x_gradient(n))
        allocate(D%height_y_gradient(n))

        ! Geometry
        allocate(D%centroid_x(n))
        allocate(D%centroid_y(n))
        allocate(D%edge_midpoint_x(3*n))
        allocate(D%edge_midpoint_y(3*n))

        ! Mesh connectivity
        allocate(D%neighbours(3*n))
        allocate(D%edgelengths(3*n))
        allocate(D%normals(6*n))
        allocate(D%areas(n))
        allocate(D%radii(n))
        allocate(D%max_speed(n))
    end subroutine allocate_domain

    subroutine deallocate_domain(D)
        type(domain_t), intent(inout) :: D

        deallocate(D%stage_centroid_values, D%xmom_centroid_values)
        deallocate(D%ymom_centroid_values, D%bed_centroid_values)
        deallocate(D%height_centroid_values)
        deallocate(D%stage_explicit_update, D%xmom_explicit_update)
        deallocate(D%ymom_explicit_update)
        deallocate(D%stage_edge_values, D%height_edge_values)
        deallocate(D%xmom_edge_values, D%ymom_edge_values)
        deallocate(D%stage_x_gradient, D%stage_y_gradient)
        deallocate(D%xmom_x_gradient, D%xmom_y_gradient)
        deallocate(D%ymom_x_gradient, D%ymom_y_gradient)
        deallocate(D%height_x_gradient, D%height_y_gradient)
        deallocate(D%centroid_x, D%centroid_y)
        deallocate(D%edge_midpoint_x, D%edge_midpoint_y)
        deallocate(D%neighbours, D%edgelengths, D%normals)
        deallocate(D%areas, D%radii, D%max_speed)
    end subroutine deallocate_domain

    subroutine generate_mesh(D, grid_size, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: grid_size, n

        integer :: nx, ny, k, cell, tri_in_cell, cell_x, cell_y, i
        real(dp) :: dx, dy, area, edgelen, radius
        real(dp) :: x0, y0, x1, y1, x2, y2, cx, cy

        nx = grid_size
        ny = grid_size
        dx = D%domain_length / (nx - 1)
        dy = D%domain_length / (ny - 1)

        area = 0.5d0 * dx * dy
        edgelen = dx
        radius = area / (1.5d0 * edgelen)

        !$omp parallel do private(i, cell, tri_in_cell, cell_x, cell_y, &
        !$omp&                    x0, y0, x1, y1, x2, y2, cx, cy)
        do k = 1, n
            D%areas(k) = area
            D%radii(k) = radius

            do i = 1, 3
                D%edgelengths(3*(k-1) + i) = edgelen
            end do

            D%normals(6*(k-1) + 1) = 1.0d0
            D%normals(6*(k-1) + 2) = 0.0d0
            D%normals(6*(k-1) + 3) = 0.0d0
            D%normals(6*(k-1) + 4) = 1.0d0
            D%normals(6*(k-1) + 5) = -0.707d0
            D%normals(6*(k-1) + 6) = -0.707d0

            ! Compute cell indices
            cell = (k - 1) / 2
            tri_in_cell = mod(k - 1, 2)
            cell_x = mod(cell, nx - 1)
            cell_y = cell / (nx - 1)

            ! Vertex coordinates for this triangle
            if (tri_in_cell == 0) then
                ! Lower-left triangle
                x0 = cell_x * dx
                y0 = cell_y * dy
                x1 = (cell_x + 1) * dx
                y1 = cell_y * dy
                x2 = cell_x * dx
                y2 = (cell_y + 1) * dy
            else
                ! Upper-right triangle
                x0 = (cell_x + 1) * dx
                y0 = (cell_y + 1) * dy
                x1 = cell_x * dx
                y1 = (cell_y + 1) * dy
                x2 = (cell_x + 1) * dx
                y2 = cell_y * dy
            end if

            ! Centroid
            cx = (x0 + x1 + x2) / 3.0d0
            cy = (y0 + y1 + y2) / 3.0d0
            D%centroid_x(k) = cx
            D%centroid_y(k) = cy

            ! Edge midpoints (edge i is opposite vertex i)
            D%edge_midpoint_x(3*(k-1) + 1) = (x1 + x2) / 2.0d0
            D%edge_midpoint_y(3*(k-1) + 1) = (y1 + y2) / 2.0d0
            D%edge_midpoint_x(3*(k-1) + 2) = (x0 + x2) / 2.0d0
            D%edge_midpoint_y(3*(k-1) + 2) = (y0 + y2) / 2.0d0
            D%edge_midpoint_x(3*(k-1) + 3) = (x0 + x1) / 2.0d0
            D%edge_midpoint_y(3*(k-1) + 3) = (y0 + y1) / 2.0d0

            ! Neighbours
            if (tri_in_cell == 0) then
                D%neighbours(3*(k-1) + 1) = k + 1
                if (cell_y > 0) then
                    D%neighbours(3*(k-1) + 2) = 2 * ((cell_y - 1) * (nx - 1) + cell_x) + 2
                else
                    D%neighbours(3*(k-1) + 2) = -1
                end if
                if (cell_x > 0) then
                    D%neighbours(3*(k-1) + 3) = 2 * (cell_y * (nx - 1) + (cell_x - 1)) + 2
                else
                    D%neighbours(3*(k-1) + 3) = -1
                end if
            else
                D%neighbours(3*(k-1) + 1) = k - 1
                if (cell_y < ny - 2) then
                    D%neighbours(3*(k-1) + 2) = 2 * ((cell_y + 1) * (nx - 1) + cell_x) + 1
                else
                    D%neighbours(3*(k-1) + 2) = -1
                end if
                if (cell_x < nx - 2) then
                    D%neighbours(3*(k-1) + 3) = 2 * (cell_y * (nx - 1) + (cell_x + 1)) + 1
                else
                    D%neighbours(3*(k-1) + 3) = -1
                end if
            end if
        end do
        !$omp end parallel do
    end subroutine generate_mesh

    subroutine init_quantities(D, n, initial_height)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n
        real(dp), intent(in) :: initial_height
        integer :: k, i

        !$omp parallel do private(i)
        do k = 1, n
            D%bed_centroid_values(k) = 0.0d0
            D%stage_centroid_values(k) = initial_height
            D%height_centroid_values(k) = initial_height
            D%xmom_centroid_values(k) = 0.0d0
            D%ymom_centroid_values(k) = 0.0d0
            D%stage_explicit_update(k) = 0.0d0
            D%xmom_explicit_update(k) = 0.0d0
            D%ymom_explicit_update(k) = 0.0d0
            D%max_speed(k) = 0.0d0

            ! Initialize gradients to zero
            D%stage_x_gradient(k) = 0.0d0
            D%stage_y_gradient(k) = 0.0d0
            D%xmom_x_gradient(k) = 0.0d0
            D%xmom_y_gradient(k) = 0.0d0
            D%ymom_x_gradient(k) = 0.0d0
            D%ymom_y_gradient(k) = 0.0d0
            D%height_x_gradient(k) = 0.0d0
            D%height_y_gradient(k) = 0.0d0

            do i = 1, 3
                D%stage_edge_values(3*(k-1) + i) = initial_height
                D%height_edge_values(3*(k-1) + i) = initial_height
                D%xmom_edge_values(3*(k-1) + i) = 0.0d0
                D%ymom_edge_values(3*(k-1) + i) = 0.0d0
            end do
        end do
        !$omp end parallel do
    end subroutine init_quantities

    subroutine map_to_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        !$omp target enter data map(to: D, &
        !$omp&    D%stage_centroid_values(1:n), &
        !$omp&    D%xmom_centroid_values(1:n), &
        !$omp&    D%ymom_centroid_values(1:n), &
        !$omp&    D%bed_centroid_values(1:n), &
        !$omp&    D%height_centroid_values(1:n), &
        !$omp&    D%stage_explicit_update(1:n), &
        !$omp&    D%xmom_explicit_update(1:n), &
        !$omp&    D%ymom_explicit_update(1:n), &
        !$omp&    D%stage_edge_values(1:3*n), &
        !$omp&    D%height_edge_values(1:3*n), &
        !$omp&    D%xmom_edge_values(1:3*n), &
        !$omp&    D%ymom_edge_values(1:3*n), &
        !$omp&    D%stage_x_gradient(1:n), &
        !$omp&    D%stage_y_gradient(1:n), &
        !$omp&    D%xmom_x_gradient(1:n), &
        !$omp&    D%xmom_y_gradient(1:n), &
        !$omp&    D%ymom_x_gradient(1:n), &
        !$omp&    D%ymom_y_gradient(1:n), &
        !$omp&    D%height_x_gradient(1:n), &
        !$omp&    D%height_y_gradient(1:n), &
        !$omp&    D%centroid_x(1:n), &
        !$omp&    D%centroid_y(1:n), &
        !$omp&    D%edge_midpoint_x(1:3*n), &
        !$omp&    D%edge_midpoint_y(1:3*n), &
        !$omp&    D%neighbours(1:3*n), &
        !$omp&    D%edgelengths(1:3*n), &
        !$omp&    D%normals(1:6*n), &
        !$omp&    D%areas(1:n), &
        !$omp&    D%radii(1:n), &
        !$omp&    D%max_speed(1:n))
    end subroutine map_to_gpu

    subroutine unmap_from_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        !$omp target exit data map(delete: D, &
        !$omp&    D%stage_centroid_values(1:n), &
        !$omp&    D%xmom_centroid_values(1:n), &
        !$omp&    D%ymom_centroid_values(1:n), &
        !$omp&    D%bed_centroid_values(1:n), &
        !$omp&    D%height_centroid_values(1:n), &
        !$omp&    D%stage_explicit_update(1:n), &
        !$omp&    D%xmom_explicit_update(1:n), &
        !$omp&    D%ymom_explicit_update(1:n), &
        !$omp&    D%stage_edge_values(1:3*n), &
        !$omp&    D%height_edge_values(1:3*n), &
        !$omp&    D%xmom_edge_values(1:3*n), &
        !$omp&    D%ymom_edge_values(1:3*n), &
        !$omp&    D%stage_x_gradient(1:n), &
        !$omp&    D%stage_y_gradient(1:n), &
        !$omp&    D%xmom_x_gradient(1:n), &
        !$omp&    D%xmom_y_gradient(1:n), &
        !$omp&    D%ymom_x_gradient(1:n), &
        !$omp&    D%ymom_y_gradient(1:n), &
        !$omp&    D%height_x_gradient(1:n), &
        !$omp&    D%height_y_gradient(1:n), &
        !$omp&    D%centroid_x(1:n), &
        !$omp&    D%centroid_y(1:n), &
        !$omp&    D%edge_midpoint_x(1:3*n), &
        !$omp&    D%edge_midpoint_y(1:3*n), &
        !$omp&    D%neighbours(1:3*n), &
        !$omp&    D%edgelengths(1:3*n), &
        !$omp&    D%normals(1:6*n), &
        !$omp&    D%areas(1:n), &
        !$omp&    D%radii(1:n), &
        !$omp&    D%max_speed(1:n))
    end subroutine unmap_from_gpu

    ! Minmod limiter function
    pure real(dp) function minmod(a, b)
        real(dp), intent(in) :: a, b
        if (a * b <= 0.0d0) then
            minmod = 0.0d0
        else if (abs(a) < abs(b)) then
            minmod = a
        else
            minmod = b
        end if
    end function minmod

    ! Compute gradients using neighbours (Green-Gauss style)
    subroutine compute_gradients_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        integer :: k, i, nb
        real(dp) :: cx, cy, nx_c, ny_c
        real(dp) :: dstage_dx, dstage_dy, dxmom_dx, dxmom_dy
        real(dp) :: dymom_dx, dymom_dy, dheight_dx, dheight_dy
        real(dp) :: stage_k, xmom_k, ymom_k, height_k
        real(dp) :: stage_nb, xmom_nb, ymom_nb, height_nb
        real(dp) :: dx_nb, dy_nb, dist_sq, weight, sum_weight

        !$omp target teams distribute parallel do &
        !$omp& private(i, nb, cx, cy, nx_c, ny_c, &
        !$omp&         dstage_dx, dstage_dy, dxmom_dx, dxmom_dy, &
        !$omp&         dymom_dx, dymom_dy, dheight_dx, dheight_dy, &
        !$omp&         stage_k, xmom_k, ymom_k, height_k, &
        !$omp&         stage_nb, xmom_nb, ymom_nb, height_nb, &
        !$omp&         dx_nb, dy_nb, dist_sq, weight, sum_weight)
        do k = 1, n
            cx = D%centroid_x(k)
            cy = D%centroid_y(k)
            stage_k = D%stage_centroid_values(k)
            xmom_k = D%xmom_centroid_values(k)
            ymom_k = D%ymom_centroid_values(k)
            height_k = D%height_centroid_values(k)

            dstage_dx = 0.0d0; dstage_dy = 0.0d0
            dxmom_dx = 0.0d0;  dxmom_dy = 0.0d0
            dymom_dx = 0.0d0;  dymom_dy = 0.0d0
            dheight_dx = 0.0d0; dheight_dy = 0.0d0
            sum_weight = 0.0d0

            do i = 1, 3
                nb = D%neighbours(3*(k-1) + i)
                if (nb >= 1) then
                    nx_c = D%centroid_x(nb)
                    ny_c = D%centroid_y(nb)
                    dx_nb = nx_c - cx
                    dy_nb = ny_c - cy
                    dist_sq = dx_nb * dx_nb + dy_nb * dy_nb

                    if (dist_sq > 1.0d-20) then
                        weight = 1.0d0 / sqrt(dist_sq)
                        sum_weight = sum_weight + weight

                        stage_nb = D%stage_centroid_values(nb)
                        xmom_nb = D%xmom_centroid_values(nb)
                        ymom_nb = D%ymom_centroid_values(nb)
                        height_nb = D%height_centroid_values(nb)

                        ! Weighted least-squares gradient contribution
                        dstage_dx = dstage_dx + weight * (stage_nb - stage_k) * dx_nb / dist_sq
                        dstage_dy = dstage_dy + weight * (stage_nb - stage_k) * dy_nb / dist_sq
                        dxmom_dx = dxmom_dx + weight * (xmom_nb - xmom_k) * dx_nb / dist_sq
                        dxmom_dy = dxmom_dy + weight * (xmom_nb - xmom_k) * dy_nb / dist_sq
                        dymom_dx = dymom_dx + weight * (ymom_nb - ymom_k) * dx_nb / dist_sq
                        dymom_dy = dymom_dy + weight * (ymom_nb - ymom_k) * dy_nb / dist_sq
                        dheight_dx = dheight_dx + weight * (height_nb - height_k) * dx_nb / dist_sq
                        dheight_dy = dheight_dy + weight * (height_nb - height_k) * dy_nb / dist_sq
                    end if
                end if
            end do

            if (sum_weight > 1.0d-20) then
                D%stage_x_gradient(k) = dstage_dx / sum_weight
                D%stage_y_gradient(k) = dstage_dy / sum_weight
                D%xmom_x_gradient(k) = dxmom_dx / sum_weight
                D%xmom_y_gradient(k) = dxmom_dy / sum_weight
                D%ymom_x_gradient(k) = dymom_dx / sum_weight
                D%ymom_y_gradient(k) = dymom_dy / sum_weight
                D%height_x_gradient(k) = dheight_dx / sum_weight
                D%height_y_gradient(k) = dheight_dy / sum_weight
            else
                D%stage_x_gradient(k) = 0.0d0
                D%stage_y_gradient(k) = 0.0d0
                D%xmom_x_gradient(k) = 0.0d0
                D%xmom_y_gradient(k) = 0.0d0
                D%ymom_x_gradient(k) = 0.0d0
                D%ymom_y_gradient(k) = 0.0d0
                D%height_x_gradient(k) = 0.0d0
                D%height_y_gradient(k) = 0.0d0
            end if
        end do
        !$omp end target teams distribute parallel do
    end subroutine compute_gradients_gpu

    ! Second-order extrapolation with minmod limiter
    subroutine extrapolate_second_order_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        integer :: k, i, ki, nb
        real(dp) :: cx, cy, ex, ey, dx_e, dy_e
        real(dp) :: stage_c, xmom_c, ymom_c, height_c
        real(dp) :: stage_grad_x, stage_grad_y
        real(dp) :: xmom_grad_x, xmom_grad_y
        real(dp) :: ymom_grad_x, ymom_grad_y
        real(dp) :: height_grad_x, height_grad_y
        real(dp) :: stage_e, xmom_e, ymom_e, height_e
        real(dp) :: stage_nb, xmom_nb, ymom_nb, height_nb
        real(dp) :: dstage, dxmom, dymom, dheight
        real(dp) :: dstage_nb, dxmom_nb, dymom_nb, dheight_nb

        !$omp target teams distribute parallel do &
        !$omp& private(i, ki, nb, cx, cy, ex, ey, dx_e, dy_e, &
        !$omp&         stage_c, xmom_c, ymom_c, height_c, &
        !$omp&         stage_grad_x, stage_grad_y, &
        !$omp&         xmom_grad_x, xmom_grad_y, &
        !$omp&         ymom_grad_x, ymom_grad_y, &
        !$omp&         height_grad_x, height_grad_y, &
        !$omp&         stage_e, xmom_e, ymom_e, height_e, &
        !$omp&         stage_nb, xmom_nb, ymom_nb, height_nb, &
        !$omp&         dstage, dxmom, dymom, dheight, &
        !$omp&         dstage_nb, dxmom_nb, dymom_nb, dheight_nb)
        do k = 1, n
            cx = D%centroid_x(k)
            cy = D%centroid_y(k)

            stage_c = D%stage_centroid_values(k)
            xmom_c = D%xmom_centroid_values(k)
            ymom_c = D%ymom_centroid_values(k)
            height_c = D%height_centroid_values(k)

            stage_grad_x = D%stage_x_gradient(k)
            stage_grad_y = D%stage_y_gradient(k)
            xmom_grad_x = D%xmom_x_gradient(k)
            xmom_grad_y = D%xmom_y_gradient(k)
            ymom_grad_x = D%ymom_x_gradient(k)
            ymom_grad_y = D%ymom_y_gradient(k)
            height_grad_x = D%height_x_gradient(k)
            height_grad_y = D%height_y_gradient(k)

            do i = 1, 3
                ki = 3*(k-1) + i
                ex = D%edge_midpoint_x(ki)
                ey = D%edge_midpoint_y(ki)
                dx_e = ex - cx
                dy_e = ey - cy

                ! Unlimited extrapolation
                dstage = stage_grad_x * dx_e + stage_grad_y * dy_e
                dxmom = xmom_grad_x * dx_e + xmom_grad_y * dy_e
                dymom = ymom_grad_x * dx_e + ymom_grad_y * dy_e
                dheight = height_grad_x * dx_e + height_grad_y * dy_e

                ! Apply minmod limiter using neighbor value
                nb = D%neighbours(ki)
                if (nb >= 1) then
                    stage_nb = D%stage_centroid_values(nb)
                    xmom_nb = D%xmom_centroid_values(nb)
                    ymom_nb = D%ymom_centroid_values(nb)
                    height_nb = D%height_centroid_values(nb)

                    dstage_nb = stage_nb - stage_c
                    dxmom_nb = xmom_nb - xmom_c
                    dymom_nb = ymom_nb - ymom_c
                    dheight_nb = height_nb - height_c

                    ! Minmod between gradient-based and neighbor-based differences
                    if (dstage * dstage_nb > 0.0d0) then
                        if (abs(dstage) > abs(dstage_nb)) dstage = dstage_nb
                    else
                        dstage = 0.0d0
                    end if
                    if (dxmom * dxmom_nb > 0.0d0) then
                        if (abs(dxmom) > abs(dxmom_nb)) dxmom = dxmom_nb
                    else
                        dxmom = 0.0d0
                    end if
                    if (dymom * dymom_nb > 0.0d0) then
                        if (abs(dymom) > abs(dymom_nb)) dymom = dymom_nb
                    else
                        dymom = 0.0d0
                    end if
                    if (dheight * dheight_nb > 0.0d0) then
                        if (abs(dheight) > abs(dheight_nb)) dheight = dheight_nb
                    else
                        dheight = 0.0d0
                    end if
                end if

                stage_e = stage_c + dstage
                xmom_e = xmom_c + dxmom
                ymom_e = ymom_c + dymom
                height_e = max(height_c + dheight, 0.0d0)

                D%stage_edge_values(ki) = stage_e
                D%xmom_edge_values(ki) = xmom_e
                D%ymom_edge_values(ki) = ymom_e
                D%height_edge_values(ki) = height_e
            end do
        end do
        !$omp end target teams distribute parallel do
    end subroutine extrapolate_second_order_gpu

    subroutine compute_fluxes_gpu(D, n, max_speed_out)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n
        real(dp), intent(out) :: max_speed_out

        integer :: k, i, ki, nb, ki_nb
        real(dp) :: g, epsilon
        real(dp) :: stage_accum, xmom_accum, ymom_accum, speed_max
        real(dp) :: h_left, h_right, uh_left, uh_right, vh_left, vh_right
        real(dp) :: edgelen, nx, ny, c_left, c_right, c_max
        real(dp) :: global_max_speed

        g = D%g
        epsilon = D%epsilon
        global_max_speed = 0.0d0

        !$omp target teams distribute parallel do &
        !$omp& private(i, ki, nb, ki_nb, stage_accum, xmom_accum, ymom_accum, speed_max, &
        !$omp&         h_left, h_right, uh_left, uh_right, vh_left, vh_right, &
        !$omp&         edgelen, nx, ny, c_left, c_right, c_max) &
        !$omp& reduction(max: global_max_speed)
        do k = 1, n
            stage_accum = 0.0d0
            xmom_accum = 0.0d0
            ymom_accum = 0.0d0
            speed_max = 0.0d0

            do i = 1, 3
                ki = 3*(k-1) + i
                nb = D%neighbours(ki)

                ! Left state (from this element's edge)
                h_left = D%height_edge_values(ki)
                uh_left = D%xmom_edge_values(ki)
                vh_left = D%ymom_edge_values(ki)

                ! Right state (from neighbor or boundary)
                if (nb >= 1) then
                    ! Find which edge of neighbor shares this edge
                    ! For structured mesh, it's typically edge 0
                    ki_nb = 3*(nb-1) + 1
                    h_right = D%height_edge_values(ki_nb)
                    uh_right = D%xmom_edge_values(ki_nb)
                    vh_right = D%ymom_edge_values(ki_nb)
                else
                    ! Reflective boundary
                    h_right = h_left
                    uh_right = -uh_left
                    vh_right = -vh_left
                end if

                edgelen = D%edgelengths(ki)
                nx = D%normals(6*(k-1) + 2*(i-1) + 1)
                ny = D%normals(6*(k-1) + 2*(i-1) + 2)

                c_left = sqrt(g * max(h_left, 0.0d0))
                c_right = sqrt(g * max(h_right, 0.0d0))
                c_max = max(c_left, c_right)

                ! Rusanov flux
                stage_accum = stage_accum + c_max * (h_left - h_right) * edgelen
                xmom_accum = xmom_accum + c_max * (uh_left - uh_right) * edgelen * nx
                ymom_accum = ymom_accum + c_max * (vh_left - vh_right) * edgelen * ny

                if (c_max > epsilon) speed_max = max(speed_max, c_max)
            end do

            D%stage_explicit_update(k) = stage_accum / D%areas(k)
            D%xmom_explicit_update(k) = xmom_accum / D%areas(k)
            D%ymom_explicit_update(k) = ymom_accum / D%areas(k)
            D%max_speed(k) = speed_max

            global_max_speed = max(global_max_speed, speed_max)
        end do
        !$omp end target teams distribute parallel do

        max_speed_out = global_max_speed
    end subroutine compute_fluxes_gpu

    subroutine protect_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        integer :: k
        real(dp) :: hc, minimum_allowed_height, bmin

        minimum_allowed_height = D%minimum_allowed_height

        !$omp target teams distribute parallel do private(hc, bmin)
        do k = 1, n
            hc = D%stage_centroid_values(k) - D%bed_centroid_values(k)

            if (hc < minimum_allowed_height) then
                D%xmom_centroid_values(k) = 0.0d0
                D%ymom_centroid_values(k) = 0.0d0

                if (hc <= 0.0d0) then
                    bmin = D%bed_centroid_values(k)
                    if (D%stage_centroid_values(k) < bmin) then
                        D%stage_centroid_values(k) = bmin
                    end if
                end if
            end if
            D%height_centroid_values(k) = max(D%stage_centroid_values(k) - D%bed_centroid_values(k), 0.0d0)
        end do
        !$omp end target teams distribute parallel do
    end subroutine protect_gpu

    subroutine update_gpu(D, n, dt)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n
        real(dp), intent(in) :: dt
        integer :: k

        !$omp target teams distribute parallel do
        do k = 1, n
            D%stage_centroid_values(k) = D%stage_centroid_values(k) + dt * D%stage_explicit_update(k)
            D%xmom_centroid_values(k) = D%xmom_centroid_values(k) + dt * D%xmom_explicit_update(k)
            D%ymom_centroid_values(k) = D%ymom_centroid_values(k) + dt * D%ymom_explicit_update(k)
        end do
        !$omp end target teams distribute parallel do
    end subroutine update_gpu

    subroutine print_progress(current, total, elapsed, sim_time)
        integer, intent(in) :: current, total
        real(dp), intent(in) :: elapsed, sim_time

        integer :: bar_width, filled, i
        real(dp) :: progress
        character(len=100) :: bar

        bar_width = 40
        progress = real(current, dp) / real(total, dp)
        filled = int(bar_width * progress)

        bar = ''
        do i = 1, bar_width
            if (i < filled) then
                bar = trim(bar) // '='
            else if (i == filled) then
                bar = trim(bar) // '>'
            else
                bar = trim(bar) // ' '
            end if
        end do

        write(*, '(A,A,A,I3,A,I0,A,I0,A,F8.2,A,F10.4,A)', advance='no') &
            char(13), '  [', trim(bar), '] ', int(progress * 100), '% (', &
            current, '/', total, ') ', elapsed, 's sim_t=', sim_time, 's'
    end subroutine print_progress

end program sw_cfl2
