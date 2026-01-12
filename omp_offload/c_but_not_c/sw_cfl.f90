! sw_cfl.f90 - Shallow water mini app with CFL-based timestepping
! Estimates wall-clock time needed for physical simulation (e.g., 2 days)
! Uses adaptive dt from CFL condition: dt = CFL * dx / max_wave_speed

program sw_cfl
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

        ! Edge values (3 per triangle)
        real(dp), allocatable :: stage_edge_values(:)
        real(dp), allocatable :: height_edge_values(:)

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
        real(dp) :: cfl              ! CFL number (0.5-0.9 typical)
        real(dp) :: domain_length    ! Physical domain size (meters)
        real(dp) :: char_length      ! Characteristic element size
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
        print '(A)', 'Usage: sw_cfl N [niter] [domain_length_km] [initial_height_m]'
        print '(A)', '  N                = grid size (creates 2*(N-1)^2 triangles)'
        print '(A)', '  niter            = number of iterations to benchmark (default: 1000)'
        print '(A)', '  domain_length_km = physical domain size in km (default: 100)'
        print '(A)', '  initial_height_m = initial water depth in meters (default: 10)'
        print '(A)', ''
        print '(A)', 'Example for Sydney Harbour scale:'
        print '(A)', '  ./sw_cfl 10000 1000 20 15'
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
        domain_length = domain_length * 1000.0d0  ! Convert km to m
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

    ! Target: 2-day simulation
    target_time = 2.0d0 * 24.0d0 * 3600.0d0  ! 172800 seconds

    print '(A)', '============================================================'
    print '(A)', '  SHALLOW WATER CFL TIMING ESTIMATOR'
    print '(A)', '============================================================'
    print '(A)', ''
    print '(A)', 'Physical Setup:'
    print '(A,F10.2,A,F10.2,A)', '  Domain size:      ', domain_length/1000.0d0, ' km x ', domain_length/1000.0d0, ' km'
    print '(A,F10.2,A)', '  Initial depth:    ', initial_height, ' m'
    print '(A,F10.2,A)', '  Gravity:           ', 9.81d0, ' m/s^2'
    print '(A,F10.2,A)', '  Wave speed:        ', sqrt(9.81d0 * initial_height), ' m/s (shallow water)'
    print '(A)', ''
    print '(A)', 'Mesh:'
    print '(A,I0,A,I0)', '  Grid:             ', grid_size, ' x ', grid_size
    print '(A,I0)', '  Triangles:        ', n
    print '(A,ES12.2)', '  Elements:         ', real(n, dp)
    print '(A)', ''
    print '(A)', 'Target Simulation:'
    print '(A,F10.2,A)', '  Duration:         ', target_time / 3600.0d0, ' hours (2 days)'
    print '(A)', ''

    ! Allocate domain
    call allocate_domain(D, n)

    ! Set physical parameters
    D%domain_length = domain_length
    D%cfl = 0.9d0  ! Typical CFL number

    ! Characteristic length from domain and elements
    ! For triangular mesh: area ~ (L/sqrt(N))^2, so char_length ~ L/sqrt(N)
    D%char_length = domain_length / sqrt(real(n, dp) / 2.0d0)

    print '(A,F12.4,A)', '  Element size:     ', D%char_length, ' m'
    print '(A,F12.4,A)', '  CFL number:       ', D%cfl
    print '(A)', ''

    ! Initialize on host
    t0 = omp_get_wtime()
    call generate_mesh(D, grid_size, n)
    call init_quantities(D, n, initial_height)
    t_init = omp_get_wtime() - t0

    ! Transfer to GPU
    t0 = omp_get_wtime()
    call map_to_gpu(D, n)
    t_to_gpu = omp_get_wtime() - t0

    ! Run benchmark iterations with CFL timestep
    sim_time = 0.0d0
    dt = 0.0d0

    print '(A,I0,A)', 'Running ', niter, ' iterations to measure performance...'
    print '(A)', ''

    t0 = omp_get_wtime()
    do iter = 1, niter
        call extrapolate_to_edges_gpu(D, n)
        call compute_fluxes_gpu(D, n, max_speed_global)

        ! CFL-based timestep: dt = CFL * characteristic_length / max_wave_speed
        if (max_speed_global > D%epsilon) then
            dt_cfl = D%cfl * D%char_length / max_speed_global
        else
            dt_cfl = D%cfl * D%char_length / sqrt(D%g * initial_height)
        end if
        dt = dt_cfl

        call protect_gpu(D, n)
        call update_gpu(D, n, dt)
        sim_time = sim_time + dt

        if (mod(iter, 100) == 0) then
            call print_progress(iter, niter, omp_get_wtime() - t0, sim_time)
        end if
    end do
    t_compute = omp_get_wtime() - t0
    print '(A)', ''
    print '(A)', ''

    ! Cleanup GPU
    call unmap_from_gpu(D, n)

    ! Calculate estimates
    time_per_step = t_compute / real(niter, dp)
    steps_per_second = 1.0d0 / time_per_step

    ! Average dt from simulation
    dt = sim_time / real(niter, dp)

    ! Estimate steps needed for target time
    estimated_steps = target_time / dt
    estimated_wallclock = estimated_steps * time_per_step

    print '(A)', '============================================================'
    print '(A)', '  BENCHMARK RESULTS'
    print '(A)', '============================================================'
    print '(A)', ''
    print '(A,I0,A)', 'Ran ', niter, ' iterations:'
    print '(A,F12.4,A)', '  Wall-clock time:  ', t_compute, ' s'
    print '(A,F12.6,A)', '  Time per step:    ', time_per_step * 1000.0d0, ' ms'
    print '(A,F12.2,A)', '  Steps per second: ', steps_per_second, ''
    print '(A)', ''
    print '(A)', 'CFL Timestep:'
    print '(A,ES12.4,A)', '  Average dt:       ', dt, ' s'
    print '(A,F12.4,A)', '  Simulated time:   ', sim_time, ' s'
    print '(A,F12.6,A)', '  Max wave speed:   ', max_speed_global, ' m/s'
    print '(A)', ''

    print '(A)', '============================================================'
    print '(A)', '  2-DAY SIMULATION ESTIMATE'
    print '(A)', '============================================================'
    print '(A)', ''
    print '(A,F14.2,A)', '  Target sim time:  ', target_time, ' s (2 days)'
    print '(A,ES14.4)',  '  Estimated steps:  ', estimated_steps
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

    ! Feasibility assessment
    print '(A)', 'Feasibility:'
    if (estimated_wallclock < 3600.0d0) then
        print '(A)', '  EXCELLENT - Simulation completes in under 1 hour'
    else if (estimated_wallclock < 86400.0d0) then
        print '(A)', '  GOOD - Simulation completes in under 1 day'
    else if (estimated_wallclock < 7.0d0 * 86400.0d0) then
        print '(A)', '  ACCEPTABLE - Simulation completes in under 1 week'
    else
        print '(A)', '  CHALLENGING - Simulation takes more than 1 week'
        print '(A)', '  Consider: coarser mesh, larger CFL, or more GPUs'
    end if
    print '(A)', ''

    ! Cleanup
    call deallocate_domain(D)

contains

    subroutine allocate_domain(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        D%number_of_elements = n
        D%g = 9.81d0
        D%epsilon = 1.0d-12
        D%minimum_allowed_height = 1.0d-6

        allocate(D%stage_centroid_values(n))
        allocate(D%xmom_centroid_values(n))
        allocate(D%ymom_centroid_values(n))
        allocate(D%bed_centroid_values(n))
        allocate(D%height_centroid_values(n))
        allocate(D%stage_explicit_update(n))
        allocate(D%xmom_explicit_update(n))
        allocate(D%ymom_explicit_update(n))
        allocate(D%stage_edge_values(3*n))
        allocate(D%height_edge_values(3*n))
        allocate(D%neighbours(3*n))
        allocate(D%edgelengths(3*n))
        allocate(D%normals(6*n))
        allocate(D%areas(n))
        allocate(D%radii(n))
        allocate(D%max_speed(n))
    end subroutine allocate_domain

    subroutine deallocate_domain(D)
        type(domain_t), intent(inout) :: D

        deallocate(D%stage_centroid_values)
        deallocate(D%xmom_centroid_values)
        deallocate(D%ymom_centroid_values)
        deallocate(D%bed_centroid_values)
        deallocate(D%height_centroid_values)
        deallocate(D%stage_explicit_update)
        deallocate(D%xmom_explicit_update)
        deallocate(D%ymom_explicit_update)
        deallocate(D%stage_edge_values)
        deallocate(D%height_edge_values)
        deallocate(D%neighbours)
        deallocate(D%edgelengths)
        deallocate(D%normals)
        deallocate(D%areas)
        deallocate(D%radii)
        deallocate(D%max_speed)
    end subroutine deallocate_domain

    subroutine generate_mesh(D, grid_size, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: grid_size, n

        integer :: nx, ny, k, cell, tri_in_cell, cell_x, cell_y, i
        real(dp) :: dx, dy, area, edgelen, radius

        nx = grid_size
        ny = grid_size

        ! Physical spacing based on domain length
        dx = D%domain_length / (nx - 1)
        dy = D%domain_length / (ny - 1)

        area = 0.5d0 * dx * dy
        edgelen = dx
        radius = area / (1.5d0 * edgelen)

        !$omp parallel do private(i)
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
        end do
        !$omp end parallel do

        !$omp parallel do private(cell, tri_in_cell, cell_x, cell_y)
        do k = 1, n
            cell = (k - 1) / 2
            tri_in_cell = mod(k - 1, 2)
            cell_x = mod(cell, nx - 1)
            cell_y = cell / (nx - 1)

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

            do i = 1, 3
                D%stage_edge_values(3*(k-1) + i) = initial_height
                D%height_edge_values(3*(k-1) + i) = initial_height
            end do
        end do
        !$omp end parallel do
    end subroutine init_quantities

    subroutine map_to_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        !$omp target enter data map(to: D, D%stage_centroid_values(1:n), &
        !$omp&    D%xmom_centroid_values(1:n), &
        !$omp&    D%ymom_centroid_values(1:n), &
        !$omp&    D%bed_centroid_values(1:n), &
        !$omp&    D%height_centroid_values(1:n), &
        !$omp&    D%stage_explicit_update(1:n), &
        !$omp&    D%xmom_explicit_update(1:n), &
        !$omp&    D%ymom_explicit_update(1:n), &
        !$omp&    D%stage_edge_values(1:3*n), &
        !$omp&    D%height_edge_values(1:3*n), &
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

        !$omp target exit data map(delete: D, D%stage_centroid_values(1:n), &
        !$omp&    D%xmom_centroid_values(1:n), &
        !$omp&    D%ymom_centroid_values(1:n), &
        !$omp&    D%bed_centroid_values(1:n), &
        !$omp&    D%height_centroid_values(1:n), &
        !$omp&    D%stage_explicit_update(1:n), &
        !$omp&    D%xmom_explicit_update(1:n), &
        !$omp&    D%ymom_explicit_update(1:n), &
        !$omp&    D%stage_edge_values(1:3*n), &
        !$omp&    D%height_edge_values(1:3*n), &
        !$omp&    D%neighbours(1:3*n), &
        !$omp&    D%edgelengths(1:3*n), &
        !$omp&    D%normals(1:6*n), &
        !$omp&    D%areas(1:n), &
        !$omp&    D%radii(1:n), &
        !$omp&    D%max_speed(1:n))
    end subroutine unmap_from_gpu

    subroutine extrapolate_to_edges_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n
        integer :: k, i, ki

        !$omp target teams distribute parallel do private(i, ki)
        do k = 1, n
            do i = 1, 3
                ki = 3*(k-1) + i
                D%stage_edge_values(ki) = D%stage_centroid_values(k)
                D%height_edge_values(ki) = D%height_centroid_values(k)
            end do
        end do
        !$omp end target teams distribute parallel do
    end subroutine extrapolate_to_edges_gpu

    subroutine compute_fluxes_gpu(D, n, max_speed_out)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n
        real(dp), intent(out) :: max_speed_out

        integer :: k, i, ki, nb
        real(dp) :: g, epsilon
        real(dp) :: stage_accum, xmom_accum, ymom_accum, speed_max
        real(dp) :: uh_k, vh_k, h_left, h_right, uh_right, vh_right
        real(dp) :: edgelen, nx, ny, c_left, c_right, c_max, h_diff
        real(dp) :: global_max_speed

        g = D%g
        epsilon = D%epsilon
        global_max_speed = 0.0d0

        !$omp target teams distribute parallel do &
        !$omp& private(i, ki, nb, stage_accum, xmom_accum, ymom_accum, speed_max, &
        !$omp&         uh_k, vh_k, h_left, h_right, uh_right, vh_right, &
        !$omp&         edgelen, nx, ny, c_left, c_right, c_max, h_diff) &
        !$omp& reduction(max: global_max_speed)
        do k = 1, n
            stage_accum = 0.0d0
            xmom_accum = 0.0d0
            ymom_accum = 0.0d0
            speed_max = 0.0d0

            uh_k = D%xmom_centroid_values(k)
            vh_k = D%ymom_centroid_values(k)

            do i = 1, 3
                ki = 3*(k-1) + i
                nb = D%neighbours(ki)

                h_left = D%height_edge_values(ki)

                if (nb >= 1) then
                    h_right = D%height_centroid_values(nb)
                    uh_right = D%xmom_centroid_values(nb)
                    vh_right = D%ymom_centroid_values(nb)
                else
                    h_right = h_left
                    uh_right = -uh_k
                    vh_right = -vh_k
                end if

                edgelen = D%edgelengths(ki)
                nx = D%normals(6*(k-1) + 2*(i-1) + 1)
                ny = D%normals(6*(k-1) + 2*(i-1) + 2)

                c_left = sqrt(g * max(h_left, 0.0d0))
                c_right = sqrt(g * max(h_right, 0.0d0))
                c_max = max(c_left, c_right)

                h_diff = h_left - h_right
                stage_accum = stage_accum + c_max * h_diff * edgelen
                xmom_accum = xmom_accum + c_max * (uh_k - uh_right) * edgelen * nx
                ymom_accum = ymom_accum + c_max * (vh_k - vh_right) * edgelen * ny

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

end program sw_cfl
