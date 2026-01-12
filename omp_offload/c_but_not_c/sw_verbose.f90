! sw_verbose.f90 - Shallow water mini app with VERBOSE output transfers
! Fortran translation of sw_verbose.c using allocatable arrays
! Transfers many arrays at each yieldstep to simulate:
! - Full visualization dumps
! - Debugging with all intermediate quantities
! - Model coupling requiring full state

program sw_verbose
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
    end type domain_t

    ! Verbose output type
    type :: verbose_output_t
        ! Primary quantities
        real(dp), allocatable :: stage(:)
        real(dp), allocatable :: height(:)
        real(dp), allocatable :: xmom(:)
        real(dp), allocatable :: ymom(:)

        ! Derived quantities
        real(dp), allocatable :: xvel(:)
        real(dp), allocatable :: yvel(:)
        real(dp), allocatable :: speed(:)
        real(dp), allocatable :: froude(:)

        ! Flux/update info
        real(dp), allocatable :: stage_update(:)
        real(dp), allocatable :: xmom_update(:)
        real(dp), allocatable :: ymom_update(:)

        ! Edge values (3x size)
        real(dp), allocatable :: stage_edge(:)
        real(dp), allocatable :: height_edge(:)

        ! Per-element diagnostics
        real(dp), allocatable :: max_speed_elem(:)

        ! Scalars
        real(dp) :: time
        real(dp) :: total_mass
        real(dp) :: total_momentum_x
        real(dp) :: total_momentum_y
        real(dp) :: max_speed_scalar
        real(dp) :: max_froude
        real(dp) :: min_height
        real(dp) :: max_height
    end type verbose_output_t

    ! Local variables
    type(domain_t) :: D
    type(verbose_output_t) :: out
    integer :: grid_size, niter, yieldstep, n, num_yields
    integer :: iter, yield_count, nargs
    real(dp) :: dt, sim_time, t_yield_transfer
    real(dp) :: t0, t_init, t_to_gpu, t_compute_total, t_compute_pure
    real(dp) :: t_from_gpu, t_transfer_total, t_total
    real(dp) :: mb_per_array, mb_per_edge_array, total_mb
    real(dp) :: yield_data_mb, yield_bandwidth, bytes_per_iter, bandwidth
    character(len=32) :: arg

    ! Parse command line arguments
    nargs = command_argument_count()
    if (nargs < 1 .or. nargs > 3) then
        print '(A)', 'Usage: sw_verbose_f90 N [niter] [yieldstep]'
        print '(A)', '  N         = grid size (creates 2*(N-1)^2 triangles)'
        print '(A)', '  niter     = number of iterations (default: 1000)'
        print '(A)', '  yieldstep = transfer output every N iterations (default: 100)'
        stop 1
    end if

    call get_command_argument(1, arg)
    read(arg, *) grid_size

    niter = 1000
    yieldstep = 100
    if (nargs >= 2) then
        call get_command_argument(2, arg)
        read(arg, *) niter
    end if
    if (nargs >= 3) then
        call get_command_argument(3, arg)
        read(arg, *) yieldstep
    end if

    if (grid_size < 3) then
        print '(A)', 'Error: Grid size must be at least 3'
        stop 1
    end if

    n = 2 * (grid_size - 1) * (grid_size - 1)
    num_yields = niter / yieldstep

    mb_per_array = real(n, 8) * 8.0d0 / (1024.0d0 * 1024.0d0)
    mb_per_edge_array = real(3 * n, 8) * 8.0d0 / (1024.0d0 * 1024.0d0)

    print '(A)', '=== SW_VERBOSE (Fortran): Heavy output transfers ==='
    print '(A,I0,A,I0)', 'Grid size: ', grid_size, ' x ', grid_size
    print '(A,I0)', 'Number of triangular elements: ', n
    print '(A,I0,A,I0,A,I0,A)', 'Iterations: ', niter, ', Yieldstep: ', yieldstep, ' (yields: ', num_yields, ')'
    print '(A)', ''

    print '(A)', 'Arrays transferred at each yieldstep:'
    print '(A)', '  Centroid arrays (n elements each):'
    print '(A,F6.2,A,F6.2,A)', '    - stage, height, xmom, ymom      : 4 x ', mb_per_array, ' MB = ', 4*mb_per_array, ' MB'
    print '(A,F6.2,A,F6.2,A)', '    - stage_update, xmom_update, ymom_update: 3 x ', mb_per_array, ' MB = ', 3*mb_per_array, ' MB'
    print '(A,F6.2,A,F6.2,A)', '    - xvel, yvel, speed, froude      : 4 x ', mb_per_array, ' MB = ', 4*mb_per_array, ' MB'
    print '(A,F6.2,A,F6.2,A)', '    - max_speed_elem                 : 1 x ', mb_per_array, ' MB = ', mb_per_array, ' MB'
    print '(A)', '  Edge arrays (3n elements each):'
    print '(A,F6.2,A,F6.2,A)', '    - stage_edge, height_edge        : 2 x ', mb_per_edge_array, ' MB = ', 2*mb_per_edge_array, ' MB'
    print '(A)', '  -----------------------------------------'
    total_mb = 12.0d0 * mb_per_array + 2.0d0 * mb_per_edge_array
    print '(A,F8.2,A)', '  TOTAL per yieldstep:                 ', total_mb, ' MB'
    print '(A)', ''

    ! Allocate domain
    call allocate_domain(D, n)
    call allocate_verbose_output(out, n)

    ! Initialize on host
    t0 = omp_get_wtime()
    call generate_mesh(D, grid_size, n)
    call init_quantities(D, n)
    t_init = omp_get_wtime() - t0

    ! Transfer to GPU
    print '(A)', 'Transferring data to GPU...'
    t0 = omp_get_wtime()
    call map_to_gpu(D, n)
    t_to_gpu = omp_get_wtime() - t0

    ! Run iterations
    dt = 0.001d0
    sim_time = 0.0d0
    t_yield_transfer = 0.0d0
    yield_count = 0

    print '(A,I0,A)', 'Computing with VERBOSE output every ', yieldstep, ' iterations:'

    t0 = omp_get_wtime()
    do iter = 1, niter
        call extrapolate_to_edges_gpu(D, n)
        call compute_fluxes_gpu(D, n)
        call protect_gpu(D, n)
        call update_gpu(D, n, dt)
        sim_time = sim_time + dt

        ! Verbose yieldstep transfer
        if (mod(iter, yieldstep) == 0) then
            call transfer_verbose_output(D, out, n, sim_time, t_yield_transfer)
            yield_count = yield_count + 1
        end if

        if (mod(iter, 100) == 0 .or. iter == niter) then
            call print_progress(iter, niter, omp_get_wtime() - t0, yield_count)
        end if
    end do
    t_compute_total = omp_get_wtime() - t0
    t_compute_pure = t_compute_total - t_yield_transfer
    print '(A)', ''
    print '(A)', ''

    ! Final cleanup
    t0 = omp_get_wtime()
    call unmap_from_gpu(D, n)
    t_from_gpu = omp_get_wtime() - t0

    t_transfer_total = t_to_gpu + t_yield_transfer + t_from_gpu
    t_total = t_init + t_to_gpu + t_compute_total + t_from_gpu

    ! Results
    print '(A)', 'Timing breakdown:'
    print '(A,F12.4,A)', '  Host init:           ', t_init * 1000.0d0, ' ms'
    print '(A,F12.4,A)', '  Transfer to GPU:     ', t_to_gpu * 1000.0d0, ' ms'
    print '(A,F12.4,A,F8.4,A)', '  Pure compute:        ', t_compute_pure * 1000.0d0, ' ms (', &
          t_compute_pure * 1000.0d0 / niter, ' ms/iter)'
    print '(A,F12.4,A,I0,A,F8.4,A)', '  Yieldstep transfers: ', t_yield_transfer * 1000.0d0, ' ms (', &
          yield_count, ' yields, ', t_yield_transfer * 1000.0d0 / max(yield_count, 1), ' ms/yield)'
    print '(A,F12.4,A)', '  Cleanup:             ', t_from_gpu * 1000.0d0, ' ms'
    print '(A)', '  --------------------------------'
    print '(A,F12.4,A)', '  Total:               ', t_total * 1000.0d0, ' ms'
    print '(A,F12.4,A)', '  Transfer/Total:      ', 100.0d0 * t_transfer_total / t_total, ' %'
    print '(A,F12.4,A)', '  Yieldstep overhead:  ', 100.0d0 * t_yield_transfer / t_compute_pure, ' % of compute time'

    ! Data transfer rate
    yield_data_mb = total_mb * yield_count
    yield_bandwidth = yield_data_mb / t_yield_transfer / 1000.0d0  ! GB/s
    print '(A)', ''
    print '(A)', 'Yieldstep transfer stats:'
    print '(A,F8.2,A,I0,A,F6.2,A)', '  Total data transferred: ', yield_data_mb, ' MB (', &
          yield_count, ' yields x ', total_mb, ' MB)'
    print '(A,F8.2,A)', '  Effective PCIe bandwidth: ', yield_bandwidth, ' GB/s'

    ! Bandwidth estimate
    bytes_per_iter = real(n, 8) * 8.0d0 * 32.0d0
    bandwidth = (niter * bytes_per_iter) / t_compute_pure / 1.0d9
    print '(A,F8.2,A)', '  GPU memory bandwidth: ', bandwidth, ' GB/s'

    ! Last output
    print '(A)', ''
    print '(A,F8.4,A)', 'Last yieldstep diagnostics (t=', out%time, '):'
    print '(A,F12.6,A)', '  Total mass: ', out%total_mass, ' (conservation check)'
    print '(A,F12.6,A,F12.6)', '  Max speed: ', out%max_speed_scalar, ', Max Froude: ', out%max_froude
    print '(A,F12.6,A,F12.6,A)', '  Height range: [', out%min_height, ', ', out%max_height, ']'

    ! Cleanup
    call deallocate_domain(D)
    call deallocate_verbose_output(out)

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

    subroutine allocate_verbose_output(out, n)
        type(verbose_output_t), intent(inout) :: out
        integer, intent(in) :: n

        allocate(out%stage(n))
        allocate(out%height(n))
        allocate(out%xmom(n))
        allocate(out%ymom(n))
        allocate(out%xvel(n))
        allocate(out%yvel(n))
        allocate(out%speed(n))
        allocate(out%froude(n))
        allocate(out%stage_update(n))
        allocate(out%xmom_update(n))
        allocate(out%ymom_update(n))
        allocate(out%stage_edge(3*n))
        allocate(out%height_edge(3*n))
        allocate(out%max_speed_elem(n))
    end subroutine allocate_verbose_output

    subroutine deallocate_verbose_output(out)
        type(verbose_output_t), intent(inout) :: out

        deallocate(out%stage)
        deallocate(out%height)
        deallocate(out%xmom)
        deallocate(out%ymom)
        deallocate(out%xvel)
        deallocate(out%yvel)
        deallocate(out%speed)
        deallocate(out%froude)
        deallocate(out%stage_update)
        deallocate(out%xmom_update)
        deallocate(out%ymom_update)
        deallocate(out%stage_edge)
        deallocate(out%height_edge)
        deallocate(out%max_speed_elem)
    end subroutine deallocate_verbose_output

    subroutine generate_mesh(D, grid_size, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: grid_size, n

        integer :: nx, ny, k, cell, tri_in_cell, cell_x, cell_y, i
        real(dp) :: dx, dy, area, edgelen, radius

        nx = grid_size
        ny = grid_size
        dx = 1.0d0 / (nx - 1)
        dy = 1.0d0 / (ny - 1)

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
                D%neighbours(3*(k-1) + 1) = k + 1  ! k+1 in Fortran 1-based
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

    subroutine init_quantities(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n
        integer :: k, i

        !$omp parallel do private(i)
        do k = 1, n
            D%bed_centroid_values(k) = 0.0d0
            D%stage_centroid_values(k) = 1.0d0
            D%height_centroid_values(k) = 1.0d0
            D%xmom_centroid_values(k) = 0.0d0
            D%ymom_centroid_values(k) = 0.0d0
            D%stage_explicit_update(k) = 0.0d0
            D%xmom_explicit_update(k) = 0.0d0
            D%ymom_explicit_update(k) = 0.0d0
            D%max_speed(k) = 0.0d0

            do i = 1, 3
                D%stage_edge_values(3*(k-1) + i) = 1.0d0
                D%height_edge_values(3*(k-1) + i) = 1.0d0
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

        !$omp target exit data map(delete:D, D%stage_centroid_values(1:n), &
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

    subroutine compute_fluxes_gpu(D, n)
        type(domain_t), intent(inout) :: D
        integer, intent(in) :: n

        integer :: k, i, ki, nb
        real(dp) :: g, epsilon
        real(dp) :: stage_accum, xmom_accum, ymom_accum, speed_max
        real(dp) :: uh_k, vh_k, h_left, h_right, uh_right, vh_right
        real(dp) :: edgelen, nx, ny, c_left, c_right, c_max, h_diff

        g = D%g
        epsilon = D%epsilon

        !$omp target teams distribute parallel do &
        !$omp& private(i, ki, nb, stage_accum, xmom_accum, ymom_accum, speed_max, &
        !$omp&         uh_k, vh_k, h_left, h_right, uh_right, vh_right, &
        !$omp&         edgelen, nx, ny, c_left, c_right, c_max, h_diff)
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
        end do
        !$omp end target teams distribute parallel do
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

    subroutine compute_derived_quantities_gpu(D, out, n)
        type(domain_t), intent(inout) :: D
        type(verbose_output_t), intent(inout) :: out
        integer, intent(in) :: n

        integer :: k
        real(dp) :: g, epsilon, h, u, v, spd, c

        g = D%g
        epsilon = D%epsilon

        !$omp target teams distribute parallel do &
        !$omp& map(from: out%xvel(1:n), out%yvel(1:n), out%speed(1:n), out%froude(1:n)) &
        !$omp& private(h, u, v, spd, c)
        do k = 1, n
            h = D%height_centroid_values(k)
            if (h > epsilon) then
                u = D%xmom_centroid_values(k) / h
                v = D%ymom_centroid_values(k) / h
                spd = sqrt(u*u + v*v)
                c = sqrt(g * h)

                out%xvel(k) = u
                out%yvel(k) = v
                out%speed(k) = spd
                out%froude(k) = spd / c
            else
                out%xvel(k) = 0.0d0
                out%yvel(k) = 0.0d0
                out%speed(k) = 0.0d0
                out%froude(k) = 0.0d0
            end if
        end do
        !$omp end target teams distribute parallel do
    end subroutine compute_derived_quantities_gpu

    subroutine compute_diagnostics_gpu(D, out, n)
        type(domain_t), intent(inout) :: D
        type(verbose_output_t), intent(inout) :: out
        integer, intent(in) :: n

        integer :: k
        real(dp) :: g, epsilon, h, area, u, v, spd, c
        real(dp) :: total_mass, total_xmom, total_ymom
        real(dp) :: max_spd, max_fr, min_h, max_h

        g = D%g
        epsilon = D%epsilon

        total_mass = 0.0d0
        total_xmom = 0.0d0
        total_ymom = 0.0d0
        max_spd = 0.0d0
        max_fr = 0.0d0
        min_h = 1.0d100
        max_h = 0.0d0

        !$omp target teams distribute parallel do &
        !$omp& reduction(+: total_mass, total_xmom, total_ymom) &
        !$omp& reduction(max: max_spd, max_fr, max_h) &
        !$omp& reduction(min: min_h) &
        !$omp& private(h, area, u, v, spd, c)
        do k = 1, n
            h = D%height_centroid_values(k)
            area = D%areas(k)

            total_mass = total_mass + h * area
            total_xmom = total_xmom + D%xmom_centroid_values(k) * area
            total_ymom = total_ymom + D%ymom_centroid_values(k) * area

            max_spd = max(max_spd, D%max_speed(k))

            if (h > epsilon) then
                u = D%xmom_centroid_values(k) / h
                v = D%ymom_centroid_values(k) / h
                spd = sqrt(u*u + v*v)
                c = sqrt(g * h)
                max_fr = max(max_fr, spd / c)
            end if

            min_h = min(min_h, h)
            max_h = max(max_h, h)
        end do
        !$omp end target teams distribute parallel do

        out%total_mass = total_mass
        out%total_momentum_x = total_xmom
        out%total_momentum_y = total_ymom
        out%max_speed_scalar = max_spd
        out%max_froude = max_fr
        out%min_height = min_h
        out%max_height = max_h
    end subroutine compute_diagnostics_gpu

    subroutine transfer_verbose_output(D, out, n, sim_time, t_transfer_accum)
        type(domain_t), intent(inout) :: D
        type(verbose_output_t), intent(inout) :: out
        integer, intent(in) :: n
        real(dp), intent(in) :: sim_time
        real(dp), intent(inout) :: t_transfer_accum

        real(dp) :: t0

        t0 = omp_get_wtime()

        ! Transfer primary quantities (4 arrays)
        !$omp target update from(D%stage_centroid_values(1:n))
        !$omp target update from(D%height_centroid_values(1:n))
        !$omp target update from(D%xmom_centroid_values(1:n))
        !$omp target update from(D%ymom_centroid_values(1:n))

        ! Copy to output struct
        out%stage(1:n) = D%stage_centroid_values(1:n)
        out%height(1:n) = D%height_centroid_values(1:n)
        out%xmom(1:n) = D%xmom_centroid_values(1:n)
        out%ymom(1:n) = D%ymom_centroid_values(1:n)

        ! Transfer update/flux arrays (3 arrays)
        !$omp target update from(D%stage_explicit_update(1:n))
        !$omp target update from(D%xmom_explicit_update(1:n))
        !$omp target update from(D%ymom_explicit_update(1:n))

        out%stage_update(1:n) = D%stage_explicit_update(1:n)
        out%xmom_update(1:n) = D%xmom_explicit_update(1:n)
        out%ymom_update(1:n) = D%ymom_explicit_update(1:n)

        ! Transfer edge values (2 arrays x 3n)
        !$omp target update from(D%stage_edge_values(1:3*n))
        !$omp target update from(D%height_edge_values(1:3*n))

        out%stage_edge(1:3*n) = D%stage_edge_values(1:3*n)
        out%height_edge(1:3*n) = D%height_edge_values(1:3*n)

        ! Transfer max_speed per element
        !$omp target update from(D%max_speed(1:n))
        out%max_speed_elem(1:n) = D%max_speed(1:n)

        ! Compute and transfer derived quantities (4 arrays computed on GPU)
        call compute_derived_quantities_gpu(D, out, n)

        ! Compute scalar diagnostics (reductions)
        call compute_diagnostics_gpu(D, out, n)

        out%time = sim_time

        t_transfer_accum = t_transfer_accum + (omp_get_wtime() - t0)
    end subroutine transfer_verbose_output

    subroutine print_progress(current, total, elapsed, yields)
        integer, intent(in) :: current, total, yields
        real(dp), intent(in) :: elapsed

        integer :: bar_width, filled, i
        real(dp) :: progress
        character(len=100) :: bar

        bar_width = 40
        progress = real(current, 8) / real(total, 8)
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

        write(*, '(A,A,A,I3,A,I0,A,I0,A,F6.2,A,I0,A)', advance='no') &
            char(13), '  [', trim(bar), '] ', int(progress * 100), '% (', &
            current, '/', total, ') ', elapsed, 's [', yields, ' yields]'
    end subroutine print_progress

end program sw_verbose
