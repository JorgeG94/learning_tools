program nbody_sync
    use omp_lib
    use iso_fortran_env, only: dp => real64
    implicit none
    
    ! Simulation parameters
    integer, parameter :: n_bodies = 70000
    integer, parameter :: max_steps = 200
    integer, parameter :: output_interval = 100
    real(dp), parameter :: dt = 0.01_dp
    real(dp), parameter :: G = 1.0_dp
    real(dp), parameter :: softening = 0.1_dp
    
    ! State vectors
    real(dp), allocatable :: x(:), y(:), z(:)      ! Positions
    real(dp), allocatable :: vx(:), vy(:), vz(:)   ! Velocities
    real(dp), allocatable :: ax(:), ay(:), az(:)   ! Velocities
    real(dp), allocatable :: mass(:)               ! Masses
    
    integer :: step
    real(dp) :: start_time, end_time, step_start, step_end, curr_time
    
    ! Initialize
    allocate(x(n_bodies), y(n_bodies), z(n_bodies))
    allocate(vx(n_bodies), vy(n_bodies), vz(n_bodies))
    allocate(ax(n_bodies), ay(n_bodies), az(n_bodies))
    allocate(mass(n_bodies))

    
    
    print *, "Starting N-body simulation (SYNC)..."
    print *, "Bodies:", n_bodies
    print *, "Steps:", max_steps
    print *, "dt:", dt
    print *, "Max OMP threads available:", omp_get_max_threads()
    print *, ""

    start_time = omp_get_wtime()
    !$omp target enter data map(alloc: x, y, z, vx, vy, vz, mass, ax,ay,az)
    call initialize_bodies(n_bodies, x, y, z, vx, vy, vz, mass, ax, ay, az)
    curr_time = omp_get_wtime()
    print *, "Done allocating and initializing variables! Took: ", curr_time - start_time, " seconds"
    
    
    step_start = omp_get_wtime()
    do step = 1, max_steps
        call evolve_system(n_bodies, x, y, z, vx, vy, vz, mass, dt, G, softening, az, ay, az)
        
        if (mod(step, output_interval) == 0) then
        !$omp target update from(x,y,z,vx,vy,vz)
            call write_output(step, n_bodies, x, y, z, vx, vy, vz)
            step_end = omp_get_wtime()
            print '(a,i6,a,f8.5,a)', "Step:", step, " (evolve took ", &
                step_end - step_start, "s)"
            step_start = step_end
        end if
    end do
    
    end_time = omp_get_wtime()
    
    print *, ""
    print *, "Simulation complete!"
    print *, "Time elapsed:", end_time - start_time, "seconds"
    
    !$omp target exit data map(release: x, y, z, vx, vy, vz, mass, ax, ay,az)

    deallocate(x, y, z, vx, vy, vz, mass,ax,ay,az)
    
contains

    subroutine initialize_bodies(n, x, y, z, vx, vy, vz, mass, ax, ay, az)
        integer, intent(in) :: n
        real(dp), intent(out) :: x(n), y(n), z(n)
        real(dp), intent(out) :: vx(n), vy(n), vz(n)
        real(dp), intent(out) :: ax(n), ay(n), az(n)
        real(dp), intent(out) :: mass(n)
        integer :: i
        real(dp) :: rand_val
        
        !$omp target teams loop
        do i = 1, n
            x(i) = real(i) * 15.0_dp
            y(i) = real(2*i) * 2.0_dp
            z(i) = real(3*i) * -6.0_dp
            
            vx(i) = 0.0_dp
            vy(i) = 0.0_dp
            vz(i) = 0.0_dp
            ax(i) = 0.0_dp
            ay(i) = 0.0_dp
            az(i) = 0.0_dp
            
            mass(i) = 1750.0_dp 
        end do
        !$omp end target teams loop
    end subroutine

    subroutine evolve_system(n, x, y, z, vx, vy, vz, mass, dt, G, eps, ax, ay, az)
        integer, intent(in) :: n
        real(dp), intent(inout) :: x(n), y(n), z(n)
        real(dp), intent(inout) :: vx(n), vy(n), vz(n)
        real(dp), intent(inout) :: ax(n), ay(n), az(n)
        real(dp), intent(in) :: mass(n)
        real(dp), intent(in) :: dt, G, eps
        
        real(dp) :: dx, dy, dz, dist_sq, dist, force_mag
        real(dp) :: ax_local, ay_local, az_local
        integer :: i, j
        
        ! Compute accelerations
        !$omp target teams loop
        do i = 1, n
          ax(i) = 0.0_dp
          ay(i) = 0.0_dp
          az(i) = 0.0_dp
        end do 
        
        ! Parallel loop over particles (each thread computes forces on subset)
        !$omp target teams loop 
        do i = 1, n
            ax_local = 0.0_dp
            ay_local = 0.0_dp
            az_local = 0.0_dp
            
            ! Compute force on particle i from all other particles
            do j = 1, n
                if (i /= j) then
                    dx = x(j) - x(i)
                    dy = y(j) - y(i)
                    dz = z(j) - z(i)
                    
                    dist_sq = dx*dx + dy*dy + dz*dz + eps*eps
                    dist = sqrt(dist_sq)
                    force_mag = G * mass(j) / (dist * dist_sq)
                    
                    ax_local = ax_local + force_mag * dx
                    ay_local = ay_local + force_mag * dy
                    az_local = az_local + force_mag * dz
                end if
            end do
            
            ax(i) = ax_local
            ay(i) = ay_local
            az(i) = az_local
        end do
        !$omp end target teams loop
        
        ! Update velocities and positions (Euler integration)
        !$omp target teams loop 
        do i = 1, n
            vx(i) = vx(i) + ax(i) * dt
            vy(i) = vy(i) + ay(i) * dt
            vz(i) = vz(i) + az(i) * dt
            
            x(i) = x(i) + vx(i) * dt
            y(i) = y(i) + vy(i) * dt
            z(i) = z(i) + vz(i) * dt
        end do
        !$omp end target teams loop

    end subroutine

    subroutine write_output(step, n, x, y, z, vx, vy, vz)
        integer, intent(in) :: step, n
        real(dp), intent(in) :: x(n), y(n), z(n)
        real(dp), intent(in) :: vx(n), vy(n), vz(n)
        character(len=50) :: filename
        integer :: unit_num, i
        real(dp) :: ke
        real(dp) :: io_start, io_end
        
        io_start = omp_get_wtime()
        
        ! Write snapshot to file
        write(filename, '(a,i6.6,a)') 'output_', step, '.dat'
        
        open(newunit=unit_num, file=trim(filename), status='replace')
        write(unit_num, '(a)') '# x, y, z, vx, vy, vz'
        do i = 1, n
            write(unit_num, '(6e16.8)') x(i), y(i), z(i), vx(i), vy(i), vz(i)
        end do
        close(unit_num)
        
        ! Compute kinetic energy
        ke = 0.0_dp
        do i = 1, n
            ke = ke + 0.5_dp * (vx(i)**2 + vy(i)**2 + vz(i)**2)
        end do
        
        io_end = omp_get_wtime()
        
        print '(a,i6,a,f8.4,a,e12.5)', &
            "[I/O] Step ", step, &
            " written in ", io_end - io_start, &
            "s, KE = ", ke
    end subroutine

end program nbody_sync
