!! Driver: run the SWE model and show what the ledger knows.
!!
!! Usage: swe [nx] [ny] [nsteps] [--leak]
!!   --leak  skip destroying the state, to show the shutdown leak report
program swe
   use, intrinsic :: iso_fortran_env, only: dp => real64, int64, output_unit
   use mem_ledger, only: mem_report, mem_check_leaks, mem_device_bytes, mem_host_bytes, &
                         mem_human_bytes, mem_register_tag
   use mem_device, only: device_init, device_free_bytes, device_enabled
   use mem_collective, only: mem_checksum_t, mem_checksum, mem_update_host, mem_report_values
   use swe_model, only: swe_params_t, swe_state_t, swe_work_t, swe_step, &
                        swe_total_mass, swe_total_energy
   implicit none

   type(swe_params_t) :: p
   type(swe_state_t) :: state
   type(swe_work_t) :: work
   integer :: nsteps, n, nleaks, nargs, k
   logical :: leak
   character(len=64) :: arg
   integer(int64) :: free_before, free_after, measured, ledger_dev, t0, t1, rate
   real(dp) :: mass0, mass1, e0, e1, secs
   type(mem_checksum_t) :: cs0, cs1
   integer :: tag_state

   nsteps = 1000
   leak = .false.
   nargs = 0
   do k = 1, command_argument_count()
      call get_command_argument(k, arg)
      if (arg == '--leak') then
         leak = .true.
         cycle
      end if
      nargs = nargs + 1
      select case (nargs)
      case (1); read (arg, *) p%nx
      case (2); read (arg, *) p%ny
      case (3); read (arg, *) nsteps
      end select
   end do
   call p%finalize()

   write (output_unit, '(a, i0, a, i0, a, i0, a, es10.3, a)') 'swe: ', p%nx, ' x ', p%ny, &
      ', ', nsteps, ' steps, dt = ', p%dt, ' s'
   write (output_unit, '(a, l1)') 'OpenACC: ', device_enabled()

   call device_init()
   free_before = device_free_bytes()

   call state%init(p)
   call work%init(p)

   free_after = device_free_bytes()
   call mem_report(title="after init")
   if (device_enabled()) then
      measured = free_before - free_after
      ledger_dev = mem_device_bytes()
      write (output_unit, '(a, a)') 'ledger device total     : ', mem_human_bytes(ledger_dev)
      write (output_unit, '(a, a)') 'runtime free-mem delta  : ', mem_human_bytes(measured)
      write (output_unit, '(a, f8.4)') 'ratio measured/ledger   : ', &
         real(measured, dp)/real(max(ledger_dev, 1_int64), dp)
   end if

   mass0 = swe_total_mass(state, p)
   e0 = swe_total_energy(state, p)

   ! Collective ops through the ledger: one call pulls every state array
   ! back, whatever it is, and summarises it. No list of arrays here.
   tag_state = mem_register_tag("swe_state")
   call mem_update_host(tag_state)
   cs0 = mem_checksum(tag_state)

   call system_clock(t0, rate)
   do n = 1, nsteps
      call swe_step(state, work, p)
   end do
   !$acc wait
   call system_clock(t1)
   secs = real(t1 - t0, dp)/real(rate, dp)

   mass1 = swe_total_mass(state, p)
   e1 = swe_total_energy(state, p)

   write (output_unit, '(/, a, f10.4, a, f10.2, a)') 'time loop: ', secs, ' s  (', &
      real(p%nx, dp)*p%ny*nsteps/secs/1.0e6_dp, ' Mcell-steps/s)'
   write (output_unit, '(a, es12.4)') 'relative mass change   : ', (mass1 - mass0)/mass0
   write (output_unit, '(a, es12.4)') 'relative energy change : ', (e1 - e0)/e0

   call mem_update_host(tag_state)
   cs1 = mem_checksum(tag_state)
   write (output_unit, '(a, f12.6)') 'max h after run        : ', maxval(state%h%data(1:p%nx, 1:p%ny))
   write (output_unit, '(/, a, i0, a, i0, a)') 'swe_state checksum (', cs1%nobj, ' arrays, ', cs1%nelem, ' values)'
   write (output_unit, '(a, es24.16, a, z16.16)') '  before: sum ', cs0%sum, '  hash ', cs0%hash
   write (output_unit, '(a, es24.16, a, z16.16)') '  after : sum ', cs1%sum, '  hash ', cs1%hash
   call mem_report_values(tag_state)

   call work%destroy()
   if (.not. leak) call state%destroy()

   call mem_report(title="after destroy")
   nleaks = mem_check_leaks()
   if (nleaks > 0) then
      write (output_unit, '(i0, a)') nleaks, ' leaked object(s)'
      error stop 2
   end if
   write (output_unit, '(a)') 'no leaks'
end program swe
