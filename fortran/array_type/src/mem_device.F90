!! Device memory queries from the OpenACC runtime.
module mem_device
   !! The ledger counts what *we* asked for. These report what the runtime
   !! says is free, which also includes the CUDA context, pool rounding and
   !! every other process on the device: use the difference across a mapping
   !! phase as the ground truth to compare the ledger against.
   !! Without OpenACC the queries return -1 and the budget check passes.
   use, intrinsic :: iso_fortran_env, only: int64
#ifdef _OPENACC
   use openacc
#endif
   use mem_ledger_core, only: mem_fatal, mem_human_bytes
   implicit none
   private
   public :: device_init, device_free_bytes, device_total_bytes, device_enabled
   public :: mem_require_device

contains

   logical function device_enabled()
#ifdef _OPENACC
      device_enabled = .true.
#else
      device_enabled = .false.
#endif
   end function device_enabled

   subroutine device_init()
      !! Create the CUDA context up front so that it is not counted inside
      !! the first measured mapping delta.
#ifdef _OPENACC
      call acc_init(acc_get_device_type())
#endif
   end subroutine device_init

   integer(int64) function device_free_bytes() result(b)
#ifdef _OPENACC
      integer(acc_device_kind) :: dt
      dt = acc_get_device_type()
      b = int(acc_get_property(acc_get_device_num(dt), dt, acc_property_free_memory), int64)
#else
      b = -1
#endif
   end function device_free_bytes

   integer(int64) function device_total_bytes() result(b)
#ifdef _OPENACC
      integer(acc_device_kind) :: dt
      dt = acc_get_device_type()
      b = int(acc_get_property(acc_get_device_num(dt), dt, acc_property_memory), int64)
#else
      b = -1
#endif
   end function device_total_bytes

   subroutine mem_require_device(nbytes, what)
      !! Fatal, with a readable message, if `nbytes` more cannot fit on the
      !! device. Called by every devarray mapping; also useful once at init
      !! with a whole-model estimate, before anything is mapped.
      integer(int64), intent(in) :: nbytes
      character(len=*), intent(in) :: what
      integer(int64) :: free
      free = device_free_bytes()
      if (free < 0) return
      if (nbytes > free) &
         call mem_fatal("not enough device memory for "//what//": need "// &
                        mem_human_bytes(nbytes)//", "//mem_human_bytes(free)// &
                        " free (call mem_report to see who holds the rest)")
   end subroutine mem_require_device

end module mem_device
