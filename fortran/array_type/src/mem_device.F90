!! Device memory queries from the OpenACC runtime, for cross-checking the ledger.
module mem_device
   !! The ledger counts what *we* asked for. These report what the runtime
   !! says is free, which also includes the CUDA context, pool rounding
   !! and every other process on the device. Use the difference across a
   !! mapping phase as the ground truth to compare the ledger against.
   !! Without OpenACC every query returns -1.
   use, intrinsic :: iso_fortran_env, only: int64
#ifdef _OPENACC
   use openacc
#endif
   implicit none
   private
   public :: device_init, device_free_bytes, device_total_bytes, device_enabled

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

end module mem_device
