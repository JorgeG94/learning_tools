!! Device memory queries for the active backend.
module mem_device
   !! The ledger counts what *we* asked for. These report what the runtime
   !! says is free, which also includes the CUDA context, pool rounding and
   !! every other process on the device: use the difference across a mapping
   !! phase as the ground truth to compare the ledger against.
   !!
   !! Backends (chosen at build time):
   !!   DEVARRAY_ACC  OpenACC: acc_get_property free/total memory.
   !!   DEVARRAY_OMP  OpenMP target: OpenMP has no memory query, so on NVIDIA
   !!                 (DEVARRAY_CUDA_MEMINFO) this calls cudaMemGetInfo;
   !!                 elsewhere the queries return -1.
   !!   neither       CPU only: queries return -1.
   !! A query that returns -1 makes the budget check pass silently.
   use, intrinsic :: iso_fortran_env, only: int64
#if defined(DEVARRAY_ACC)
   use openacc
#elif defined(DEVARRAY_OMP)
   use omp_lib, only: omp_get_num_devices
#endif
#ifdef DEVARRAY_CUDA_MEMINFO
   use, intrinsic :: iso_c_binding, only: c_int, c_size_t
#endif
   use mem_ledger_core, only: mem_fatal, mem_human_bytes
   implicit none
   private
   public :: device_init, device_free_bytes, device_total_bytes, device_enabled, device_backend
   public :: mem_require_device

#ifdef DEVARRAY_CUDA_MEMINFO
   interface
      integer(c_int) function cudaMemGetInfo(free, total) bind(c, name="cudaMemGetInfo")
         import :: c_int, c_size_t
         integer(c_size_t), intent(out) :: free, total
      end function cudaMemGetInfo
   end interface
#endif

contains

   logical function device_enabled()
#ifdef DEVARRAY_DEVICE
      device_enabled = .true.
#else
      device_enabled = .false.
#endif
   end function device_enabled

   function device_backend() result(name)
      character(len=:), allocatable :: name
#if defined(DEVARRAY_ACC)
      name = "OpenACC"
#elif defined(DEVARRAY_OMP)
      name = "OpenMP target"
#else
      name = "none (host only)"
#endif
   end function device_backend

   subroutine device_init()
      !! Create the device context up front so that it is not counted inside
      !! the first measured mapping delta, and fail early if there is no
      !! device at all.
#if defined(DEVARRAY_ACC)
      call acc_init(acc_get_device_type())
#elif defined(DEVARRAY_OMP)
      if (omp_get_num_devices() < 1) &
         call mem_fatal("device_init: built for OpenMP offload but no device is available")
      !$omp target
      !$omp end target
#endif
   end subroutine device_init

   integer(int64) function device_free_bytes() result(b)
#if defined(DEVARRAY_ACC)
      integer(acc_device_kind) :: dt
      dt = acc_get_device_type()
      b = int(acc_get_property(acc_get_device_num(dt), dt, acc_property_free_memory), int64)
#elif defined(DEVARRAY_CUDA_MEMINFO)
      integer(c_size_t) :: free, total
      b = -1
      if (cudaMemGetInfo(free, total) == 0) b = int(free, int64)
#else
      b = -1
#endif
   end function device_free_bytes

   integer(int64) function device_total_bytes() result(b)
#if defined(DEVARRAY_ACC)
      integer(acc_device_kind) :: dt
      dt = acc_get_device_type()
      b = int(acc_get_property(acc_get_device_num(dt), dt, acc_property_memory), int64)
#elif defined(DEVARRAY_CUDA_MEMINFO)
      integer(c_size_t) :: free, total
      b = -1
      if (cudaMemGetInfo(free, total) == 0) b = int(total, int64)
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
