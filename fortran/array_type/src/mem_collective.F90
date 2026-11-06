!! Collective operations over every ledger-tracked array with a given tag.
module mem_collective
   !! The ledger knows the host address, kind and size of every tracked
   !! array, so it acts as the "array of arrays" without a type hierarchy or
   !! an arena: each operation walks the live entries with the tag (all
   !! entries if the tag is absent), checks each one is still intact, and
   !! views its memory through `c_f_pointer`.
   !!
   !!   call mem_update_host(tag)            ! device -> host, any kind or rank
   !!   call mem_update_device(tag)          ! host -> device
   !!   call mem_fill(0.0_dp, tag)           ! where the data lives (device if mapped)
   !!   cs = mem_checksum(tag)               ! real(dp) arrays: sum, sum |x|, bit hash
   !!   call mem_report_values(tag)          ! min / max / non-finite per array
   !!   call mem_for_each_r64(visit, tag)    ! callback(name, rank-1 view)
   !!   call mem_validate(tag)               ! fatal if any array was reallocated
   !!
   !! Inspection reads host memory. By default it first pulls mapped arrays
   !! back from the device (`sync=.true.`), since in a GPU run the device
   !! copy is the current one. Pass `sync=.false.` to read the host copies
   !! as they are.
   !!
   !! All loops here are plain DO on purpose: under -stdpar=gpu a DO
   !! CONCURRENT in this module would be offloaded.
   use, intrinsic :: iso_fortran_env, only: int8, int32, int64, real64, output_unit
   use, intrinsic :: iso_c_binding, only: c_f_pointer
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use mem_ledger, only: mem_tag_t
   use mem_ledger_backend, only: mem_object_t, mem_live_ids, mem_get_object, mem_check_object, &
                                 MEM_KIND_R64, MEM_KIND_I32
   implicit none
   private

   integer, parameter :: dp = real64

   public :: mem_checksum_t, mem_visit_r64
   public :: mem_update_host, mem_update_device, mem_fill, mem_validate
   public :: mem_checksum, mem_for_each_r64, mem_report_values

   type :: mem_checksum_t
      integer :: nobj = 0              !! real(dp) arrays included
      integer(int64) :: nelem = 0
      real(dp) :: sum = 0.0_dp
      real(dp) :: abs_sum = 0.0_dp
      integer(int64) :: hash = 0       !! order-dependent, exact: changes if any bit changes
   end type mem_checksum_t

   ! gfortran 15.1 BUG: -Wexternal-argument-mismatch (enabled by -Wall)
   ! treats `procedure(mem_visit_r64) :: visit` as an external procedure and
   ! infers a second interface from the call site. Callers then fail with
   ! "Interface mismatch in dummy procedure 'visit' ... '_formal_1'", and
   ! where it does compile the call is miscompiled (segfault on first access
   ! of `a`). Build with -Wno-external-argument-mismatch on gfortran 15.
   abstract interface
      subroutine mem_visit_r64(name, a)
         import :: dp
         character(len=*), intent(in) :: name
         real(dp), intent(inout) :: a(:)
      end subroutine mem_visit_r64
   end interface

   interface mem_fill
      module procedure fill_r64, fill_i32
   end interface mem_fill

contains

   ! ------------------------------------------------------------ transfers

   subroutine mem_update_host(tag)
      !! Device -> host for every mapped array with `tag` (all if absent).
      type(mem_tag_t), intent(in), optional :: tag
      call update_all(.true., "mem_update_host", tag)
   end subroutine mem_update_host

   subroutine mem_update_device(tag)
      !! Host -> device for every mapped array with `tag` (all if absent).
      type(mem_tag_t), intent(in), optional :: tag
      call update_all(.false., "mem_update_device", tag)
   end subroutine mem_update_device

   subroutine update_all(to_host, where, tag)
      logical, intent(in) :: to_host
      character(len=*), intent(in) :: where
      type(mem_tag_t), intent(in), optional :: tag
      type(mem_object_t) :: obj
      integer(int64), allocatable :: ids(:)
      integer :: k
      integer(int8), pointer :: bytes(:)
      ids = mem_live_ids(tag)
      do k = 1, size(ids)
         call mem_check_object(ids(k), where)
         obj = mem_get_object(ids(k))
         if (.not. obj%mapped .or. obj%nelem == 0) cycle
         ! A byte view covers any kind/rank; the runtime finds the mapping
         ! by host address range.
         call c_f_pointer(obj%addr, bytes, [obj%host])
         if (to_host) then
#if defined(DEVARRAY_ACC)
            !$acc update self(bytes)
#elif defined(DEVARRAY_OMP)
            !$omp target update from(bytes)
#endif
         else
#if defined(DEVARRAY_ACC)
            !$acc update device(bytes)
#elif defined(DEVARRAY_OMP)
            !$omp target update to(bytes)
#endif
         end if
      end do
   end subroutine update_all

   subroutine fill_r64(value, tag)
      !! Set every real(dp) array with `tag` to `value`, where it lives:
      !! on the device if mapped (host copy untouched), else on the host.
      real(dp), intent(in) :: value
      type(mem_tag_t), intent(in), optional :: tag
      type(mem_object_t) :: obj
      integer(int64), allocatable :: ids(:)
      integer(int64) :: i
      integer :: k
      logical :: dev
      real(dp), pointer :: a(:)
      ids = mem_live_ids(tag)
      do k = 1, size(ids)
         call mem_check_object(ids(k), "mem_fill")
         obj = mem_get_object(ids(k))
         if (obj%kind_code /= MEM_KIND_R64 .or. obj%nelem == 0) cycle
         call c_f_pointer(obj%addr, a, [obj%nelem])
         dev = obj%mapped
#if defined(DEVARRAY_ACC)
         !$acc parallel loop present(a) if(dev)
#elif defined(DEVARRAY_OMP)
         !$omp target teams distribute parallel do if(target: dev)
#endif
         do i = 1, obj%nelem
            a(i) = value
         end do
      end do
   end subroutine fill_r64

   subroutine fill_i32(value, tag)
      !! Integer counterpart of `fill_r64`, for integer(int32) arrays.
      integer(int32), intent(in) :: value
      type(mem_tag_t), intent(in), optional :: tag
      type(mem_object_t) :: obj
      integer(int64), allocatable :: ids(:)
      integer(int64) :: i
      integer :: k
      logical :: dev
      integer(int32), pointer :: a(:)
      ids = mem_live_ids(tag)
      do k = 1, size(ids)
         call mem_check_object(ids(k), "mem_fill")
         obj = mem_get_object(ids(k))
         if (obj%kind_code /= MEM_KIND_I32 .or. obj%nelem == 0) cycle
         call c_f_pointer(obj%addr, a, [obj%nelem])
         dev = obj%mapped
#if defined(DEVARRAY_ACC)
         !$acc parallel loop present(a) if(dev)
#elif defined(DEVARRAY_OMP)
         !$omp target teams distribute parallel do if(target: dev)
#endif
         do i = 1, obj%nelem
            a(i) = value
         end do
      end do
   end subroutine fill_i32

   subroutine mem_validate(tag)
      !! Fatal, naming the array, if any tracked array with `tag` was
      !! reallocated or deallocated behind its owner.
      type(mem_tag_t), intent(in), optional :: tag
      integer(int64), allocatable :: ids(:)
      integer :: k
      ids = mem_live_ids(tag)
      do k = 1, size(ids)
         call mem_check_object(ids(k), "mem_validate")
      end do
   end subroutine mem_validate

   ! ----------------------------------------------------------- inspection

   function mem_checksum(tag, sync) result(cs)
      !! Summary of the real(dp) arrays with `tag`. `sync` (default .true.)
      !! pulls mapped arrays from the device first.
      type(mem_tag_t), intent(in), optional :: tag
      logical, intent(in), optional :: sync
      type(mem_checksum_t) :: cs
      call mem_for_each_r64(accumulate, tag, sync)
   contains
      subroutine accumulate(name, a)
         character(len=*), intent(in) :: name
         real(dp), intent(inout) :: a(:)
         integer(int64) :: i
         cs%nobj = cs%nobj + 1
         cs%nelem = cs%nelem + size(a, kind=int64)
         do i = 1, size(a, kind=int64)
            cs%sum = cs%sum + a(i)
            cs%abs_sum = cs%abs_sum + abs(a(i))
            ! rotate-xor: no integer overflow, sensitive to every bit
            cs%hash = ieor(ishftc(cs%hash, 13), transfer(a(i), cs%hash))
         end do
         if (len(name) < 0) continue
      end subroutine accumulate
   end function mem_checksum

   subroutine mem_for_each_r64(visit, tag, sync, sync_back)
      !! Call `visit(name, a)` for every real(dp) array with `tag`, where `a`
      !! is a rank-1 view of its host memory. `sync` (default .true.) pulls
      !! mapped arrays from the device first; `sync_back` (default .false.)
      !! pushes them back afterwards. Set it if `visit` modifies `a`.
      procedure(mem_visit_r64) :: visit
      type(mem_tag_t), intent(in), optional :: tag
      logical, intent(in), optional :: sync, sync_back
      type(mem_object_t) :: obj
      integer(int64), allocatable :: ids(:)
      integer :: k
      real(dp), pointer :: a(:)
      if (opt(sync, .true.)) call update_all(.true., "mem_for_each_r64", tag)
      ids = mem_live_ids(tag)
      do k = 1, size(ids)
         call mem_check_object(ids(k), "mem_for_each_r64")
         obj = mem_get_object(ids(k))
         if (obj%kind_code /= MEM_KIND_R64 .or. obj%nelem == 0) cycle
         call c_f_pointer(obj%addr, a, [obj%nelem])
         call visit(trim(obj%name), a)
      end do
      if (opt(sync_back, .false.)) call update_all(.false., "mem_for_each_r64", tag)
   end subroutine mem_for_each_r64

   subroutine mem_report_values(tag, unit, sync)
      !! min / max / max |x| / non-finite count of each real(dp) array.
      type(mem_tag_t), intent(in), optional :: tag
      integer, intent(in), optional :: unit
      logical, intent(in), optional :: sync
      integer :: u
      u = output_unit
      if (present(unit)) u = unit
      write (u, '(a24, 3a15, a10)') 'array', 'min', 'max', 'max |x|', 'nonfinite'
      call mem_for_each_r64(row, tag, sync)
   contains
      subroutine row(name, a)
         character(len=*), intent(in) :: name
         real(dp), intent(inout) :: a(:)
         integer(int64) :: i, nbad
         real(dp) :: lo, hi, amax
         lo = huge(1.0_dp)
         hi = -huge(1.0_dp)
         amax = 0.0_dp
         nbad = 0
         do i = 1, size(a, kind=int64)
            if (.not. ieee_is_finite(a(i))) then
               nbad = nbad + 1
               cycle
            end if
            lo = min(lo, a(i))
            hi = max(hi, a(i))
            amax = max(amax, abs(a(i)))
         end do
         write (u, '(a24, 3es15.6, i10)') name, lo, hi, amax, nbad
      end subroutine row
   end subroutine mem_report_values

   pure logical function opt(x, default)
      logical, intent(in), optional :: x
      logical, intent(in) :: default
      opt = default
      if (present(x)) opt = x
   end function opt

end module mem_collective
