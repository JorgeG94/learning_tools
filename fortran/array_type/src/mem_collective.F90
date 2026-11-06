!! Collective operations over every ledger-tracked array with a given tag.
module mem_collective
   !! The ledger knows the host address, kind and size of every tracked
   !! array, so it can act as the "array of arrays" without a type
   !! hierarchy or an arena: each operation walks the live entries, filters
   !! by tag, and views the memory through `c_f_pointer`.
   !!
   !! * `mem_update_host` / `mem_update_device` move every mapped array with
   !!   the tag (or all of them), whatever its kind or rank.
   !! * `mem_checksum` summarises the real(dp) arrays: sum, sum of |x| and an
   !!   exact bit hash. It reads HOST memory: call `mem_update_host` first if
   !!   the device copy is the current one.
   !! * `mem_for_each_r64` hands every real(dp) array to a callback as a
   !!   contiguous rank-1 view (internal procedures work, so the callback
   !!   can capture state).
   !! * `mem_report_values` prints min/max/non-finite count per array.
   !!
   !! All loops here are plain DO on the host on purpose: under -stdpar=gpu
   !! a DO CONCURRENT in this module would be offloaded.
   use, intrinsic :: iso_fortran_env, only: int8, int64, real64, output_unit
   use, intrinsic :: iso_c_binding, only: c_associated, c_f_pointer
   use, intrinsic :: ieee_arithmetic, only: ieee_is_finite
   use mem_ledger, only: mem_object_t, mem_num_objects, mem_get_object, MEM_KIND_R64
   implicit none
   private

   integer, parameter :: dp = real64

   public :: mem_checksum_t, mem_update_host, mem_update_device, mem_checksum
   public :: mem_for_each_r64, mem_report_values, mem_visit_r64

   type :: mem_checksum_t
      integer :: nobj = 0              !! real(dp) arrays included
      integer(int64) :: nelem = 0
      real(dp) :: sum = 0.0_dp
      real(dp) :: abs_sum = 0.0_dp
      integer(int64) :: hash = 0       !! order-dependent, exact: equal iff bits equal (w.h.p.)
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

contains

   logical function selected(obj, tag)
      type(mem_object_t), intent(in) :: obj
      integer, intent(in), optional :: tag
      selected = obj%live .and. obj%nelem > 0 .and. c_associated(obj%addr)
      if (present(tag)) selected = selected .and. obj%tag == tag
   end function selected

   ! ------------------------------------------------------------ transfers

   subroutine mem_update_host(tag)
      !! Device -> host for every mapped array with `tag` (all if absent).
      integer, intent(in), optional :: tag
      call update_all(.true., tag)
   end subroutine mem_update_host

   subroutine mem_update_device(tag)
      !! Host -> device for every mapped array with `tag` (all if absent).
      integer, intent(in), optional :: tag
      call update_all(.false., tag)
   end subroutine mem_update_device

   subroutine update_all(to_host, tag)
      logical, intent(in) :: to_host
      integer, intent(in), optional :: tag
      type(mem_object_t) :: obj
      integer(int64) :: id
      integer(int8), pointer :: bytes(:)
      do id = 1, mem_num_objects()
         obj = mem_get_object(id)
         if (.not. selected(obj, tag)) cycle
         if (obj%dev == 0) cycle                ! host-only: nothing to move
         ! A byte view covers any kind/rank; the runtime finds the mapping
         ! by host address range.
         call c_f_pointer(obj%addr, bytes, [obj%host])
         if (to_host) then
            !$acc update self(bytes)
         else
            !$acc update device(bytes)
         end if
      end do
   end subroutine update_all

   ! ----------------------------------------------------------- inspection

   function mem_checksum(tag) result(cs)
      !! Summary of the HOST copies of the real(dp) arrays with `tag`.
      integer, intent(in), optional :: tag
      type(mem_checksum_t) :: cs
      type(mem_object_t) :: obj
      integer(int64) :: id, i
      real(dp), pointer :: a(:)
      do id = 1, mem_num_objects()
         obj = mem_get_object(id)
         if (.not. selected(obj, tag)) cycle
         if (obj%kind_code /= MEM_KIND_R64) cycle
         call c_f_pointer(obj%addr, a, [obj%nelem])
         cs%nobj = cs%nobj + 1
         cs%nelem = cs%nelem + obj%nelem
         do i = 1, obj%nelem
            cs%sum = cs%sum + a(i)
            cs%abs_sum = cs%abs_sum + abs(a(i))
            ! rotate-xor: no integer overflow, sensitive to every bit
            cs%hash = ieor(ishftc(cs%hash, 13), transfer(a(i), cs%hash))
         end do
      end do
   end function mem_checksum

   subroutine mem_for_each_r64(visit, tag)
      !! Call `visit(name, a)` for every real(dp) array with `tag`, where `a`
      !! is a rank-1 view of its HOST memory.
      procedure(mem_visit_r64) :: visit
      integer, intent(in), optional :: tag
      type(mem_object_t) :: obj
      integer(int64) :: id
      real(dp), pointer, contiguous :: a(:)
      do id = 1, mem_num_objects()
         obj = mem_get_object(id)
         if (.not. selected(obj, tag)) cycle
         if (obj%kind_code /= MEM_KIND_R64) cycle
         call c_f_pointer(obj%addr, a, [obj%nelem])
         call visit(trim(obj%name), a)
      end do
   end subroutine mem_for_each_r64

   subroutine mem_report_values(tag, unit)
      !! min / max / non-finite count of each real(dp) array (HOST copies).
      integer, intent(in), optional :: tag
      integer, intent(in), optional :: unit
      integer :: u
      u = output_unit
      if (present(unit)) u = unit
      write (u, '(a24, 3a15, a10)') 'array', 'min', 'max', 'max |x|', 'nonfinite'
      call mem_for_each_r64(row, tag)
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

end module mem_collective
