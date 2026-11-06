!! Byte ledger for tracked allocations, host and device.
module mem_ledger
   !! A process-wide ledger that knows how many bytes every tracked object
   !! holds on the host and on the device, grouped by an owner *tag*
   !! ("swe_state", "barotropic", "halo", ...).
   !!
   !! The ledger is kind- and rank-agnostic: it only ever sees byte counts.
   !! The typed wrappers in `devarray` (fypp-generated) are the intended
   !! callers, but anything can register itself.
   !!
   !! Every tracked object gets a unique, never-reused `id`. Objects are
   !! never removed from the registry, only marked dead, so that a second
   !! `mem_untrack` of the same id (a double destroy, or the destroy of a
   !! copy made by intrinsic assignment, which duplicates the id) is caught
   !! and reported by name instead of silently corrupting the counts.
   !!
   !! Not thread safe by design: tracking happens at init and teardown,
   !! never inside parallel regions.
   use, intrinsic :: iso_fortran_env, only: int64, real64, error_unit, output_unit
   use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
   implicit none
   private

   public :: MEM_TAG_NAME_LEN, MEM_OBJ_NAME_LEN, MEM_UNTAGGED
   public :: MEM_KIND_RAW, MEM_KIND_R64, MEM_KIND_R32, MEM_KIND_I32, MEM_KIND_I64
   public :: mem_object_t, mem_num_objects, mem_get_object
   public :: mem_register_tag, mem_tag_name, mem_num_tags
   public :: mem_track, mem_untrack, mem_track_device, mem_untrack_device
   public :: mem_is_live, mem_object_name
   public :: mem_host_bytes, mem_device_bytes, mem_host_peak, mem_device_peak
   public :: mem_live_count
   public :: mem_report, mem_check_leaks, mem_fatal, mem_human_bytes

   integer, parameter :: MEM_TAG_NAME_LEN = 32
   integer, parameter :: MEM_OBJ_NAME_LEN = 64
   integer, parameter :: MEM_UNTAGGED = 0
   integer, parameter :: MAX_TAGS = 64

   ! Element kind of a tracked object, so collective operations know how to
   ! view its memory. RAW objects can be moved (update) but not interpreted.
   integer, parameter :: MEM_KIND_RAW = 0
   integer, parameter :: MEM_KIND_R64 = 1
   integer, parameter :: MEM_KIND_R32 = 2
   integer, parameter :: MEM_KIND_I32 = 3
   integer, parameter :: MEM_KIND_I64 = 4

   type :: tag_stats_t
      character(len=MEM_TAG_NAME_LEN) :: name = ''
      integer(int64) :: host = 0, host_peak = 0
      integer(int64) :: dev = 0, dev_peak = 0
      integer :: live = 0
   end type tag_stats_t

   type :: mem_object_t
      !! One ledger entry. `mem_get_object` hands out copies; the registry
      !! itself is private.
      character(len=MEM_OBJ_NAME_LEN) :: name = ''
      integer :: tag = MEM_UNTAGGED
      integer(int64) :: host = 0          !! host bytes
      integer(int64) :: dev = 0           !! device bytes (0 = not mapped)
      logical :: live = .false.
      type(c_ptr) :: addr = c_null_ptr    !! host address of the first element
      integer :: kind_code = MEM_KIND_RAW
      integer(int64) :: nelem = 0
   end type mem_object_t

   type(tag_stats_t), save :: tags(0:MAX_TAGS)
   integer, save :: n_tags = 0
   type(mem_object_t), allocatable, save :: objs(:)
   integer(int64), save :: n_objs = 0
   integer(int64), save :: tot_host = 0, tot_host_peak = 0
   integer(int64), save :: tot_dev = 0, tot_dev_peak = 0

contains

   ! ------------------------------------------------------------------ tags

   integer function mem_register_tag(name) result(tag)
      !! Return the tag for `name`, creating it on first use. Idempotent, so
      !! each component can simply ask for its own tag in its init.
      character(len=*), intent(in) :: name
      integer :: t
      do t = 1, n_tags
         if (tags(t)%name == name) then
            tag = t
            return
         end if
      end do
      if (n_tags >= MAX_TAGS) call mem_fatal("mem_register_tag: too many tags (MAX_TAGS)")
      n_tags = n_tags + 1
      tags(n_tags)%name = name
      tag = n_tags
   end function mem_register_tag

   function mem_tag_name(tag) result(name)
      integer, intent(in) :: tag
      character(len=:), allocatable :: name
      call check_tag(tag, "mem_tag_name")
      if (tag == MEM_UNTAGGED) then
         name = "(untagged)"
      else
         name = trim(tags(tag)%name)
      end if
   end function mem_tag_name

   integer function mem_num_tags()
      mem_num_tags = n_tags
   end function mem_num_tags

   ! ------------------------------------------------------------- tracking

   integer(int64) function mem_track(name, tag, nbytes, addr, kind_code, nelem) result(id)
      !! Register a new live object holding `nbytes` of host memory.
      !! `addr`/`kind_code`/`nelem` make it visible to `mem_collective`; the
      !! address must stay valid until `mem_untrack`, so the owner must never
      !! reallocate behind the ledger's back.
      character(len=*), intent(in) :: name
      integer, intent(in) :: tag
      integer(int64), intent(in) :: nbytes
      type(c_ptr), intent(in), optional :: addr
      integer, intent(in), optional :: kind_code
      integer(int64), intent(in), optional :: nelem
      call check_tag(tag, "mem_track")
      call grow_registry()
      n_objs = n_objs + 1
      id = n_objs
      objs(id)%name = name
      objs(id)%tag = tag
      objs(id)%host = nbytes
      objs(id)%dev = 0
      objs(id)%live = .true.
      objs(id)%addr = c_null_ptr
      objs(id)%kind_code = MEM_KIND_RAW
      objs(id)%nelem = 0
      if (present(addr)) objs(id)%addr = addr
      if (present(kind_code)) objs(id)%kind_code = kind_code
      if (present(nelem)) objs(id)%nelem = nelem

      tags(tag)%live = tags(tag)%live + 1
      tags(tag)%host = tags(tag)%host + nbytes
      tags(tag)%host_peak = max(tags(tag)%host_peak, tags(tag)%host)
      tot_host = tot_host + nbytes
      tot_host_peak = max(tot_host_peak, tot_host)
   end function mem_track

   subroutine mem_untrack(id)
      !! Retire a live object. It must not still hold device memory.
      integer(int64), intent(in) :: id
      integer :: tag
      call check_live(id, "mem_untrack")
      if (objs(id)%dev /= 0) call mem_fatal("mem_untrack: '"//trim(objs(id)%name)// &
                                            "' still holds device memory; untrack the device copy first")
      tag = objs(id)%tag
      tags(tag)%live = tags(tag)%live - 1
      tags(tag)%host = tags(tag)%host - objs(id)%host
      tot_host = tot_host - objs(id)%host
      objs(id)%live = .false.
      objs(id)%addr = c_null_ptr
   end subroutine mem_untrack

   subroutine mem_track_device(id, nbytes)
      integer(int64), intent(in) :: id
      integer(int64), intent(in) :: nbytes
      integer :: tag
      call check_live(id, "mem_track_device")
      if (objs(id)%dev /= 0) call mem_fatal("mem_track_device: '"//trim(objs(id)%name)// &
                                            "' is already mapped to the device")
      tag = objs(id)%tag
      objs(id)%dev = nbytes
      tags(tag)%dev = tags(tag)%dev + nbytes
      tags(tag)%dev_peak = max(tags(tag)%dev_peak, tags(tag)%dev)
      tot_dev = tot_dev + nbytes
      tot_dev_peak = max(tot_dev_peak, tot_dev)
   end subroutine mem_track_device

   subroutine mem_untrack_device(id)
      integer(int64), intent(in) :: id
      integer :: tag
      call check_live(id, "mem_untrack_device")
      tag = objs(id)%tag
      tags(tag)%dev = tags(tag)%dev - objs(id)%dev
      tot_dev = tot_dev - objs(id)%dev
      objs(id)%dev = 0
   end subroutine mem_untrack_device

   logical function mem_is_live(id)
      integer(int64), intent(in) :: id
      mem_is_live = .false.
      if (id >= 1 .and. id <= n_objs) mem_is_live = objs(id)%live
   end function mem_is_live

   function mem_object_name(id) result(name)
      !! Name of any object ever tracked, live or dead.
      integer(int64), intent(in) :: id
      character(len=:), allocatable :: name
      if (id >= 1 .and. id <= n_objs) then
         name = trim(objs(id)%name)
      else
         name = "(unknown id)"
      end if
   end function mem_object_name

   integer(int64) function mem_num_objects()
      !! Number of objects ever tracked; valid ids are 1..mem_num_objects().
      mem_num_objects = n_objs
   end function mem_num_objects

   function mem_get_object(id) result(obj)
      !! Copy of entry `id` (live or dead). Iterate 1..mem_num_objects().
      integer(int64), intent(in) :: id
      type(mem_object_t) :: obj
      if (id < 1 .or. id > n_objs) call mem_fatal("mem_get_object: unknown ledger id")
      obj = objs(id)
   end function mem_get_object

   ! -------------------------------------------------------------- queries
   ! Without `tag` these return the process total.

   integer(int64) function mem_host_bytes(tag) result(b)
      integer, intent(in), optional :: tag
      b = tot_host
      if (present(tag)) then
         call check_tag(tag, "mem_host_bytes")
         b = tags(tag)%host
      end if
   end function mem_host_bytes

   integer(int64) function mem_device_bytes(tag) result(b)
      integer, intent(in), optional :: tag
      b = tot_dev
      if (present(tag)) then
         call check_tag(tag, "mem_device_bytes")
         b = tags(tag)%dev
      end if
   end function mem_device_bytes

   integer(int64) function mem_host_peak(tag) result(b)
      integer, intent(in), optional :: tag
      b = tot_host_peak
      if (present(tag)) then
         call check_tag(tag, "mem_host_peak")
         b = tags(tag)%host_peak
      end if
   end function mem_host_peak

   integer(int64) function mem_device_peak(tag) result(b)
      integer, intent(in), optional :: tag
      b = tot_dev_peak
      if (present(tag)) then
         call check_tag(tag, "mem_device_peak")
         b = tags(tag)%dev_peak
      end if
   end function mem_device_peak

   integer function mem_live_count(tag) result(n)
      integer, intent(in), optional :: tag
      integer :: t
      if (present(tag)) then
         call check_tag(tag, "mem_live_count")
         n = tags(tag)%live
      else
         n = 0
         do t = 0, n_tags
            n = n + tags(t)%live
         end do
      end if
   end function mem_live_count

   ! ------------------------------------------------------------ reporting

   subroutine mem_report(unit, title)
      !! Per-tag table of live objects, current and peak bytes.
      integer, intent(in), optional :: unit
      character(len=*), intent(in), optional :: title
      integer :: u, t
      character(len=*), parameter :: hdr = '(a20, a7, 4a13)'
      character(len=*), parameter :: row = '(a20, i7, 4a13)'
      u = output_unit
      if (present(unit)) u = unit
      write (u, '(a)') ''
      if (present(title)) write (u, '(a)') '== memory ledger: '//title
      write (u, hdr) 'tag', 'live', 'host', 'host peak', 'device', 'dev peak'
      do t = 0, n_tags
         if (t == MEM_UNTAGGED .and. tags(t)%host_peak == 0) cycle
         write (u, row) mem_tag_name(t), tags(t)%live, &
            mem_human_bytes(tags(t)%host), mem_human_bytes(tags(t)%host_peak), &
            mem_human_bytes(tags(t)%dev), mem_human_bytes(tags(t)%dev_peak)
      end do
      write (u, row) 'TOTAL', mem_live_count(), &
         mem_human_bytes(tot_host), mem_human_bytes(tot_host_peak), &
         mem_human_bytes(tot_dev), mem_human_bytes(tot_dev_peak)
   end subroutine mem_report

   integer function mem_check_leaks(unit) result(n)
      !! List every object still live. Call at shutdown: this is what stands
      !! in for finalizers. Returns the number of leaked objects.
      integer, intent(in), optional :: unit
      integer :: u
      integer(int64) :: i
      u = output_unit
      if (present(unit)) u = unit
      n = 0
      do i = 1, n_objs
         if (.not. objs(i)%live) cycle
         n = n + 1
         write (u, '(a, a, a, a, a, a, a, a)') 'LEAK: ', trim(objs(i)%name), &
            '  [', mem_tag_name(objs(i)%tag), ']  host ', trim(adjustl(mem_human_bytes(objs(i)%host))), &
            '  device ', trim(adjustl(mem_human_bytes(objs(i)%dev)))
      end do
   end function mem_check_leaks

   function mem_human_bytes(b) result(s)
      integer(int64), intent(in) :: b
      character(len=12) :: s
      real(real64) :: x
      character(len=3), parameter :: units(0:4) = ['B  ', 'KiB', 'MiB', 'GiB', 'TiB']
      integer :: k
      x = real(b, real64)
      k = 0
      do while (abs(x) >= 1024.0_real64 .and. k < 4)
         x = x/1024.0_real64
         k = k + 1
      end do
      if (k == 0) then
         write (s, '(i8, 1x, a3)') b, units(k)
      else
         write (s, '(f8.2, 1x, a3)') x, units(k)
      end if
   end function mem_human_bytes

   subroutine mem_fatal(msg)
      character(len=*), intent(in) :: msg
      write (error_unit, '(a)') 'mem_ledger FATAL: '//msg
      error stop 1
   end subroutine mem_fatal

   ! ------------------------------------------------------------ internals

   subroutine check_tag(tag, where)
      integer, intent(in) :: tag
      character(len=*), intent(in) :: where
      if (tag < 0 .or. tag > n_tags) call mem_fatal(where//": unregistered tag")
   end subroutine check_tag

   subroutine check_live(id, where)
      integer(int64), intent(in) :: id
      character(len=*), intent(in) :: where
      if (id < 1 .or. id > n_objs) call mem_fatal(where//": unknown ledger id")
      if (.not. objs(id)%live) call mem_fatal(where//": '"//trim(objs(id)%name)// &
                                              "' is already dead. Double destroy, or the object was"// &
                                              " copied by intrinsic assignment (a = b duplicates the ledger id)")
   end subroutine check_live

   subroutine grow_registry()
      type(mem_object_t), allocatable :: tmp(:)
      if (.not. allocated(objs)) allocate (objs(64))
      if (n_objs < size(objs, kind=int64)) return
      allocate (tmp(2*size(objs)))
      tmp(1:size(objs)) = objs
      call move_alloc(tmp, objs)
   end subroutine grow_registry

end module mem_ledger
