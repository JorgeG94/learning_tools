!! Ledger state and implementation. INTERNAL: use `mem_ledger` (user API)
!! or `mem_ledger_backend` (for authors of tracked types) instead.
module mem_ledger_core
   !! A process-wide ledger of tracked objects: how many bytes each holds on
   !! the host and on the device, grouped by an owner tag. One ledger per
   !! process (per MPI rank); reduce across ranks with `mem_tag_stats`.
   !!
   !! Object ids encode (generation, slot). Slots are reused after an object
   !! is untracked, so the registry stays as small as the number of live
   !! objects, while a stale id (double destroy, or a copy that duplicated
   !! the id) still never matches the new occupant of its slot.
   !!
   !! Not thread safe by design: tracking happens at init and teardown,
   !! never inside parallel regions.
   use, intrinsic :: iso_fortran_env, only: int64, real64, error_unit, output_unit
   use, intrinsic :: iso_c_binding, only: c_ptr, c_null_ptr
   implicit none
   private

   integer, parameter, public :: MEM_TAG_NAME_LEN = 32
   integer, parameter, public :: MEM_OBJ_NAME_LEN = 64

   ! Element kind of a tracked object, so collective operations know how to
   ! view its memory. RAW objects can be moved (update) but not interpreted.
   integer, parameter, public :: MEM_KIND_RAW = 0
   integer, parameter, public :: MEM_KIND_R64 = 1
   integer, parameter, public :: MEM_KIND_I32 = 2

   integer, parameter :: MAX_TAGS = 64
   integer(int64), parameter :: SLOT_MASK = 4294967295_int64   ! 2**32 - 1

   type, public :: mem_tag_t
      !! Opaque owner tag. Get one with `mem_tag("name")`. A declared but
      !! unset tag, or one from before `mem_reset`, is rejected loudly.
      private
      integer :: id = -1
      integer :: epoch = -1
   end type mem_tag_t

   type, public :: mem_tag_stats_t
      !! Snapshot of one tag, e.g. for an MPI reduction across ranks.
      character(len=MEM_TAG_NAME_LEN) :: name = ''
      integer :: live = 0
      integer(int64) :: host = 0, host_peak = 0
      integer(int64) :: device = 0, device_peak = 0
   end type mem_tag_stats_t

   abstract interface
      logical function mem_validate_i(owner)
         !! Return .false. if the object behind `owner` no longer matches
         !! what it registered (e.g. its storage was reallocated).
         import :: c_ptr
         type(c_ptr), value :: owner
      end function mem_validate_i
   end interface

   type, public :: mem_object_t
      !! One ledger entry (backend view).
      character(len=MEM_OBJ_NAME_LEN) :: name = ''
      integer :: tag = 0                     !! internal tag index
      integer(int64) :: host = 0             !! host bytes
      integer(int64) :: dev = 0              !! device bytes
      logical :: mapped = .false.            !! has a device copy (independent of size)
      logical :: live = .false.
      integer(int64) :: gen = 0
      type(c_ptr) :: addr = c_null_ptr       !! host address of the first element
      integer :: kind_code = MEM_KIND_RAW
      integer(int64) :: nelem = 0
      type(c_ptr) :: owner = c_null_ptr      !! the tracked object, for `validate`
      procedure(mem_validate_i), pointer, nopass :: validate => null()
   end type mem_object_t

   type :: tag_rec_t
      character(len=MEM_TAG_NAME_LEN) :: name = ''
      integer :: live = 0
      integer(int64) :: host = 0, host_peak = 0
      integer(int64) :: dev = 0, dev_peak = 0
   end type tag_rec_t

   type(tag_rec_t), save :: tags(MAX_TAGS)
   integer, save :: n_tags = 0
   integer, save :: epoch = 1
   type(mem_object_t), allocatable, save :: objs(:)
   integer, save :: n_slots = 0
   integer, allocatable, save :: free_slots(:)
   integer, save :: n_free = 0
   integer(int64), save :: tot_host = 0, tot_host_peak = 0
   integer(int64), save :: tot_dev = 0, tot_dev_peak = 0

   public :: mem_validate_i
   public :: operator(==)
   public :: mem_tag, mem_find_tag, mem_tag_name, mem_tag_stats
   public :: mem_track, mem_untrack, mem_track_device, mem_untrack_device
   public :: mem_is_live, mem_object_name, mem_get_object, mem_live_ids, mem_num_slots
   public :: mem_check_object, mem_tag_matches
   public :: mem_host_bytes, mem_device_bytes, mem_host_peak, mem_device_peak, mem_live_count
   public :: mem_report, mem_check_leaks, mem_reset, mem_fatal, mem_human_bytes

   interface operator(==)
      module procedure tag_equal
   end interface

contains

   ! ------------------------------------------------------------------ tags

   function mem_tag(name) result(tag)
      !! The tag called `name`, created on first use. Idempotent, so a
      !! component can ask for its own tag in its init.
      character(len=*), intent(in) :: name
      type(mem_tag_t) :: tag
      integer :: t
      if (len_trim(name) == 0) call mem_fatal("mem_tag: empty tag name")
      if (len_trim(name) > MEM_TAG_NAME_LEN) &
         call mem_fatal("mem_tag: name '"//trim(name)//"' is longer than MEM_TAG_NAME_LEN")
      t = find_tag_index(name)
      if (t == 0) then
         if (n_tags >= MAX_TAGS) call mem_fatal("mem_tag: too many tags")
         n_tags = n_tags + 1
         tags(n_tags) = tag_rec_t(name=name)
         t = n_tags
      end if
      tag%id = t
      tag%epoch = epoch
   end function mem_tag

   function mem_find_tag(name) result(tag)
      !! The existing tag called `name`; fatal if no component created it.
      !! Use this wherever you look a tag up by name (drivers, diagnostics),
      !! so a typo stops the run instead of creating an empty tag.
      character(len=*), intent(in) :: name
      type(mem_tag_t) :: tag
      integer :: t
      t = find_tag_index(name)
      if (t == 0) call mem_fatal("mem_find_tag: no tag named '"//trim(name)//"'")
      tag%id = t
      tag%epoch = epoch
   end function mem_find_tag

   function mem_tag_name(tag) result(name)
      type(mem_tag_t), intent(in) :: tag
      character(len=:), allocatable :: name
      call check_tag(tag, "mem_tag_name")
      name = trim(tags(tag%id)%name)
   end function mem_tag_name

   function mem_tag_stats(tag) result(s)
      type(mem_tag_t), intent(in) :: tag
      type(mem_tag_stats_t) :: s
      call check_tag(tag, "mem_tag_stats")
      associate (r => tags(tag%id))
         s = mem_tag_stats_t(r%name, r%live, r%host, r%host_peak, r%dev, r%dev_peak)
      end associate
   end function mem_tag_stats

   elemental logical function tag_equal(a, b)
      type(mem_tag_t), intent(in) :: a, b
      tag_equal = a%id == b%id .and. a%epoch == b%epoch
   end function tag_equal

   logical function mem_tag_matches(obj, tag)
      !! Backend helper: does entry `obj` belong to `tag` (any if absent)?
      type(mem_object_t), intent(in) :: obj
      type(mem_tag_t), intent(in), optional :: tag
      mem_tag_matches = .true.
      if (present(tag)) then
         call check_tag(tag, "mem_tag_matches")
         mem_tag_matches = obj%tag == tag%id
      end if
   end function mem_tag_matches

   ! ------------------------------------------------------------- tracking

   integer(int64) function mem_track(name, tag, nbytes, kind_code, addr, nelem, owner, validate) result(id)
      !! Register a new live object holding `nbytes` of host memory.
      !! `addr`/`kind_code`/`nelem` make it visible to collective operations;
      !! `owner`/`validate` let the ledger check it is still intact before
      !! touching `addr`.
      character(len=*), intent(in) :: name
      type(mem_tag_t), intent(in) :: tag
      integer(int64), intent(in) :: nbytes
      integer, intent(in), optional :: kind_code
      type(c_ptr), intent(in), optional :: addr
      integer(int64), intent(in), optional :: nelem
      type(c_ptr), intent(in), optional :: owner
      procedure(mem_validate_i), optional :: validate
      integer :: s
      integer(int64) :: gen

      call check_tag(tag, "mem_track('"//trim(name)//"')")
      if (len_trim(name) > MEM_OBJ_NAME_LEN) &
         call mem_fatal("mem_track: name '"//trim(name)//"' is longer than MEM_OBJ_NAME_LEN")
      s = take_slot()
      gen = objs(s)%gen + 1
      objs(s) = mem_object_t(name=name, tag=tag%id, host=nbytes, live=.true., gen=gen)
      if (present(kind_code)) objs(s)%kind_code = kind_code
      if (present(addr)) objs(s)%addr = addr
      if (present(nelem)) objs(s)%nelem = nelem
      if (present(owner)) objs(s)%owner = owner
      if (present(validate)) objs(s)%validate => validate
      id = ior(shiftl(gen, 32), int(s, int64))

      associate (r => tags(tag%id))
         r%live = r%live + 1
         r%host = r%host + nbytes
         r%host_peak = max(r%host_peak, r%host)
      end associate
      tot_host = tot_host + nbytes
      tot_host_peak = max(tot_host_peak, tot_host)
   end function mem_track

   subroutine mem_untrack(id)
      !! Retire a live object. It must not still be mapped to the device.
      integer(int64), intent(in) :: id
      integer :: s
      s = check_live(id, "mem_untrack")
      if (objs(s)%mapped) call mem_fatal("mem_untrack: '"//trim(objs(s)%name)// &
                                         "' is still mapped to the device; untrack the device copy first")
      associate (r => tags(objs(s)%tag))
         r%live = r%live - 1
         r%host = r%host - objs(s)%host
      end associate
      tot_host = tot_host - objs(s)%host
      objs(s)%live = .false.
      objs(s)%addr = c_null_ptr
      objs(s)%owner = c_null_ptr
      objs(s)%validate => null()
      call push_free(s)
   end subroutine mem_untrack

   subroutine mem_track_device(id, nbytes)
      integer(int64), intent(in) :: id
      integer(int64), intent(in) :: nbytes
      integer :: s
      s = check_live(id, "mem_track_device")
      if (objs(s)%mapped) call mem_fatal("mem_track_device: '"//trim(objs(s)%name)// &
                                         "' is already mapped to the device")
      objs(s)%mapped = .true.
      objs(s)%dev = nbytes
      associate (r => tags(objs(s)%tag))
         r%dev = r%dev + nbytes
         r%dev_peak = max(r%dev_peak, r%dev)
      end associate
      tot_dev = tot_dev + nbytes
      tot_dev_peak = max(tot_dev_peak, tot_dev)
   end subroutine mem_track_device

   subroutine mem_untrack_device(id)
      integer(int64), intent(in) :: id
      integer :: s
      s = check_live(id, "mem_untrack_device")
      if (.not. objs(s)%mapped) call mem_fatal("mem_untrack_device: '"//trim(objs(s)%name)// &
                                               "' is not mapped to the device")
      tags(objs(s)%tag)%dev = tags(objs(s)%tag)%dev - objs(s)%dev
      tot_dev = tot_dev - objs(s)%dev
      objs(s)%dev = 0
      objs(s)%mapped = .false.
   end subroutine mem_untrack_device

   logical function mem_is_live(id)
      integer(int64), intent(in) :: id
      integer :: s
      mem_is_live = .false.
      s = slot_of(id)
      if (s > 0) mem_is_live = objs(s)%live
   end function mem_is_live

   function mem_object_name(id) result(name)
      !! Name for messages: works for live and dead ids.
      integer(int64), intent(in) :: id
      character(len=:), allocatable :: name
      integer :: s
      s = slot_of(id)
      if (s > 0) then
         name = trim(objs(s)%name)
      else if (raw_slot(id) >= 1 .and. raw_slot(id) <= n_slots) then
         name = "(destroyed object; its ledger slot has been reused)"
      else
         name = "(unknown ledger id)"
      end if
   end function mem_object_name

   function mem_get_object(id) result(obj)
      integer(int64), intent(in) :: id
      type(mem_object_t) :: obj
      integer :: s
      s = slot_of(id)
      if (s == 0) call mem_fatal("mem_get_object: stale or unknown ledger id")
      obj = objs(s)
   end function mem_get_object

   function mem_live_ids(tag) result(ids)
      !! Ids of the live objects with `tag` (all if absent), in slot order.
      type(mem_tag_t), intent(in), optional :: tag
      integer(int64), allocatable :: ids(:)
      integer :: s, n
      if (present(tag)) call check_tag(tag, "mem_live_ids")
      n = 0
      do s = 1, n_slots
         if (selected(s)) n = n + 1
      end do
      allocate (ids(n))
      n = 0
      do s = 1, n_slots
         if (.not. selected(s)) cycle
         n = n + 1
         ids(n) = ior(shiftl(objs(s)%gen, 32), int(s, int64))
      end do
   contains
      logical function selected(slot)
         integer, intent(in) :: slot
         selected = objs(slot)%live
         if (present(tag)) selected = selected .and. objs(slot)%tag == tag%id
      end function selected
   end function mem_live_ids

   integer function mem_num_slots()
      !! Registry size; stays near the peak number of live objects.
      mem_num_slots = n_slots
   end function mem_num_slots

   subroutine mem_check_object(id, where)
      !! Fatal unless `id` is live and its owner still validates.
      integer(int64), intent(in) :: id
      character(len=*), intent(in) :: where
      integer :: s
      s = check_live(id, where)
      if (.not. associated(objs(s)%validate)) return
      if (.not. objs(s)%validate(objs(s)%owner)) &
         call mem_fatal(where//": '"//trim(objs(s)%name)//"' was reallocated or deallocated"// &
                        " behind its owner (realloc-on-assignment? write x%data(:,:) = ..."// &
                        " instead of x%data = ...)")
   end subroutine mem_check_object

   ! -------------------------------------------------------------- queries
   ! Without `tag` these return the process total.

   integer(int64) function mem_host_bytes(tag) result(b)
      type(mem_tag_t), intent(in), optional :: tag
      b = tot_host
      if (present(tag)) then
         call check_tag(tag, "mem_host_bytes")
         b = tags(tag%id)%host
      end if
   end function mem_host_bytes

   integer(int64) function mem_device_bytes(tag) result(b)
      type(mem_tag_t), intent(in), optional :: tag
      b = tot_dev
      if (present(tag)) then
         call check_tag(tag, "mem_device_bytes")
         b = tags(tag%id)%dev
      end if
   end function mem_device_bytes

   integer(int64) function mem_host_peak(tag) result(b)
      type(mem_tag_t), intent(in), optional :: tag
      b = tot_host_peak
      if (present(tag)) then
         call check_tag(tag, "mem_host_peak")
         b = tags(tag%id)%host_peak
      end if
   end function mem_host_peak

   integer(int64) function mem_device_peak(tag) result(b)
      type(mem_tag_t), intent(in), optional :: tag
      b = tot_dev_peak
      if (present(tag)) then
         call check_tag(tag, "mem_device_peak")
         b = tags(tag%id)%dev_peak
      end if
   end function mem_device_peak

   integer function mem_live_count(tag) result(n)
      type(mem_tag_t), intent(in), optional :: tag
      if (present(tag)) then
         call check_tag(tag, "mem_live_count")
         n = tags(tag%id)%live
      else
         n = sum(tags(1:n_tags)%live)
      end if
   end function mem_live_count

   ! ------------------------------------------------------------ reporting

   subroutine mem_report(unit, title, objects)
      !! Per-tag table of live objects, current and peak bytes.
      !! `objects=.true.` also lists each live object, largest first.
      integer, intent(in), optional :: unit
      character(len=*), intent(in), optional :: title
      logical, intent(in), optional :: objects
      integer :: u, t, w, k, s
      logical :: list
      character(len=64) :: hdr, row, orow
      integer, allocatable :: order(:)

      u = output_unit
      if (present(unit)) u = unit
      list = .false.
      if (present(objects)) list = objects
      w = 5
      do t = 1, n_tags
         w = max(w, len_trim(tags(t)%name) + 2)
      end do
      if (list) then
         do s = 1, n_slots
            if (objs(s)%live) w = max(w, len_trim(objs(s)%name) + 4)
         end do
      end if
      hdr = '(a, a7, 4a13)'
      row = '(a, i7, 4a13)'
      orow = '(a, 7x, a13, 13x, a13)'

      write (u, '(a)') ''
      if (present(title)) write (u, '(a)') '== memory ledger: '//title
      write (u, hdr) lj('tag'), 'live', 'host', 'host peak', 'device', 'dev peak'
      do t = 1, n_tags
         associate (r => tags(t))
            write (u, row) lj(trim(r%name)), r%live, rj(r%host), rj(r%host_peak), rj(r%dev), rj(r%dev_peak)
         end associate
         if (.not. list) cycle
         order = live_slots_of_tag_by_size(t)
         do k = 1, size(order)
            s = order(k)
            write (u, orow) lj('  '//trim(objs(s)%name)), rj(objs(s)%host), rj(objs(s)%dev)
         end do
      end do
      write (u, row) lj('TOTAL'), mem_live_count(), rj(tot_host), rj(tot_host_peak), rj(tot_dev), rj(tot_dev_peak)
   contains
      function lj(name) result(col)
         !! left-justified in the name column
         character(len=*), intent(in) :: name
         character(len=:), allocatable :: col
         col = repeat(' ', w)
         col(1:min(w, len(name))) = name
      end function lj

      function rj(b) result(sb)
         !! right-justified in a 13-character column
         integer(int64), intent(in) :: b
         character(len=13) :: sb
         character(len=:), allocatable :: s
         s = mem_human_bytes(b)
         sb = ''
         sb(max(1, 14 - len(s)):) = s
      end function rj
   end subroutine mem_report

   integer function mem_check_leaks(unit) result(n)
      !! List every object still live. Call at shutdown: this is what stands
      !! in for finalizers. Returns the number of leaked objects.
      integer, intent(in), optional :: unit
      integer :: u, s
      u = output_unit
      if (present(unit)) u = unit
      n = 0
      do s = 1, n_slots
         if (.not. objs(s)%live) cycle
         n = n + 1
         write (u, '(10a)') 'LEAK: ', trim(objs(s)%name), '  [', trim(tags(objs(s)%tag)%name), &
            ']  host ', mem_human_bytes(objs(s)%host), '  device ', mem_human_bytes(objs(s)%dev)
      end do
   end function mem_check_leaks

   subroutine mem_reset()
      !! Forget everything (for unit tests). Fatal if anything is still
      !! live. Every existing mem_tag_t becomes invalid.
      if (mem_live_count() /= 0) call mem_fatal("mem_reset: objects are still live (see mem_check_leaks)")
      tags = tag_rec_t()
      n_tags = 0
      epoch = epoch + 1
      if (allocated(objs)) deallocate (objs)
      if (allocated(free_slots)) deallocate (free_slots)
      n_slots = 0
      n_free = 0
      tot_host = 0
      tot_host_peak = 0
      tot_dev = 0
      tot_dev_peak = 0
   end subroutine mem_reset

   function mem_human_bytes(b) result(s)
      !! "640.63 MiB", no padding.
      integer(int64), intent(in) :: b
      character(len=:), allocatable :: s
      character(len=24) :: buf
      character(len=3), parameter :: units(0:4) = ['B  ', 'KiB', 'MiB', 'GiB', 'TiB']
      real(real64) :: x
      integer :: k
      x = real(b, real64)
      k = 0
      do while (abs(x) >= 1024.0_real64 .and. k < 4)
         x = x/1024.0_real64
         k = k + 1
      end do
      if (k == 0) then
         write (buf, '(i0, 1x, a)') b, trim(units(k))
      else
         write (buf, '(f0.2, 1x, a)') x, trim(units(k))
      end if
      s = trim(adjustl(buf))
      if (s(1:1) == '.') s = '0'//s
   end function mem_human_bytes

   subroutine mem_fatal(msg)
      character(len=*), intent(in) :: msg
      write (error_unit, '(a)') 'mem_ledger FATAL: '//msg
      error stop 1
   end subroutine mem_fatal

   ! ------------------------------------------------------------ internals

   integer function find_tag_index(name) result(t)
      character(len=*), intent(in) :: name
      do t = 1, n_tags
         if (tags(t)%name == name) return
      end do
      t = 0
   end function find_tag_index

   subroutine check_tag(tag, where)
      type(mem_tag_t), intent(in) :: tag
      character(len=*), intent(in) :: where
      if (tag%id < 1 .or. tag%id > n_tags .or. tag%epoch /= epoch) &
         call mem_fatal(where//": invalid mem_tag_t (never set with mem_tag(""name""),"// &
                        " or created before mem_reset)")
   end subroutine check_tag

   pure integer function raw_slot(id)
      integer(int64), intent(in) :: id
      raw_slot = int(iand(id, SLOT_MASK))
   end function raw_slot

   integer function slot_of(id) result(s)
      !! Slot of `id` if the id is current (live or dead), else 0.
      integer(int64), intent(in) :: id
      s = raw_slot(id)
      if (s < 1 .or. s > n_slots) then
         s = 0
      else if (objs(s)%gen /= shiftr(id, 32)) then
         s = 0
      end if
   end function slot_of

   integer function check_live(id, where) result(s)
      integer(int64), intent(in) :: id
      character(len=*), intent(in) :: where
      s = slot_of(id)
      if (s == 0) then
         if (raw_slot(id) >= 1 .and. raw_slot(id) <= n_slots) &
            call mem_fatal(where//": the object was already destroyed and its slot reused."// &
                           " Double destroy, or a copy made by intrinsic assignment")
         call mem_fatal(where//": unknown ledger id")
      end if
      if (.not. objs(s)%live) call mem_fatal(where//": '"//trim(objs(s)%name)// &
                                             "' is already dead. Double destroy, or the object was"// &
                                             " copied by intrinsic assignment (a = b duplicates the ledger id)")
   end function check_live

   integer function take_slot() result(s)
      type(mem_object_t), allocatable :: tmp(:)
      if (n_free > 0) then
         s = free_slots(n_free)
         n_free = n_free - 1
         return
      end if
      if (.not. allocated(objs)) allocate (objs(64))
      if (n_slots == size(objs)) then
         allocate (tmp(2*size(objs)))
         tmp(1:n_slots) = objs(1:n_slots)
         call move_alloc(tmp, objs)
      end if
      n_slots = n_slots + 1
      s = n_slots
   end function take_slot

   subroutine push_free(s)
      integer, intent(in) :: s
      integer, allocatable :: tmp(:)
      if (.not. allocated(free_slots)) allocate (free_slots(64))
      if (n_free == size(free_slots)) then
         allocate (tmp(2*size(free_slots)))
         tmp(1:n_free) = free_slots(1:n_free)
         call move_alloc(tmp, free_slots)
      end if
      n_free = n_free + 1
      free_slots(n_free) = s
   end subroutine push_free

   function live_slots_of_tag_by_size(t) result(order)
      integer, intent(in) :: t
      integer, allocatable :: order(:)
      integer :: s, n, i, j, key
      n = 0
      do s = 1, n_slots
         if (objs(s)%live .and. objs(s)%tag == t) n = n + 1
      end do
      allocate (order(n))
      n = 0
      do s = 1, n_slots
         if (.not. (objs(s)%live .and. objs(s)%tag == t)) cycle
         n = n + 1
         order(n) = s
      end do
      do i = 2, n                       ! insertion sort, largest host first
         key = order(i)
         j = i - 1
         do while (j >= 1)
            if (objs(order(j))%host >= objs(key)%host) exit
            order(j + 1) = order(j)
            j = j - 1
         end do
         order(j + 1) = key
      end do
   end function live_slots_of_tag_by_size

end module mem_ledger_core
