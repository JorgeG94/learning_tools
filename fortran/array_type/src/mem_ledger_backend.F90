!! Registration API of the memory ledger, for authors of tracked types
!! (devarray, mem_collective). Application code should not need this:
!! calling these by hand can make the counts disagree with reality.
module mem_ledger_backend
   use mem_ledger_core, only: mem_object_t, mem_validate_i, mem_track, mem_untrack, &
                              mem_track_device, mem_untrack_device, mem_is_live, &
                              mem_object_name, mem_get_object, mem_live_ids, mem_num_slots, &
                              mem_check_object, mem_tag_matches, mem_fatal, &
                              MEM_KIND_RAW, MEM_KIND_R64, MEM_KIND_I32
   implicit none
   private
   public :: mem_object_t, mem_validate_i
   public :: mem_track, mem_untrack, mem_track_device, mem_untrack_device
   public :: mem_is_live, mem_object_name, mem_get_object, mem_live_ids, mem_num_slots
   public :: mem_check_object, mem_tag_matches, mem_fatal
   public :: MEM_KIND_RAW, MEM_KIND_R64, MEM_KIND_I32
end module mem_ledger_backend
