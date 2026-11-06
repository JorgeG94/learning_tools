!! User API of the memory ledger: tags, queries, reports.
module mem_ledger
   !! What application code needs:
   !!
   !!   type(mem_tag_t) :: tag
   !!   tag = mem_tag("barotropic")        ! in a component's init
   !!   tag = mem_find_tag("barotropic")   ! in a driver: fatal on a typo
   !!   call mem_report(title="after init", objects=.true.)
   !!   if (mem_check_leaks() > 0) error stop
   !!
   !! Low-level registration lives in `mem_ledger_backend`, for authors of
   !! tracked types only.
   use mem_ledger_core, only: mem_tag_t, mem_tag_stats_t, operator(==), &
                              mem_tag, mem_find_tag, mem_tag_name, mem_tag_stats, &
                              mem_host_bytes, mem_device_bytes, mem_host_peak, mem_device_peak, &
                              mem_live_count, mem_report, mem_check_leaks, mem_reset, &
                              mem_human_bytes, MEM_TAG_NAME_LEN, MEM_OBJ_NAME_LEN
   implicit none
   private
   public :: mem_tag_t, mem_tag_stats_t, operator(==)
   public :: mem_tag, mem_find_tag, mem_tag_name, mem_tag_stats
   public :: mem_host_bytes, mem_device_bytes, mem_host_peak, mem_device_peak, mem_live_count
   public :: mem_report, mem_check_leaks, mem_reset, mem_human_bytes
   public :: MEM_TAG_NAME_LEN, MEM_OBJ_NAME_LEN
end module mem_ledger
