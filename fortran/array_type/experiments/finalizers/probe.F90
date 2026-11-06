!! Finalizer probe, modelled on the ledger.
!!
!! Every `init` issues a fresh ledger id. The final procedure retires the
!! id of an object whose data is allocated. After each scenario we count
!!   leaked  = ids issued but never retired      (finalizer missed)
!!   doubles = ids retired more than once         (copy shared an id)
!! Both must be zero for a finalizer-driven ledger to be correct.
!!
!! Build with -DDEFINED_ASSIGN to give the type a defined assignment that
!! deep-copies and issues a NEW id for the copy (a "copy constructor").
module probe_mod
   implicit none
   private
   public :: res_t, parent_t, reset, make_res, leaked, doubles, nfinal

   integer, parameter :: MAXID = 1000
   logical :: live(MAXID) = .false.
   integer :: next_id = 0
   integer :: doubles = 0
   integer :: nfinal = 0

   type :: res_t
      real, allocatable :: data(:)
      integer :: id = 0
   contains
      procedure :: init
#ifdef DEFINED_ASSIGN
      procedure, private :: assign_res
      generic :: assignment(=) => assign_res
#endif
      final :: res_final_r0, res_final_r1
   end type res_t

   type :: parent_t
      type(res_t) :: child
   end type parent_t

contains

   subroutine reset()
      live = .false.
      next_id = 0
      doubles = 0
      nfinal = 0
   end subroutine reset

   integer function leaked()
      leaked = count(live)
   end function leaked

   integer function issue()
      next_id = next_id + 1
      issue = next_id
      live(issue) = .true.
   end function issue

   subroutine init(this, val)
      class(res_t), intent(inout) :: this
      integer, intent(in) :: val
      allocate (this%data(4))
      this%data = real(val)
      this%id = issue()
   end subroutine init

#ifdef DEFINED_ASSIGN
   subroutine assign_res(lhs, rhs)
      !! Deep copy with a fresh id. Retires whatever lhs held first.
      class(res_t), intent(inout) :: lhs
      class(res_t), intent(in) :: rhs
      call res_final_r0_impl(lhs)
      if (allocated(rhs%data)) then
         lhs%data = rhs%data
         lhs%id = issue()
      end if
   end subroutine assign_res
#endif

   subroutine res_final_r0(this)
      type(res_t), intent(inout) :: this
      nfinal = nfinal + 1
      call res_final_r0_impl(this)
   end subroutine res_final_r0

   subroutine res_final_r0_impl(this)
      class(res_t), intent(inout) :: this
      if (.not. allocated(this%data)) return
      if (this%id < 1 .or. this%id > MAXID) then
         doubles = doubles + 1
      else if (.not. live(this%id)) then
         doubles = doubles + 1
      else
         live(this%id) = .false.
      end if
      deallocate (this%data)
      this%id = 0
   end subroutine res_final_r0_impl

   subroutine res_final_r1(this)
      type(res_t), intent(inout) :: this(:)
      integer :: i
      do i = 1, size(this)
         call res_final_r0(this(i))
      end do
   end subroutine res_final_r1

   function make_res(val) result(r)
      integer, intent(in) :: val
      type(res_t) :: r
      call r%init(val)
   end function make_res

end module probe_mod

program probe
   use probe_mod
   implicit none
   integer :: nfail
   nfail = 0

#ifdef DEFINED_ASSIGN
   print '(a)', '--- variant: defined assignment (copy gets a new id)'
#else
   print '(a)', '--- variant: intrinsic assignment (copy shares the id)'
#endif
   call scenario("S1  local scalar, end of subroutine", s1)
   call scenario("S2  local in BLOCK, end of block", s2)
   call scenario("S3  allocatable scalar, deallocate", s3)
   call scenario("S4  component of local parent", s4)
   call scenario("S5  local rank-1 array(3)", s5)
   call scenario("S6  a = b, both live", s6)
   call scenario("S7  intent(out) dummy", s7)
   call scenario("S8  a = make_res()", s8)
   call scenario("S9  allocatable parent, deallocate", s9)
   call scenario("S10 unallocated allocatable LHS a = b", s10)
   call scenario("S11 allocatable array, deallocate", s11)
   call scenario("S12 parent assign p = q (component copy)", s12)
   call scenario("S13 pass by value-ish: call f(make_res())", s13)
   call scenario("S14 intent(out) dummy, SAVEd actual", s14)

   if (nfail == 0) then
      print '(a)', 'ALL PASS'
   else
      print '(i0, a)', nfail, ' scenario(s) FAIL'
   end if

contains

   subroutine scenario(label, body)
      character(len=*), intent(in) :: label
      interface
         subroutine body()
         end subroutine body
      end interface
      character(len=4) :: verdict
      call reset()
      call body()
      verdict = 'ok'
      if (leaked() /= 0 .or. doubles /= 0) then
         verdict = 'FAIL'
         nfail = nfail + 1
      end if
      print '(a4, 2x, a42, a, i0, a, i0, a, i0)', verdict, label, &
         '  leaked=', leaked(), ' doubles=', doubles, '  (final calls ', nfinal
   end subroutine scenario

   subroutine s1()
      type(res_t) :: a
      call a%init(1)
   end subroutine s1

   subroutine s2()
      block
         type(res_t) :: a
         call a%init(2)
      end block
   end subroutine s2

   subroutine s3()
      type(res_t), allocatable :: a
      allocate (a)
      call a%init(3)
      deallocate (a)
   end subroutine s3

   subroutine s4()
      type(parent_t) :: p
      call p%child%init(4)
   end subroutine s4

   subroutine s5()
      type(res_t) :: a(3)
      integer :: i
      do i = 1, 3
         call a(i)%init(i)
      end do
   end subroutine s5

   subroutine s6()
      type(res_t) :: a, b
      call a%init(61)
      call b%init(62)
      a = b
   end subroutine s6

   subroutine s7()
      type(res_t) :: a
      call a%init(7)
      call take_out(a)
   end subroutine s7

   subroutine take_out(x)
      type(res_t), intent(out) :: x
      if (x%id /= 0) continue
   end subroutine take_out

   subroutine s8()
      type(res_t) :: a
      call a%init(81)
      a = make_res(82)
   end subroutine s8

   subroutine s9()
      type(parent_t), allocatable :: p
      allocate (p)
      call p%child%init(9)
      deallocate (p)
   end subroutine s9

   subroutine s10()
      type(res_t), allocatable :: a
      type(res_t) :: b
      call b%init(10)
#ifdef DEFINED_ASSIGN
      ! Defined assignment never auto-allocates the LHS (unlike intrinsic
      ! assignment). Without this ALLOCATE the program is invalid and
      ! segfaults on every compiler.
      allocate (a)
#endif
      a = b
   end subroutine s10

   subroutine s11()
      type(res_t), allocatable :: a(:)
      integer :: i
      allocate (a(3))
      do i = 1, 3
         call a(i)%init(i)
      end do
      deallocate (a)
   end subroutine s11

   subroutine s12()
      type(parent_t) :: p, q
      call p%child%init(121)
      call q%child%init(122)
      p = q
   end subroutine s12

   subroutine s13()
      call peek(make_res(13))
   end subroutine s13

   subroutine peek(x)
      type(res_t), intent(in) :: x
      if (x%id < 0) error stop
   end subroutine peek

   subroutine s14()
      ! Main-program variables are implicitly SAVE, so this is also the
      ! "pass a main-program variable to intent(out)" case.
      type(res_t), save :: a
      call a%init(14)
      call take_out(a)
   end subroutine s14

end program probe
