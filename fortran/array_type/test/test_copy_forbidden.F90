!! Must FAIL: assigning a dp2_t is a fatal error (the copy would share the
!! ledger id and the device mapping). `make test` checks the nonzero exit.
program test_copy_forbidden
   use devarray, only: dp2_t
   implicit none
   type(dp2_t) :: a, b
   call a%create("original", [4, 4])
   b = a
   print '(a)', 'copy was allowed -- this is a bug'
end program test_copy_forbidden
