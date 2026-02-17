module test_saxpy
use iso_fortran_env, only: dp => real64
implicit none

private

public :: saxpy_t
public :: init_saxpy, init_saxpy_arrays, delete_saxpy, do_saxpy

type :: saxpy_t
  real(dp), allocatable :: a(:,:,:), b(:,:,:), c(:,:,:)
  real(dp) :: alpha = 0.0_dp
  integer :: m
  logical :: is_init = .false.
end type saxpy_t

contains

subroutine init_saxpy(s_t, alpha, m)
type(saxpy_t), intent(inout) :: s_t
real(dp), intent(in) :: alpha
integer, intent(in) :: m

s_t%m = m
s_t%alpha = alpha

allocate(s_t%a(s_t%m,s_t%m,s_t%m))
allocate(s_t%b(s_t%m,s_t%m,s_t%m))
allocate(s_t%c(s_t%m,s_t%m,s_t%m))

!$omp target enter data map(to: s_t)
!$omp target enter data map(alloc: s_t%a, s_t%b, s_t%c)

s_t%is_init = .true.

end subroutine init_saxpy

subroutine init_saxpy_arrays(s_t, val_a, val_b, val_c)
type(saxpy_t), intent(inout) :: s_t
real(dp), intent(in) :: val_a, val_b, val_c
integer :: i,j,k

if(s_t%is_init == .false.) then
  error stop " initializing arrays without allocating memory first!"
end if

do concurrent (j=1:s_t%m, i=1:s_t%m, k=1:s_t%m)
  s_t%a(i,j,k) = val_a
  s_t%b(i,j,k) = val_b
  s_t%c(i,j,k) = val_c
end do

end subroutine init_saxpy_arrays

subroutine delete_saxpy(s_t)
type(saxpy_t), intent(inout) :: s_t

!$omp target exit data map(delete: s_t)
!$omp target exit data map(delete: s_t%a, s_t%b, s_t%c)

deallocate(s_t%a)
deallocate(s_t%b)
deallocate(s_t%c)

s_t%is_init = .false.

end subroutine delete_saxpy

subroutine do_saxpy(s_t)
type(saxpy_t), intent(inout) :: s_t
integer :: i,j,k

do concurrent(j=1:s_t%m, i=1:s_t%m, k=1:s_t%m)
  s_t%c(i,j,k) = s_t%alpha * s_t%a(i,j,k) + s_t%b(i,j,k)
end do

end subroutine do_saxpy

end module test_saxpy

program main
use iso_fortran_env, only: dp => real64
use test_saxpy, only: saxpy_t, init_saxpy, init_saxpy_arrays, delete_saxpy, do_saxpy
implicit none
type(saxpy_t) :: saxpy_1, saxpy_2, saxpy_3, saxpy_4, saxpy_5, saxpy_6
integer :: dim_1, dim_2, dim_3, dim_4, dim_5, dim_6
real(dp) :: a_1, a_2, a_3, a_4, a_5, a_6


dim_1 = 128
dim_2 = 64
dim_3 = 37
dim_4 = 256
dim_5 = 45
dim_6 = 42

a_1 = 12.0_dp
a_2 = 22.0_dp
a_3 = 17.0_dp
a_4 = 0.8_dp
a_5 = 2.0_dp
a_6 = 1.0_dp

call init_saxpy(saxpy_1, a_1, dim_1)
call init_saxpy(saxpy_2, a_2, dim_2)
call init_saxpy(saxpy_3, a_3, dim_3)
call init_saxpy(saxpy_4, a_4, dim_4)
call init_saxpy(saxpy_5, a_5, dim_5)
call init_saxpy(saxpy_6, a_6, dim_6)


call init_saxpy_arrays(saxpy_1, 2.0_dp, 3.0_dp, 0.0_dp)
call init_saxpy_arrays(saxpy_2, 3.0_dp, 2.0_dp, 0.0_dp)
call init_saxpy_arrays(saxpy_3, 4.0_dp, 1.0_dp, 0.0_dp)
call init_saxpy_arrays(saxpy_4, 5.0_dp, 13.0_dp, 0.0_dp)
call init_saxpy_arrays(saxpy_5, 6.0_dp, 32.0_dp, 0.0_dp)
call init_saxpy_arrays(saxpy_6, 7.0_dp, 17.0_dp, 0.0_dp)

block
integer, parameter :: max_it = 1000
integer :: i

do i = 1, max_it
call do_saxpy(saxpy_1)
call do_saxpy(saxpy_2)
call do_saxpy(saxpy_3)
call do_saxpy(saxpy_4)
call do_saxpy(saxpy_5)
call do_saxpy(saxpy_6)
end do
end block


call delete_saxpy(saxpy_1)
call delete_saxpy(saxpy_2)
call delete_saxpy(saxpy_3)
call delete_saxpy(saxpy_4)
call delete_saxpy(saxpy_5)
call delete_saxpy(saxpy_6)

end program main
