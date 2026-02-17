module cuda_interface
  use iso_c_binding
  implicit none
  interface
    subroutine launch_mega_kernel( &
        a1, b1, c1, va1, vb1, alpha1, n1, &
        a2, b2, c2, va2, vb2, alpha2, n2, &
        a3, b3, c3, va3, vb3, alpha3, n3, &
        a4, b4, c4, va4, vb4, alpha4, n4, &
        a5, b5, c5, va5, vb5, alpha5, n5, &
        a6, b6, c6, va6, vb6, alpha6, n6, &
        nmax, max_it) bind(C, name="launch_mega_kernel")
      import :: c_ptr, c_double, c_int
      type(c_ptr), value :: a1, b1, c1, a2, b2, c2
      type(c_ptr), value :: a3, b3, c3, a4, b4, c4
      type(c_ptr), value :: a5, b5, c5, a6, b6, c6
      real(c_double), value :: va1, vb1, alpha1
      real(c_double), value :: va2, vb2, alpha2
      real(c_double), value :: va3, vb3, alpha3
      real(c_double), value :: va4, vb4, alpha4
      real(c_double), value :: va5, vb5, alpha5
      real(c_double), value :: va6, vb6, alpha6
      integer(c_int), value :: n1, n2, n3, n4, n5, n6, nmax, max_it
    end subroutine
  end interface
end module cuda_interface


program main
  use iso_fortran_env, only: dp => real64
  use iso_c_binding
  use cuda_interface
  implicit none

  real(dp), allocatable, target :: a1(:,:,:), b1(:,:,:), c1(:,:,:)
  real(dp), allocatable, target :: a2(:,:,:), b2(:,:,:), c2(:,:,:)
  real(dp), allocatable, target :: a3(:,:,:), b3(:,:,:), c3(:,:,:)
  real(dp), allocatable, target :: a4(:,:,:), b4(:,:,:), c4(:,:,:)
  real(dp), allocatable, target :: a5(:,:,:), b5(:,:,:), c5(:,:,:)
  real(dp), allocatable, target :: a6(:,:,:), b6(:,:,:), c6(:,:,:)

  integer :: m1, m2, m3, m4, m5, m6
  integer :: n1, n2, n3, n4, n5, n6, nmax

  m1 = 128;  m2 = 64;  m3 = 37;  m4 = 256;  m5 = 45;  m6 = 42
  n1 = m1**3; n2 = m2**3; n3 = m3**3
  n4 = m4**3; n5 = m5**3; n6 = m6**3
  nmax = max(n1, n2, n3, n4, n5, n6)

  allocate(a1(m1,m1,m1), b1(m1,m1,m1), c1(m1,m1,m1))
  allocate(a2(m2,m2,m2), b2(m2,m2,m2), c2(m2,m2,m2))
  allocate(a3(m3,m3,m3), b3(m3,m3,m3), c3(m3,m3,m3))
  allocate(a4(m4,m4,m4), b4(m4,m4,m4), c4(m4,m4,m4))
  allocate(a5(m5,m5,m5), b5(m5,m5,m5), c5(m5,m5,m5))
  allocate(a6(m6,m6,m6), b6(m6,m6,m6), c6(m6,m6,m6))

  !$omp target enter data map(alloc: a1, b1, c1)
  !$omp target enter data map(alloc: a2, b2, c2)
  !$omp target enter data map(alloc: a3, b3, c3)
  !$omp target enter data map(alloc: a4, b4, c4)
  !$omp target enter data map(alloc: a5, b5, c5)
  !$omp target enter data map(alloc: a6, b6, c6)

  !$omp target data use_device_ptr(a1, b1, c1, a2, b2, c2, &
  !$omp&  a3, b3, c3, a4, b4, c4, a5, b5, c5, a6, b6, c6)
  call launch_mega_kernel( &
      c_loc(a1), c_loc(b1), c_loc(c1), 2.0_dp,  3.0_dp, 12.0_dp, n1, &
      c_loc(a2), c_loc(b2), c_loc(c2), 3.0_dp,  2.0_dp, 22.0_dp, n2, &
      c_loc(a3), c_loc(b3), c_loc(c3), 4.0_dp,  1.0_dp, 17.0_dp, n3, &
      c_loc(a4), c_loc(b4), c_loc(c4), 5.0_dp, 13.0_dp,  0.8_dp, n4, &
      c_loc(a5), c_loc(b5), c_loc(c5), 6.0_dp, 32.0_dp,  2.0_dp, n5, &
      c_loc(a6), c_loc(b6), c_loc(c6), 7.0_dp, 17.0_dp,  1.0_dp, n6, &
      nmax, 1000)
  !$omp end target data

  !$omp target exit data map(delete: a1, b1, c1)
  !$omp target exit data map(delete: a2, b2, c2)
  !$omp target exit data map(delete: a3, b3, c3)
  !$omp target exit data map(delete: a4, b4, c4)
  !$omp target exit data map(delete: a5, b5, c5)
  !$omp target exit data map(delete: a6, b6, c6)

  deallocate(a1, b1, c1)
  deallocate(a2, b2, c2)
  deallocate(a3, b3, c3)
  deallocate(a4, b4, c4)
  deallocate(a5, b5, c5)
  deallocate(a6, b6, c6)

end program main
