program main
  use iso_fortran_env, only: dp => real64
  implicit none

  real(dp), allocatable :: a1(:,:,:), b1(:,:,:), c1(:,:,:)
  real(dp), allocatable :: a2(:,:,:), b2(:,:,:), c2(:,:,:)
  real(dp), allocatable :: a3(:,:,:), b3(:,:,:), c3(:,:,:)
  real(dp), allocatable :: a4(:,:,:), b4(:,:,:), c4(:,:,:)
  real(dp), allocatable :: a5(:,:,:), b5(:,:,:), c5(:,:,:)
  real(dp), allocatable :: a6(:,:,:), b6(:,:,:), c6(:,:,:)

  real(dp) :: alpha1, alpha2, alpha3, alpha4, alpha5, alpha6
  integer  :: m1, m2, m3, m4, m5, m6
  integer  :: i, j, k

  m1 = 128;  m2 = 64;  m3 = 37;  m4 = 256;  m5 = 45;  m6 = 42
  alpha1 = 12.0_dp; alpha2 = 22.0_dp; alpha3 = 17.0_dp
  alpha4 =  0.8_dp; alpha5 =  2.0_dp; alpha6 =  1.0_dp

  ! Allocate
  allocate(a1(m1,m1,m1), b1(m1,m1,m1), c1(m1,m1,m1))
  allocate(a2(m2,m2,m2), b2(m2,m2,m2), c2(m2,m2,m2))
  allocate(a3(m3,m3,m3), b3(m3,m3,m3), c3(m3,m3,m3))
  allocate(a4(m4,m4,m4), b4(m4,m4,m4), c4(m4,m4,m4))
  allocate(a5(m5,m5,m5), b5(m5,m5,m5), c5(m5,m5,m5))
  allocate(a6(m6,m6,m6), b6(m6,m6,m6), c6(m6,m6,m6))

  ! Map to GPU
  !$omp target enter data map(alloc: a1, b1, c1)
  !$omp target enter data map(alloc: a2, b2, c2)
  !$omp target enter data map(alloc: a3, b3, c3)
  !$omp target enter data map(alloc: a4, b4, c4)
  !$omp target enter data map(alloc: a5, b5, c5)
  !$omp target enter data map(alloc: a6, b6, c6)

  ! Initialize arrays on GPU
  do concurrent (k=1:m1, j=1:m1, i=1:m1)
    a1(i,j,k) = 2.0_dp;  b1(i,j,k) = 3.0_dp;  c1(i,j,k) = 0.0_dp
  end do

  do concurrent (k=1:m2, j=1:m2, i=1:m2)
    a2(i,j,k) = 3.0_dp;  b2(i,j,k) = 2.0_dp;  c2(i,j,k) = 0.0_dp
  end do

  do concurrent (k=1:m3, j=1:m3, i=1:m3)
    a3(i,j,k) = 4.0_dp;  b3(i,j,k) = 1.0_dp;  c3(i,j,k) = 0.0_dp
  end do

  do concurrent (k=1:m4, j=1:m4, i=1:m4)
    a4(i,j,k) = 5.0_dp;  b4(i,j,k) = 13.0_dp; c4(i,j,k) = 0.0_dp
  end do

  do concurrent (k=1:m5, j=1:m5, i=1:m5)
    a5(i,j,k) = 6.0_dp;  b5(i,j,k) = 32.0_dp; c5(i,j,k) = 0.0_dp
  end do

  do concurrent (k=1:m6, j=1:m6, i=1:m6)
    a6(i,j,k) = 7.0_dp;  b6(i,j,k) = 17.0_dp; c6(i,j,k) = 0.0_dp
  end do

  ! SAXPY loop
  block
    integer, parameter :: max_it = 1000
    integer , parameter :: outside_it = 30
    integer :: it, jit

    do jit = 1, outside_it
    do it = 1, max_it
      do concurrent (k=1:m1, j=1:m1, i=1:m1)
        c1(i,j,k) = alpha1 * a1(i,j,k) + b1(i,j,k)
      end do

      do concurrent (k=1:m2, j=1:m2, i=1:m2)
        c2(i,j,k) = alpha2 * a2(i,j,k) + b2(i,j,k)
      end do

      do concurrent (k=1:m3, j=1:m3, i=1:m3)
        c3(i,j,k) = alpha3 * a3(i,j,k) + b3(i,j,k)
      end do

      do concurrent (k=1:m4, j=1:m4, i=1:m4)
        c4(i,j,k) = alpha4 * a4(i,j,k) + b4(i,j,k)
      end do

      do concurrent (k=1:m5, j=1:m5, i=1:m5)
        c5(i,j,k) = alpha5 * a5(i,j,k) + b5(i,j,k)
      end do

      do concurrent (k=1:m6, j=1:m6, i=1:m6)
        c6(i,j,k) = alpha6 * a6(i,j,k) + b6(i,j,k)
      end do
    end do
    end do
  end block

  ! Unmap from GPU
  !$omp target exit data map(delete: a1, b1, c1)
  !$omp target exit data map(delete: a2, b2, c2)
  !$omp target exit data map(delete: a3, b3, c3)
  !$omp target exit data map(delete: a4, b4, c4)
  !$omp target exit data map(delete: a5, b5, c5)
  !$omp target exit data map(delete: a6, b6, c6)

  ! Deallocate
  deallocate(a1, b1, c1)
  deallocate(a2, b2, c2)
  deallocate(a3, b3, c3)
  deallocate(a4, b4, c4)
  deallocate(a5, b5, c5)
  deallocate(a6, b6, c6)

end program main
