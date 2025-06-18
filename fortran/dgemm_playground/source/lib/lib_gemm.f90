module lib_gemm
   use lib_types
   use iso_c_binding
   implicit none

contains

   subroutine naive_omp_dgemm(A, B, C, m, n, k)
      implicit none
      integer, intent(in) :: m, n, k
      real(rk), intent(in) :: A(m, k)
      real(rk), intent(in) :: B(k, n)
      real(rk), intent(inout) :: C(m, n)
      integer :: i, j, l, rep
!!$omp parallel do private(i,j,l) schedule(static)
!$omp target teams distribute parallel do private(i,j,l)
      do i = 1, m
         do j = 1, n
            do l = 1, k
               C(i, j) = C(i, j) + A(i, l)*B(l, j)
            end do
         end do
      end do
!$omp end target teams distribute parallel do
!!$omp end parallel do
   end subroutine naive_omp_dgemm

   subroutine blocked_dgemm(A, B, C, m, n, k)
      integer, intent(in) :: m, n, k
      real(rk), intent(in)  :: A(m, k), B(k, n)
      real(rk), intent(inout) :: C(m, n)
      integer :: i, j, l, ii, jj, ll, rep
      integer, parameter :: block_size = 64

!$omp parallel do private(ii,jj,ll,i,j,l) schedule(static)
      do ii = 1, m, block_size
         do jj = 1, n, block_size
            do ll = 1, k, block_size

               do i = ii, min(ii + block_size - 1, m)
                  do j = jj, min(jj + block_size - 1, n)
                     do l = ll, min(ll + block_size - 1, k)
                        C(i, j) = C(i, j) + A(i, l)*B(l, j)
                     end do
                  end do
               end do

            end do
         end do
      end do
!$omp end parallel do

   end subroutine blocked_dgemm

   subroutine blas_dgemm(A, B, C, m, n, k)
      integer, intent(in) :: m, n, k
      real(rk), intent(in)  :: A(m, k), B(k, n)
      real(rk), intent(inout) :: C(m, n)

      integer :: rep
      real(rk) :: alpha, beta
      character(len=1) :: transa, transb

      transa = 'N'
      transb = 'N'
      alpha = 1.0d0
      beta = 1.0d0

!    call dgemm(transa, transb, m, n, k, alpha, A, m, B, k, beta, C, m)
   end subroutine blas_dgemm

   subroutine simd_dgemm(A, B, C, m, n, k)
      use omp_lib
      use lib_types
      implicit none

      integer, intent(in) :: m, n, k
      real(rk), intent(in)  :: A(m, k), B(k, n)
      real(rk), intent(inout) :: C(m, n)
      integer :: i, j, l, rep
      real(rk) :: tmp

!$omp parallel do private(i,j,l,tmp) schedule(static)
      do i = 1, m
         do j = 1, n
            tmp = 0.0d0
!$omp simd reduction(+:tmp)
            do l = 1, k
               tmp = tmp + A(i, l)*B(l, j)
            end do
            C(i, j) = C(i, j) + tmp
         end do
      end do
!$omp end parallel do

   end subroutine simd_dgemm

   subroutine do_concurrent_dgemm(A, B, C, m, n, k)
      implicit none
      integer, intent(in) :: m, n, k
      real(rk), intent(in) :: A(m, k)
      real(rk), intent(in) :: B(k, n)
      real(rk), intent(inout) :: C(m, n)
      real(rk) :: tmp
      integer :: i, j, l, rep
      do concurrent(i=1:m, j=1:n)
         tmp = 0.0d0
         do l = 1, k
            tmp = tmp + A(i, l)*B(l, j)
         end do
         C(i, j) = tmp
      end do

   end subroutine do_concurrent_dgemm



end module lib_gemm
