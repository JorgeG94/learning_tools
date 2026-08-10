module density_mod
  use iso_fortran_env, only: real64
  implicit none
contains

  !----------------------------------------
  pure subroutine calculate_density_local(T, S, p, rho)
  !----------------------------------------
  ! Compute density from local scalar T, S, p
  real(real64), intent(in)  :: T, S, p
  real(real64), intent(out) :: rho

  ! simple dummy equation of state:
  rho = 1000.0_real64 + 0.8_real64 * S - 0.2_real64 * T + 0.0001_real64 * p

  end subroutine calculate_density_local


  !----------------------------------------
  pure function calculate_density(T, S, p) result(rho)
  !----------------------------------------
  ! Generic 3D array version
  real(real64), intent(in) :: T(:,:,:), S(:,:,:), p(:,:,:)
  real(real64)             :: rho(size(T,1), size(T,2), size(T,3))
  integer :: i, j, k

  do concurrent (k = 1:size(T,3), j = 1:size(T,2), i = 1:size(T,1))
     call calculate_density_local(T(i,j,k), S(i,j,k), p(i,j,k), rho(i,j,k))
  end do

  end function calculate_density


  !----------------------------------------
  pure function calculate_density_2d(T, S, p) result(rho)
  !----------------------------------------
  ! Same idea, but 2D variant
  real(real64), intent(in) :: T(:,:), S(:,:), p(:,:)
  real(real64)             :: rho(size(T,1), size(T,2))
  integer :: i, j

  do concurrent (j = 1:size(T,2), i = 1:size(T,1))
     call calculate_density_local(T(i,j), S(i,j), p(i,j), rho(i,j))
  end do

  end function calculate_density_2d

end module density_mod

