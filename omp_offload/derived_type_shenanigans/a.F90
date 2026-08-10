module module_one
implicit none 

  type structure_types 
    integer, allocatable :: vector_1(:)
    integer, allocatable :: vector_2(:)
    integer :: size_of_vectors
  
  
    contains 
  
    procedure :: initialize_vectors
    procedure :: finalize_vectors
  
  end type 

contains 

  subroutine initialize_vectors(self, n_elements, init_value)
    class(structure_types), intent(inout) :: self 
    integer, intent(in) :: n_elements
    integer, intent(in) :: init_value
    
    allocate(self%vector_1(n_elements))
    allocate(self%vector_2(n_elements))
    self%size_of_vectors = n_elements

    self%vector_1 = init_value
    self%vector_2 = init_value
  
  end subroutine initialize_vectors
  
  
  subroutine finalize_vectors(self)
    class(structure_types), intent(inout) :: self 
    
    deallocate(self%vector_1)
    deallocate(self%vector_2)
  
  end subroutine finalize_vectors

end module module_one 


module module_two
use module_one, only: structure_types
implicit none 

  type contrl_structure 
    type(structure_types), pointer :: stype => NULL()
  end type 

end module module_two 


module module_three
use module_one, only: structure_types
implicit none 

contains 

  subroutine pass_derived_struct_in(struct_type)
    type(structure_types), intent(in) :: struct_type
    integer :: i
    integer, allocatable :: sum_total(:) 

    print *, " begginning derived struct in test"
    allocate(sum_total(size(struct_type%vector_1)))

#ifdef OMP
    !$omp target teams loop map(tofrom:sum_total) map(to:struct_type%vector_1, struct_type%vector_2)
    do i = 1, size(struct_type%vector_1)
      sum_total(i) = struct_type%vector_1(i) + struct_type%vector_2(i)
    end do 
#else
    do concurrent (i=1:size(struct_type%vector_1))
      sum_total(i) = struct_type%vector_1(i) + struct_type%vector_2(i)
    end do 
#endif

    if( sum(sum_total) /= 40) then 
      error  stop 
    else 
    print *, "derived struct in test passed!"
    end if

  end subroutine pass_derived_struct_in 
  
  subroutine use_array_in_struct(array)
    integer, intent(in) :: array(:)
    integer, allocatable :: sum_total(:) 
    integer :: i 

    print *, "beginning use array in struct test"
    allocate(sum_total(size(array)))

#ifdef OMP
    !$omp target teams loop 
    do i = 1, size(array)
      sum_total(i) = array(i)
    end do 
#else
    do concurrent (i=1:size(array))
      sum_total(i) = array(i)
    end do 
#endif

    if( sum(sum_total) /= 20) then 
      error  stop 
    else 
    print *, "array intent in test passed!"
    end if
  
  end subroutine use_array_in_struct 
  
  subroutine read_write_array_in_struct(array1)
    integer,  intent(inout) :: array1(:)
    integer :: alpha 
    integer :: i 

    alpha = 3.0

    print *, "beginning read write arrain intent inout atest"

#ifdef OMP
    !$omp target teams loop 
    do i = 1, size(array1)
      array1(i) = alpha * array1(i)
    end do 
#else
    do concurrent (i=1:size(array1))
      array1(i) = alpha * array1(i)
    end do 
#endif

    if( sum(array1) /= 60) then 
      error  stop 
    else 
    print *, "inout test passed!"
    end if
  
  end subroutine read_write_array_in_struct
  
  subroutine pass_derived_struct_as_pointer(struct_type)
    type(structure_types), pointer :: struct_type
    integer, dimension(struct_type%size_of_vectors) :: sum_total
    integer, dimension(struct_type%size_of_vectors) :: alpha
    integer :: i 

    print *, "beginning derived struct as pointer"
    !allocate(sum_total(struct_type%size_of_vectors))
    !!$omp target update to(struct_type%vector_1, struct_type%vector_2)
    !!$omp target enter data map(to: struct_type%vector_1, struct_type%vector_2)
    !$omp target enter data map(alloc: sum_total, alpha)

    !$omp target teams loop
    do i = 1, struct_type%size_of_vectors
      alpha(i) = 1
    end do
    !$omp end target teams loop

#ifdef OMP
    !$omp target teams loop map(tofrom: sum_total) map(to: struct_type%vector_1, struct_type%vector_2)
    do i = 1, struct_type%size_of_vectors
      sum_total(i) = alpha(i) * struct_type%vector_1(i) + struct_type%vector_2(i)
    end do 
#else
    do concurrent (i=1:struct_type%size_of_vectors)
      sum_total(i) = alpha(i) * struct_type%vector_1(i) + struct_type%vector_2(i)
    end do 
#endif
    !$omp target exit data map(from: sum_total)
    !$omp target exit data map(delete: sum_total, alpha)
    !!$omp target update from(struct_type%vector_1, struct_type%vector_2)

    print *, sum(sum_total)
    if( sum(sum_total) /= 80) then 
      error  stop 
    else 
    print *, "pointer in test passed!"
    end if
      
  end subroutine pass_derived_struct_as_pointer

end module module_three

program main 
use module_two, only: contrl_structure
use module_three
implicit none 
type(contrl_structure) :: my_control_structure

allocate(my_control_structure%stype)
call my_control_structure%stype%initialize_vectors(10, 2)
!
call pass_derived_struct_in(my_control_structure%stype)
!
call use_array_in_struct(my_control_structure%stype%vector_1)
!
call read_write_array_in_struct(my_control_structure%stype%vector_2)
call pass_derived_struct_as_pointer(my_control_structure%stype)

  
call my_control_structure%stype%finalize_vectors()



end program main
