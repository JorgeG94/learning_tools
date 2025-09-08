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
    if( sum(sum_total) /= 40) then 
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

call pass_derived_struct_as_pointer(my_control_structure%stype)

  
call my_control_structure%stype%finalize_vectors()



end program main
