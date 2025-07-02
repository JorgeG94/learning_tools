program hello_from_omp 
use omp_lib 
implicit none 

integer :: id_private, id_shared
  !! variables are declared here and are global to the lifetime of the program, this is important 

print *, ""
print *, "Starting the hello_form_omp program"

!$omp parallel 
id_shared = omp_get_thread_num()
  !! here by setting id_shared to omp get thread num can lead to some annoying things, since by default the variables are shared across threads
  !! once you run this, you might see that the id is the same of repeated for many threads! 
print *, " Hi from parallel block in thread num", id_shared
!$omp end parallel 

print *, "*********************************************************"

!$omp parallel private(id_private)
id_private = omp_get_thread_num()
 !! here, we used the private() construct in OMP to say that this variable is private to each thread that is in the parallel region 
 !! therefore, each thread gets its _own_ thread id 
print *, "Hi from parallel block with private thread id", id_private
!$omp end parallel 

print *, "*********************************************************"

!$omp parallel  
block
 !! here by using a block _within_ the parallel region we ensure that the scope of the id is local to the parallel region/block and therefore each thread gets its own copy
 !! this is similar to writing a loop in C where id is delcared within the parallel region. See example at the end of the program
 !! however, the output will most likely be different every time you run it since the printing of the thread ids depends on how the threads are scheduled
  integer :: id
  id = omp_get_thread_num()
  print *, "Hi from block and in thread num ", id
end block
!$omp end parallel 

end program hello_from_omp 

!#pragma omp parallel 
!{
!  int id = omp_get_thread_num();
!  printf("Hi from %d ", id);
!}
