program main
    implicit none

    procedure(work_proc), pointer :: pwork => null()

    abstract interface
        subroutine work_proc(id)
            integer, intent(in) :: id
        end subroutine work_proc
    end interface

    ! map the pointer 
    pwork => do_task

    ! pass the function pointer
    call worker(42, pwork)

contains


    subroutine do_task(id)
        integer, intent(in) :: id
        print *, "Doing work on task", id
    end subroutine do_task

    subroutine worker(task_id, work_routine)
        integer, intent(in) :: task_id
        integer :: local_task
        procedure(work_proc) :: work_routine

        print *, "Worker starting task", task_id
        local_task = task_id + 4
        call work_routine(local_task)
    end subroutine worker

end program main

