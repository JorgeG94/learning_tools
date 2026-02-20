# Fortran Essentials: A Quick Reference for New Readers

This guide covers the core concepts you need to understand Fortran code, from modern idioms to legacy patterns you'll encounter in older codebases.

---

## 1. Source Format: Fixed vs Free Form

Fortran has two source formats, and recognizing which one you're looking at is the first step.

### Fixed Form (Fortran 77 and earlier)

- File extension: `.f`, `.for`, `.F`
- Columns matter:
  - Column 1: `C`, `c`, or `*` means comment
  - Column 6: Any character (except `0`) means continuation
  - Columns 7-72: Actual code
  - Columns 73+: Ignored (historically for card sequence numbers)

```fortran
C     This is a comment in fixed form
      PROGRAM HELLO
      INTEGER I
      DO 10 I = 1, 10
         WRITE(*,*) I
   10 CONTINUE
      END
```

### Free Form (Fortran 90+)

- File extension: `.f90`, `.f95`, `.f03`, `.f08`, `.F90`
- No column restrictions
- `!` starts a comment
- `&` at end of line means continuation

```fortran
! This is a comment in free form
program hello
    integer :: i
    do i = 1, 10
        write(*,*) i
    end do
end program
```

**Key point**: Most legacy scientific codes (GAMESS, older BLAS/LAPACK) use fixed form. Modern codes use free form.

---

## 2. Program Units

Fortran code is organized into **program units**. Every executable must have exactly one `program`, but can have many modules, functions, and subroutines.

```fortran
program main
    implicit none
    call greet()

contains 

subroutine greet()
    print *, "Hello, World!"
end subroutine greet

end program main
```

---

## 3. `implicit none`

**Always use this.** Without it, Fortran uses implicit typing where variables starting with I-N are integers, others are real. This is a major source of bugs.

```fortran
program bad_example
    ! No implicit none - dangerous!
    x = 1.5     ! Implicitly real
    icount = 3  ! Implicitly integer
end program

program good_example
    implicit none
    real :: x
    integer :: icount
    x = 1.5
    icount = 3
end program
```

---

## 4. Variables and Basic Types

### Declaration Syntax

Modern style uses `::` separator:

```fortran
integer :: count
real :: temperature
double precision :: precise_value   ! Or: real(8), real(kind=8)
complex :: z
logical :: is_valid
character(len=20) :: name
```

### Kind Parameters

Fortran uses "kinds" to specify precision:

```fortran
use, intrinsic :: iso_fortran_env, only: real32, real64, int32, int64

real(real64) :: double_precision_value   ! 64-bit float
integer(int64) :: big_integer            ! 64-bit integer
```

### Parameters (Constants)

```fortran
real, parameter :: pi = 3.14159265359
integer, parameter :: max_size = 1000
```

---

## 5. Functions and Subroutines

### Functions

Return a single value. The result can be named explicitly or use the function name.

```fortran
function add(a, b) result(c)
    real, intent(in) :: a, b
    real :: c
    c = a + b
end function add

! Alternative: result stored in function name
function multiply(a, b)
    real, intent(in) :: a, b
    real :: multiply
    multiply = a * b
end function multiply
```

**Calling a function:**
```fortran
x = add(1.0, 2.0)
```

### Subroutines

Don't return a value directly; use arguments for input/output.

```fortran
subroutine swap(a, b)
    real, intent(inout) :: a, b
    real :: temp
    temp = a
    a = b
    b = temp
end subroutine swap
```

**Calling a subroutine:**
```fortran
call swap(x, y)
```

### Intent Attributes

**Critical for clarity and compiler optimization:**

| Intent | Meaning |
|--------|---------|
| `intent(in)` | Read-only; procedure cannot modify it |
| `intent(out)` | Write-only; undefined on entry, must be set |
| `intent(inout)` | Read-write; can be read and modified |

```fortran
subroutine process(input, output, workspace)
    real, intent(in) :: input(:)
    real, intent(out) :: output(:)
    real, intent(inout) :: workspace(:)
    ! ...
end subroutine
```

---

## 6. Modules

Modules are Fortran's primary mechanism for organizing code, sharing data, and providing explicit interfaces.

```fortran
module math_utils
    implicit none
    
    real, parameter :: pi = 3.14159265359
    
contains
    
    function circle_area(radius) result(area)
        real, intent(in) :: radius
        real :: area
        area = pi * radius**2
    end function circle_area
    
    function circle_circumference(radius) result(circ)
        real, intent(in) :: radius
        real :: circ
        circ = 2.0 * pi * radius
    end function circle_circumference
    
end module math_utils
```

**Using a module:**
```fortran
program main
    use math_utils                    ! Import everything
    use math_utils, only: pi          ! Import only pi
    use math_utils, only: area => circle_area  ! Rename on import
    implicit none
    
    print *, circle_area(2.0)
end program
```

**Key benefits of modules:**
- Automatic explicit interfaces for contained procedures
- Encapsulation of related functionality
- Namespace management via `only` and renaming

---

## 7. Derived Types (Structs)

User-defined composite types, similar to structs in C.

```fortran
module particle_module
    implicit none
    
    type :: particle
        real :: x, y, z           ! Position
        real :: vx, vy, vz        ! Velocity
        real :: mass
        character(len=10) :: name
    end type particle
    
contains
    
    function kinetic_energy(p) result(ke)
        type(particle), intent(in) :: p
        real :: ke
        ke = 0.5 * p%mass * (p%vx**2 + p%vy**2 + p%vz**2)
    end function kinetic_energy
    
end module particle_module

program main
    use particle_module
    implicit none
    
    type(particle) :: electron
    
    electron%mass = 9.109e-31
    electron%vx = 1.0e6
    electron%vy = 0.0
    electron%vz = 0.0
    electron%name = "electron"
    
    print *, "KE = ", kinetic_energy(electron)
end program
```

**Note:** Use `%` to access components (not `.` like C).

---

## 8. Arrays

Arrays are Fortran's strength. They're column-major (first index varies fastest in memory).

### Declaration

```fortran
real :: vector(10)              ! 1D array, indices 1 to 10
real :: matrix(3, 3)            ! 2D array, 3x3
real :: custom(-5:5)            ! Custom bounds: -5 to 5
real, dimension(100) :: data    ! Alternative syntax
```

### Array Operations

Fortran supports whole-array operations:

```fortran
real :: a(100), b(100), c(100)

c = a + b           ! Element-wise addition
c = a * b           ! Element-wise multiplication
c = sin(a)          ! Apply sin to every element
c = a + 1.0         ! Add scalar to every element

where (a > 0)
    c = sqrt(a)
elsewhere
    c = 0.0
end where
```

### Array Sections (Slicing)

```fortran
real :: matrix(10, 10)

matrix(1, :)        ! First row
matrix(:, 1)        ! First column
matrix(1:5, 1:5)    ! Upper-left 5x5 submatrix
matrix(1:10:2, :)   ! Odd rows (stride of 2)
matrix(10:1:-1, :)  ! Rows in reverse order
```

---

## 9. Allocatable Arrays

Dynamic arrays that can be sized at runtime.

```fortran
program dynamic_arrays
    implicit none
    
    real, allocatable :: data(:)
    real, allocatable :: matrix(:,:)
    integer :: n
    
    print *, "Enter array size:"
    read *, n
    
    allocate(data(n))
    allocate(matrix(n, n))
    
    data = 0.0
    matrix = 0.0
    
    ! Check allocation status
    if (allocated(data)) print *, "data is allocated"
    
    ! Resize (Fortran 2003+)
    deallocate(data)
    allocate(data(2*n))
    
    ! Automatic deallocation when going out of scope (F2003+)
    ! But explicit deallocation is good practice:
    deallocate(data)
    deallocate(matrix)
end program
```

### Allocatable Components in Derived Types

```fortran
type :: flexible_container
    real, allocatable :: values(:)
    integer :: size
end type
```

---

## 10. Assumed-Shape vs Assumed-Size Arrays

This distinction is crucial when passing arrays to procedures.

### Assumed-Shape Arrays (Modern, Preferred)

The procedure knows the shape via a descriptor. **Requires explicit interface.**

```fortran
subroutine process_modern(arr)
    real, intent(in) :: arr(:,:)    ! Assumed-shape: knows dimensions
    
    print *, "Shape:", shape(arr)
    print *, "Size:", size(arr)
    print *, "Rows:", size(arr, 1)
    print *, "Cols:", size(arr, 2)
end subroutine
```

### Assumed-Size Arrays (Legacy)

Only the last dimension is assumed (`*`). The procedure **does not know** the actual size—you must pass it explicitly.

```fortran
subroutine process_legacy(arr, m, n)
    integer, intent(in) :: m, n
    real, intent(in) :: arr(m, *)   ! Assumed-size: last dim unknown
    
    ! Cannot use: shape(arr), size(arr), arr(:,:)
    ! Must manually track dimensions
end subroutine
```

### Explicit-Shape Arrays (Also Legacy)

Dimensions passed as arguments:

```fortran
subroutine process_explicit(arr, m, n)
    integer, intent(in) :: m, n
    real, intent(in) :: arr(m, n)   ! Explicit-shape
    
    ! Can use size() and shape() here
end subroutine
```

**Summary table:**

| Style | Syntax | Knows Shape? | Interface Required? |
|-------|--------|--------------|---------------------|
| Assumed-shape | `arr(:,:)` | Yes | Yes (explicit) |
| Assumed-size | `arr(m,*)` | No | No |
| Explicit-shape | `arr(m,n)` | Yes | No |

---

## 11. Array Intrinsic Functions

Essential built-in functions for working with arrays:

### Shape and Size

```fortran
real :: a(3, 4, 5)

size(a)         ! Total elements: 60
size(a, 1)      ! Size along dimension 1: 3
size(a, dim=2)  ! Size along dimension 2: 4
shape(a)        ! Array of dimensions: [3, 4, 5]
rank(a)         ! Number of dimensions: 3 (Fortran 2008)
lbound(a)       ! Lower bounds: [1, 1, 1]
ubound(a)       ! Upper bounds: [3, 4, 5]
lbound(a, 1)    ! Lower bound of dim 1: 1
```

### Reduction Operations

```fortran
real :: x(100)

sum(x)              ! Sum of all elements
product(x)          ! Product of all elements
maxval(x)           ! Maximum value
minval(x)           ! Minimum value
maxloc(x)           ! Location of maximum (array)
minloc(x)           ! Location of minimum (array)

sum(x, mask=x>0)    ! Sum of positive elements only
count(x > 0)        ! Count of positive elements
any(x > 0)          ! True if any element > 0
all(x > 0)          ! True if all elements > 0
```

### Array Construction and Manipulation

```fortran
real :: a(3,3), b(9)

reshape(b, [3,3])         ! Reshape 1D to 2D
transpose(a)              ! Matrix transpose
spread(x, dim=1, ncopies=3)  ! Replicate along new dimension
pack(a, mask=a>0)         ! Extract elements where mask is true
merge(a, b, mask)         ! Choose from a or b based on mask

matmul(a, b)              ! Matrix multiplication
dot_product(x, y)         ! Vector dot product
```

---

## 12. Explicit Interfaces

An **interface** tells the compiler the signature of a procedure. Without it, the compiler cannot check argument types, shapes, or counts.

### Why Explicit Interfaces Matter

```fortran
! Without explicit interface - DANGEROUS
program no_interface
    implicit none
    real :: x(10)
    
    call process(x)   ! Compiler has no idea what process() expects
                      ! Wrong arguments = silent bugs or crashes
end program
```

### How to Get Explicit Interfaces

**1. Put procedures in modules (best practice):**
```fortran
module my_procedures
    implicit none
contains
    subroutine process(arr)
        real, intent(inout) :: arr(:)
        ! ...
    end subroutine
end module

program main
    use my_procedures   ! Interface automatically available
    implicit none
    real :: x(10)
    call process(x)     ! Compiler checks this call
end program
```

**2. Use an interface block:**
```fortran
program main
    implicit none
    
    interface
        subroutine external_process(arr, n)
            real, intent(inout) :: arr(*)
            integer, intent(in) :: n
        end subroutine
    end interface
    
    real :: x(10)
    call external_process(x, 10)
end program
```

**3. Use `contains` for internal procedures:**
```fortran
program main
    implicit none
    real :: x(10)
    
    call process(x)
    
contains
    subroutine process(arr)
        real, intent(inout) :: arr(:)
        ! ...
    end subroutine
end program
```

### When Explicit Interfaces are Required

- Assumed-shape arrays (`arr(:)`)
- Allocatable or pointer arguments
- Optional arguments
- Keyword arguments
- Functions returning arrays or pointers
- `pure` or `elemental` procedures

---

## 13. Control Flow

### If Statements

```fortran
! Simple if
if (x > 0) print *, "positive"

! If-then-else
if (x > 0) then
    print *, "positive"
else if (x < 0) then
    print *, "negative"
else
    print *, "zero"
end if

! Named blocks (useful for nested ifs)
outer: if (condition1) then
    inner: if (condition2) then
        ! ...
    end if inner
end if outer
```

### Logical Operators

| Modern | Legacy | Meaning |
|--------|--------|---------|
| `==` | `.eq.` | Equal |
| `/=` | `.ne.` | Not equal |
| `<` | `.lt.` | Less than |
| `<=` | `.le.` | Less or equal |
| `>` | `.gt.` | Greater than |
| `>=` | `.ge.` | Greater or equal |
| `.and.` | | Logical AND |
| `.or.` | | Logical OR |
| `.not.` | | Logical NOT |
| `.eqv.` | | Logical equivalence |
| `.neqv.` | | Logical XOR |

### Select Case (Switch)

```fortran
select case (grade)
    case ('A')
        print *, "Excellent"
    case ('B', 'C')
        print *, "Good"
    case ('D':'F')      ! Range
        print *, "Needs improvement"
    case default
        print *, "Invalid grade"
end select
```

---

## 14. Do Loops

### Counted Do Loop

```fortran
do i = 1, 10
    print *, i
end do

do i = 10, 1, -1    ! Count down (step = -1)
    print *, i
end do

do i = 1, 100, 2    ! Odd numbers (step = 2)
    print *, i
end do
```

### Do While

```fortran
do while (error > tolerance)
    call iterate(x, error)
end do
```

### Infinite Loop with Exit

```fortran
do
    call compute(x, converged)
    if (converged) exit
end do
```

### Cycle (Continue to Next Iteration)

```fortran
do i = 1, 100
    if (mod(i, 2) == 0) cycle   ! Skip even numbers
    print *, i
end do
```

### Named Loops

```fortran
outer: do i = 1, 10
    inner: do j = 1, 10
        if (condition) exit outer   ! Exit the outer loop
        if (other) cycle inner      ! Continue inner loop
    end do inner
end do outer
```

---

## 15. Do Concurrent (Fortran 2008+)

Declares that loop iterations are independent and can execute in parallel.

```fortran
do concurrent (i = 1:n)
    a(i) = b(i) + c(i)
end do

! Multiple indices
do concurrent (i = 1:m, j = 1:n)
    matrix(i,j) = i + j
end do

! With mask
do concurrent (i = 1:n, x(i) > 0)
    y(i) = sqrt(x(i))
end do

! With locality specifiers (Fortran 2018)
do concurrent (i = 1:n) local(temp) shared(a, b)
    temp = a(i) * 2
    b(i) = temp + 1
end do
```

**Restrictions inside `do concurrent`:**
- No `exit`, `cycle` (to outer loops), `return`, or `stop`
- No I/O statements (in standard; some compilers allow it)
- No procedure calls with side effects
- Each iteration must be independent

---

## 16. Legacy Features

These features appear in older codes like GAMESS. Understanding them helps you read legacy code, but **avoid using them in new code**.

### Common Blocks

Shared global memory regions. Predates modules.

```fortran
! In one file
subroutine set_values()
    common /shared_data/ x, y, z
    real :: x, y, z
    x = 1.0
    y = 2.0
    z = 3.0
end subroutine

! In another file - must match EXACTLY
subroutine use_values()
    common /shared_data/ a, b, c   ! Same memory, different names
    real :: a, b, c
    print *, a, b, c               ! Prints 1.0, 2.0, 3.0
end subroutine
```

**Problems with common blocks:**
- No type checking across files
- Names don't have to match, only memory layout
- Easy to get out of sync
- Global mutable state is error-prone

**Modern replacement:** Use modules with module variables.

### Hollerith Constants

Ancient way to embed strings in numeric variables. You'll see these in GAMESS.

```fortran
! Hollerith: nH followed by n characters
data label /6HRESULT/    ! 6 characters: "RESULT"

! These were used to pass "strings" before character types existed
call old_sub(4HTEST)
```

The number before `H` indicates how many characters follow. These get stored in integer or real variables by reinterpreting the bytes.

**Modern replacement:** Use `character` type.

### GOTO Statements

Unconditional jump. Still sometimes used, but `exit`, `cycle`, and structured constructs are preferred.

```fortran
      if (error) goto 999
      ! ... normal processing ...
      return
  999 continue
      print *, "Error occurred"
      stop
```

**Computed GOTO (obsolete):**
```fortran
      goto (100, 200, 300), index   ! Jump to label based on index value
  100 continue
      ! ... case 1 ...
      goto 400
  200 continue
      ! ... case 2 ...
      goto 400
  300 continue
      ! ... case 3 ...
  400 continue
```

### Arithmetic IF (Obsolete)

Three-way branch based on sign of expression:

```fortran
      if (x) 10, 20, 30   ! Negative, zero, positive
   10 continue
      print *, "x is negative"
      goto 40
   20 continue
      print *, "x is zero"
      goto 40
   30 continue
      print *, "x is positive"
   40 continue
```

This is equivalent to:
```fortran
if (x < 0) then
    print *, "x is negative"
else if (x == 0) then
    print *, "x is zero"
else
    print *, "x is positive"
end if
```

### Statement Labels

Numeric labels at the start of statements (columns 1-5 in fixed form):

```fortran
   10 format(F10.3)
  100 continue
  999 stop 'Fatal error'
```

### Obsolete Loop Syntax

```fortran
C     Old-style DO with label and CONTINUE
      DO 10 I = 1, 100
         X(I) = I * 2.0
   10 CONTINUE

C     Shared termination (multiple loops, one label) - VERY confusing
      DO 20 I = 1, 10
      DO 20 J = 1, 10
         A(I,J) = 0.0
   20 CONTINUE
```

---

## 17. Basic I/O

### Print and Read

```fortran
print *, "Hello"              ! List-directed output
print '(A,I5)', "Value:", n   ! Formatted output

read *, x, y, z               ! List-directed input
read '(3F10.5)', a, b, c      ! Formatted input
```

### File I/O

```fortran
integer :: unit_num
real :: data(100)

! Open file
open(newunit=unit_num, file='data.txt', status='old', action='read')

! Read
read(unit_num, *) data

! Close
close(unit_num)

! Write to new file
open(newunit=unit_num, file='output.txt', status='replace', action='write')
write(unit_num, '(F12.6)') data
close(unit_num)
```

### Format Specifiers

| Specifier | Meaning |
|-----------|---------|
| `I5` | Integer, 5 characters wide |
| `F10.3` | Float, 10 wide, 3 decimal places |
| `E12.4` | Scientific notation |
| `A` | Character string |
| `A20` | Character, 20 wide |
| `X` | Skip one space |
| `3X` | Skip three spaces |
| `/` | New line |

---

## 18. Additional Modern Features

### Associate Construct

Create local aliases for complex expressions:

```fortran
associate (vel => particles(i)%velocity, &
           pos => particles(i)%position)
    pos = pos + vel * dt
    vel = vel + acceleration * dt
end associate
```

### Block Construct

Local scope within a procedure:

```fortran
block
    real :: temp_array(1000)   ! Only exists in this block
    temp_array = compute_something()
    result = sum(temp_array)
end block   ! temp_array deallocated here
```

### Elemental Procedures

Functions that work on scalars but automatically apply to arrays:

```fortran
elemental function celsius_to_fahrenheit(c) result(f)
    real, intent(in) :: c
    real :: f
    f = c * 9.0/5.0 + 32.0
end function

! Usage:
real :: temps_c(100), temps_f(100)
temps_f = celsius_to_fahrenheit(temps_c)   ! Works on whole array
```

### Pure Procedures

No side effects—can be used in `do concurrent` and `forall`:

```fortran
pure function square(x) result(y)
    real, intent(in) :: x
    real :: y
    y = x * x
end function
```

---

## 19. Quick Reference: What's Modern vs Legacy

| Feature | Legacy | Modern |
|---------|--------|--------|
| Source format | Fixed form | Free form |
| Declarations | `REAL X` | `real :: x` |
| Type safety | Implicit typing | `implicit none` |
| Sharing data | Common blocks | Modules |
| Strings | Hollerith | `character(len=n)` |
| Branching | GOTO, arithmetic IF | `if/else`, `select case` |
| Loops | DO with labels | `do/end do`, `do concurrent` |
| Arrays | Assumed-size | Assumed-shape, allocatable |
| Procedures | External | Module procedures |
| Memory | Static/common | Allocatable |

---

## 20. Compiler Flags You Should Know

When building Fortran code, these flags help catch errors:

**gfortran:**
```bash
gfortran -Wall -Wextra -fcheck=all -fbacktrace -g -O0  # Debug
gfortran -O3 -march=native                              # Release
```

**Intel (ifx/ifort):**
```bash
ifx -warn all -check all -traceback -g -O0   # Debug
ifx -O3 -xHost                               # Release
```

**NVIDIA (nvfortran):**
```bash
nvfortran -Minform=warn -Mbounds -traceback -g -O0   # Debug
nvfortran -O3 -fast                                   # Release
```

---

## Summary

When reading Fortran code:

1. **Check the file extension** to know if it's fixed or free form
2. **Look for `implicit none`**—if missing, be extra careful about variable types
3. **Identify modules** to understand code organization
4. **Watch array syntax** closely—Fortran arrays are powerful but have subtleties
5. **Note interface blocks** or module procedures for type-safe calls
6. **Recognize legacy patterns** (common blocks, GOTO, Hollerith) in older codes

When writing Fortran code:

1. **Always use `implicit none`**
2. **Put procedures in modules**
3. **Use assumed-shape arrays** with `intent` attributes
4. **Use allocatable** for dynamic sizing
5. **Prefer modern control structures** (`do/end do`, `if/end if`)
6. **Avoid GOTO** except for error handling in legacy code

---

*This guide covers Fortran 77 through Fortran 2018. Language features continue to evolve—Fortran 2023 adds more parallelism features and other enhancements.*
