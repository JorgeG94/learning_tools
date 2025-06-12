# learning_tools
Learning experiences with new tools at my new job at the NCI 

## How to compile things 

### GCC


Installing GCC to `/shared/compilers/gcc/x.y.z`
```
tar -xvf gcc-x.y.z.tar.gz
cd gcc-x.y.z
contrib/download_prerequisites
./configure --prefix=/shared/compilers/gcc/x.y.z --enable-languages=c,c++,fortran --enable-libgomp --enable-bootstrap --enable-shared --enable-threads=posix --with-tune=generic --disable-multilib
```
*  Exceptions:
```
./configure --prefix=/shared/compilers/gcc/#.#.#-aarch64 --enable-languages=c,c++,fortran --enable-libgomp --enable-bootstrap --enable-shared --enable-threads=posix  --disable-multilib
```
*  `make && make install`

Activating a specific GCC version
```
export GCCVERSION=x.y.z
export GCC_ROOT=/shared/compilers/gcc/$GCCVERSION
export PATH=$GCC_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$GCC_ROOT/lib64:$LD_LIBRARY_PATH
export LD_RUN_PATH=$GCC_ROOT/lib64:$LD_RUN_PATH
export CPATH=$GCC_ROOT/include:$CPATH
export INCLUDEPATH=$GCC_ROOT/include:$INCLUDEPATH
export CXX=$GCC_ROOT/bin/g++
export CC=$GCC_ROOT/bin/gcc
export FC=$GCC_ROOT/bin/gfortran
```

### CMake 

```
wget https://github.com/Kitware/CMake/releases/download/v3.30.5/cmake-3.30.5.tar.gz 
tar -xvf cmake-3.30.5.tar.gz
cd cmake-3.30.5/
./bootstrap --prefix=$HOME/install/cmake 
make -j install
export PATH=$HOME/install/cmake:$PATH 
```

### Julia on ARM 

```
wget https://julialang-s3.julialang.org/bin/linux/aarch64/1.11/julia-1.11.0-linux-aarch64.tar.gz
tar -xvf julia-1.11.0-linux-aarch64.tar.gz
export JULIA_ROOT=$HOME/path/to/julia
export PATH=$PATH:$JULIA_ROOT/bin
```
### MAGMA 

#### Frontier build verified Jan 2nd 2025
```
wget https://icl.utk.edu/projectsfiles/magma/downloads/magma-2.8.0.tar.gz 
tar -xvf magma-2.8.0.tar.gz 
module load cpe/24.07
module load cmake/3.27.9
module load gcc-native/13.2
module load rocm/6.3.1
module load cmake/3.27.9
module load cray-hdf5/1.14.3.1
module load craype-accel-amd-gfx90a
module load cray-libsci/23.12.5
export MAGMA_ROOT=/lustre/orion/proj-shared/chm213/software/magma-2.8.0-wrocm-6.3.1
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
export MPI_ROOT=$MPICH_DIR

cmake -DMAGMA_ENABLE_HIP=ON -DCMAKE_CXX_COMPILER=hipcc -DCMAKE_INSTALL_PREFIX=$MAGMA_ROOT -DGPU_TARGET=gfx90a -DFORTRAN_CONVENTION=-DADD_ -DBUILD_SHARED_LIBS=off -DCMAKE_POSITION_INDEPENDENT_CODE=ON ../
```

#### Pawsey magma spack oriented build

```
salloc -p gpu-dev --exclusive -A pawsey0799-gpu
module load spack/0.21.0
cp -r /software/setonix/2024.05/spack/var/spack/repos/pawsey/packages/magma $MYSOFTWARE/setonix/2024.05/spack_repo/packages
spack edit magma
Add the following version at line-28 
    version("2.8.0", sha256="f4e5e75350743fe57f49b615247da2cc875e5193cc90c11b43554a7c82cc4348")

spack install -j64 magma@2.8.0 +rocm ~cuda amdgpu_target=gfx90a
spack install -j64 magma@2.8.0 %cce@16.0.1 +rocm ~cuda amdgpu_target=gfx90a

```
### MPI 4.x

This won't build any GPU aware MPI versions. It is a pain with OpenMPI 4.x
```
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.6.tar.gz
tar -xvf openmpi-4.1.6.tar.gz
cd openmpi-4.1.6
./configure --enable-mpi-fortran --enable-mpi-cxx --prefix=$HOME/install/openmpi-4.1.6
make -j install
```

#### with nvhpc 
```
CC=nvc CXX=nvc++ FC=nvfortran FCFLAGS="-Mstandard -fPIC" CFLAGS="-mno-hle -fPIC" CXXFLAGS="-fPIC" ./configure --enable-mpi-fortran --enable-mpi-cxx --prefix=$HOME/install/nvhpc/24.5/openmpi/4.1.4
```
### MPI 5.x 

This will build a cuda aware version of MPI. Point to the correct lcoation of your cuda install
```
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-5.0.5.tar.gz
tar -xvf openmpi-5.0.5.tar.gz
cd openmpi-5.0.5
./configure --prefix=$HOME/install/openmpi-5.0.5 --with-cuda=/usr/local/cuda-11.7 -with-cuda-libdir=/usr/local/cuda-11.7/lib64/stubs  --enable-mca-dso=btl-smcuda,rcache-rgpusm,rcache-gpusm,accelerator-cuda
make -j install
```

If configuring MPI 5.x fails with pmix not found append `--with-pmix=internal

#### with nvhpc 

```
CC=nvc CXX=nvc++ FC=nvfortran FCFLAGS="-Mstandard -fPIC" CFLAGS="-mno-hle -fPIC" CXXFLAGS="-fPIC" ./configure --enable-mpi-fortran --prefix=$HOME/install/nvhpc/24.5/openmpi/5.0.5
```

### HDF5 

(probably needs to be updated...) 

In order to install this version of HDF5 follow the instructions below:

1. Go to: https://github.com/HDFGroup/hdf5/releases
2. Download the version you'd like
3. Untar the file (tar -xvf) and cd into the resulting directory 
4. run the cmake command listed below

```
cmake -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_SHARED_LIBS:BOOL=OFF -DBUILD_TESTING:BOOL=ON -DHDF5_BUILD_TOOLS:BOOL=ON -DCMAKE_INSTALL_PREFIX=$HOME/install/hdf5/ -DHDF5_BUILD_CPP_LIB=ON -DHDF5_BUILD_FORTRAN=ON ../
make -j install
```

This will install HDF5 into the directory where you untared. You can change this by following the instructions in the documentation of HDF5. For convience, we recommend defining an environment variable called HDF5_ROOT that holds the path of your installation. Some OS need you to add the library path to the LD_LIBRARY_PATH variable. You can do so by doing:
```
export HDF5_ROOT=/path/to/hdf5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF5_ROOT/lib
```

#### HDF5 parallel i/o with fortran 

```
CC=mpicc FC=mpif90 CFLAGS=-fPIC ./configure --enable-shared --enable-parallel --enable-fortran --enable-fortran2003 --prefix=$HOME/install/hdf5-mpi-fortran
```

### Julia 

You can download and install julia by following the instructions in their webpage: https://julialang.org/downloads/ 

After you have done it, you just need to create an alias to the julia executable or add it to your PATH. 

export JULIA_ROOT=/path/to/julia
export PATH=$JULIA_ROOT/bin:$PATH


### OpenBLAS 

```
git clone git@github.com:OpenMathLib/OpenBLAS.git 
cd OpenBlas 
git checkout v0.3.26 
make PREFIX=/path/to/install 
make install PREFIX=/path/to/install
```

#### Attention if linking to a Fortran app

Some old Fortran applications depend on setting the default integer size to be 8 bytes. If you don't enable this at build, your openblas will die all the time. 

Do not try to use the CMake as it currently is still beta and it will fail to install certain dependencies such as LAPACK. Open Makefile.rule to find the ability to turn on INTERFACE64

### CLANG and FLANG (classic) 
```
cd /where/you/want/to/build/flang
mkdir install

INSTALL_PREFIX=$HOME/install/clang/

# Targets to build should be one of: X86 PowerPC AArch64
CMAKE_OPTIONS="-DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX \
    -DCMAKE_CXX_COMPILER=$INSTALL_PREFIX/bin/clang++ \
    -DCMAKE_C_COMPILER=$INSTALL_PREFIX/bin/clang \
    -DGCC_INSTALL_PREFIX=$GCC_DIR \
    -DCMAKE_Fortran_COMPILER=$INSTALL_PREFIX/bin/flang 
    -DCMAKE_Fortran_COMPILER_ID=Flang \
    -DLLVM_TARGETS_TO_BUILD=X86"
```

```
. setup.sh

if [[ ! -d classic-flang-llvm-project ]]; then
    git clone -b release_16x https://github.com/flang-compiler/classic-flang-llvm-project.git
fi

cd classic-flang-llvm-project
mkdir -p build && cd build
cmake $CMAKE_OPTIONS -DCMAKE_C_COMPILER=$GCC_DIR/bin/gcc -DCMAKE_CXX_COMPILER=$GCC_DIR/bin/g++ \
      -DLLVM_ENABLE_CLASSIC_FLANG=ON -DLLVM_ENABLE_PROJECTS="clang;openmp" ../llvm
make
make install

```

```
. setup.sh

if [[ ! -d flang ]]; then
    git clone https://github.com/flang-compiler/flang.git
fi

(cd flang/runtime/libpgmath
 mkdir -p build && cd build
 cmake $CMAKE_OPTIONS ..
 make
 sudo make install)

cd flang
mkdir -p build && cd build
cmake $CMAKE_OPTIONS -DFLANG_LLVM_EXTENSIONS=ON ..
make
make install
```


https://stackoverflow.com/questions/26333823/clang-doesnt-see-basic-headers
