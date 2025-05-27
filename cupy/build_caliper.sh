#!/bin/bash 

# How to build Caliper on Gadi 

git clone git@github.com:LLNL/Caliper.git
cd Caliper
git checkout releases/v2.12.0
module load gcc/12.2.0
module load cuda/12.8.0
module load openmpi/4.1.7 
module load python3/3.12.1
pip install pybind11 
# run python3 -m pybind11 --cmakedir 
export PYBIND_DIR=/your/path/to/pybind11/share/cmake/pybind11 
export CALIPER_INSTALL_DIR=/path/to/install
mkdir build 
cd build
cmake -DWITH_FORTRAN=ON -DWITH_MPI=ON -DBUILD_TESTING=ON -DWITH_NVTX=ON -DWITH_PYTHON_BINDINGS=ON -DWITH_CUPTI=ON -DCMAKE_PREFIX_PATH=$PYBIND_DIR -DCMAKE_INSTALL_PREFIX=$CALIPER_INSTALL_DIR ../
make -j install 

export PYTHONPATH=$PYTHONPATH:$CALIPER_INSTALL_DIR/lib/python3.12/site-packages/
export PYTHONPATH=$PYTHONPATH:$CALIPER_INSTALL_DIR/lib64/python3.12/site-packages/
