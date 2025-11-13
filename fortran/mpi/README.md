To compile use: mpifort -mp=gpu -O3 -acc=gpu -stdpar=gpu -gpu=mem:separate dc_scatter.f90 -cudalib=nvtx 
