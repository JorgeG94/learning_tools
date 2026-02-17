FC = nvfortran
FFLAGS = -mp=gpu -acc=gpu -stdpar=gpu
NVCC = nvcc
NVCCFLAGS = -arch=sm_70

all: ovh no_ovh hybrid

ovh: ovh.f90
	$(FC) $(FFLAGS) -o ovh ovh.f90

no_ovh: no_ovh.f90
	$(FC) $(FFLAGS) -o no_ovh no_ovh.f90

hybrid: mega_kernel.o hybrid.f90
	$(FC) $(FFLAGS) -cuda -o hybrid hybrid.f90 mega_kernel.o -lcudart

mega_kernel.o: mega_kernel.cu
	$(NVCC) $(NVCCFLAGS) -c mega_kernel.cu

clean:
	rm -f ovh no_ovh hybrid *.o *.mod
