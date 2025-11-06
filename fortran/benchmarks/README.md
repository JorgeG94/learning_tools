# Loop ordering benchmarks

This is a (stupid) attempt to understand loop ordering behavior in the 
vertical viscocity module in MOM6.

For a set nx, ny I loop over multiple values of nz (vertical layers) to 
see how the code behaves. The loop orderings are:

```
 vertical->i->j elapsed:     0.0020 s
 i->j->vertical elapsed:      0.0016 s
 j->vertical->i elapsed:      0.0011 s
 vertical->j->i elapsed:      0.0015 s
```

## Compile

The Makefile right now is jerry rigged to use nvfortan. Will extend later. Just `make`.


### Flags for GPU offloading 

- NVIDIA: `nvfortran -mp=gpu -stdpar=gpu -gpu=mem:separate`
- AMD:    `amdflang -fopenmp --offload-arch=gfx90a -fdo-concurrent-to-openmp=device`
- INTEL:  `ifx -fiopenmp -fopenmp-targets=spir64  -fopenmp-target-do-concurrent  -fopenmp-do-concurrent-maptype-modifier=none`

## Details of implementations 

I do a very simple stencil like computation over the layers and horizontal points. The 
benchmarks are:

- serial do (this is do, do, do) triple nested 
- serial do concurrent. (do concurrent ijk) without `-stdpar=multicore`
- multicore do concurrent: with `-stdpar=multicore`
  - Here it is interesting to run the code with `ACC_NUM_CORES=1`, this replicates serial dododo performance 
- gpu do concurrent with openmp memory management 

Here are some benchmark results with 1024
```
! serial, triple nested do
 Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i
   10,    0.019742,    0.014636,    0.014633,    0.019711
   25,    0.055599,    0.040107,    0.039818,    0.056021
   50,    0.115540,    0.082892,    0.081659,    0.115319
  100,    0.238410,    0.167818,    0.169176,    0.235396
  200,    0.473674,    0.343457,    0.340936,    0.472812
  400,    0.947998,    0.741353,    0.731510,    0.947599
! serial, dc no stdpar
 Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i
   10,    0.187092,    0.179741,    0.014913,    0.021410
   25,    0.605634,    0.634475,    0.039990,    0.061305
   50,    1.346150,    1.558753,    0.081965,    0.127570
  100,    2.842295,    3.410667,    0.165134,    0.263256
  200,    5.975079,    6.923923,    0.336171,    0.532912
  400,   14.469462,   13.467266,    0.682110,    1.121559
! serial, dc stdpar acc cores 1
 Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i
   10,    0.022786,    0.014929,    0.014752,    0.022717
   25,    0.064856,    0.040659,    0.040264,    0.065112
   50,    0.136139,    0.083670,    0.082530,    0.136767
  100,    0.278299,    0.169777,    0.168358,    0.283645
  200,    0.551403,    0.349195,    0.344826,    0.560834
  400,    1.111615,    0.732145,    0.725876,    1.134471
! parallel, dc stdpar acc cores 20
 Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i
   10,    0.004373,    0.003514,    0.003328,    0.003969
   25,    0.010702,    0.010480,    0.010368,    0.010172
   50,    0.023186,    0.020458,    0.020390,    0.022932
  100,    0.057726,    0.045676,    0.041948,    0.057277
  200,    0.125344,    0.096001,    0.086213,    0.125260
  400,    0.251209,    0.197070,    0.170460,    0.250801
! parallel, dc gpu v100
 Nz,vertical->i->j,i->j->vertical,j->vertical->i,vertical->j->i
   10,    0.000486,    0.000332,    0.000335,    0.000487
   25,    0.000988,    0.000610,    0.000615,    0.000987
   50,    0.002047,    0.001168,    0.001204,    0.002072
  100,    0.004212,    0.002366,    0.002419,    0.004217
  200,    0.008493,    0.004672,    0.004841,    0.008504
  400,    0.018087,    0.010600,    0.010362,    0.017297
```
