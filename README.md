# dtb, diffusion toolbox

## MAPMRI
MAPMRI.cpp, a multithreaded C++ implementation of Mean apparent propagator (MAP) MRI using OpenMP.\
For MAPMRI, refer to [1].

### Dependencies:
1. nifticlib
2. Gurobi optimization library
3. EIGEN library

### Compile:
DT=~/diffusion_toolbox\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
GUROBI=$DT/gurobi751\
g++ $DT/cpp/MAPMRI.cpp -o $DT/bin/MAPMRI -I$EIGEN -L$NIFTICLIB/linux/lib -I$NIFTICLIB/linux/niftilib -I$NIFTICLIB/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ -lniftiio -lznz -lz -lgurobi_c++ -lgurobi75 -O3 -march=native -std=c++17 -fopenmp -w

### Run
MAPMRI dwi.nii.gz brain_mask.nii.gz bvals.txt bvecs.txt -grid_size 15 -order 6 -small_delta 62e-3 -big_delta 62e-3 -threads 10 


[1] Özarslan, Evren, et al. "Mean apparent propagator (MAP) MRI: a novel diffusion imaging method for mapping tissue microstructure." NeuroImage 78 (2013): 16-32.
