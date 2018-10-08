# dtb, diffusion toolbox

## MAPMRI
MAPMRI.cpp, a C++ implementation of Mean apparent propagator (MAP) MRI [1]

### dependencies:
1. nifticlib
2. Gurobi optimization library
3. EIGEN library

### compile:
DT=~/diffusion_toolbox\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
GUROBI=$DT/gurobi751\
g++ $DT/cpp/MAPMRI.cpp -o $DT/bin/MAPMRI \
-I$EIGEN -L$NIFTICLIB/linux/lib -I$NIFTICLIB/linux/niftilib -I$NIFTICLIB/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ \
-lniftiio -lznz -lz -lgurobi_c++ -lgurobi75 -O3 -march=native -std=c++17 -fopenmp -w



[1] Özarslan, Evren, et al. "Mean apparent propagator (MAP) MRI: a novel diffusion imaging method for mapping tissue microstructure." NeuroImage 78 (2013): 16-32.
