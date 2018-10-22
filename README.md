# diffusion toolbox

## MAPMRI
MAPMRI.cpp, a multithreaded C++ implementation of Mean apparent propagator (MAP) MRI using OpenMP\
For MAPMRI, refer to [1]

### Dependencies:
1. nifticlib
2. Gurobi optimization library
3. EIGEN library  

After installing Gurobi, you might need to recompile the libgurobi_c++.a by  
cd ${GUROBI_HOME}/src/build   
make  
cd ${GUROBI_HOME}/lib  
mv ./libgurobi_c++.a ./libgurobi_c++.a.bak  
ln -s ${GUROBI_HOME}/src/build/libgurobi_c++.a ./libgurobi_c++.a  

### Compile:
DT=~/dtb\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
GUROBI=$DT/gurobi751\
g++ $DT/cpp/MAPMRI.cpp -o $DT/bin/MAPMRI -I$EIGEN -L$NIFTICLIB/linux/lib -I$NIFTICLIB/linux/niftilib -I$NIFTICLIB/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ -lniftiio -lznz -lz -lgurobi_c++ -lgurobi75 -O3 -march=native -std=c++17 -fopenmp -w

### Run:
MAPMRI dwi.nii.gz brain_mask.nii.gz bvals.txt bvecs.txt -grid_size 15 -order 6 -small_delta 62e-3 -big_delta 62e-3 -threads 10 




## dti_fit
dti_fit.m, a MATLAB implementation of diffusion tensor fitting

### Run:
dti_parameters = dti_fit('data',dwi,'bvals',bvals,'bvecs',bvecs,'brain_mask',brain_mask);


## References
[1] Özarslan, Evren, et al. "Mean apparent propagator (MAP) MRI: a novel diffusion imaging method for mapping tissue microstructure." NeuroImage 78 (2013): 16-32.
