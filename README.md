# diffusion toolbox

## MAPMRI
MAPMRI.cpp, a multithreaded C++ implementation of Mean apparent propagator (MAP) MRI using OpenMP\
For MAPMRI, refer to [1]

### Dependencies:
1. nifticlib
2. Gurobi optimization library
3. EIGEN library  

### How to use
#### 1. Set PATH in ~/.bashrc
export PATH=$PATH:/home/xuagu37/diffusion_toolbox/bin
export PATH=$PATH:/home/xuagu37/diffusion_toolbox/bash
export GUROBI_HOME=/home/xuagu37/diffusion_toolbox/gurobi751/linux64
export PATH=$PATH:${GUROBI_HOME}/bin
export LD_LIBRARY_PATH=${GUROBI_HOME}/lib
export GRB_LICENSE_FILE=$HOME/gurobi.lic
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib
#### 2. Copy to local: git clone https://github.com/xuagu37/diffusion_toolbox.git
#### 3. Get a Gurobi licence
#### 4. Activate Gurobi licence: grbgetkey
#### 5. Compile:
DT=~/diffusion_toolbox\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
GUROBI=$DT/gurobi751\
g++ $DT/cpp/MAPMRI.cpp -o $DT/bin/MAPMRI -I$EIGEN -L$NIFTICLIB/linux/lib -I$NIFTICLIB/linux/niftilib -I$NIFTICLIB/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ -lniftiio -lznz -lz -lgurobi_c++ -lgurobi75 -O3 -march=native -std=c++17 -fopenmp -w
#### 6. After installing Gurobi, you might need to recompile the libgurobi_c++.a by  
cd ${GUROBI_HOME}/src/build   
make  
cd ${GUROBI_HOME}/lib  
mv ./libgurobi_c++.a ./libgurobi_c++.a.bak  
ln -s ${GUROBI_HOME}/src/build/libgurobi_c++.a ./libgurobi_c++.a  
#### 7. Run MAPMRI:
MAPMRI dwi.nii.gz brain_mask.nii.gz bvals.txt bvecs.txt -grid_size 15 -order 6 -small_delta 62e-3 -big_delta 62e-3 -threads 10 

## DiffusionTensorFit

### Dependencies:
1. nifticlib
2. Gurobi optimization library
3. EIGEN library  

### Compile:
DT=~/diffusion_toolbox\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
g++ $DT/cpp/DiffusionTensorFit.cpp -lniftiio -lznz -lz -I$DT/eigen/ -L$DT/nifticlib-2.0.0/linux/lib -I$DT/nifticlib-2.0.0/linux/niftilib -I$DT/nifticlib-2.0.0/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ -lniftiio -lznz -lz -lgurobi_c++ -lgurobi75  -O3  -march=native -std=c++17 -fopenmp -o $DT/bin/DiffusionTensorFit -w 

### Run:
DiffusionTensorFit dwi.nii.gz brain_mask.nii.gz bvals.txt bvecs.txt -threads 10

## dti_fit
dti_fit.m, a MATLAB implementation of diffusion tensor fitting

### Run:
dti_parameters = dti_fit('data',dwi,'bvals',bvals,'bvecs',bvecs,'brain_mask',brain_mask);


## References
[1] Özarslan, Evren, et al. "Mean apparent propagator (MAP) MRI: a novel diffusion imaging method for mapping tissue microstructure." NeuroImage 78 (2013): 16-32.
