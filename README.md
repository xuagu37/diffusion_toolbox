# Diffusion toolbox

## MAPMRI
We provide a multithreaded c++ implementation for mean apparent propagator MRI (MAP-MRI) [1].   
The code was used in our MAP-MRI bootstrap paper [2].

### Dependencies:
1. nifticlib
2. Gurobi optimization library
3. EIGEN library  

### Getting started
#### 1. Set PATH in ~/.bashrc
```bash
export PATH=$PATH:/home/xuagu37/diffusion_toolbox/bin\
export PATH=$PATH:/home/xuagu37/diffusion_toolbox/bash\
export GUROBI_HOME=/home/xuagu37/diffusion_toolbox/gurobi751/linux64\
export PATH=$PATH:${GUROBI_HOME}/bin\
export LD_LIBRARY_PATH=${GUROBI_HOME}/lib\
export GRB_LICENSE_FILE=$HOME/gurobi.lic\
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GUROBI_HOME}/lib
```

#### 2. Copy to local: git clone https://github.com/xuagu37/diffusion_toolbox.git

#### 3. Get a Gurobi licence
https://www.gurobi.com/downloads/end-user-license-agreement-academic/  
#### 4. Activate Gurobi licence 
```bash
grbgetkey
```

#### 5. Compile
```bash
DT=~/diffusion_toolbox\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
GUROBI=$DT/gurobi751\
g++ $DT/cpp/MAPMRI.cpp -o $DT/bin/MAPMRI -I$EIGEN -L$NIFTICLIB/linux/lib -I$NIFTICLIB/linux/niftilib -I$NIFTICLIB/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ -lniftiio -lznz -lz -lgurobi_c++ -lgurobi75 -O3 -march=native -std=c++17 -fopenmp -w
```

#### 6. After installing Gurobi, you might need to recompile the libgurobi_c++.a 
```bash
cd ${GUROBI_HOME}/src/build\   
make  \
cd ${GUROBI_HOME}/lib  \
mv ./libgurobi_c++.a ./libgurobi_c++.a.bak  \
ln -s ${GUROBI_HOME}/src/build/libgurobi_c++.a ./libgurobi_c++.a  \
```

#### 7. Run MAPMRI
```bash
MAPMRI dwi.nii.gz brain_mask.nii.gz bvals.txt bvecs.txt -grid_size 15 -order 6 -small_delta 62e-3 -big_delta 62e-3 -threads 10 
```

## DiffusionTensorFit
We provide a multithreaded c++ implementation for diffusion tensor fit [3].


### Dependencies
1. nifticlib
2. Gurobi optimization library
3. EIGEN library  

### Compile
```bash
DT=~/diffusion_toolbox\
EIGEN=$DT/eigen\
NIFTICLIB=$DT/nifticlib-2.0.0\
g++ $DT/cpp/DiffusionTensorFit.cpp -lniftiio -lznz -lz -I$DT/eigen/ -L$DT/nifticlib-2.0.0/linux/lib -I$DT/nifticlib-2.0.0/linux/niftilib -I$DT/nifticlib-2.0.0/linux/znzlib -I$GUROBI/linux64/include -L$GUROBI/linux64/lib/ -lniftiio -lznz -lz -lgurobi_c++ -lgurobi75  -O3  -march=native -std=c++17 -fopenmp -o $DT/bin/DiffusionTensorFit -w 
```

### Run
```bash
DiffusionTensorFit dwi.nii.gz brain_mask.nii.gz bvals.txt bvecs.txt -threads 10
```

## dti_fit
We provide a MATLAB implementation of diffusion tensor fitting [3].

### Run
```matlab
dti_parameters = dti_fit('data',dwi,'bvals',bvals,'bvecs',bvecs,'brain_mask',brain_mask);
```

## References
[1] Özarslan, E., Koay, C.G., Shepherd, T.M., Komlosh, M.E., İrfanoğlu, M.O., Pierpaoli, C. and Basser, P.J., 2013. Mean apparent propagator (MAP) MRI: a novel diffusion imaging method for mapping tissue microstructure. NeuroImage, 78, pp.16-32.
[2] Gu, X., Eklund, A., Özarslan, E. and Knutsson, H., 2019. Using the wild bootstrap to quantify uncertainty in mean apparent propagator MRI. Frontiers in Neuroinformatics, 13, p.43.  
[3] Pierpaoli, C., Jezzard, P., Basser, P.J., Barnett, A. and Di Chiro, G., 1996. Diffusion tensor MR imaging of the human brain. Radiology, 201(3), pp.637-648.
