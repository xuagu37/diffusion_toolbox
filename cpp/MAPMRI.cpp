#include <iostream>
#include <cmath>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/Polynomials>
#include "fstream"
#include "nifti1_io.h"
#include "HelpFunctions.cpp"
#include <functional>
#include <sstream>
#include <omp.h>
#include <math.h>       /* asin */
#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::MatrixXcd;
using Eigen::Vector3cd;
using Eigen::VectorXd;
using Eigen::SelfAdjointEigenSolver;
using Eigen::PolynomialSolver;
using Eigen::Map;
using Eigen::Tensor;

#define EIGEN_DONT_PARALLELIZE
#define ADD_FILENAME true
#define DONT_CHECK_EXISTING_FILE false

int main(int argc, char **argv)
{
    printf("\nAuthored by X. Gu, Linkoping University. \n\n");
    // Size of the diffusion dataset X Y Z T
    size_t        DWI_DATA_H, DWI_DATA_W, DWI_DATA_D, DWI_DATA_T;
    // DWI_VOLUMES_SIZE = XYZT*sizeof(float), N = XYZ
    size_t        DWI_VOLUMES_SIZE, N, DWI_MASK_SIZE, MAPMRI_RTOP_SIZE, MAPMRI_RTAP_SIZE, MAPMRI_RTPP_SIZE, MAPMRI_NG_SIZE, MAPMRI_PA_SIZE, MAPMRI_RESIDUAL_SIZE;
    // Memory allocation related
    size_t        allocatedHostMemory = 0;
    int           numberOfMemoryPointers = 0, numberOfNiftiImages = 0;
    void          *allMemoryPointers[500];
    // Size of the voxel
    float         DWI_VOXEL_SIZE_X, DWI_VOXEL_SIZE_Y, DWI_VOXEL_SIZE_Z;
    // Input pointers
    float         *DWI_Data, *DWI_Mask;
    // Output pointers
    float          *MAPMRI_RTOP, *MAPMRI_RTAP, *MAPMRI_RTPP, *MAPMRI_NG, *MAPMRI_PA, *MAPMRI_RESIDUAL;
    // If data if less than MIN_SIGNAL, then replace with MIN_SIGNAL
    float   MIN_SIGNAL = 1;
    // If eigenvalue if less than MIN_DIFFUSIVITY, then replace with MIN_DIFFUSIVITY
    double  MIN_DIFFUSIVITY = 1e-4;
    // Order of Hermite polynomial
    int         order = 6;
    // Number of coefficients
    size_t         ncoeff;
    // Big delta and small delta for MAP-MRI, If unknown, calculate tau instead.
    float         big_delta = 0, small_delta = 0;
    // Diffusion time
    float         tau = 0;
    // Gyromagnetic ratio, 1/sec/G
    const float   GYRO = 2*M_PI*42.576e2;
    // Resolution of propagator constraints
    int           grid_size = 21;
    // Output
    const char*		outputFilename;
    // Use first n_dti volumes to do DTI fitting
    int     b_threshold = 2000;
    // DTI fit method
    const char*		dti_fit = "WLS";
    // Number of threads for OpemMP
    int     NUM_THREADS = 5;
    // Other
    bool			    VERBOSE = false;
    bool			    CHANGE_OUTPUT_FILENAME = false;    
    
    for (int i = 0; i < 500; i++)
    {
        allMemoryPointers[i] = NULL;
    }
    nifti_image*	allNiftiImages[500];
    for (int i = 0; i < 500; i++)
    {
        allNiftiImages[i] = NULL;
    }    
    
    FILE *fp = NULL;
    // No inputs, so print help text
    if (argc == 1)
    {
        printf("\nThis function applies MAP-MRI to a diffusion dataset.\n");
        printf("Usage:\n");
        printf("MAPMRI dwi.nii.gz brain_mask.nii.gz bvals bvecs [options]\n");
        printf("Options:\n");
        printf(" -order               Order of Hermite polynomial, default 6 \n");
        printf(" -grid_size           Displacement space grid size, default 15 \n");
        printf(" -big_delta           Separation of the diffusion-weighting gradients \n");
        printf(" -small_delta         Duration of the diffusion-weighting gradients \n");
        printf(" -MIN_SIGNAL          Replace data value below, default 1 \n");
        printf(" -MIN_DIFFUSIVITY     Replace eigenvalues below, default 1e-4 \n");
        printf(" -b_threshold         Use volumes below b_threshold for tensor fitting, default 2000 \n");
        printf(" -dti_fit             DTI fit method, default WLS \n");
        printf(" -output              Set output filename, optional\n");
        printf(" -threads             Number of threads for OpenMP, default 5 \n");
        printf(" -verbose             Print extra stuff (default false) \n");
        return EXIT_SUCCESS;
    }    
    else if (argc > 1)
    {
        std::string   extension;
        bool          extensionOK;
        // Check that file extension is .nii or .nii.gz
        CheckFileExtension(argv[1],extensionOK,extension);
        if (!extensionOK)
        {
            printf("File extension is not .nii or .nii.gz, %s is not allowed!\n",extension.c_str());
            return EXIT_FAILURE;
        }
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);
        // Check that file extension is .nii or .nii.gz
        CheckFileExtension(argv[2],extensionOK,extension);
        if (!extensionOK)
        {
            printf("File extension is not .nii or .nii.gz, %s is not allowed!\n",extension.c_str());
            return EXIT_FAILURE;
        }
        fp = fopen(argv[1],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[1]);
            return EXIT_FAILURE;
        }
        fclose(fp);
        // Check that file extension is .bvals
        CheckFileExtension(argv[3],extensionOK,extension);
        fp = fopen(argv[2],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[2]);
            return EXIT_FAILURE;
        }
        fclose(fp);
        fp = fopen(argv[2],"r");
        if (fp == NULL)
        {
            printf("Could not open file %s !\n",argv[2]);
            return EXIT_FAILURE;
        }
        fclose(fp);
    }
    
    // Loop over additional inputs
    int i = 5;
    while (i < argc)
    {
        char *input = argv[i];
        char *p;
        if (strcmp(input,"-verbose") == 0)
        {
            VERBOSE = true;
            i += 1;
        }
        else if (strcmp(input,"-big_delta") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read value after -big_delta !\n");
                return EXIT_FAILURE;
            }
            big_delta = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-small_delta") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read value after -small_delta !\n");
                return EXIT_FAILURE;
            }
            small_delta = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-order") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read value after -order !\n");
                return EXIT_FAILURE;
            }
            order = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-grid_size") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read value after -grid_size !\n");
                return EXIT_FAILURE;
            }
            grid_size = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-threads") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read name after -threads !\n");
                return EXIT_FAILURE;
            }
            NUM_THREADS = (int)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-output") == 0)
        {
            CHANGE_OUTPUT_FILENAME = true;
            if ( (i+1) >= argc  )
            {
                printf("Unable to read name after -output !\n");
                return EXIT_FAILURE;
            }
            outputFilename = argv[i+1];
            i += 2;
        }
        else if (strcmp(input,"-MIN_SIGNAL") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read name after -MIN_SIGNAL !\n");
                return EXIT_FAILURE;
            }
            MIN_SIGNAL = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-MIN_DIFFUSIVITY") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read name after -MIN_DIFFUSIVITY !\n");
                return EXIT_FAILURE;
            }
            MIN_DIFFUSIVITY = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-b_threshold") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read name after -b_threshold !\n");
                return EXIT_FAILURE;
            }
            b_threshold = (float)strtod(argv[i+1], &p);
            i += 2;
        }
        else if (strcmp(input,"-dti_fit") == 0)
        {
            if ( (i+1) >= argc  )
            {
                printf("Unable to read name after -dti_fit !\n");
                return EXIT_FAILURE;
            }
            dti_fit = argv[i+1];
            i += 2;
        }
        else
        {
            printf("Unrecognized option! %s \n",argv[i]);
            return EXIT_FAILURE;
        }
    }
    
    if (VERBOSE) {
        printf("MAPMRI fitting with constraints. \n\n");
    }
    
    double startTime = GetWallTime();
    // Read the diffusion data
    nifti_image *inputDWI = nifti_image_read(argv[1],1);
    if (inputDWI == NULL)
    {
        printf("Could not open diffusion data!\n");
        return EXIT_FAILURE;
    }
    allNiftiImages[numberOfNiftiImages] = inputDWI;
    numberOfNiftiImages++;
    // Read the brain mask
    nifti_image *inputMask = nifti_image_read(argv[2],1);
    if (inputMask == NULL)
    {
        printf("Could not open brain mask!\n");
        return EXIT_FAILURE;
    }
    allNiftiImages[numberOfNiftiImages] = inputMask;
    numberOfNiftiImages++;
    double endTime = GetWallTime();
    if (VERBOSE)
    {
        printf("It took %f seconds to read the diffusion data and brain mask.\n\n",(float)(endTime - startTime));
    }
    
    // Get data dimensions from input diffusion data
    DWI_DATA_W = inputDWI->nx;
    DWI_DATA_H = inputDWI->ny;
    DWI_DATA_D = inputDWI->nz;
    DWI_DATA_T = inputDWI->nt;
    N = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D; // Number of voxels
    DWI_VOXEL_SIZE_X = inputDWI->dx;
    DWI_VOXEL_SIZE_Y = inputDWI->dy;
    DWI_VOXEL_SIZE_Z = inputDWI->dz;
    
    // Check  if mask has same dimensions as reference volume
    size_t TEMP_DATA_W = inputMask->nx;
    size_t TEMP_DATA_H = inputMask->ny;
    size_t TEMP_DATA_D = inputMask->nz;
    if ( (TEMP_DATA_W != DWI_DATA_W) || (TEMP_DATA_H != DWI_DATA_H) || (TEMP_DATA_D != DWI_DATA_D) )
    {
        printf("Diffusion data has the dimensions %zu x %zu x %zu, while the brain mask has the dimensions %zu x %zu x %zu. Aborting! \n",DWI_DATA_W,DWI_DATA_H,DWI_DATA_D,TEMP_DATA_W,TEMP_DATA_H,TEMP_DATA_D);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }    
    
    ncoeff = round(1.0/6*(order/2+1)*(order/2+2)*(2*order+3)); // only even orders supported (symmtric pdf)
    // Memory size
    DWI_VOLUMES_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T * sizeof(float);
    DWI_MASK_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 1 * sizeof(float);
    MAPMRI_RTOP_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 1 * sizeof(float);
    MAPMRI_RTAP_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3 * sizeof(float);
    MAPMRI_RTPP_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3 * sizeof(float);
    MAPMRI_NG_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 7 * sizeof(float);
    MAPMRI_PA_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3 * sizeof(float);
    MAPMRI_RESIDUAL_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T * sizeof(float);
    
    // Read b-values
    MatrixXd bvals(DWI_DATA_T, 1);
    MatrixXd bvecs(DWI_DATA_T, 3);
    ifstream inputbvals(argv[3]);
    float read_value;
    // test file open
    if (inputbvals)
    {
        i = 0;
        // read the elements in the file into a vector
        while ( inputbvals >> read_value )
        {
            if (i > DWI_DATA_T - 1)
            {
                cout << "Number of B values should be the same as the number of data volumes!" << endl;
                return EXIT_FAILURE;
            }
            bvals(i) = read_value;
            i = i + 1;
        }
    }
    else
    {
        cout << "B values file does not exist." << endl;
        return 0;
    }
    inputbvals.close();
    // Check the how many lines in bvecs
    string line{"line"};
    int N_Bvecs_lines = 0;
    ifstream inputbvecs2(argv[4]);
    while (getline(inputbvecs2, line)) {
        N_Bvecs_lines++;
    }
    // Read bvecs
    ifstream inputbvecs(argv[4]);
    // test file open
    if (inputbvecs)
    {
        i = 0;
        // read the elements in the file into a matrix
        while ( inputbvecs >> read_value )
        {
            if (N_Bvecs_lines > DWI_DATA_T)
            {
                cout << "Number of B vectors should be the same as the number of data volumes!";
                return EXIT_FAILURE;
            }
            
            else if (N_Bvecs_lines == DWI_DATA_T)
            {
                bvecs(i/3, i%3) = read_value;
                i = i + 1;
            }
            else if (N_Bvecs_lines == 3)
            {
                if (i > 3*DWI_DATA_T - 1)
                {
                    cout << "Number of B vectors should be the same as the number of data volumes!" << endl;
                    return EXIT_FAILURE;
                }
                bvecs(i%DWI_DATA_T, i/DWI_DATA_T) = read_value;
                i = i + 1;
            }
        }
    }
    else
    {
        cout << "B vectors file does not exist." << endl;
        return 0;
    }
    
    // Print some info
    printf("DWI data size: %zu x %zu x %zu x %zu \n",  DWI_DATA_W, DWI_DATA_H, DWI_DATA_D, DWI_DATA_T);
    printf("DWI data voxel size: %f x %f x %f mm \n\n", DWI_VOXEL_SIZE_X, DWI_VOXEL_SIZE_Y, DWI_VOXEL_SIZE_Z);
    if (!VERBOSE){
        printf("\n");
    }
    
    startTime = GetWallTime();
    AllocateMemory(DWI_Data, DWI_VOLUMES_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DWI");
    AllocateMemory(DWI_Mask, DWI_MASK_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_MASK");
    AllocateMemory(MAPMRI_RTOP, MAPMRI_RTOP_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_RTOP");
    AllocateMemory(MAPMRI_RTAP, MAPMRI_RTAP_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_RTAP");
    AllocateMemory(MAPMRI_RTPP, MAPMRI_RTPP_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_RTPP");
    AllocateMemory(MAPMRI_NG, MAPMRI_NG_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_NG");
    AllocateMemory(MAPMRI_PA, MAPMRI_PA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_PA");
    AllocateMemory(MAPMRI_RESIDUAL, MAPMRI_RESIDUAL_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_RESIDUAL");
    
    endTime = GetWallTime();
    if (VERBOSE)
    {
        printf("It took %f seconds to allocate memory.\n",(float)(endTime - startTime));
    }
    
    startTime = GetWallTime();
    // Convert diffusion data to floats
    if ( inputDWI->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputDWI->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T; i++)
        {
            DWI_Data[i] = (float)p[i];
        }
    }
    else if ( inputDWI->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputDWI->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T; i++)
        {
            DWI_Data[i] = (float)p[i];
        }
    }
    else if ( inputDWI->datatype == DT_FLOAT )
    {
        float *p = (float*)inputDWI->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T; i++)
        {
            DWI_Data[i] = p[i];
        }
    }
    else
    {
        printf("Unknown data type in input diffusion data, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    // Convert brain mask to floats and get the number of voxels within the mask
    int N_in_mask = 0;
    if ( inputMask->datatype == DT_SIGNED_SHORT )
    {
        short int *p = (short int*)inputMask->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D; i++)
        {
            DWI_Mask[i] = (float)p[i];
            if (DWI_Mask[i] != 0){
                N_in_mask = N_in_mask + 1;
            }
        }
    }
    else if ( inputMask->datatype == DT_UINT8 )
    {
        unsigned char *p = (unsigned char*)inputMask->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D; i++)
        {
            DWI_Mask[i] = (float)p[i];
            if (DWI_Mask[i] != 0){
                N_in_mask = N_in_mask + 1;
            }
        }
    }
    else if ( inputMask->datatype == DT_FLOAT )
    {
        float *p = (float*)inputMask->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D; i++)
        {
            DWI_Mask[i] = p[i];
            if (DWI_Mask[i] != 0){
                N_in_mask = N_in_mask + 1;
            }
        }
    }
    else if ( inputMask->datatype == DT_DOUBLE )
    {
        float *p = (float*)inputMask->data;
        for (size_t i = 0; i < DWI_DATA_W * DWI_DATA_H * DWI_DATA_D; i++)
        {
            DWI_Mask[i] = p[i];
            if (DWI_Mask[i] != 0){
                N_in_mask = N_in_mask + 1;
            }
        }
    }
    else
    {
        printf("Unknown data type in input brain mask, aborting!\n");
        FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
        FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);
        return EXIT_FAILURE;
    }
    
    endTime = GetWallTime();
    if (VERBOSE)
    {
        printf("It took %f seconds to convert data to floats.\n\n",(float)(endTime - startTime));
    }
    
    // Get diffusion time
    if ((big_delta == 0) || (small_delta == 0) ){
        tau = 1 / (4 * pow(M_PI, 2));
    }
    else{
        tau = big_delta - small_delta / 3;
    }
    
    startTime = GetWallTime();
    
    
    int n_b0 = 0;  // Number of b0 volumes
    int n_subdata = 0;  // Number of sub volumes
    for (int i = 0; i < DWI_DATA_T; i++)
    {
        if (bvals(i) < 10)
        {
            n_b0++;
        }
        if (bvals(i) < b_threshold+100)
        {
            n_subdata++;
        }
    }
    
    MatrixXd X(DWI_DATA_T, 7);
    MatrixXd X2(n_subdata, 7);
    MatrixXd S_hat(DWI_DATA_T, 1);
    MatrixXd S_hat2(n_subdata, 1);
    MatrixXd W(DWI_DATA_T, DWI_DATA_T);
    MatrixXd W2(n_subdata, n_subdata);
    MatrixXd pinv_XW(7, DWI_DATA_T);
    MatrixXd pinv_XW2(7, n_subdata);
    MatrixXd bvecs2(n_subdata, 3);
    MatrixXd bvals2(n_subdata, 1);
    
    int k = 0;
    for (int i = 0; i < DWI_DATA_T; i++)
    {
        
        if (bvals(i) < b_threshold+100)
        {
            bvals2(k) = bvals(i);
            bvecs2.row(k) = bvecs.row(i);
            k++;
        }
    }    
    
    X.col(0) = (bvecs.col(0).array()*bvecs.col(0).array()*1*bvals.array()).matrix();
    X.col(1) = (bvecs.col(0).array()*bvecs.col(1).array()*2*bvals.array()).matrix();
    X.col(2) = (bvecs.col(1).array()*bvecs.col(1).array()*1*bvals.array()).matrix();
    X.col(3) = (bvecs.col(0).array()*bvecs.col(2).array()*2*bvals.array()).matrix();
    X.col(4) = (bvecs.col(1).array()*bvecs.col(2).array()*2*bvals.array()).matrix();
    X.col(5) = (bvecs.col(2).array()*bvecs.col(2).array()*1*bvals.array()).matrix();
    X.col(6).setOnes();
    X.col(6) = - X.col(6);
    X = -X;
    
    X2.col(0) = (bvecs2.col(0).array()*bvecs2.col(0).array()*1*bvals2.array()).matrix();
    X2.col(1) = (bvecs2.col(0).array()*bvecs2.col(1).array()*2*bvals2.array()).matrix();
    X2.col(2) = (bvecs2.col(1).array()*bvecs2.col(1).array()*1*bvals2.array()).matrix();
    X2.col(3) = (bvecs2.col(0).array()*bvecs2.col(2).array()*2*bvals2.array()).matrix();
    X2.col(4) = (bvecs2.col(1).array()*bvecs2.col(2).array()*2*bvals2.array()).matrix();
    X2.col(5) = (bvecs2.col(2).array()*bvecs2.col(2).array()*1*bvals2.array()).matrix();
    X2.col(6).setOnes();
    X2.col(6) = - X2.col(6);
    X2 = -X2;
    
    // Get q
    MatrixXd q(3,DWI_DATA_T);
    q = (GYRO/2/M_PI*(bvecs.array()*(bvals.replicate(1,3)/tau/pow(GYRO,2)).array().sqrt())).transpose();  // 1/um, 3xT
    MatrixXd pinv_X(7, DWI_DATA_T);
    MatrixXd pinv_X2(7, n_subdata);
    pinv_X = X.completeOrthogonalDecomposition().pseudoInverse();
    pinv_X2 = X2.completeOrthogonalDecomposition().pseudoInverse();
    
    MatrixXd Y(DWI_DATA_T, 1); // Data for one voxel
    MatrixXd Y2(n_subdata, 1); // Data for one subdata voxel
    MatrixXd Y_norm(DWI_DATA_T, 1); // Data for one voxel
    MatrixXd Y_norm1b0(DWI_DATA_T-n_b0+1, 1); // Data for one voxel    
    MatrixXd Y_norm2(n_subdata, 1); // Data for one subdata voxel
    MatrixXd logY(DWI_DATA_T, 1); // Data for one voxel
    MatrixXd logY2(n_subdata, 1); // Data for one subdata voxel
    
    float S0 = 0;
    MatrixXd tensor_elements(6, 1);
    MatrixXd tensor(3, 3);
    double eigenval1, eigenval2, eigenval3; // eigenvalues    
    MatrixXd R(3, 3); // rotate matrix    
    // Some coefficients used for Hermite polynomial
    MatrixXd n1_full(161, 1);
    MatrixXd n2_full(161, 1);
    MatrixXd n3_full(161, 1);
    n1_full << 0,2,0,0,1,1,0,4,0,0,3,3,1,1,0,0,2,2,0,2,1,1,6,0,0,5,5,1,1,0,0,4,4,2,2,0,0,4,1,1,3,3,0,3,3,2,2,1,1,2,8,0,0,7,7,1,1,0,0,6,6,2,2,0,0,6,1,1,5,5,3,3,0,0,5,5,2,2,1,1,4,4,0,4,4,3,3,1,1,4,2,2,3,3,2,10,0,0,9,9,1,1,0,0,8,8,2,2,0,0,8,1,1,7,7,3,3,0,0,7,7,2,2,1,1,6,6,4,4,0,0,6,6,3,3,1,1,6,2,2,5,5,0,5,5,4,4,1,1,5,5,3,3,2,2,4,4,2,4,3,3;
    n2_full << 0,0,2,0,1,0,1,0,4,0,1,0,3,0,3,1,2,0,2,1,2,1,0,6,0,1,0,5,0,5,1,2,0,4,0,4,2,1,4,1,3,0,3,2,1,3,1,3,2,2,0,8,0,1,0,7,0,7,1,2,0,6,0,6,2,1,6,1,3,0,5,0,5,3,2,1,5,1,5,2,4,0,4,3,1,4,1,4,3,2,4,2,3,2,3,0,10,0,1,0,9,0,9,1,2,0,8,0,8,2,1,8,1,3,0,7,0,7,3,2,1,7,1,7,2,4,0,6,0,6,4,3,1,6,1,6,3,2,6,2,5,0,5,4,1,5,1,5,4,3,2,5,2,5,3,4,2,4,3,4,3;
    n3_full << 0,0,0,2,0,1,1,0,0,4,0,1,0,3,1,3,0,2,2,1,1,2,0,0,6,0,1,0,5,1,5,0,2,0,4,2,4,1,1,4,0,3,3,1,2,1,3,2,3,2,0,0,8,0,1,0,7,1,7,0,2,0,6,2,6,1,1,6,0,3,0,5,3,5,1,2,1,5,2,5,0,4,4,1,3,1,4,3,4,2,2,4,2,3,3,0,0,10,0,1,0,9,1,9,0,2,0,8,2,8,1,1,8,0,3,0,7,3,7,1,2,1,7,2,7,0,4,0,6,4,6,1,3,1,6,3,6,2,2,6,0,5,5,1,4,1,5,4,5,2,3,2,5,3,5,2,4,4,3,3,4;
        
    MatrixXd n1(ncoeff, 1);
    MatrixXd n2(ncoeff, 1);
    MatrixXd n3(ncoeff, 1);
    n1 = n1_full.topRows(ncoeff);
    n2 = n2_full.topRows(ncoeff);
    n3 = n3_full.topRows(ncoeff);
    int N_constraints = grid_size*grid_size*(grid_size+1)/2;
    
    MatrixXd K(N_constraints, ncoeff);
    complex<double> complex_i(0.0,1.0);
    MatrixXd Q(DWI_DATA_T, ncoeff); // Design matrix
    MatrixXd Qiso(DWI_DATA_T, 3); // Design matrix
    MatrixXd Kiso(N_constraints, 3);
    MatrixXd constraint_grid_iso(N_constraints, 3);    
    MatrixXd Q1b0(DWI_DATA_T-n_b0+1, ncoeff); // Design matrix    
    MatrixXd Q0 = MatrixXd::Zero(1, ncoeff);
    MatrixXd Q0iso = MatrixXd::Zero(1, order/2+1);
    MatrixXd Q1b0iso(DWI_DATA_T-n_b0+1, order/2+1); // Design matrix    
    MatrixXd pinv_Q(ncoeff, DWI_DATA_T); // Design matrix
    MatrixXd mu(3, 1);
    MatrixXd constraint_grid(N_constraints, 3);
    MatrixXd dr(1,3); // Step size in x,y,z
    MatrixXd A(N_constraints + 1, ncoeff);  // constraint matrix
    MatrixXd H(ncoeff, ncoeff); // quadratic portion of objective function
    MatrixXd c(ncoeff, 1);   // linear portion of objective function
    MatrixXd Aiso(N_constraints + 1, order/2+1);  // constraint matrix
    MatrixXd Hiso(order/2+1, order/2+1); // quadratic portion of objective function
    MatrixXd ciso(order/2+1, 1);   // linear portion of objective function
    MatrixXd mapc(ncoeff, 1); // MAPMRI coefficients
    MatrixXd mapc_iso(order/2+1, 1); // MAPMRI coefficients
    // Settings for Gurobi dense_optimize
    if (VERBOSE)
    {
        printf("Gurobi is used to solve the quadratic optimization.\n");
    }

    GRBException e;
    GRBEnv env;
    cout << endl;
    env.set(GRB_IntParam_Threads, 1); // Set the number of threads
    bool    success;
    double  objval;
    char    sense[N_constraints + 1];
    for (int i = 0; i < N_constraints ; i++){
        sense[i] = '>';
    }
    sense[N_constraints] = '=';
    double  rhs[N_constraints + 1];
    for (int i = 0; i < N_constraints ; i++){
        rhs[i] = 0;
    }
    rhs[N_constraints] = 1;
    double lb[ncoeff];
    for (int i = 0; i < ncoeff; i++){
        lb[i] = -9999999;
    }    
    
    Vector3cd roots(3, 1);
    VectorXd polynomial(4, 1);
    PolynomialSolver<double,3> psolver;
    double mu0 = 0;
    double theta_PO_DTI = 0;
    double theta_PO = 0;    
    // Loop over all voxels
    int analyzed_voxels = 0;
    float analyzed_portion = 0.0f;
    float previous_analyzed_portion = 0.0f;    
    
    startTime = GetWallTime();
    int t = 0;
    float timeLeft = 0;
    SelfAdjointEigenSolver<MatrixXd> es;
    double currentTime;
#pragma omp parallel for shared (order, bvals, n_b0, N,  DWI_Mask, DWI_DATA_T, DWI_Data,X,X2, pinv_X,pinv_X2, tau, ncoeff, complex_i, n1, n2, n3, grid_size, N_constraints, sense, rhs, lb, MAPMRI_RTOP, MAPMRI_RTAP, MAPMRI_RTPP, MAPMRI_NG, MAPMRI_PA, MAPMRI_RESIDUAL, VERBOSE, analyzed_voxels, analyzed_portion, N_in_mask, timeLeft, startTime, previous_analyzed_portion) private (i, t, k,eigenval1, eigenval2, eigenval3, es, psolver, dr, success,  objval, pinv_Q,  currentTime) firstprivate (Y_norm,Y_norm1b0,Y_norm2, Y,Y2, logY,logY2, W,W2, pinv_XW,pinv_XW2, S_hat,S_hat2, tensor_elements, tensor, R, Q,Qiso,Kiso,constraint_grid, constraint_grid_iso,Q1b0,Q1b0iso, Q0, Q0iso, mu,mu0, K,e, env,mapc,mapc_iso,A, H, c,Aiso,Hiso,ciso, S0,  roots, polynomial, theta_PO_DTI, theta_PO) num_threads(NUM_THREADS)
    for (i = 0; i < N; i++)
    {
        if (DWI_Mask[i])
        {
            S0 = 0;
            k = 0;
            for (t = 0; t < DWI_DATA_T; t++)
            {
                Y(t) = max(DWI_Data[i + t*N], MIN_SIGNAL);
                
                if (bvals(t) < b_threshold + 100)
                {
                    Y2(k) = max(DWI_Data[i + t*N], MIN_SIGNAL);
                    k++;
                }
                if (bvals(t) < 10)
                {
                    S0 = S0 + Y(t);
                }
            }            
            S0 = S0/n_b0;
            Y_norm = Y/S0;
            Y_norm2 = Y2/S0;
            // Get the tensor for voxel i
            logY = Y_norm.array().log();
            logY2 = Y_norm2.array().log();            
            
            if (strcmp(dti_fit,"LS") == 0)
            {
                tensor_elements = (pinv_X2*logY2).topRows(6);
            }
            if (strcmp(dti_fit,"WLS") == 0)
            {
                tensor_elements = (pinv_X2*logY2);
                S_hat2 = ((X2*tensor_elements).array().exp())*((X2*tensor_elements).array().exp());
                W2 = S_hat2.asDiagonal();
                pinv_XW2 = ((X2.transpose()*W2*X2).inverse())*X2.transpose()*W2;
                tensor_elements = (pinv_XW2*logY2).topRows(6);
            }
            
            tensor << tensor_elements(0), tensor_elements(1), tensor_elements(3),
                    tensor_elements(1), tensor_elements(2), tensor_elements(4),
                    tensor_elements(3), tensor_elements(4), tensor_elements(5);
            
            es.compute(tensor);
            eigenval1 = max(es.eigenvalues()(2), MIN_DIFFUSIVITY);
            eigenval2 = max(es.eigenvalues()(1), MIN_DIFFUSIVITY);
            eigenval3 = max(es.eigenvalues()(0), MIN_DIFFUSIVITY);
            
            mu << eigenval1,
                    eigenval2,
                    eigenval3;
            mu = (2 * mu * tau).array().sqrt();
            R << es.eigenvectors()(0,2), es.eigenvectors()(0,1), es.eigenvectors()(0,0),
                    es.eigenvectors()(1,2), es.eigenvectors()(1,1), es.eigenvectors()(1,0),
                    es.eigenvectors()(2,2), es.eigenvectors()(2,1), es.eigenvectors()(2,0);
            Q = mapmri_phi_matrix(order, mu, R, q, n1, n2, n3);
            constraint_grid = create_space(grid_size, sqrt(5)*mu(0));
            dr = sqrt(5)*mu/grid_size;
            K = mapmri_psi_matrix(grid_size, constraint_grid, mu, n1, n2, n3);
            k = 1;
            Q0 = MatrixXd::Zero(1, ncoeff);
            for (t = 0; t < DWI_DATA_T; t++)
            {
                if (bvals(t) < 10)
                {
                    Q0 = Q.row(t) + Q0;
                }
                else
                {
                    Q1b0.row(k) = Q.row(t);
                    Y_norm1b0(k) = Y_norm(t);
                    k++;
                }
            }
            Q0 = Q0/n_b0;
            Q1b0.row(0) = Q0.row(0);
            Y_norm1b0(0) = S0;
            A << K*dr(0)*dr(1)*dr(2),
                    Q0;
            A = (abs(A.array() - 0) < 1e-13).select(0, A);
            H = 0.5*Q1b0.transpose()*Q1b0;
            c = -Y_norm1b0.transpose()*Q1b0;
            try
            {
                success = dense_optimize(&env, N_constraints + 1, ncoeff, c, H, A, sense, rhs, lb, NULL, NULL, mapc, &objval);
            }
            catch(GRBException e)
            {
                mapc = Q.completeOrthogonalDecomposition().pseudoInverse()*Y;
                cout << "Error code = " << e.getErrorCode() << endl;
                cout << e.getMessage() << endl;
            }
            catch(...)
            {
                mapc = Q.completeOrthogonalDecomposition().pseudoInverse()*Y;
                cout << "Exception during optimization" << endl;
            }
            
            for (t = 0; t < DWI_DATA_T; t++)
            {
                // Get data for voxel i
                MAPMRI_RESIDUAL[i + t*N] = (Y - Q*mapc*S0)(t);
            }
            
            MAPMRI_RTOP[i] = mapmri_rtop(mapc, mu, n1, n2, n3); cout<<"voxel i "<<i << " RTOP is " << MAPMRI_RTOP[i]<<endl;
            MAPMRI_RTAP[i] =  mapmri_rtap(mapc, mu, n1, n2, n3)(0); cout<<"voxel i "<<i << " RTAP is " << MAPMRI_RTAP[i]<<endl;
            MAPMRI_RTAP[i+N]  =  mapmri_rtap(mapc, mu, n1, n2, n3)(1);
            MAPMRI_RTAP[i+2*N] =  mapmri_rtap(mapc, mu, n1, n2, n3)(2);
            MAPMRI_RTPP[i] =  mapmri_rtpp(mapc, mu, n1, n2, n3)(0); cout<<"voxel i "<<i << " RTPP is " << MAPMRI_RTPP[i]<<endl;
            MAPMRI_RTPP[i+N]  =  mapmri_rtpp(mapc, mu, n1, n2, n3)(1);
            MAPMRI_RTPP[i+2*N] =  mapmri_rtpp(mapc, mu, n1, n2, n3)(2);
            MAPMRI_NG[i] = mapmri_ng(mapc, mu, n1, n2, n3, order)(0); cout<<"voxel i "<<i << " NG is " << MAPMRI_NG[i]<<endl;
            MAPMRI_NG[i+N] = mapmri_ng(mapc, mu, n1, n2, n3, order)(1);
            MAPMRI_NG[i+2*N] = mapmri_ng(mapc, mu, n1, n2, n3, order)(2);
            MAPMRI_NG[i+3*N] = mapmri_ng(mapc, mu, n1, n2, n3, order)(3);
            MAPMRI_NG[i+4*N] = mapmri_ng(mapc, mu, n1, n2, n3, order)(4);
            MAPMRI_NG[i+5*N] = mapmri_ng(mapc, mu, n1, n2, n3, order)(5);
            MAPMRI_NG[i+6*N] = mapmri_ng(mapc, mu, n1, n2, n3, order)(6);
            
            polynomial(0) = 3*mu(0)*mu(0)*mu(1)*mu(1)*mu(2)*mu(2);
            polynomial(1) = mu(0)*mu(0)*mu(1)*mu(1)+mu(0)*mu(0)*mu(2)*mu(2)+mu(1)*mu(1)*mu(2)*mu(2);
            polynomial(2) = -(mu(0)*mu(0)+mu(1)*mu(1)+mu(2)*mu(2));
            polynomial(3) = -3;
            psolver.compute(polynomial);
            roots = psolver.roots();
            if (real(roots(0)) > 0)
            {
                mu0 = sqrt(real(roots(0)));
            }
            else if(real(roots(1)) > 0)
            {
                mu0 = sqrt(real(roots(1)));
            }
            else
            {
                mu0 = sqrt(real(roots(2)));
            }
            Qiso = mapmri_isotropic_phi_matrix_pa(order, mu0, q.transpose());
            constraint_grid_iso = create_space(grid_size, sqrt(5)*mu0);
            Kiso = mapmri_isotropic_psi_matrix_pa(order, mu0, constraint_grid_iso);
            k = 1;
            Q0iso = MatrixXd::Zero(1, order/2+1);
            for (t = 0; t < DWI_DATA_T; t++)
            {                
                if (bvals(t) < 10)
                {
                    Q0iso = Qiso.row(t) + Q0iso;
                }
                else
                {
                    Q1b0iso.row(k) = Qiso.row(t);
                    k++;
                }
            }
            Q0iso = Q0iso/n_b0; 
            Q1b0iso.row(0) = Q0iso.row(0);
            Aiso << Kiso,
                    Q0iso;
            Aiso = (abs(Aiso.array() - 0) < 1e-13).select(0, Aiso);  
            Hiso = 0.5*Q1b0iso.transpose()*Q1b0iso; 
            ciso = -Y_norm1b0.transpose()*Q1b0iso;           
            try
            {
                success = dense_optimize(&env, N_constraints + 1, order/2+1, ciso, Hiso, Aiso, sense, rhs, lb, NULL, NULL, mapc_iso, &objval);
            }
            catch(GRBException e)
            {
                mapc_iso = Qiso.completeOrthogonalDecomposition().pseudoInverse()*Y;
                cout << "Error code = " << e.getErrorCode() << endl;
                cout << e.getMessage() << endl;
            }
            catch(...)
            {
                mapc_iso = Qiso.completeOrthogonalDecomposition().pseudoInverse()*Y;
                cout << "Exception during optimization" << endl;
            }
            
            MAPMRI_PA[i] = mapmri_pa(mapc, mapc_iso, mu, mu0, n1, n2, n3, order);   cout<<"voxel i "<<i << " PA is " << MAPMRI_PA[i]<<endl<<endl; 
            MAPMRI_PA[i+N] = sqrt(1 - 8*mu0*mu0*mu0*mu(0)*mu(1)*mu(2)/(mu(0)*mu(0)+mu0*mu0)/(mu(1)*mu(1)+mu0*mu0)/(mu(2)*mu(2)+mu0*mu0));            
            theta_PO_DTI = asin(MAPMRI_PA[i+N]);
            theta_PO = asin(MAPMRI_PA[i]);            
            // THETA_PO, THETA_PO_DTI: 0 - pi
            MAPMRI_PA[i+2*N] = theta_PO - theta_PO_DTI; 
                        
            if (VERBOSE)
            {
#pragma omp critical
                {
                // Get info about processing time
                analyzed_voxels++;
                currentTime = GetWallTime();
                analyzed_portion = (float)analyzed_voxels/N_in_mask * 100.0f;
                timeLeft = (float)(currentTime - startTime) / analyzed_portion * (100.0f - analyzed_portion);
                if ((analyzed_portion - previous_analyzed_portion) > 0.0f)
                {
                    if (timeLeft < 60) {
                        printf("Analyzed %.2f %% of %i voxels in mask in %.2f minutes, expected time remaining %.2f seconds. \r",analyzed_portion,N_in_mask,(float)(currentTime - startTime)/60.0f,timeLeft/1.0f);
                        cout.flush();
                    }
                    else if (timeLeft > 3600){
                        printf("Analyzed %.2f %% of %i voxels in mask in %.2f minutes, expected time remaining %.2f hours. \r",analyzed_portion,N_in_mask,(float)(currentTime - startTime)/60.0f,timeLeft/3600.0f);
                        cout.flush();
                    }
                    else {
                        printf("Analyzed %.2f %% of %i voxels in mask in %.2f minutes, expected time remaining %.2f minutes. \r",analyzed_portion,N_in_mask,(float)(currentTime - startTime)/60.0f,timeLeft/60.0f);
                        cout.flush();
                    }
                    previous_analyzed_portion = analyzed_portion;
                }
                }
            }
            
        }
        else
        {  // If out of mask
            
            MAPMRI_RTOP[i] = 0;
            MAPMRI_RTAP[i] = 0;
            MAPMRI_RTAP[i+N] = 0;
            MAPMRI_RTAP[i+2*N] = 0;
            MAPMRI_RTPP[i] = 0;
            MAPMRI_RTPP[i+N] = 0;
            MAPMRI_RTPP[i+2*N] = 0;
            MAPMRI_NG[i] = 0;
            MAPMRI_NG[i+N] = 0;
            MAPMRI_NG[i+2*N] = 0;
            MAPMRI_NG[i+3*N] = 0;
            MAPMRI_NG[i+4*N] = 0;
            MAPMRI_NG[i+5*N] = 0;
            MAPMRI_NG[i+6*N] = 0;
            MAPMRI_PA[i] = 0;
            MAPMRI_PA[i+N] = 0;
            MAPMRI_PA[i+2*N] = 0;
            for (t = 0; t < DWI_DATA_T; t++)
            {
                // Get data for voxel i
                MAPMRI_RESIDUAL[i + t*N] = 0; //cout << rand() % DWI_DATA_T<<" ";
            }
        }
        
    }
    
    endTime = GetWallTime();
    if (VERBOSE)
    {
        cout << endl;
        if ((endTime - startTime) < 60) {
            printf("It took %f seconds to fit the MAPMRI model.\n",(float)(endTime - startTime)/1.0f);
        }
        else if ((endTime - startTime) > 60) {
            printf("It took %f minutes to fit the MAPMRI model.\n",(float)(endTime - startTime)/60.0f);
        }
        else if ((endTime - startTime) > 3600) {
            printf("It took %f hours to fit the MAPMRI model.\n",(float)(endTime - startTime)/3600.0f);
        }
        else {
            printf("It took %f seconds to fit the MAPMRI model.\n",(float)(endTime - startTime));
        }        
    }
    
    startTime = GetWallTime();
// Create new nifti image for output
    nifti_image *outputNifti = nifti_copy_nim_info(inputDWI);
    allNiftiImages[numberOfNiftiImages] = outputNifti;
    numberOfNiftiImages++;    
// Change dimensions for MAPMRI_RTOP    
    outputNifti->nt = 1;
    outputNifti->ndim = 3;
    outputNifti->dim[0] = 3;
    outputNifti->dim[4] = 1;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D;    
// Copy information from input data
    if (!CHANGE_OUTPUT_FILENAME)
    {
        nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
    }
    else
    {
        nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    }
    
    WriteNifti(outputNifti,MAPMRI_RTOP,"_RTOP",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);    
// Change dimensions for MAPMRI_RTAP    
    outputNifti->nt = 3;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[4] = 3;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3;    
// Copy information from input data
    if (!CHANGE_OUTPUT_FILENAME)
    {
        nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
    }
    else
    {
        nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    }
    
    WriteNifti(outputNifti,MAPMRI_RTAP,"_RTAP",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
    
// Change dimensions for MAPMRI_RTPP    
    outputNifti->nt = 3;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[4] = 3;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3;    
// Copy information from input data
    if (!CHANGE_OUTPUT_FILENAME)
    {
        nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
    }
    else
    {
        nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    }
    WriteNifti(outputNifti,MAPMRI_RTPP,"_RTPP",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);  
   
// Change dimensions for MAPMRI_NG    
    outputNifti->nt = 7;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[4] = 7;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 7;    
// Copy information from input data
    if (!CHANGE_OUTPUT_FILENAME)
    {
        nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
    }
    else
    {
        nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    }
    WriteNifti(outputNifti,MAPMRI_NG,"_NG",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);   
// Change dimensions for MAPMRI_PA
    
    outputNifti->nt = 3;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[3] = 3;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3;    
// Copy information from input data
    if (!CHANGE_OUTPUT_FILENAME)
    {
        nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
    }
    else
    {
        nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    }
    WriteNifti(outputNifti,MAPMRI_PA,"_PA",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);   
    
// Change dimensions for MAPMRI_RESIDUAL
    
    outputNifti->nt = DWI_DATA_T;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[3] = DWI_DATA_D;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T;    
// Copy information from input data
    if (!CHANGE_OUTPUT_FILENAME)
    {
        nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
    }
    else
    {
        nifti_set_filenames(outputNifti, outputFilename, 0, 1);
    }
    WriteNifti(outputNifti,MAPMRI_RESIDUAL,"_RESIDUAL",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
    
    endTime = GetWallTime();
    if (VERBOSE)
    {
        printf("It took %f seconds to write the nifti files.\n\n",(float)(endTime - startTime));
    }
    
// Free all memory
    FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
    FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);    
    return EXIT_SUCCESS;
    
}
