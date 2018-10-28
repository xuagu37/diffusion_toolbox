#include <iostream>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/QR>
#include <Eigen/Eigenvalues>
#include "fstream"
#include "nifti1_io.h"
#include "HelpFunctions.cpp"
using namespace std;
using Eigen::MatrixXd;
using Eigen::SelfAdjointEigenSolver;

#define ADD_FILENAME true
#define DONT_CHECK_EXISTING_FILE false

int main(int argc, char **argv)
{
  printf("\nAuthored by X. Gu, Linkoping University. \n\n");
  // Size of the diffusion dataset X Y Z T
  size_t        DWI_DATA_H, DWI_DATA_W, DWI_DATA_D, DWI_DATA_T;
  // DWI_VOLUMES_SIZE = XYZT*sizeof(float), N = XYZ
  size_t        DWI_VOLUMES_SIZE, N, DWI_TENSORS_SIZE, DWI_EIGENVALS_SIZE, DWI_EIGENVECS_SIZE, DWI_FA_SIZE, DWI_MASK_SIZE, DWI_MD_SIZE;
  // Memory allocation related
  size_t        allocatedHostMemory = 0;
  int           numberOfMemoryPointers = 0, numberOfNiftiImages = 0;
  void          *allMemoryPointers[500];
  // Size of the voxel
  float         DWI_VOXEL_SIZE_X, DWI_VOXEL_SIZE_Y, DWI_VOXEL_SIZE_Z;
  // Input pointers
  float         *DWI_Data, *DWI_Mask;
  // Output pointers
  float         *DWI_Tensors, *DWI_Eigenvals, *DWI_Eigenvecs, *DWI_FA, *DWI_MD;
  // If data if less than MIN_SIGNAL, then replace with MIN_SIGNAL
  float   MIN_SIGNAL = 1;
  // If eigenvalue if less than MIN_DIFFUSIVITY, then replace with MIN_DIFFUSIVITY
  double  MIN_DIFFUSIVITY = 1e-6;
  // Use first n_dti volumes to do DTI fitting
  int     b_threshold = 2000;
  // Fitting method
  const char* dti_fit = "WLS";
  // Number of threads for OpemMP
  int     NUM_THREADS = 5;
  // Other
  bool			    VERBOSE = false;
  bool			    CHANGE_OUTPUT_FILENAME = false;
  // Output
  const char*		outputFilename;

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
    printf("\nThis function applies diffusion tensor fitting.\n");
    printf("Usage:\n");
    printf("DiffusionTensorFit dwi.nii brain_mask.nii bvals bvecs [options]\n");
    printf("Options:\n");
    printf(" -dti_fit                   Fitting method LS, WLS (default WLS) \n");
    printf(" -b_threshold               Use volumes below b_threshold for tensor fitting, default 2000 \n");
    printf(" -MIN_SIGNAL                Replace data value below, default 1 \n");
    printf(" -MIN_DIFFUSIVITY           Replace eigenvalues below, default 1e-6 \n");
    printf(" -verbose                   Print extra stuff (default false) \n");
    printf(" -threads                   Number of threads for OpenMP, default 5 \n");
    printf(" -output                    Set output filename \n");

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
    // // Check that file extension is .bval
    // CheckFileExtension(argv[3],extensionOK,extension);
    // fp = fopen(argv[2],"r");
    // if (fp == NULL)
    // {
    //   printf("Could not open file %s !\n",argv[2]);
    //   return EXIT_FAILURE;
    // }
    // fclose(fp);
    // // Check that file extension is .bvec
    // CheckFileExtension(argv[4],extensionOK,extension);
    // fp = fopen(argv[2],"r");
    // if (fp == NULL)
    // {
    //   printf("Could not open file %s !\n",argv[2]);
    //   return EXIT_FAILURE;
    // }
    // fclose(fp);
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
    else if (strcmp(input,"-dti_fit") == 0)
    {
      if ( (i+1) >= argc  )
      {
        printf("Unable to read value after -dti_fit !\n");
        return EXIT_FAILURE;
      }
      dti_fit = argv[i+1];
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
    else
    {
      printf("Unrecognized option! %s \n",argv[i]);
      return EXIT_FAILURE;
    }
  }

  if (VERBOSE) {
    printf("DTI fitting. \n\n");
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
    printf("It took %f seconds to read the diffusion data and brain mask files\n",(float)(endTime - startTime));
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

  // Memory size
  DWI_VOLUMES_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * DWI_DATA_T * sizeof(float);
  DWI_TENSORS_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 6 * sizeof(float);
  DWI_EIGENVALS_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3 * sizeof(float);
  DWI_EIGENVECS_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 9 * sizeof(float);
  DWI_FA_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 1 * sizeof(float);
  DWI_MASK_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 1 * sizeof(float);
  DWI_MD_SIZE = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 1 * sizeof(float);


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
        cout << "Number of b values should be the same as the number of data volumes!" << endl;
        return EXIT_FAILURE;
      }
      bvals(i) = read_value;
      i = i + 1;
    }
  }
  else
  {
    cout << "b values file does not exist." << endl;
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
        cout << "Number of b vectors should be the same as the number of data volumes!";
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
          cout << "Number of b vectors should be the same as the number of data volumes!" << endl;
          return EXIT_FAILURE;
        }
        bvecs(i%DWI_DATA_T, i/DWI_DATA_T) = read_value;
        i = i + 1;
      }
    }
  }
  else
  {
    cout << "b vectors file does not exist." << endl;
    return 0;
  }

  // Print some info
  printf("DWI data size: %zu x %zu x %zu x %zu \n",  DWI_DATA_W, DWI_DATA_H, DWI_DATA_D, DWI_DATA_T);
  printf("DWI data voxel size: %f x %f x %f mm \n", DWI_VOXEL_SIZE_X, DWI_VOXEL_SIZE_Y, DWI_VOXEL_SIZE_Z);
  if (!VERBOSE){
    printf("\n");
  }

  startTime = GetWallTime();
  AllocateMemory(DWI_Data, DWI_VOLUMES_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_DWI");
  AllocateMemory(DWI_Mask, DWI_MASK_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "INPUT_MASK");
  AllocateMemory(DWI_Tensors, DWI_TENSORS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_TENSORS");
  AllocateMemory(DWI_Eigenvals, DWI_EIGENVALS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_EIGENVALS");
  AllocateMemory(DWI_Eigenvecs, DWI_EIGENVECS_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_EIGENVECS");
  AllocateMemory(DWI_FA, DWI_FA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_FA");
  AllocateMemory(DWI_MD, DWI_FA_SIZE, allMemoryPointers, numberOfMemoryPointers, allNiftiImages, numberOfNiftiImages, allocatedHostMemory, "OUTPUT_MD");
  endTime = GetWallTime();
  if (VERBOSE)
  {
    printf("It took %f seconds to allocate memory\n",(float)(endTime - startTime));
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
    printf("It took %f seconds to convert data to floats\n",(float)(endTime - startTime));
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


  X2.col(0) = (bvecs2.col(0).array()*bvecs2.col(0).array()*1*bvals2.array()).matrix();
  X2.col(1) = (bvecs2.col(0).array()*bvecs2.col(1).array()*2*bvals2.array()).matrix();
  X2.col(2) = (bvecs2.col(1).array()*bvecs2.col(1).array()*1*bvals2.array()).matrix();
  X2.col(3) = (bvecs2.col(0).array()*bvecs2.col(2).array()*2*bvals2.array()).matrix();
  X2.col(4) = (bvecs2.col(1).array()*bvecs2.col(2).array()*2*bvals2.array()).matrix();
  X2.col(5) = (bvecs2.col(2).array()*bvecs2.col(2).array()*1*bvals2.array()).matrix();
  X2.col(6).setOnes();
  X2.col(6) = - X2.col(6);
  X2 = -X2;

  MatrixXd pinv_X2(7, n_subdata);

  pinv_X2 = X2.completeOrthogonalDecomposition().pseudoInverse();
  MatrixXd Y2(n_subdata, 1); // Data for one subdata voxel
  MatrixXd logY2(n_subdata, 1); // Data for one subdata voxel
  MatrixXd tensor_elements(6, 1);
  MatrixXd tensor(3, 3);
  double eigenval1, eigenval2, eigenval3; // eigenvalues
  // Loop over all voxels
  int analyzed_voxels = 0;
  float analyzed_portion = 0.0f;
  float previous_analyzed_portion = 0.0f;

  startTime = GetWallTime();
  int t = 0;
  float timeLeft = 0;
  SelfAdjointEigenSolver<MatrixXd> es;
  double currentTime;

  #pragma omp parallel for shared (bvals, DWI_Mask, DWI_DATA_T, DWI_Data,X2,pinv_X2, DWI_Tensors, DWI_Eigenvals, DWI_Eigenvecs, DWI_FA, DWI_MD,  VERBOSE, analyzed_voxels, analyzed_portion, N_in_mask, timeLeft, startTime, previous_analyzed_portion) private (i, t,k, eigenval1, eigenval2, eigenval3, es, currentTime) firstprivate (Y2,logY2, W2,pinv_XW2,S_hat2, tensor_elements, tensor) num_threads(NUM_THREADS)

  for (int i = 0; i < N; i++){

    if (DWI_Mask[i])
    {
      k = 0;
      for (t = 0; t < DWI_DATA_T; t++)
      {
        if (bvals(t) < b_threshold + 100)
        {
          Y2(k) = max(DWI_Data[i + t*N], MIN_SIGNAL);
          k++;
        }
      }

      // Get the tensor for voxel i
      logY2 = Y2.array().log();

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


      DWI_Tensors[i + 0*N] = tensor_elements(0);
      DWI_Tensors[i + 1*N] = tensor_elements(1);
      DWI_Tensors[i + 2*N] = tensor_elements(2);
      DWI_Tensors[i + 3*N] = tensor_elements(3);
      DWI_Tensors[i + 4*N] = tensor_elements(4);
      DWI_Tensors[i + 5*N] = tensor_elements(5);

      DWI_Eigenvals[i + 0*N] = eigenval1;
      DWI_Eigenvals[i + 1*N] = eigenval2;
      DWI_Eigenvals[i + 2*N] = eigenval3;

      DWI_Eigenvecs[i + 0*N] = es.eigenvectors()(0,0);
      DWI_Eigenvecs[i + 1*N] = es.eigenvectors()(1,0);
      DWI_Eigenvecs[i + 2*N] = es.eigenvectors()(2,0);
      DWI_Eigenvecs[i + 3*N] = es.eigenvectors()(0,1);
      DWI_Eigenvecs[i + 4*N] = es.eigenvectors()(1,1);
      DWI_Eigenvecs[i + 5*N] = es.eigenvectors()(2,1);
      DWI_Eigenvecs[i + 6*N] = es.eigenvectors()(0,2);
      DWI_Eigenvecs[i + 7*N] = es.eigenvectors()(1,2);
      DWI_Eigenvecs[i + 8*N] = es.eigenvectors()(2,2);

      DWI_FA[i] = sqrt( 0.5*(pow((DWI_Eigenvals[i + 0*N] - DWI_Eigenvals[i + 1*N]),2) + pow((DWI_Eigenvals[i + 1*N] - DWI_Eigenvals[i + 2*N]),2) + pow((DWI_Eigenvals[i + 2*N] - DWI_Eigenvals[i + 0*N]),2)) / (pow(DWI_Eigenvals[i + 0*N], 2) + pow(DWI_Eigenvals[i + 1*N], 2) + pow(DWI_Eigenvals[i + 2*N], 2)) );
      DWI_MD[i] = (DWI_Eigenvals[i + 0]+DWI_Eigenvals[i + 1]+DWI_Eigenvals[i + 2])/3.0;
      //cout << DWI_FA[i]  << endl;
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
    else{
      DWI_Tensors[i + 0*N] = 0;
      DWI_Tensors[i + 1*N] = 0;
      DWI_Tensors[i + 2*N] = 0;
      DWI_Tensors[i + 3*N] = 0;
      DWI_Tensors[i + 4*N] = 0;
      DWI_Tensors[i + 5*N] = 0;
      DWI_Eigenvals[i + 0*N] = 0;
      DWI_Eigenvals[i + 1*N] = 0;
      DWI_Eigenvals[i + 2*N] = 0;
      DWI_Eigenvecs[i + 0*N] = 0;
      DWI_Eigenvecs[i + 1*N] = 0;
      DWI_Eigenvecs[i + 2*N] = 0;
      DWI_Eigenvecs[i + 3*N] = 0;
      DWI_Eigenvecs[i + 4*N] = 0;
      DWI_Eigenvecs[i + 5*N] = 0;
      DWI_Eigenvecs[i + 6*N] = 0;
      DWI_Eigenvecs[i + 7*N] = 0;
      DWI_Eigenvecs[i + 8*N] = 0;
      DWI_FA[i] = 0;
      DWI_MD[i] = 0;
    }
  }
  endTime = GetWallTime();
  if (VERBOSE)
  {
    printf("It took %f seconds to form tensors and do eigen-decomposition\n",(float)(endTime - startTime));
  }

  startTime = GetWallTime();
  // Create new nifti image for output
  nifti_image *outputNifti = nifti_copy_nim_info(inputDWI);
  allNiftiImages[numberOfNiftiImages] = outputNifti;
  numberOfNiftiImages++;
  // Change dimensions for DWI_Tensors
  if (DWI_DATA_T > 1)
  {
    outputNifti->nt = 6;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[4] = 6;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 6;
  }
  // Copy information from input data
      if (!CHANGE_OUTPUT_FILENAME)
      {
          nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
      }
      else
      {
          nifti_set_filenames(outputNifti, outputFilename, 0, 1);
      }
  // Copy information from input data
  // nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
  WriteNifti(outputNifti,DWI_Tensors,"_tensors",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

  // Change dimensions for DWI_Eigenvals
  if (DWI_DATA_T > 1)
  {
    outputNifti->nt = 3;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[4] = 3;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 3;
  }
  // Copy information from input data
      if (!CHANGE_OUTPUT_FILENAME)
      {
          nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
      }
      else
      {
          nifti_set_filenames(outputNifti, outputFilename, 0, 1);
      }
  // Copy information from input data
  // nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
  WriteNifti(outputNifti,DWI_Eigenvals,"_eigenvals",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

  // Change dimensions for DWI_Eigenvecs
  if (DWI_DATA_T > 1)
  {
    outputNifti->nt = 9;
    outputNifti->ndim = 4;
    outputNifti->dim[0] = 4;
    outputNifti->dim[4] = 9;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D * 9;
  }
  // Copy information from input data
      if (!CHANGE_OUTPUT_FILENAME)
      {
          nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
      }
      else
      {
          nifti_set_filenames(outputNifti, outputFilename, 0, 1);
      }
  // Copy information from input data
  // nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
  WriteNifti(outputNifti,DWI_Eigenvecs,"_eigenvecs",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

  // Change dimensions for DWI_FA
  if (DWI_DATA_T > 1)
  {
    outputNifti->nt = 1;
    outputNifti->ndim = 3;
    outputNifti->dim[0] = 3;
    // outputNifti->dim[4] = 9;
    outputNifti->nvox = DWI_DATA_W * DWI_DATA_H * DWI_DATA_D;
  }
  // Copy information from input data
      if (!CHANGE_OUTPUT_FILENAME)
      {
          nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
      }
      else
      {
          nifti_set_filenames(outputNifti, outputFilename, 0, 1);
      }
  // Copy information from input data
  // nifti_set_filenames(outputNifti, inputDWI->fname, 0, 1);
  WriteNifti(outputNifti,DWI_FA,"_FA",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);
  WriteNifti(outputNifti,DWI_MD,"_MD",ADD_FILENAME,DONT_CHECK_EXISTING_FILE);

  endTime = GetWallTime();
  if (VERBOSE)
  {
    printf("It took %f seconds to write the nifti files\n\n",(float)(endTime - startTime));
  }

  // Free all memory
  FreeAllMemory(allMemoryPointers,numberOfMemoryPointers);
  FreeAllNiftiImages(allNiftiImages,numberOfNiftiImages);

  return EXIT_SUCCESS;

}
