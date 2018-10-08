#include <string.h>
#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <Eigen/Dense>
#include "gurobi_c++.h"
#include <complex>
#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Tensor;
using Eigen::MatrixXcd;
using namespace std;
using Eigen::Map;

void CheckFileExtension(const char* filename, bool& extensionOK, std::string& extension)
{
  const char* p = filename;
  int dotPosition = 0;
  while ( (p != NULL) && ((*p) != '.') )
  {
    p++;
  }
  extension = "";
  extension.append(p);

  // Compare extension to OK values
  std::string extension1 = ".nii";
  std::string extension2 = ".nii.gz";
  std::string extension3 = ".bvals";
  std::string extension4 = ".bvecs";

  if ( (extension.compare(extension1) != 0) && (extension.compare(extension2) != 0) && (extension.compare(extension3) != 0) && (extension.compare(extension4) != 0))
  {
    extensionOK = false;
  }
  else
  {
    extensionOK = true;
  }
}

void FreeAllMemory(void **pointers, int N)
{
  for (int i = 0; i < N; i++)
  {
    if (pointers[i] != NULL)
    {
      free(pointers[i]);
    }
  }
}

void FreeAllNiftiImages(nifti_image **niftiImages, int N)
{
  for (int i = 0; i < N; i++)
  {
    if (niftiImages[i] != NULL)
    {
      nifti_image_free(niftiImages[i]);
    }
  }
}

void AllocateMemory(float *& pointer, size_t size, void** pointers, int& Npointers, nifti_image** niftiImages, int Nimages, size_t& allocatedMemory, const char* variable)
{
  pointer = (float*)malloc(size);
  if (pointer != NULL)
  {
    pointers[Npointers] = (void*)pointer;
    Npointers++;
    allocatedMemory += size;
  }
  else
  {
    perror ("The following error occurred");
    printf("Could not allocate host memory for variable %s ! \n",variable);
    FreeAllMemory(pointers, Npointers);
    FreeAllNiftiImages(niftiImages, Nimages);
    exit(EXIT_FAILURE);
  }
}

float mymin(float* data, int N)
{
  float min = 100000.0f;
  for (int i = 0; i < N; i++)
  {
    if (data[i] < min)
    min = data[i];
  }

  return min;
}

float mymax(float* data, int N)
{
  float max = -100000.0f;
  for (int i = 0; i < N; i++)
  {
    if (data[i] > max)
    max = data[i];
  }

  return max;
}

bool WriteNifti(nifti_image* inputNifti, float* data, const char* filename, bool addFilename, bool checkFilename)
{
  if (data == NULL)
  {
    printf("The provided data pointer for file %s is NULL, aborting writing nifti file! \n",filename);
    return false;
  }
  if (inputNifti == NULL)
  {
    printf("The provided nifti pointer for file %s is NULL, aborting writing nifti file! \n",filename);
    return false;
  }

  char* filenameWithExtension;
  // Add the provided filename extension to the original filename, before the dot
  if (addFilename)
  {
    // Find the dot in the original filename
    const char* p = inputNifti->fname;
    int dotPosition = 0;
    while ( (p != NULL) && ((*p) != '.') )
    {
      p++;
      dotPosition++;
    }

    // Allocate temporary array
    filenameWithExtension = (char*)malloc(strlen(inputNifti->fname) + strlen(filename) + 1);
    if (filenameWithExtension == NULL)
    {
      printf("Could not allocate temporary host memory! \n");
      return false;
    }

    // Copy filename to the dot
    strncpy(filenameWithExtension,inputNifti->fname,dotPosition);
    filenameWithExtension[dotPosition] = '\0';
    // Add the extension
    strcat(filenameWithExtension,filename);
    // Add the rest of the original filename
    strcat(filenameWithExtension,inputNifti->fname+dotPosition);
  }

  // Copy information from input data
  nifti_image *outputNifti = nifti_copy_nim_info(inputNifti);
  // Set data pointer
  outputNifti->data = (void*)data;
  // Set data type to float
  outputNifti->datatype = DT_FLOAT;
  outputNifti->nbyper = 4;

  // Change cal_min and cal_max, to get the scaling right in AFNI and FSL
  int N = inputNifti->nx * inputNifti->ny * inputNifti->nz * inputNifti->nt;
  outputNifti->cal_min = mymin(data,N);
  outputNifti->cal_max = mymax(data,N);

  // Change filename and write
  bool written = false;
  if (addFilename)
  {
    if ( nifti_set_filenames(outputNifti, filenameWithExtension, checkFilename, 1) == 0)
    {
      nifti_image_write(outputNifti);
      written = true;
    }
  }
  else if (!addFilename)
  {
    if ( nifti_set_filenames(outputNifti, filename, checkFilename, 1) == 0)
    {
      nifti_image_write(outputNifti);
      written = true;
    }
  }

  outputNifti->data = NULL;
  nifti_image_free(outputNifti);

  if (addFilename)
  {
    free(filenameWithExtension);
  }

  if (written)
  {
    return true;
  }
  else
  {
    return false;
  }
}

double GetWallTime()
{
  struct timeval time;
  if (gettimeofday(&time,NULL))
  {
    //  Handle error
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

double factorial(double n)
{
  if(n > 1)
  return n * factorial(n - 1);
  else
  return 1;
}

double factorial2(double n)
{
  if(n > 1)
  return n * factorial2(n - 2);
  else
  return 1;
}

MatrixXd polyval(MatrixXd p, MatrixXd x){
  int n = p.size();  // Number of polynomial coefficients
  int m = x.size();  // Number of x to be evaluated
  MatrixXd y = MatrixXd::Zero(1, m);
  for (int i = 0; i < m; i++){ // for one of x
    for (int j = 0; j < n; j++){ // sum of p(j)*x(i)^(n-1)
      y(i) = y(i) + p(j) * pow(x(i), n-j-1);
    }
  }
  return y;
}

MatrixXd hermite_rec(int n){
  MatrixXd h(1, n+1);
  MatrixXd h1 = MatrixXd::Zero(1, n+1);
  MatrixXd h2 = MatrixXd::Zero(1, n+1);
  if (n == 0){
    h(0) = 1;
    return h;
  }
  else if(n == 1){
    h(0) = 2;
    h(1) = 0;
    return h;
  }
  else {
    h1 << 2*hermite_rec(n-1), 0;
    h2 << 0,0,2*(n-1)*hermite_rec(n-2);
    h = h1 - h2;
    return h;
  }
}

MatrixXd hermite(int n){
  // check n
  if(n < 0){
    cerr << "ERROR: The order of Hermite polynomial must be greater than or equal to 0.\n";
  }
  // again check n is an integer
  if( n - round(n) != 0 ){
    cerr << "ERROR: The order of Hermite polynomial must be an integer.\n";
  }
  // call the hermite recursive function.
  MatrixXd h(1, n+1);
  h = hermite_rec(n);
  return h;
}

MatrixXd cart2sphere(MatrixXd X){
    MatrixXd x = X.col(0);
    MatrixXd y = X.col(1);
    MatrixXd z = X.col(2);
    MatrixXd r = (x.array()*x.array()+y.array()*y.array()+z.array()*z.array()).array().sqrt();
    //(z.array()/r.array()).acos();
    int n = x.rows();
    MatrixXd theta = MatrixXd::Zero(n,1);
    MatrixXd phi(n,1);
    for (int i=0;i<n;i++)
    {   if (r(i)>0)
        {
            theta(i) = acos(z(i)/r(i));
        }
        phi(i) = atan2(y(i),x(i));
    }
    MatrixXd Y(n,3);
    Y.col(0) = r.col(0);
    Y.col(1) = theta.col(0);
    Y.col(2) = phi.col(0);
    return Y;
}

// Generalized Laguerre polynomial
double L0(double a, double x){
    return 1;
}
 
double L1(double a, double x){
    return 1+a-x;
}
double laguerre_poly(int n, double a, double x){
    if(n==0){
        return L0(a,x);
    }
    else if(n==1){
        return L1(a,x);
    }
    else{
        return ((2*n-1+a-x)*laguerre_poly(n-1,a,x)-(n-1+a)*laguerre_poly(n-2,a,x)) / n;
    }
}

std::complex<double> spherical_harmonics(int m, int n, double theta, double phi){
    
    double val = assoc_legendre(n,m,cos(phi));
    val = val*sqrt((2*n+1)/4.0/M_PI);   
    val = val*exp(0.5*(lgamma(n-m+1)-lgamma(n+m+1) ));   
    std::complex<double> sh;
    sh = val * exp(1i*(double)m*theta);
    return sh;
    
}

double real_sph_harm(int m, int n, double theta, double phi){
    std::complex<double> sh = spherical_harmonics(abs(m),n,phi,theta);
    double real_sh;
    if (m>0)
    {
        real_sh = sh.imag();
    }
    else
    {
        real_sh = sh.real();
    }
    if (m!=0)
    {
        real_sh = real_sh*sqrt(2);
    }
    return real_sh;
    
}

MatrixXd B123_matrix(MatrixXd n1, MatrixXd n2, MatrixXd n3){
    MatrixXd K123 = n1.unaryExpr([](const int x){return double((x+1)%2);}).array() * n2.unaryExpr([](const int x){return double((x+1)%2);}).array() * n3.unaryExpr([](const int x){return double((x+1)%2);}).array();
    MatrixXd B123 = K123.array() * (n1.unaryExpr(ptr_fun(factorial)).array()*n2.unaryExpr(ptr_fun(factorial)).array()*n3.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n1.unaryExpr(ptr_fun(factorial2)).array() / n2.unaryExpr(ptr_fun(factorial2)).array() / n3.unaryExpr(ptr_fun(factorial2)).array();
    return B123;
    
}

MatrixXd mapmri_isotropic_radial_signal_basis(int j, int l, double mu, MatrixXd qval){    
    MatrixXd pi2_mu2_q2 = 2*M_PI*M_PI*mu*mu*qval.array()*qval.array();
    int n = qval.rows();
    MatrixXd const1(n,1);
    const1 = pow(-1,l/2)*sqrt(4*M_PI)*pi2_mu2_q2.array().pow(l/2)*(-pi2_mu2_q2).array().exp();
    for (int i=0;i<n;i++){
        const1(i) = const1(i)*laguerre_poly(j-1, l+0.5,2*pi2_mu2_q2(i));
    }
    return const1;
    
}


MatrixXd mapmri_phi_matrix(int radial_order, MatrixXd mu, MatrixXd R, MatrixXd q, MatrixXd n1, MatrixXd n2, MatrixXd n3){
    int ncoeff = n1.rows(); 
    int DWI_DATA_T = q.cols();
    MatrixXd M(DWI_DATA_T, ncoeff);
    MatrixXcd phi1(DWI_DATA_T, ncoeff);
    MatrixXcd phi2(DWI_DATA_T, ncoeff);
    MatrixXcd phi3(DWI_DATA_T, ncoeff);
    MatrixXd harg(3, DWI_DATA_T);
    MatrixXd q_rotate(3, DWI_DATA_T);
    complex<double> complex_i(0.0,1.0);
    q_rotate = R.transpose()*q;
    harg = 2*M_PI*q_rotate.array()*mu.replicate(1, DWI_DATA_T).array();
    for (int k = 0; k < ncoeff; k++){
        phi1.col(k) = pow(complex_i, -n1(k))/sqrt(pow(2, n1(k)) * factorial(n1(k))) * (-harg.row(0).array().pow(2)/2).exp() * polyval(hermite(n1(k)),harg.row(0)).array();
        phi2.col(k) = pow(complex_i, -n2(k))/sqrt(pow(2, n2(k)) * factorial(n2(k))) * (-harg.row(1).array().pow(2)/2).exp() * polyval(hermite(n2(k)),harg.row(1)).array();
        phi3.col(k) = pow(complex_i, -n3(k))/sqrt(pow(2, n3(k)) * factorial(n3(k))) * (-harg.row(2).array().pow(2)/2).exp() * polyval(hermite(n3(k)),harg.row(2)).array();
    }
    M = (phi1.array()*phi2.array()*phi3.array()).real();
    return M;    
}


MatrixXd mapmri_psi_matrix(int grid_size, MatrixXd constraint_grid, MatrixXd mu, MatrixXd n1, MatrixXd n2, MatrixXd n3){
    int ncoeff = n1.rows();
    int N_constraints = grid_size*grid_size*(grid_size+1)/2;
    VectorXd pre_cal1(N_constraints);
    VectorXd pre_cal2(N_constraints);
    VectorXd pre_cal3(N_constraints);
    pre_cal1 = (-constraint_grid.col(0).array()*constraint_grid.col(0).array()/mu(0)/mu(0)/2).exp();
    pre_cal2 = (-constraint_grid.col(1).array()*constraint_grid.col(1).array()/mu(1)/mu(1)/2).exp();
    pre_cal3 = (-constraint_grid.col(2).array()*constraint_grid.col(2).array()/mu(2)/mu(2)/2).exp();
    MatrixXd psi1(N_constraints, ncoeff);
    MatrixXd psi2(N_constraints, ncoeff);
    MatrixXd psi3(N_constraints, ncoeff);
    MatrixXd M(N_constraints, ncoeff);
    
    for (int k = 0; k < ncoeff; k++){
        psi1.col(k) = 1/sqrt(pow(2, n1(k)+1)*M_PI*factorial(n1(k))) / mu(0) * pre_cal1.array() * polyval(hermite(n1(k)),constraint_grid.col(0)/mu(0)).transpose().array();
        psi2.col(k) = 1/sqrt(pow(2, n2(k)+1)*M_PI*factorial(n2(k))) / mu(1) * pre_cal2.array() * polyval(hermite(n2(k)),constraint_grid.col(1)/mu(1)).transpose().array();
        psi3.col(k) = 1/sqrt(pow(2, n3(k)+1)*M_PI*factorial(n3(k))) / mu(2) * pre_cal3.array() * polyval(hermite(n3(k)),constraint_grid.col(2)/mu(2)).transpose().array();
    }
    M = psi1.array()*psi2.array()*psi3.array();
    return M;
}


MatrixXd mapmri_isotropic_phi_matrix_pa(int radial_order, double mu, MatrixXd q){
    
    MatrixXd Y = cart2sphere(q);
    MatrixXd qval = Y.col(0);
    MatrixXd theta = Y.col(1);
    MatrixXd phi = Y.col(2);
    int j_max = radial_order/2+1;
    MatrixXd M = MatrixXd::Zero(q.rows(), j_max);
    MatrixXd const1(q.rows(),1);
    MatrixXd real_sh(q.rows(),1);
    for (int j=0;j<q.rows();j++)
    {
        real_sh(j) = real_sph_harm(0,0,theta(j),phi(j));        
    }
    int counter = 0;
    for (int j=1;j<j_max+1;j++)
    {
        const1 = mapmri_isotropic_radial_signal_basis(j,0,mu,qval);
        M.col(counter) = const1.array()*real_sh.array();
        counter++;
    }
    return M;
    
}


MatrixXd mapmri_isotropic_radial_pdf_basis(int j, int l, double mu, MatrixXd r){    
    MatrixXd r2u2 = r.array()*r.array()/(2*mu*mu);
    int n = r.rows();
    MatrixXd const1(n,1);
    const1 = pow(-1,j-1)/(sqrt(2)*M_PI*mu*mu*mu)*r2u2.array().pow(l/2)*(-r2u2).array().exp();
    for (int i=0;i<n;i++){
        const1(i) = const1(i)*laguerre_poly(j-1, l+0.5,2*r2u2(i));
    }
    return const1;
    
}


MatrixXd mapmri_isotropic_psi_matrix_pa(int radial_order, double mu, MatrixXd rgrad){
    MatrixXd Y = cart2sphere(rgrad);
    MatrixXd r = Y.col(0);
    MatrixXd theta = Y.col(1);
    MatrixXd phi = Y.col(2);
    int j_max = radial_order/2+1;
    MatrixXd K = MatrixXd::Zero(rgrad.rows(), j_max);
    MatrixXd const1(rgrad.rows(),1);
    MatrixXd real_sh(rgrad.rows(),1);    
    for (int j=0;j<rgrad.rows();j++)
    {
        real_sh(j) = real_sph_harm(0,0,theta(j),phi(j));        
    }
    int counter = 0;
    for (int j=1;j<j_max+1;j++)
    {
        const1 = mapmri_isotropic_radial_pdf_basis(j,0,mu,r);
        K.col(counter) = const1.array()*real_sh.array();        

        counter++;
    }
    return K;    
    
}

MatrixXd create_space(int grid_size, double radius_max){
    VectorXd grid_x = VectorXd::LinSpaced(grid_size, -radius_max, radius_max);//cout << grid_x << endl;
    VectorXd grid_y = VectorXd::LinSpaced(grid_size, -radius_max, radius_max);//cout << grid_y << endl;
    VectorXd grid_z = VectorXd::LinSpaced((grid_size+1)/2, 0, radius_max);//cout << grid_z << endl;
    int N_constraints = grid_size*grid_size*(grid_size+1)/2;
    
    VectorXd rx(N_constraints);
    VectorXd ry(N_constraints);
    VectorXd rz(N_constraints);
    MatrixXd rspace(N_constraints,3);
    MatrixXd ry_temp(grid_size, grid_size);
    VectorXd ry_temp2(grid_size*grid_size);
    MatrixXd rz_temp(grid_size*grid_size, (grid_size+1)/2);
    rx = grid_y.replicate(grid_size*(grid_size+1)/2, 1); //cout << rx(16) << endl;
    ry_temp = grid_x.transpose().replicate(grid_size, 1);//cout << ry_temp.row(0) << endl;
    ry_temp2 = Map<VectorXd>(ry_temp.data(), grid_size*grid_size);//cout << ry_temp2(1087) << endl;
    ry = ry_temp2.replicate((grid_size+1)/2, 1); //cout << ry << endl;
    rz_temp = grid_z.transpose().replicate(grid_size*grid_size, 1);
    rz = Map<VectorXd> (rz_temp.data(), N_constraints);//cout << rz(1225) << endl;
    // Only keep points inside the sphere
    for (int p = 0; p < N_constraints; p++)
    {
        if (sqrt(rx(p)*rx(p) + ry(p)*ry(p) + rz(p)*rz(p)) >  radius_max)
        {
            rx(p) = 0;
            ry(p) = 0;
            rz(p) = 0;
        }
    }
    rspace.col(0) = rx;
    rspace.col(1) = ry;
    rspace.col(2) = rz;
    return rspace;
}

int K_m_plus_n(int m, int n){
 return (remainder(m,2)-1)*(remainder(n,2)-1);   
}

Tensor<double, 4> Tmn_xi_storage(int radial_order){
    Tensor<double, 4> storage_matrix(radial_order+1,radial_order+1,radial_order+1,radial_order+1);
    storage_matrix.setZero();
    double const1;
    int m, n, r, s;
    for (m=0; m<radial_order+1;m++)
    {
        for (n=0; n<radial_order+1;n++)
        {
            if (K_m_plus_n(m,n)==1)
            {
                const1 = sqrt(factorial(m)*factorial(n));
                for (r=0; r<m+1;r+=2)
                {
                    for (s=0; s<n+1;s+=2)
                    {
                        storage_matrix(m,n,r,s) = const1*pow(-1,(r+s)/2)*factorial2(m+n-r-s-1)/(factorial(m-r)*factorial(n-s)*factorial2(r)*factorial2(s));
                        //cout<<storage_matrix(m,n,r,s)<<" ";
                    }
                }
            }
        }
    }
    return storage_matrix;
}

double Tmn_xi_stor(int m, int n, double zeta, Tensor<double, 4> storage_matrix){
    double sum=0;
    for (int r=0; r<m+1;r+=2)
    {
        for (int s=0; s<n+1;s+=2)
        {
            sum = sum + pow(zeta,n-s)*pow((1+zeta*zeta)/2,-(m+n-r-s+1)/2.0)*storage_matrix(m,n,r,s);
        }
    }
    return sum;
}

MatrixXd T_zeta_matrix(MatrixXd mu, double mu0, MatrixXd n1, MatrixXd n2, MatrixXd n3, Tensor<double, 4> stor_mat){
    MatrixXd zeta = mu/mu0;
    MatrixXd T = MatrixXd::Zero(n1.rows(),n1.rows());
    //cout<<"im here "<<Tmn_xi_stor(0,0,zeta(0),stor_mat)<<endl;
    for (int m=0; m<n1.rows();m++)
    {
        for (int n=0; n<n1.rows();n++)
        {
            if ((K_m_plus_n(n1(m),n1(n))+K_m_plus_n(n2(m),n2(n))+K_m_plus_n(n3(m),n3(n)))==3)
            {
                T(m,n) = Tmn_xi_stor(n1(m),n1(n),zeta(0),stor_mat)*Tmn_xi_stor(n2(m),n2(n),zeta(1),stor_mat)*Tmn_xi_stor(n3(m),n3(n),zeta(2),stor_mat);
                
            }                        
        }               
    }
    return T;
}

double mapmri_rtop(MatrixXd mapmri_coef, MatrixXd mu, MatrixXd n1, MatrixXd n2, MatrixXd n3){
    double rtopsc = 1/sqrt(8*pow(M_PI, 3))/mu(0)/mu(1)/mu(2);
    int ncoeff = n1.rows();
    MatrixXd mo123(ncoeff, 1);
    mo123 = (((n1+n2+n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    MatrixXd B123 = B123_matrix(n1, n2, n3);
    double rtop;
    rtop = rtopsc*(mo123.array() * mapmri_coef.array() * B123.array()).sum();
    return rtop;
}

MatrixXd mapmri_rtap(MatrixXd mapmri_coef, MatrixXd mu, MatrixXd n1, MatrixXd n2, MatrixXd n3){
    int ncoeff = n1.rows();    
    double rtap1sc = 1.0/2/M_PI/mu(1)/mu(2);
    double rtap2sc = 1.0/2/M_PI/mu(0)/mu(2);
    double rtap3sc = 1.0/2/M_PI/mu(0)/mu(1);
    MatrixXd mo12(ncoeff, 1);
    MatrixXd mo23(ncoeff, 1);
    MatrixXd mo31(ncoeff, 1);
    mo12 = (((n1+n2)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo23 = (((n2+n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo31 = (((n1+n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    MatrixXd B123 = B123_matrix(n1, n2, n3);
    MatrixXd rtap(3,1);
    rtap(0) = rtap1sc * (mo23.array() * mapmri_coef.array() * B123.array()).sum();
    rtap(1) = rtap2sc * (mo31.array() * mapmri_coef.array() * B123.array()).sum();
    rtap(2) = rtap3sc * (mo12.array() * mapmri_coef.array() * B123.array()).sum();
    return rtap;
}

MatrixXd mapmri_rtpp(MatrixXd mapmri_coef, MatrixXd mu, MatrixXd n1, MatrixXd n2, MatrixXd n3){
    int ncoeff = n1.rows();    
    
    double  rtpp1sc = 1.0/sqrt(2*M_PI)/mu(0);
    double  rtpp2sc = 1.0/sqrt(2*M_PI)/mu(1);
    double  rtpp3sc = 1.0/sqrt(2*M_PI)/mu(2);
    MatrixXd mo1(ncoeff, 1);
    MatrixXd mo2(ncoeff, 1);
    MatrixXd mo3(ncoeff, 1);
    mo1 = (((n1)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo2 = (((n2)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo3 = (((n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    MatrixXd B123 = B123_matrix(n1, n2, n3);
    MatrixXd rtpp(3,1);
    rtpp(0) = rtpp1sc * (mo1.array() * mapmri_coef.array() * B123.array()).sum();
    rtpp(1) = rtpp2sc * (mo2.array() * mapmri_coef.array() * B123.array()).sum();
    rtpp(2) = rtpp3sc * (mo3.array() * mapmri_coef.array() * B123.array()).sum();
    return rtpp;
}

MatrixXd mapmri_ng(MatrixXd mapc, MatrixXd mu, MatrixXd n1, MatrixXd n2, MatrixXd n3, int order){
    int ncoeff = n1.rows();
    MatrixXd an1(order + 1, 1);
    MatrixXd an2(order + 1, 1);
    MatrixXd an3(order + 1, 1);
    MatrixXd an12((order + 1)*(order + 1), 1);
    MatrixXd an23((order + 1)*(order + 1), 1);
    MatrixXd an31((order + 1)*(order + 1), 1);
    MatrixXd B12(ncoeff, 1);
    MatrixXd B23(ncoeff, 1);
    MatrixXd B31(ncoeff, 1);
    MatrixXd B1(ncoeff, 1);
    MatrixXd B2(ncoeff, 1);
    MatrixXd B3(ncoeff, 1);
    MatrixXd mo1(ncoeff, 1);
    MatrixXd mo2(ncoeff, 1);
    MatrixXd mo3(ncoeff, 1);
    MatrixXd mo12(ncoeff, 1);
    MatrixXd mo23(ncoeff, 1);
    MatrixXd mo31(ncoeff, 1);
    MatrixXd K23(ncoeff, 1);
    MatrixXd K13(ncoeff, 1);
    MatrixXd K12(ncoeff, 1);
    MatrixXd K1(ncoeff, 1);
    MatrixXd K2(ncoeff, 1);
    MatrixXd K3(ncoeff, 1);
    mo1 = (((n1)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo2 = (((n2)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo3 = (((n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo12 = (((n1+n2)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo23 = (((n2+n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    mo31 = (((n1+n3)*0.5).unaryExpr([](const int x){return double((x+1)%2);})).array()*2-1;
    K23 = n2.unaryExpr([](const int x){return double((x+1)%2);}).array() * n3.unaryExpr([](const int x){return double((x+1)%2);}).array();
    K13 = n1.unaryExpr([](const int x){return double((x+1)%2);}).array() * n3.unaryExpr([](const int x){return double((x+1)%2);}).array();
    K12 = n1.unaryExpr([](const int x){return double((x+1)%2);}).array() * n2.unaryExpr([](const int x){return double((x+1)%2);}).array();
    K1 = n1.unaryExpr([](const int x){return double((x+1)%2);}).array();
    K2 = n2.unaryExpr([](const int x){return double((x+1)%2);}).array();
    K3 = n3.unaryExpr([](const int x){return double((x+1)%2);}).array();
    B12 = K1.array() * K2.array() * (n1.unaryExpr(ptr_fun(factorial)).array()*n2.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n1.unaryExpr(ptr_fun(factorial2)).array() / n2.unaryExpr(ptr_fun(factorial2)).array();
    B23 = K2.array() * K3.array() * (n2.unaryExpr(ptr_fun(factorial)).array()*n3.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n2.unaryExpr(ptr_fun(factorial2)).array() / n3.unaryExpr(ptr_fun(factorial2)).array();
    B31 = K3.array() * K1.array() * (n3.unaryExpr(ptr_fun(factorial)).array()*n1.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n3.unaryExpr(ptr_fun(factorial2)).array() / n1.unaryExpr(ptr_fun(factorial2)).array();
    B1 = K23.array() * (n1.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n1.unaryExpr(ptr_fun(factorial2)).array();
    B2 = K13.array() * (n2.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n2.unaryExpr(ptr_fun(factorial2)).array();
    B3 = K12.array() * (n3.unaryExpr(ptr_fun(factorial)).array()).sqrt() / n3.unaryExpr(ptr_fun(factorial2)).array();
    for (int k = 0; k < order + 1; k++)
    {
        an1(k) = (mapc.array() * mo23.array() * B23.array() * ((n1.array() == k).select(MatrixXd::Ones(ncoeff,1), 0)).array()).sum();
        an2(k) = (mapc.array() * mo31.array() * B31.array() * ((n2.array() == k).select(MatrixXd::Ones(ncoeff,1), 0)).array()).sum();
        an3(k) = (mapc.array() * mo12.array() * B12.array() * ((n3.array() == k).select(MatrixXd::Ones(ncoeff,1), 0)).array()).sum();
    }    
    for (int k1 = 0; k1 < order + 1; k1++)
    {
        for (int k2 = 0; k2 < order + 1; k2++)
        {           
            an12(k1*(order+1) + k2) = (mapc.array() * mo3.array() * B3.array() * (((n1.array() == k1) && (n2.array() == k2)).select(MatrixXd::Ones(ncoeff,1), 0)).array()).sum();
            an23(k1*(order+1) + k2) = (mapc.array() * mo1.array() * B1.array() * (((n2.array() == k1) && (n3.array() == k2)).select(MatrixXd::Ones(ncoeff,1), 0)).array()).sum();
            an31(k1*(order+1) + k2) = (mapc.array() * mo2.array() * B2.array() * (((n3.array() == k1) && (n1.array() == k2)).select(MatrixXd::Ones(ncoeff,1), 0)).array()).sum();
        }
    }
    MatrixXd ng(7,1);
    ng(0) = sqrt(1 - mapc(0)*mapc(0)/((mapc.transpose()*mapc)(0,0)));
    ng(1) = sqrt(1 - an1(0)*an1(0)/((an1.transpose()*an1)(0,0)));
    ng(2) = sqrt(1 - an2(0)*an2(0)/((an2.transpose()*an2)(0,0)));
    ng(3) = sqrt(1 - an3(0)*an3(0)/((an3.transpose()*an3)(0,0)));
    ng(4) = sqrt(1 - an12(0)*an12(0)/((an12.transpose()*an12)(0,0)));
    ng(5) = sqrt(1 - an23(0)*an23(0)/((an23.transpose()*an23)(0,0)));
    ng(6) = sqrt(1 - an31(0)*an31(0)/((an31.transpose()*an31)(0,0)));
    return ng;
}


double mapmri_pa(MatrixXd mapmri_coef, MatrixXd mapmri_coef_iso, MatrixXd mu, double mu0, MatrixXd n1, MatrixXd n2, MatrixXd n3, int radial_order){
    MatrixXd N = n1+n2+n3;
    MatrixXd o(n1.rows(),1);
    MatrixXd B123 = B123_matrix(n1, n2, n3);
    for (int n=0; n<n1.rows();n++)
    {
        o(n) = B123(n)*mapmri_coef_iso(N(n)/2);
    }
     Tensor<double, 4> pa_storage_matrix = Tmn_xi_storage(radial_order);
     MatrixXd T_zeta = T_zeta_matrix(mu, mu0,n1,n2,n3,pa_storage_matrix);
     double pa;
     pa = (o.transpose()*T_zeta*mapmri_coef).array().square()(0,0)/(mapmri_coef.array().square().sum()*o.array().square().sum())*sqrt(mu.prod()/mu0/mu0/mu0);
     pa = sqrt(1-pa);
    return pa;
}

static bool
dense_optimize(GRBEnv* env,
               int     rows,
               int     cols,
              //  double* c,     /* linear portion of objective function */
              MatrixXd& c,
              //  double* Q,     /* quadratic portion of objective function */
              MatrixXd& Q,
              //  double* A,     /* constraint matrix */
              MatrixXd& A,
               char*   sense, /* constraint senses */
               double* rhs,   /* RHS vector */
               double* lb,    /* variable lower bounds */
               double* ub,    /* variable upper bounds */
               char*   vtype, /* variable types (continuous, binary, etc.) */
              //  double* solution,
              MatrixXd& solution,
               double* objvalP)
{
  GRBModel model = GRBModel(*env);
  int i, j;
  bool success = false;
  
  //model.set(GRB_IntParam_Threads, 1);
  /* Add variables to the model */  
  GRBVar* vars = model.addVars(lb, ub, NULL, vtype, NULL, cols);

  /* Populate A matrix */

  for (i = 0; i < rows; i++) {
    GRBLinExpr lhs = 0;
    for (j = 0; j < cols; j++)
      // if (A[i*cols+j] != 0)
      if (A(i,j) != 0)
        lhs += A(i,j)*vars[j];
    model.addConstr(lhs, sense[i], rhs[i]);
  }

  GRBQuadExpr obj = 0;

  for (j = 0; j < cols; j++)
    // obj += c[j]*vars[j];
    obj += c(j)*vars[j];
  for (i = 0; i < cols; i++)
    for (j = 0; j < cols; j++)
      // if (Q[i*cols+j] != 0)
      if (Q(i, j) != 0)
        obj += Q(i, j)*vars[i]*vars[j];

  model.setObjective(obj);
  model.getEnv().set(GRB_IntParam_OutputFlag, 0);

  model.optimize();

  //model.write("dense.lp");

  if (model.get(GRB_IntAttr_Status) == GRB_OPTIMAL) {
    *objvalP = model.get(GRB_DoubleAttr_ObjVal);
    for (i = 0; i < cols; i++)
      solution(i) = vars[i].get(GRB_DoubleAttr_X);
    success = true;
  }
  delete[] vars;
  return success;
}
