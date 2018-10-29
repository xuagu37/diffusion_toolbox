function  dti_parameters = dti_fit(varargin)
% This function applies diffusion tensor fitting to a diffusion dataset.

% Input:
% -----------------------------------------------------------------------------------------------
% X, Y, Z is the size of the diffusion volume, T is the number of volumes.
% data: diffusion data, XxYxZxT
% b_vals: b-values, Tx1.
% b_vecs: b-vectors, Tx3.
% brain_mask: use a brain mask to avoid calculation for background voxels.
% fit_method: diffusion tensor fitting method, LS/WLS/... The default is LS.
% min_signal: Replace data values below min_signal with min_signal, min_signal is positive value, the default is 1e-4.

% Output:
% -----------------------------------------------------------------------------------------------
% tensors: 3x3xXxYxZ
% [Dxx, Dxy, Dxz;
%  Dxy, Dyy, Dyz;
%  Dxz, Dyz, Dzz].
% eigenvals: 3xXxYxZ. In descending order. Negative eigenvalues are replaced by min_diffusivity.
% eigenvecs: 3x3xXxYxZ. Eigenvectors are columnar e.g. eigenvecs(:, p, i, j, k) is associated with eigenvals(p, i, j, k).
% FA: Fractional anisotropy, XxYxZ.
% MD: XxYxZ, mean diffusivity.
% color_FA: 3xXxYxZ, FA in RGB form.

% Notes
% -----------------------------------------------------------------------------------------------
% Y = X*W + E
% B = design matrix
% (X'*X)^(-1)*X'*Y is the LS solution.
% (X'*X)^(-1)*X' is the Moore-Penrose pseudoinverse of X.
% -----------------------------------------------------------------------------------------------
% FA is equal to the variance of the 3 eigenvalues, normalised to take values between 0 and 1.
% However, if one of the eigenvalues is negative, then FA can be higher than one.
% A negative eigenvalue is physically impossible.
% This can happen in practice, e.g. with poor SNR or head motion.
% In FSL dtifit, the tensor is calculated with no positivity constraints on the eigenvalues.
% So situations where FA > 1 may happen in practice.
% -----------------------------------------------------------------------------------------------
% MD = (lambda_1+\lambda_2+\lambda_3)/3
% -----------------------------------------------------------------------------------------------
% color_FA = ev1xFA. ev1 is the principal eigenvector.



tic

p = inputParser;
addParameter(p, 'data', []);
addParameter(p, 'brain_mask', []);
addParameter(p, 'bvals', []);
addParameter(p, 'bvecs', []);
addParameter(p, 'fit_method', 'LS');
addParameter(p, 'min_diffusivity', 1e-6);

p.parse(varargin{:});
data = p.Results.data;
brain_mask = p.Results.brain_mask;
if isempty(brain_mask)
    brain_mask = ones(size(data));
end
bvals = p.Results.bvals;
bvecs = p.Results.bvecs;
% Check if bvals and bvecs are in the right form
if size(bvals, 2) ~= 1
    bvals = bvals';
end
if size(bvecs, 2) ~= 3
    bvecs = bvecs';
end
fit_method = p.Results.fit_method;
min_diffusivity = p.Results.min_diffusivity;

[sx, sy, sz, sv] = size(data);

% Reshape diffusion data XxYxZxT --> XYZxT --> TxXYZ.
data_flat = reshape(data, [], sv)';


% Construct design matrix for diffusion tensor fitting.
X = dti_X(bvals, bvecs);

switch fit_method
    case 'LS'
        % Least square fit
%         tensors = lscov(X,log(data_flat));
        tensor_elements = (pinv(X)*log(data_flat));  
    case 'WLS'    
%         tensors_LS = lscov(X,log(data_flat));
%         data_flat_hat = exp(X*tensors_LS);
%         W = diag(data_flat_hat*data_flat_hat');
%         tensors = lscov(X,log(data_flat),W);

end

% returns a 3 by 3 tensor.
tensor_elements = tensor_elements(1:6, :);
tensors = tensor_elements([1,2,4,2,3,5,4,5,6], :);
% tensor: (3, 3, XYZ) matrix.
tensors = reshape(tensors, 3, 3, sx, sy, sz);


% Do eigen decomposition
[eigenvals, eigenvecs] = eigen_decomposition('tensors', tensors, 'brain_mask', brain_mask, 'min_diffusivity', min_diffusivity);

% eigenvals = reshape(eigenvals, 3, Nx, Ny, Nz);
% eigenvecs = reshape(eigenvecs, 3, 3, Nx, Ny, Nz);


% Calculat FA.
ev1 = squeeze(eigenvals(1,:,:,:));
ev2 = squeeze(eigenvals(2,:,:,:));
ev3 = squeeze(eigenvals(3,:,:,:));
FA = sqrt( 0.5 * ( (ev1 - ev2).^2 + (ev2 - ev3).^ 2 + (ev3 - ev1).^ 2 ) ./ (ev1.^2 + ev2.^2 + ev3.^2) );
% In the background of the image the fitting will not be accurate.
% There is no signal and possibly we will find FA values with nans (not a number).
FA(isnan(FA)) = 0;
color_FA = squeeze(abs(eigenvecs(:, 1, :, :, :))) .* permute(FA, [4,1,2,3]);

MD = squeeze(mean(eigenvals,1));

AD = ev1;

RD = (ev2 + ev3) / 2;


dti_parameters.tensors = tensors;
dti_parameters.eigenvals = eigenvals;
dti_parameters.eigenvecs = eigenvecs;
dti_parameters.FA = FA;
dti_parameters.color_FA = color_FA;
dti_parameters.MD = MD;
dti_parameters.AD = AD;
dti_parameters.RD = RD;





end