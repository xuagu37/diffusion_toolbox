function  dti_parameters = dti_fit(varargin)
% This function applies diffusion tensor fitting 

% Input:
% -----------------------------------------------------------------------------------------------
% data: diffusion data, sx x sy x sz x sv
% b_vals: b-values, sv x 1.
% b_vecs: b-vectors, sv x 3.
% brain_mask: 
% fit_method: diffusion tensor fitting method, LS/WLS, default is WLS.
% min_diffusivity: negative eigenvalues are replaced by min_diffusivity, the default is 1e-6.

% Output:
% -----------------------------------------------------------------------------------------------
% tensors: 3 x 3 x sx x sy x sz
% [Dxx, Dxy, Dxz;
%  Dxy, Dyy, Dyz;
%  Dxz, Dyz, Dzz].
% eigenvals: 3 x sx x sy x sz. In descending order. 
% eigenvecs: 3 x 3 x sx x sy x sz. Eigenvectors are columnar.
% FA: Fractional anisotropy, sx x sy x sz.
% MD: mean diffusivity, sx x sy x sz.
% color_FA: FA in RGB form, 3 x sx x sy x sz.

tic

p = inputParser;
addParameter(p, 'data', []);
addParameter(p, 'brain_mask', []);
addParameter(p, 'bvals', []);
addParameter(p, 'bvecs', []);
addParameter(p, 'fit_method', 'WLS');
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

if (ndims(data)==3)
    data = permute(data, [1,2,4,3]);
end

[sx, sy, sz, sv] = size(data);

data_flat = reshape(data, [], sv)';
% Construct design matrix for diffusion tensor fitting.
X = dti_X(bvals, bvecs);

tensor_elements = zeros(7, sx*sy*sz);
switch fit_method
    case 'LS'
        tensor_elements = (pinv(X)*log(data_flat));
    case 'WLS'
        tensors_LS = (pinv(X)*log(data_flat));
        data_flat_hat = exp(X*tensors_LS);
        for i=1:sx*sy*sz
            W = diag(data_flat_hat(:,i).*data_flat_hat(:,i));
            tensor_elements(:,i) = X'*W*X\X'*W*log(data_flat(:,i));
        end
end

tensor_elements = tensor_elements(1:6, :);
tensors = tensor_elements([1,2,4,2,3,5,4,5,6], :);
tensors = reshape(tensors, 3, 3, sx, sy, sz);
[eigenvals, eigenvecs] = eigen_decomposition('tensors', tensors, 'brain_mask', brain_mask, 'min_diffusivity', min_diffusivity);

% Calculat FA.
ev1 = squeeze(eigenvals(1,:,:,:));
ev2 = squeeze(eigenvals(2,:,:,:));
ev3 = squeeze(eigenvals(3,:,:,:));
FA = sqrt( 0.5 * ( (ev1 - ev2).^2 + (ev2 - ev3).^ 2 + (ev3 - ev1).^ 2 ) ./ (ev1.^2 + ev2.^2 + ev3.^2) );
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