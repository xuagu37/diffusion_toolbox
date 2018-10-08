function dki_parameters = dki_fit(varargin)


p = inputParser;
p.addParameter('data', []);
addParameter(p, 'brain_mask', []);
addParameter(p, 'bvals', []);
addParameter(p, 'bvecs', []);
addParameter(p, 'min_difusivity', 1e-5);


p.parse(varargin{:});

data = p.Results.data;
brain_mask = p.Results.brain_mask;
if isempty(brain_mask)
    brain_mask = ones(size(data));
end
bvals = p.Results.bvals;
bvecs = p.Results.bvecs;
min_difusivity = p.Results.min_difusivity;




B = zeros(length(bvals), 22);
B(:, 1) = -bvals .* bvecs(:, 1) .* bvecs(:, 1);
B(:, 2) = -2 * bvals .* bvecs(:, 1) .* bvecs(:, 2);
B(:, 3) = -bvals .* bvecs(:, 2) .* bvecs(:, 2);
B(:, 4) = -2 * bvals .* bvecs(:, 1) .* bvecs(:, 3);
B(:, 5) = -2 * bvals .* bvecs(:, 2) .* bvecs(:, 3);
B(:, 6) = -bvals .* bvecs(:, 3) .* bvecs(:, 3);
B(:, 7) = bvals .* bvals .* bvecs(:, 1).^4 / 6;
B(:, 8) = bvals .* bvals .* bvecs(:, 2).^4 / 6;
B(:, 9) = bvals .* bvals .* bvecs(:, 3).^4 / 6;
B(:, 10) = 4 * bvals .* bvals .* bvecs(:, 1).^3 .* bvecs(:, 2) / 6;
B(:, 11) = 4 * bvals .* bvals .* bvecs(:, 1).^3 .* bvecs(:, 3) / 6;
B(:, 12) = 4 * bvals .* bvals .* bvecs(:, 2).^3 .* bvecs(:, 1) / 6;
B(:, 13) = 4 * bvals .* bvals .* bvecs(:, 2).^3 .* bvecs(:, 3) / 6;
B(:, 14) = 4 * bvals .* bvals .* bvecs(:, 3).^3 .* bvecs(:, 1) / 6;
B(:, 15) = 4 * bvals .* bvals .* bvecs(:, 3).^3 .* bvecs(:, 2) / 6;
B(:, 16) = bvals .* bvals .* bvecs(:, 1).^2 .* bvecs(:, 2).^2;
B(:, 17) = bvals .* bvals .* bvecs(:, 1).^2 .* bvecs(:, 3).^2;
B(:, 18) = bvals .* bvals .* bvecs(:, 2).^2 .* bvecs(:, 3).^2;
B(:, 19) = 2 * bvals .* bvals .* bvecs(:, 1).^2 .* bvecs(:, 2) .* bvecs(:, 3);
B(:, 20) = 2 * bvals .* bvals .* bvecs(:, 2).^2 .* bvecs(:, 1) .* bvecs(:, 3);
B(:, 21) = 2 * bvals .* bvals .* bvecs(:, 3).^2 .* bvecs(:, 1) .* bvecs(:, 2);
B(:, 22) = ones(length(bvals), 1);

sz = size(data);
Nx = sz(1);
Ny = sz(2);
Nz = sz(3);
T = sz(4);

% Reshape diffusion data XxYxZxT --> XYZxT --> TxXYZ.
data_flat = reshape(data, [], T)';
% Reshape brain mask XxYxZ --> XYZx1.
brain_mask_flat = reshape(brain_mask, [], 1);

tensor_elements = (pinv(B)*log(data_flat));

dki_elements = tensor_elements(7:end-1,:);

dti_elements = tensor_elements(1:6,:);
dti_tensors = dti_elements([1,2,4,2,3,5,4,5,6], :);
% tensor: (3, 3, XYZ) matrix.
dti_tensors = reshape(dti_tensors, 3, 3, Nx*Ny*Nz);


% Do eigen decomposition
[eigenvals, eigenvecs] = eigen_decomposition('tensors', dti_tensors, 'brain_mask_flat', brain_mask_flat, 'min_difusivity', min_difusivity);



eigenvals = reshape(eigenvals, 3, Nx, Ny, Nz);
eigenvecs = reshape(eigenvecs, 3, 3, Nx, Ny, Nz);
dti_tensors = reshape(dti_tensors, 3,3, Nx, Ny, Nz);
dti_elements = reshape(dti_elements, 6, Nx, Ny, Nz);
dki_elements = reshape(dki_elements, 15, Nx, Ny, Nz);
MD = squeeze(mean(eigenvals,1));

dki_elements = dki_elements ./ permute((MD.^2),[4,1,2,3]);


% Calculat FA.
ev1 = squeeze(eigenvals(1,:,:,:));
ev2 = squeeze(eigenvals(2,:,:,:));
ev3 = squeeze(eigenvals(3,:,:,:));
FA = sqrt( 0.5 * ( (ev1 - ev2).^2 + (ev2 - ev3).^ 2 + (ev3 - ev1).^ 2 ) ./ (ev1.^2 + ev2.^2 + ev3.^2) );
% In the background of the image the fitting will not be accurate.
% There is no signal and possibly we will find FA values with nans (not a number).
FA(isnan(FA)) = 0;
color_FA = squeeze(abs(eigenvecs(:, 1, :, :, :))) .* permute(FA, [4,1,2,3]);


AD = ev1;

RD = (ev2 + ev3) / 2;


MK = mean_kurtosis(dki_elements, eigenvals, eigenvecs);
AK = axial_kurtosis(dki_elements, eigenvals, eigenvecs, MD, dti_elements);
RK = radial_kurtosis(dki_elements, eigenvals, eigenvecs);

dki_parameters.dti_elements = dti_elements;
dki_parameters.dki_elements = dki_elements;
dki_parameters.FA = FA;
dki_parameters.color_FA = color_FA;
dki_parameters.MD = MD;
dki_parameters.AD = AD;
dki_parameters.RD = RD;
dki_parameters.MK = MK;
dki_parameters.AK = AK;
dki_parameters.RK = RK;



end

