function X = dti_X(bvals, bvecs)

% Check if bvals and bvecs are in the right form
if size(bvals, 2) ~= 1
    bvals = bvals';
end
if size(bvecs, 2) ~= 3
    bvecs = bvecs';
end

sv = size(bvals,1);
% Construct design matrix for diffusion tensor fitting.
X = zeros(sv, 7);
X(:, 1) = bvecs(:, 1) .* bvecs(:, 1) * 1 .* bvals;   % Bxx
X(:, 2) = bvecs(:, 1) .* bvecs(:, 2) * 2 .* bvals;   % Bxy
X(:, 3) = bvecs(:, 2) .* bvecs(:, 2) * 1 .* bvals;   % Byy
X(:, 4) = bvecs(:, 1) .* bvecs(:, 3) * 2 .* bvals;   % Bxz
X(:, 5) = bvecs(:, 2) .* bvecs(:, 3) * 2 .* bvals;   % Byz
X(:, 6) = bvecs(:, 3) .* bvecs(:, 3) * 1 .* bvals;   % Bzz
X(:, 7) = -ones(sv,1);
X = -X;

end
