function [eigenvals, eigenvecs] = eigen_decomposition(varargin)

% Input:
% -----------------------------------------------------------------------------------------------
% tensors: 3 x 3 x sx x sy x sz 
% brain_mask: sx x sy x sz
% 
% Output
% -----------------------------------------------------------------------------------------------
% eigenvals: 3 x sx x sy x sz
% eigenvecs: 3 x 3 x sx x sy x sz


p = inputParser;
addParameter(p, 'tensors', []);
addParameter(p, 'brain_mask', []);
addParameter(p, 'min_diffusivity', 1e-6);
p.parse(varargin{:});
tensors = p.Results.tensors;
brain_mask = p.Results.brain_mask;
min_diffusivity = p.Results.min_diffusivity;

if isempty(brain_mask)
    brain_mask = ones(1, size(tensors, 3));
end

[~, ~, sx, sy, sz] = size(tensors);
eigenvals = zeros(3, sx, sy, sz);
eigenvecs = zeros(3, 3, sx, sy, sz);

for i = 1:sx
    for j = 1:sy
        for k = 1:sz
            if brain_mask(i,j,k) ~= 0
                [V, D] = eig((tensors(:,:,i,j,k)));
                [D, ind] = sort(diag(D),'descend');
                V = V(:,ind);
                eigenvals(:,i,j,k) = D;
                eigenvecs(:, :,i,j,k) = V;
            end
        end
    end
end

eigenvals = max(eigenvals, min_diffusivity);


end
