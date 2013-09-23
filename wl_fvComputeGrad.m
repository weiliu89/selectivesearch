function [grad, goodIdx] = wl_fvComputeGrad(feats, encoder)
% wl_fvComputeGrad() will compute the gradients of a single feature w.r.t.
% the parameters of the GMM model according to
%   "Image Classification with the Fisher Vector: Theory and Practice" by
%   Jorge Sanchez, Florent Perronnin, Thomas Mensink, and Jakob Verbeek.
%   IJCV 2013
%
% Input:
%   feats: the feature vector nxd
%   encoder: the GMM model
% Output:
%   grad: the gradients w.r.t. GMM parameter
%   goodIdx: the index of feature which is valid
%

%% step 1: compute the posterior probability for each feature
% step 1.1: compute ux according to e.q. 9
[featDim, nGMMs] = size(encoder.covariances);
[nFeats, featDim2] = size(feats);
assert(featDim == featDim2);
u = zeros(nFeats, nGMMs);
for k = 1:nGMMs
    u(:,k) = mvnpdf(double(feats), double(encoder.means(:,k)'), double(encoder.covariances(:,k)'));
end
goodIdx = ~any(u==inf, 2);
u = u(goodIdx, :);
goodIdx = find(goodIdx);
nFeats = length(goodIdx);
% step 1.2: compute the posterior using e.q. 15
gamma = bsxfun(@times, u, double(encoder.priors'));
gamma = bsxfun(@times, gamma, 1./max(1e-5, sum(gamma,2)));
% create more sparsity
gamma(gamma<0.01) = 0;
gamma = bsxfun(@times, gamma, 1./max(1e-5, sum(gamma,2)));

%% step 2: compute first-order and second-order statistics for each feature
grad.s0 = reshape(gamma, [nFeats 1 nGMMs]);
grad.s1 = bsxfun(@times, feats(goodIdx, :), grad.s0);
grad.s2 = bsxfun(@times, feats(goodIdx, :).^2, grad.s0);

grad.s0 = sparse(reshape(grad.s0, [nFeats nGMMs]));
grad.s1 = sparse(reshape(double(grad.s1), [nFeats featDim*nGMMs]));
grad.s2 = sparse(reshape(double(grad.s2), [nFeats featDim*nGMMs]));