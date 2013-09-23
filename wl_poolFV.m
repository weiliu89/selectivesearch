function [fvFeat, goodIdx] = wl_poolFV(img, boxes, encoder)
% wl_poolFV() will pool the fisher vector from the image
%

%% step 0: compute the feature
stepSize = 4;
features = wl_getDenseSIFT(img, 'step', stepSize);

% RootSIFT
features.descr = sqrt(single(features.descr));
% L2 normalization
features.descr = bsxfun(@times, features.descr, 1./max(1e-5, sqrt(sum(features.descr.^2))));

%% step 1: project the feature
descrs = encoder.projection * bsxfun(@minus, features.descr, encoder.projectionCenter);
% descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2))));
[grad, goodIdx] = wl_fvComputeGrad(descrs', encoder);
frames = features.frame(:, goodIdx);

%% step 2: pool out for each box
nd = size(boxes, 1);
dim = (2*size(encoder.means, 1))*size(encoder.means, 2);
fvFeat = zeros(dim*4, nd);
tic
for d = 1:nd
	box = boxes(d,:);
	fvFeat(:, d) = wl_fisher(grad, frames, box, encoder);
end
toc
goodIdx = find(~any(isnan(fvFeat)));
fvFeat = sparse(fvFeat(:,goodIdx));
