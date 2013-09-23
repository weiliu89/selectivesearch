function fvFeat = wl_fisher(grad, frames, box, encoder)
% wl_fisher() will compute the feature vector of a box given all the
% information
%   "Image Classification with the Fisher Vector: Theory and Practice" by
%   Jorge Sanchez, Florent Perronnin, Thomas Mensink, and Jakob Verbeek.
%   IJCV 2013 -- Algorithm 1
%
% Input:
%   grad: the gradient w.r.t. the GMM model
%   frames: the position of the feature points
%   boxes: the boxes that needs to pool fisher vector
%   encoder: the GMM model

%% step 1: get info and prepare
[featDim, nGMMs] = size(encoder.covariances);
dim = (2*featDim)*nGMMs;
fvFeat = zeros(dim*4, 1);
w = encoder.priors';
sigma = encoder.covariances;
mu = encoder.means;
sqrtw = sqrt(w);
sqrtsigma = sqrt(sigma);
musigmadiff = (mu.^2-sigma.^2);

%% step 2: pool out fisher vector signature
% step 2.1: pool out for 3x1 sub regions
boxH = box(4)-box(2)+1;
s0 = zeros(1, nGMMs);
s1 = zeros(featDim, nGMMs);
s2 = zeros(featDim, nGMMs);
t = 0;
for i = 1:3
    % step 2.1.1: get the region
    minx = box(1);
    miny = box(2)+(i-1)*boxH/3;
    maxx = box(3);
    maxy = box(2)+i*boxH/3;
    % step 2.1.2: find feature in the region
    sel = minx <= frames(1,:) & frames(1,:) < maxx  & ...
        miny <= frames(2,:) & frames(2,:) < maxy ;
    % step 2.1.3: compute the fisher vector signature for current region
    % step 2.1.3.1: aggregate statistics
    is0 = reshape(sum(grad.s0(sel,:)), [1 nGMMs]);
    is1 = reshape(sum(grad.s1(sel,:)), [featDim nGMMs]);
    is2 = reshape(sum(grad.s2(sel,:)), [featDim nGMMs]);
    it = sum(sel);
    continue;
    % stack to the big region
    s0 = s0 + is0;
    s1 = s1 + is1;
    s2 = s2 + is2;
    t = t + it;
    % step 2.1.3.2: compute fisher vector components
    gak = 1/it*(is0 - it*w)./sqrtw;
    gmuk = 1/it*(is1-bsxfun(@times, mu, is0))./bsxfun(@times, sqrtsigma, sqrtw);
    gsigmak = 1/it*(is2 - 2*mu.*is1 + bsxfun(@times, musigmadiff, is0))./bsxfun(@times, sigma, sqrt(2)*sqrtw);
%     fvFeat(i*dim+1:(i+1)*dim) = [gak'; gmuk(:); gsigmak(:)];
    fvFeat(i*dim+1:(i+1)*dim) = [gmuk(:); gsigmak(:)];
end
return;
% step 2.2: compute fisher vector signature for the whole box
gak = 1/t*(s0 - t*w)./sqrtw;
gmuk = 1/t*(s1-bsxfun(@times, mu, s0))./bsxfun(@times, sqrtsigma, sqrtw);
gsigmak = 1/t*(s2 - 2*mu.*s1 + bsxfun(@times, musigmadiff, s0))./bsxfun(@times, sigma, sqrt(2)*sqrtw);
% fvFeat(1:dim) = [gak'; gmuk(:); gsigmak(:)];
fvFeat(1:dim) = [gmuk(:); gsigmak(:)];

%% step 3: normalization
% power normalization
fvFeat = sign(fvFeat).*sqrt(abs(fvFeat));
% l2 normalization
fvFeat = bsxfun(@times, fvFeat, 1./max(1e-5,sqrt(sum(fvFeat.^2,1))));
