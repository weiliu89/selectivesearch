function [feats, labels, boxes, ids] = wl_getFVFeat(bbsInfo)
% wl_getFVFeat() will get the information of the bounding box
%
% Input:
%   bbsInfo: a 1x6 cell array
%       bbNames: cell array
%       bbLabes: cell array
%       x_mins: cell array
%       y_mins: cell array
%       x_maxs: cell array
%       y_maxs: cell array
% Output:
%   feats: the feature vector Nxp
%   labels: the label vector Nx1
%   boxes: the corresponded boxes
%   ids: the corresponded image names
%   bbsImg: the image of the boundng boxes Nx1
%

% step 1: preprocessing
% step 1.1: setup the global variable
global VOCopts;
wl_setup();

% step 1.2: get the bounding box information
bbNames = bbsInfo{1,1}(:);
bbLabels = double(bbsInfo{1,2});
% [x_min y_min x_max y_max]
bbs = double([bbsInfo{1,3} bbsInfo{1,4} bbsInfo{1,5} bbsInfo{1,6}]);

load('data/encoder.mat');

% step 1.3: get the training images
hash = VOChash_init(bbNames);
imgNames = unique(bbNames);
nImgs = length(imgNames);

% step 2: get feature for each bounding box
nBBs = length(bbLabels);
feats = sparse(131072, nBBs);
selIdx = zeros(1,nBBs);
count = 0;
for iImg = 1:nImgs
    imgName = imgNames{iImg};
    % step 2.2: get the feature for the image
    imgFile = sprintf(VOCopts.imgpath, imgName);
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        continue;
    end
    img = imread(imgFile);
    img = wl_resizeIm(img);
    
    % step 2.3: get the bouding box information
    idx = VOChash_lookup(hash, imgName);
    
    % step 2.4: get the feature for each bounding box using max pooling
    [beta, goodIdx] = wl_poolFV(img, bbs(idx, :), encoder);
    nFeats = size(beta, 2);
    
    % step 2.5: get the features and labels
    if nFeats > 0
	    selIdx(count+1:count+nFeats) = idx(goodIdx);
	    feats(:,count+1:count+nFeats) = beta;
	    count = count + nFeats;
    end
end

if count < nBBs
	selIdx = selIdx(1:count);
	feats = feats(:,1:count);
end

% get the other information
labels = bbLabels(selIdx);
boxes = bbs(selIdx, :);
ids = bbNames(selIdx);
