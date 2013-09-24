function wl_computeFeatureAndBoxes(imageFileList, featureFileList, codebookPath, startIdx, endIdx)
% wl_computeFeatureAndBoxes() will compute the dense sift feature for every four pixels of the image and the selective search bounding box
% Input:
%	imageFileList: contains a list of image path
%	featureFileList: contains a list of feature path
%	codebookPath: the path for the codebook file
%	startIdx: the start idx from the imageFileList
%	endIdx: the end idx from the imageFileList
%

% check file existence
if ~exist(imageFileList, 'file')
    fprintf('%s does not exist!\n', imageFileList);
    return;
end
if ~exist(featureFileList, 'file')
    fprintf('%s does not exist!\n', featureFileList);
    return;
end
if ~exist(codebookPath, 'file')
    fprintf('%s does not exist!\n', codebookPath);
    return;
end

% setup the environment
wl_setup;

% read in the image and feature file list
fid = fopen(imageFileList);
imagePathList = textscan(fid, '%s');
imagePathList = imagePathList{1};
nImages = length(imagePathList);
fclose(fid);

fid = fopen(featureFileList);
featurePathList = textscan(fid, '%s');
featurePathList = featurePathList{1};
nFeatures = length(featurePathList);
fclose(fid);

% check the consistency between the two file list
if (nImages ~= nFeatures)
    fprintf('%s and %s do not have the same number of files!\n', imageFileList, featureFileList);
    return;
end
if (nImages == 0)
    fprintf('%s and %s appears empty!\n', imageFileList, featureFileList);
    return;
end

% load the codebook
a = load(codebookPath);

% set the parameter
params.codebook = single(a.codebook)';
params.cb_norm2 = sum(params.codebook.^2, 2);
params.knn     = 5;
params.maxsize = 500;
params.mode    = '10k';

% process each image
for i = startIdx:endIdx
    if (i > nImages)
        break;
    end
    featurePath = sprintf('%s/%s/Features/%s', VOCopts.datadir, VOCopts.dataset, featurePathList{i});
    if exist(featurePath, 'file')
	    % fprintf('%s already exist!\n', featurePath);
	    % continue;
    end
    % read image
    imgPath = sprintf('%s/%s/JPEGImages/%s', VOCopts.datadir, VOCopts.dataset, imagePathList{i});
    if ~exist(imgPath, 'file')
        fprintf('%s does not exist!\n', imgPath);
        continue;
    end
    img = imread(imgPath);
    % make sure the image has three channels
    isGray = false;
    if ndims(img) ~= 3
        fprintf('%s is gray!\n', imgPath);
        isGray = true;
        img = repmat(img, [1 1 3]);
    end
    tic
    % compute the feature
%    [resize_factor, llc_codes, boxes, s] = ComputeFeatureAndBoxes_DFLOAT_fast(img, params);
%    s = rmfield(s, 'feaArr');
    % save the feature
%    save(featurePath, 'isGray', 'resize_factor', 'llc_codes', 'boxes', 's');
    [resize_factor, llc_codes] = ComputeFeatureAndBoxes_DFLOAT_fast(img, params);
    save(featurePath, 'llc_codes', '-append');
    fprintf('%s: %0.1f\n', imgPath, toc);
end

function [resize_factor, llc_codes, boxes, s] = ComputeFeatureAndBoxes_DFLOAT_fast(img, params)
[resize_factor, s] = ExtractDenseSift(img, params.maxsize);
llc_codes = LLC_encoding(s, params.codebook, params.cb_norm2, params.knn);
return;
if resize_factor ~= 1
    img = imresize(img, resize_factor);
end
boxes = GetBoxes(img);
boxes = boxes(:, [2 1 4 3]); % Convert to [left top right bottom];
boxes = FilterBox(boxes, 13, 4);

function [resize_factor, s] = ExtractDenseSift(img, max_dim)
factor = 1;
max_dim = double(max_dim);
if size(img,1)>max_dim || size(img, 2)>max_dim
    factor = min(max_dim/size(img, 1), max_dim/size(img, 2));
    img = imresize(img, factor);
end
resize_factor = factor;

img_sizes = [1 0.5 0.25];
bin_sizes = [4   4    4]; % Width in pixel of a spatial bin.
% 1-pixel dense
% steps     = [1   1    1]; % The H- and V- displacement of each feature to the next.
% Not so dense
steps     = [4 2.0  1.0];

s.width   = size(img, 2);
s.height  = size(img, 1);
s.descs   = [];
s.frames  = [];
s.scales  = [];

if(size(img, 3) == 3);
    img = rgb2gray(img);
end

img = im2single(img);
for j = 1:numel(bin_sizes)
    im = imresize(img, img_sizes(j));
    [myframes mydescs] = vl_dsift(im, ...
        'Size', bin_sizes(j), ...
        'Step', steps(j));
    % FAST: vl_dsift parameter 'Fast'
    s.frames = [s.frames myframes/img_sizes(j)];
    s.descs  = [s.descs  mydescs];
    s.scales = [s.scales, bin_sizes(j) * ones(1, size(myframes, 2)) / img_sizes(j)];
end
s.descs = single(s.descs);
% RootSIFT
s.descs = sqrt(s.descs);
% L2 normalization
s.descs = bsxfun(@times, s.descs, 1./max(1e-5, sqrt(sum(s.descs.^2))));
s.feaArr = single(s.descs);
s.x      = (s.frames(1, :));
s.y      = (s.frames(2, :));
s = rmfield(s, {'frames', 'descs'});

function [Coeff] = LLC_encoding(s, cb, cb_norm2, knn)
USE_PARFOR = 0;

X = s.feaArr;    % X should be 128 x #-of-features
% cb should be dict-size x 128
nframe = size(X,2);
nbase  = size(cb,1);
X_norm2 = sum(X.^2, 1);
% Approach 1
%D = repmat(X_norm2, nbase, 1) + repmat(cb_norm2, 1, nframe) - 2*cb*X;
% Approach 2 (slightly faster and takes less memory)
D = bsxfun(@plus, bsxfun(@plus, -2*cb*X, cb_norm2), X_norm2);

IDX = zeros(nframe, knn);
for k=1:knn
    [~, mi] = min(D, [], 1);
    D(sub2ind(size(D), mi, 1:size(D,2))) = inf;
    IDX(:, k) = mi';
end
clear D;

if USE_PARFOR ~= 1
    % approach 1
    II = eye(knn, knn);
    Coeff = sparse(nbase, nframe);
    beta = 1e-4;
    for i = 1:nframe
        idx = IDX(i, :);
        z = cb(idx,:) - repmat(X(:,i)', knn, 1);           % shift ith pt to origin
        C = z*z';                                        % local covariance
        tr = trace(C);
        if tr ==0
            C = C + 1e-10;
        end
        C = C + II*beta*tr;
        w = C\ones(knn, 1);
        w = w / sum(w);
        Coeff(idx, i) = w;
    end
else
    % approach 2 (need to open matlab workers before hand);
    II = eye(knn, knn);
    beta = 1e-4;
    %num_threads = matlabpool('size');
    % modified by Wei Liu Nov. 2nd, 2012
    num_threads = 4;
    chunks = cell(1, num_threads);
    chunk_size = ceil(nframe / num_threads);
    for i = 1:num_threads
        chunks{i} = (i-1)*chunk_size+1:min(i*chunk_size, nframe);
    end
    Coeff = cell(1, num_threads);
    parfor c = 1:num_threads
        cf = sparse(nbase, length(chunks{c}));
        for k = 1:length(chunks{c})
            i = chunks{c}(k);
            idx = IDX(i, :);
            z = cb(idx,:) - repmat(X(:,i)', knn, 1);
            C = z*z';
            tr = trace(C);
            if tr ==0
                C = C + 1e-10;
            end
            C = C + II*beta*tr;
            w = C\ones(knn, 1);
            cf(idx, k) = w / sum(w);
        end
        Coeff{c} = cf;
    end
    Coeff = cell2mat(Coeff);
end

function boxes  = GetBoxes(im)
%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
%   colorTypes = {'Rgb', 'Hsv', 'RGI', 'Opp'};
colorTypes = {'Rgb', 'Hsv', 'RGI', 'Opp'};
% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
kThresholds = [100 200];
sigma = 0.8;
numHierarchy = length(colorTypes) * length(kThresholds);

idx = 1;
currBox = cell(1, numHierarchy);
for k = kThresholds
    minSize = k; % We use minSize = k.
    for colorTypeI = 1:length(colorTypes)
        colorType = colorTypes{colorTypeI};
        currBox{idx}= SelectiveSearch(im, sigma, k, minSize, colorType);
        idx = idx + 1;
    end
end
boxes = cat(1, currBox{:}); % Concatenate results of all hierarchies
boxes = unique(boxes, 'rows'); % Remove duplicate boxes

% Box coordinates are [left top right bottom]
function [goodboxes, good_ind] = FilterBox(boxes, minBoxSize, maxAspectRatio)
if nargin < 3
    maxAspectRatio = 4;
end
if nargin < 2
    minBoxSize = 13;  % Assume the 'step' for dense sift is 4,
    % and pyramid grid is [1 3];
end

width = boxes(:, 3) - boxes(:, 1);
height= boxes(:, 4) - boxes(:, 2);

good_ind = (width  >= minBoxSize) & ...
    (height >= minBoxSize) & ...
    (width ./ height <= maxAspectRatio) & ...
    (height./ width  <= maxAspectRatio);

goodboxes = boxes(good_ind, :);

