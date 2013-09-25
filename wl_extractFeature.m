function wl_extractFeature(imgFileList, featFileList, startIdx, endIdx, step, redetectFlag)
% wl_extractFeature() will extract feature from image given the image list
% Input:
%   imgFileList: the image file list
%   featFileList: the feature file list
%   startIdx: the start index of the image file list
%   endIdx: the end index of the image file list
%   step: the step size of SIFT feature
%

%% step 0: setup the environment
wl_setup;
if nargin < 5
    step = 10;
    redetectFlag = false;
end
if nargin < 6
    redetectFlag = true;
end

%% step 1: get the image/feature file path
% imgFiles = wl_getLines(imgFileList, startIdx, endIdx);
% featFiles = wl_getLines(featFileList, startIdx, endIdx);

% read in the image and feature file list
if ~exist(imgFileList, 'file')
    fprintf('%s does not exist!\n', imgFileList);
    return;
end
fid = fopen(imgFileList);
if fid == -1
    fprintf('Cannot open %s!\n', imgFileList);
    return;
end
C = textscan(fid, '%s');
imgFiles = C{1};
nImages = length(imgFiles);
clear C
fclose(fid);

if ~exist(featFileList, 'file')
    fprintf('%s does not exist!\n', featFileList);
    return;
end
fid = fopen(featFileList);
if fid == -1
    fprintf('Cannot open %s!\n', featFileList);
    return;
end
C = textscan(fid, '%s');
featFiles = C{1};
nFeatures = length(featFiles);
clear C
fclose(fid);

% check the consistency between the two file list
if (nImages ~= nFeatures)
    fprintf('%s and %s do not have the same number of files!\n', imgFileList, featFileList);
    return;
end
if (nImages == 0)
    fprintf('%s and %s appears empty!\n', imgFileList, featFileList);
    return;
end

%% step 2: read the image and extract feature from the image
nd = length(imgFiles);
for d = startIdx:endIdx
    if d > nd
	    break;
    end
    featFile = sprintf('%s/%s/Dictionary/%s', VOCopts.datadir, VOCopts.dataset, featFiles{d});
    featDir = fileparts(featFile);
    if ~exist(featDir, 'dir')
	    mkdir(featDir);
    end
    if exist(featFile, 'file') && ~redetectFlag
        continue;
    end
    % step 2.1: read the image
    th = tic;
    imgFile = sprintf('%s/%s/JPEGImages/%s', VOCopts.datadir, VOCopts.dataset, imgFiles{d});
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        continue;
    end
    img = imread(imgFile);
    % step 2.2: resize the image if necessary
    [img, resize_factor] = wl_resizeIm(img);
    % step 2.2: extract feature
    features = wl_getDenseSIFT(img, 'step', step);
    % step 2.3: save the feature
    save(featFile, 'features', 'resize_factor');
    fprintf('Detect %d features from %s in %.2f sec\n', length(features.scale), imgFile, toc(th));
end
