function wl_detectObject(modelName, dataset, iter, startIdx, endIdx, redetectFlag)
% wl_detectObject() will use all the model to detect object from an image
% using selective search method
% Reference:
%   "Segmentation As Selective Search for Object Recognition", ICCV 2011
%   by Koen E. A. van de Sande, Jasper R. R. Uijlings, Theo Gevers, Arnold W. M. Smeulders
%
% Input:
%   modelName: the name of the model. e.g.: cat, dog
%   dataset: the dataset type. e.g.: val, test
%   iter: the iteration number
%   startIdx: start index to process
%   endIdx: end index to process
%   redetectFlag: flag to redetect or not
%

%% step 0: set up the environment
global VOCopts;
wl_setup;

% by default, no redetection
if nargin < 6
    redetectFlag = false;
end

%% step 1: read the file names
fileList = sprintf(VOCopts.imgsetpath, dataset);
if ~exist(fileList, 'file')
    fprintf('%s does not exist!\n', fileList);
    return;
end
fid = fopen(fileList);
if fid == -1
    fprintf('Cannot open %s!\n', fileList);
    return;
end
fileNames = textscan(fid, '%s');
fileNames = fileNames{1};
nFiles = length(fileNames);
fclose(fid);

%% step 2: use the model to detect object on image
modelFile = [VOCopts.resdir 'models/' modelName '_iter' num2str(iter) '.mat'];
if ~exist(modelFile, 'file')
    fprintf('%s does not exist!\n', modelFile);
    return;
end
load(modelFile);
%load('data/encoder.mat');
boxesAll = cell(1,nFiles);
for d = startIdx:endIdx
    if d > nFiles
        return;
    end
    tic
    imgName = fileNames{d};
    %% step 3: predict image using all linear models
    resultFile = [VOCopts.resdir 'detections/' modelName '/' imgName '_iter' num2str(iter) '.mat'];
    if ~exist(resultFile, 'file') || redetectFlag
        % step 3.1: compute feature for the image
        % step 3.1.1: get the LLC coding file
        featFile = sprintf(VOCopts.featpath, imgName);
        if ~exist(featFile, 'file')
            fprintf('%s does not exist!\n', featFile);
            continue;
        end
        a = load(featFile);
        
        imgFile = sprintf(VOCopts.imgpath, imgName);
        if ~exist(imgFile, 'file')
            fprintf('%s does not exist!\n', imgFile);
            continue;
        end
        img = imread(imgFile);
        img = wl_resizeIm(img);
        
        % step 3.1.2: get the feature for each bounding box using max pooling
        pyramids = [1 3];
        [beta goodIdx] = LLC_pooling_mex_sparse(a.s.x, a.s.y, a.s.scales, a.llc_codes, a.boxes(:, [2 1 4 3]), pyramids);
        boxes = a.boxes(goodIdx, :);
        %[beta, goodIdx] = wl_poolFV(img, a.boxes, encoder);
        %boxes = a.boxes(goodIdx, :);
        
        % step 3.1.3: evaluate the feature using all the linear models
        dummy = -1*ones(size(beta, 2), 1);
        [plabels, acc, pscores] = predict(dummy, beta, linearModel, '-b  0', 'col');
        
        % step 3.1.4: correct the score according to the model
        pscores = pscores*linearModel.Label(1);
        
        % step 3.1.5: do nms for the detection results
        [pick, boxes] = wl_nms([boxes pscores], 0.3);
        
        % step 3.1.6: store the detection results
	if startIdx == 1 && endIdx == nFiles
		boxesAll{d} = boxes;
	else
		save(resultFile, 'boxes');
	end
        fprintf('%s evaluation time: %f\n', imgName, toc);
    end
end

if startIdx == 1 && endIdx == nFiles
	resultFile = [VOCopts.resdir 'detections/' modelName '/' modelName '_iter' num2str(iter) '.mat'];
	resultDir = fileparts(resultFile);
	if ~exist(resultDir, 'dir')
		mkdir(resultDir);
	end
	save(resultFile, 'boxesAll', '-v7.3');
end

%% step 4: evaluate the model if necessary
nd = endIdx - startIdx + 1;
if nd == nFiles && VOCopts.year == 2007
    wl_evalModel(modelName, iter, redetectFlag);
end
