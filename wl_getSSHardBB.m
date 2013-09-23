function  wl_getSSHardBB(linearModel, hardBBsFile, topK)
% wl_getsshardbb() will get the hard negative training bounding box for selective search method
% input:
%   linearModel: the linear model for a class with wordnet id; e.g. n01443537
%   hardBBsFile: contains the hard negative bounding box info for positive and negative samples
%   topK: number of negative samples to choose
% output:
%   hardBBsInfo: a 1x6 cell array
%       bbnames: cell array
%       bblabes: cell array
%       x_mins: cell array
%       y_mins: cell array
%       x_maxs: cell array
%       y_maxs: cell array
%

% step 0: setup the global variable
global VOCopts;
wl_setup();

% step 0.1: get the input parameter
if nargin < 3
	topK = 1;
end

% step 1: get all the negative image names
% step 1.1: get all the image names
trainFileList = sprintf(VOCopts.clsimgsetpath,linearModel.modelName,VOCopts.trainset);
if ~exist(trainFileList, 'file')
    fprintf('%s does not exist!\n', trainFileList);
    return;
end
fid = fopen(trainFileList);
if fid == -1
    fprintf('Cannot open %s!\n', trainFieList);
    return;
end
C = textscan(fid, '%s %d');
ids = C{1};
labels = C{2};
fclose(fid);

% step 1.2: find the image which does not have the modelName
negIdx = labels==-1;
negIds = ids(negIdx);

% step 2: open hardBBsFile to write
trainBBdir = fileparts(hardBBsFile);
if ~exist(trainBBdir, 'dir')
    mkdir(trainBBdir);
end
if ~exist(hardBBsFile, 'file')
	fid = fopen(hardBBsFile, 'w');
	if fid == -1
		fprintf('cannot open %s!\n', hardBBsFile);
		return;
	end
else
	fid = fopen(hardBBsFile, 'a+');
	if fid == -1
		fprintf('cannot open %s!\n', hardBBsFile);
		return;
	end
	C = textscan(fid, '%s %d %d %d %d %d');
	ids = unique(C{1});
	clear C
	negIds = setdiff(negIds, ids);
end

% step 3: get the top negative bounding boxes in negative images
nNegImgs = length(negIds);
%hardNegBBs = [];
for iNeg = 1:nNegImgs
    negId = negIds{iNeg};
    % step 3.1: get the feature for the negative image
    % step 3.1.1: load the llc coding feature
    negFeatFile = sprintf(VOCopts.featpath, negId);
    if ~exist(negFeatFile, 'file')
        fprintf('%s does not exist!\n', negFeatFile);
        continue;
    end
    a = load(negFeatFile);
    
    % step 3.1.2: do the llc pooling
    pyramids = [1 3];
    [beta goodIdx] = LLC_pooling_mex_sparse(a.s.x, a.s.y, a.s.scales, a.llc_codes, a.boxes(:, [2 1 4 3]), pyramids);
    boxes = a.boxes(goodIdx,:);
    
    % step 3.2: evaluate the feature using the linear model
    dummy = -1*ones(size(beta, 2), 1);
    [~, ~, scores] = predict(dummy, beta, linearModel, '-b  0', 'col');
    
    % step 3.3: correct the score according to the model
    scores = scores*linearModel.Label(1);
    
    % step 3.4: do nms on the detection results
    [pick, boxes] = wl_nms([boxes, scores], 0.5);
    nBoxes = length(pick);
    
    % step 3.5: output the top one prediction result
    for i=1:topK
	    if i > nBoxes
		    break;
	    end
	    fprintf(fid,'%s -1 %d %d %d %d\n',negId,boxes(i,1:4));
	    %hardNegBBs = [hardNegBBs; {negId -1 boxes(i,1) boxes(i,2) boxes(i,3) boxes(i,4)}];
    end
end

% step 4: close the file
%hardBBsInfo = [{hardNegBBs(:,1)} cell2mat(hardNegBBs(:,2)) cell2mat(hardNegBBs(:,3)) cell2mat(hardNegBBs(:,4)) cell2mat(hardNegBBs(:,5)) cell2mat(hardNegBBs(:,6))];
fclose(fid);
