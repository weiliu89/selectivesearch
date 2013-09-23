function wl_showDetection(imgID, modelName, iter)
% wl_showDetection() will show the detection results of using model to
% detect on a image with imgPath
% Input:
%   imgID: the ID of the image, e.g.: 000032
%   modelName: the name of the model, e.g.: aeroplane
%   iter: iteration number
%

%% step 0: setup the environment
global VOCopts;
wl_setup();

%% step 1: read in the model, image, and feature
% step 1.1: read the model
modelFile = sprintf('%s/models/%s_iter%d.mat',VOCopts.resdir,modelName,iter);
if ~exist(modelFile, 'file')
    fprintf('%s does not exist!\n', modelFile);
    return;
end
load(modelFile);

% step 1.2: read the file list
fileList = sprintf(VOCopts.clsimgsetpath, modelName, 'test');
if ~exist(fileList, 'file')
    fprintf('%s does not exist!\n', fileList);
    return;
end
fid = fopen(fileList);
if fid == -1
    fprintf('Cannot open %s!\n', fileList);
    return;
end
C = textscan(fid, '%s %d');
ids = C{1};
labels = C{2};
clear C
fclose(fid);

for i = 1:length(ids)
    if labels(i) == -1
        continue;
    end
    imgID = ids{i};
    % step 1.2: read the image
    imgFile = sprintf(VOCopts.imgpath, imgID);
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        return;
    end
    img = imread(imgFile);
    [imH, imW, ~] = size(img);
    
    % step 1.3: read the feature
    featFile = sprintf(VOCopts.featpath, imgID);
    if ~exist(featFile, 'file')
        fprintf('%s does not exist!\n', featFile);
        return;
    end
    a = load(featFile);
    
    %% step 2: detect object
    pyramids = [1 3];
    [beta goodIdx] = LLC_pooling_mex_sparse(a.s.x, a.s.y, a.s.scales, a.llc_codes, a.boxes(:, [2 1 4 3]), pyramids);
    boxes = a.boxes(goodIdx, :);
    
    % step 2.1: evaluate the feature using all the linear models
    dummy = -1*ones(size(beta, 2), 1);
    [plabels, acc, pscores] = predict(dummy, beta, linearModel, '-b  0', 'col');
    
    % step 2.2: correct the score according to the model
    pscores = pscores*linearModel.Label(1);
    
    % step 2.3: sort the value
    [pscores,si] = sort(pscores, 'descend');
    boxes = boxes(si,:);
    
    %% step 3: draw the detection result
    detMask = ones(imH, imW)*inf*-1;
    nd = size(boxes, 1);
    for d = 1:nd
        box = boxes(d,:);
        %box = [boxes(d,1:2), boxes(d,3:4)-boxes(d,1:2)];
        %rectangle('Position', box);
        %[X, Y] = meshgrid(box(2):box(4), box(1):box(3));
        %mask(X, Y)  = 1;
        %detMask = max(detMask, pscores(d)*mask);
        detMask(box(2):box(4), box(1):box(3)) = max(detMask(box(2):box(4), box(1):box(3)), pscores(d));
    end
    
    %% step 4: show the result
    subplot(2,1,1);
    imagesc(img); axis image
    subplot(2,1,2);
    imagesc(detMask); axis image
    
    pause;
end