function wl_extractShape(modelName)
% wl_extratShape() will extract shape of the object
% Input:
%	modelName: the name of the model. e.g.: cat, dog
%

%% step 0: set up the environment
global VOCopts
wl_setup;

segFile = sprintf('%s/%s/SegmentationObject/%s.mat', VOCopts.datadir, VOCopts.dataset, modelName);
if exist(segFile, 'file')
    fprintf('%s already exist!\n', segFile);
    return;
end

%% step 1: get the segmentation image names
segObjDir = sprintf('%s/%s/SegmentationObject', VOCopts.datadir, VOCopts.dataset);
segClsDir = sprintf('%s/%s/SegmentationClass', VOCopts.datadir, VOCopts.dataset);
imgDir = sprintf('%s/%s/JPEGImages', VOCopts.datadir, VOCopts.dataset);
imgNames = dir(sprintf('%s/*.png', segObjDir));
nd = length(imgNames);

%% step 2: get the segmentation shapes
modelID = find(strcmp(VOCopts.classes, modelName));
segs = [];
imgs = [];
for d = 1:nd
    % step 2.1: get the segmentation image
    [~,imgName] = fileparts(imgNames(d).name);
    objImgFile = sprintf('%s/%s.png', segObjDir, imgName);
    if ~exist(objImgFile, 'file')
        fprintf('%s does not exist!\n', objImgFile);
        continue;
    end
    objImg = imread(objImgFile);
    clsImgFile = sprintf('%s/%s.png', segClsDir, imgName);
    if ~exist(clsImgFile, 'file')
        fprintf('%s does not exist!\n', clsImgFile);
        continue;
    end
    clsImg = imread(clsImgFile);
    imgFile = sprintf('%s/%s.jpg', imgDir, imgName);
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        continue;
    end
    img = imread(imgFile);
    % step 2.2: get the segmentation ids
    ids = unique(objImg(:));
    % step 2.3: remove background (0) and unlabeled (255)
    ids(ids==0) = [];
    ids(ids==255) = [];
    % step 2.4: identify each segmentation
    ni = length(ids);
    for i = 1:ni
        id = ids(i);
        mask = objImg==id;
        clsID = unique(clsImg(mask));
        if length(clsID) > 1
            fprintf('%s: inaccurate segmentation!\n', imgName);
            continue;
        end
        if clsID ~= modelID
            continue;
        end
        [rows, cols] = find(mask);
        box = [min(cols) min(rows) max(cols) max(rows)];
        rect = [box(1:2) box(3:4)-box(1:2)+1];
        segs{end+1} = imcrop(mask, rect);
        imgs{end+1} = imcrop(img, rect);
        %imagesc(mask);
        %rectangle('Position', rect);
    end
end
save(segFile, 'segs', 'imgs', '-v7.3');