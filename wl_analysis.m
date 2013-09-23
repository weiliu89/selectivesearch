function wl_analysis()
% wl_analysis() will analyze the process of BOW for a given model
% Input:
%   modelName: the ame of the model. e.g.: cat, dog
%

%% step 0: setup the environment
global VOCopts;
wl_setup;
addpath('utils');

% step 0.1: get the AP values for all results
typeFile = sprintf('%s/%s/typeNames', VOCopts.datadir, VOCopts.dataset);
if ~exist(typeFile,'file')
    fprintf('%s does not exist!\n', typeFile);
    return;
end
fid = fopen(typeFile);
if fid == -1
    fprintf('Cannot open %s!\n', typeFile);
    return;
end
C = textscan(fid, '%s');
typeNames = C{1};
fclose(fid);

aps = zeros(length(typeNames), 3);
for k=1:length(typeNames)
    typeName = typeNames{k};
    for iter=1:2
        apFile = [VOCopts.resdir 'aps/' typeName '_iter' num2str(iter) '.mat'];
        if ~exist(apFile, 'file')
            fprintf('%s does not exist!\n', apFile);
            continue;
        end
        load(apFile, 'ap');
        aps(k,iter) = ap;
    end
    fprintf('%.3f %.3f %.3f\n', aps(k,:));
end
fprintf('%.3f %.3f %.3f\n', mean(aps));
keyboard

%% step 1: get the positive and negative training samples
posImFile = [VOCopts.resdir 'analysis/' modelName '_trainval_posTrain.png'];
negImFile = [VOCopts.resdir 'analysis/' modelName '_trainval_negTrain.png'];
if ~exist(posImFile, 'file') || ~exist(negImFile, 'file')
    % step 1.1: read in the bbInfo
    trainBBFile = [VOCopts.resdir 'trainBB/' modelName '_iter1.txt'];
    if ~exist(trainBBFile, 'file')
        fprintf('%s does not exist!\n', trainBBFile);
        return;
    end
    fid = fopen(trainBBFile);
    if fid == -1
        fprintf('Cannot open %s!\n', trainBBFile);
        return;
    end
    C = textscan(fid, '%s %d %d %d %d %d');
    ids = C{1};
    labels = C{2};
    boxes = [C{3} C{4} C{5} C{6}];
    
    % step 1.2: get the images
    nd = length(ids);
    posIms = [];
    negIms = [];
    for d = 1:nd
        id = ids{d};
        % step 1.2.1: read in the image
        imgFile = sprintf(VOCopts.imgpath, id);
        if ~exist(imgFile, 'file')
            fprintf('%s does not exist!\n', imgFile);
            continue;
        end
        img = imread(imgFile);
        
        % step 1.2.2: get the bounding box image
        rect = [boxes(d,1:2) boxes(d,3:4)-boxes(d,1:2)+1];
        cropIm = imcrop(img, rect);
        
        % step 1.2.3: store the image
        label = labels(d);
        if label == 1
            posIms{end+1} = cropIm;
        else
            negIms{end+1} = cropIm;
        end
    end
    
    % step 1.3: display the image
    imdisp(posIms);
    print('-dpng', posImFile);
    
    clf; imdisp(negIms);
    print('-dpng', negImFile);
end

%% step 2: get the bow detection results

