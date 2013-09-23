function wl_getRecallValue(modelName, dataset)
% wl_getRecallValue() will get the recall value for selective search method
% for a certain model on the dataset
% Input:
%   modelName: the name of the model. e.g.: cat, dog
%   dataset: the name of the dataset. e.g.: trainval, test
%

% step 0: setup the global variable
global VOCopts;
wl_setup();

% step 1: get the training bounding boxes filelist for a given modelName
trainFileList = sprintf(VOCopts.clsimgsetpath,modelName,dataset);
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
nFiles = length(ids);
fclose(fid);

% step 2: get the positive bounding boxes and count the recall value
bbs = [];
nRecallBBs = 0;
for iFile = 1:nFiles
    % step 2.1: get the positive bounding boxes
    % step 2.1.1: read the annotation file
    id = ids{iFile};
    label = labels(iFile);
    if label == -1
        continue;
    end
    annotationFile=sprintf(VOCopts.annopath, id);
    if ~exist(annotationFile, 'file')
        disp([annotationFile, ' does not exist!']);
        continue;
    end
    rows = numel(textread(annotationFile, '%1c%*[^\n]'));
    if rows == 15
        disp([annotationFile, ' does not have objects!']);
        continue;
    end
    rec = PASreadrecord(annotationFile);
    if isempty(rec.objects)
        continue;
    end
    clsinds = strmatch(modelName, {rec.objects(:).class}, 'exact');
    % skip difficult examples
    diff = [rec.objects(clsinds).difficult];
    clsinds(diff) = [];
    
    % step 2.1.2: get the positive bounding boxes
    iposBBs = [];
    for j = clsinds(:)'
        bbox = rec.objects(j).bbox; % [x_min y_min x_max y_max]
        iposBBs = [iposBBs; bbox];
        bbs = [bbs; {id 1 bbox(1) bbox(2) bbox(3) bbox(4)}];
    end
    
    % step 2.2: find the recall box
    % step 2.2.1: get the selective search segmented bounding boxes
    featFile = sprintf(VOCopts.featpath, id);
    if ~exist(featFile,'file')
        continue;
    end
    load(featFile, 'boxes');
    nBoxes= size(boxes,1);
    ovs = zeros(nBoxes,1);
    % step 2.2.2: get the negative bounding boxes
    recallIdx = zeros(size(iposBBs,1), 1);
    for iBox = 1:nBoxes
        ibox = boxes(iBox,:);
        max_ov = 0;
        for iPos = 1:size(iposBBs,1)
            gtBB = iposBBs(iPos,:);
            ov = wl_overlapBB(gtBB,ibox);
            if ov > 0.5
                recallIdx(iPos) = 1;
            end
        end
        if sum(recallIdx) == size(iposBBs,1)
            break;
        end
    end
    % step 2.2.3: sum up the recall value
    nRecallBBs = nRecallBBs + sum(recallIdx); 
end

% step 3: compute the recall value
nPosBBs = size(bbs,1);
recallRate = nRecallBBs/nPosBBs;
fprintf('nPosBBs: %d, nRecallBBs: %d, recall rate: %f\n', nPosBBs, nRecallBBs, recallRate);
