function wl_getSSTrainBB(modelName, trainBBFile)
% wl_getSSTrainBB() will get the initial training bounding box for selective search method
% Input:
%   modelName: the name of the model. e.g.: cat, dog
%   trainBBFile: it will contain the bounding box info for positive and negative samples
% Output:
%   bbsInfo: a 1x6 cell array
%       bbNames: cell array
%       bbLabes: cell array
%       x_mins: cell array
%       y_mins: cell array
%       x_maxs: cell array
%       y_maxs: cell array
%

% step 0: setup the global variable
global VOCopts;
wl_setup();

%% step 1: get the training bounding boxes filelist for a given modelName
trainFileList = sprintf(VOCopts.clsimgsetpath,modelName,VOCopts.trainset);
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

% step 1.2: find the positive images
posIdx = labels==1;
ids = ids(posIdx);

%% step 2: open trainBBFile to write
trainbbdir = fileparts(trainBBFile);
if ~exist(trainbbdir, 'dir')
    mkdir(trainbbdir);
end
if ~exist(trainBBFile, 'file')
    fid = fopen(trainBBFile, 'w');
    if fid == -1
        fprintf('Cannot open %s!\n', trainBBFile);
        return;
    end
else
    fid = fopen(trainBBFile, 'a+');
    if fid == -1
        fprintf('Cannot open %s!\n', trainBBFile);
        return;
    end
    C = textscan(fid, '%s %d %d %d %d %d');
    prevIds = C{1};
    clear C
    ids = setdiff(ids, prevIds);
end

% step 3: get the positive and negative bounding boxes
nFiles = length(ids);
bbs = [];
nRecallBBs = 0;
for iFile = 1:nFiles
    % step 3.1: get the positive bounding boxes
    % step 3.1.1: read the annotation file
    id = ids{iFile};
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
    
    % show the image
    imgFile = sprintf('%s/%s', VOCopts.datadir, rec.imgname);
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        continue;
    end
    img = imread(imgFile);
    [imH, imW, imC] = size(img);
    
    % step 3.1.2: get the positive bounding boxes
    iposBBs = [];
    for j = clsinds(:)'
        bbox = rec.objects(j).bbox; % [x_min y_min x_max y_max]
        fprintf(fid,'%s 1 %d %d %d %d\n',id,bbox);
        iposBBs = [iposBBs; bbox];
        bbs = [bbs; {id 1 bbox(1) bbox(2) bbox(3) bbox(4)}];
        continue;
        % generate some jittered positive bounding boxes
        jbbox = wl_jitterBB(bbox, imH, imW, 20);
        njb = size(jbbox, 1);
        for i = 1:njb
            ijbbox = jbbox(i,:);
            ov = wl_overlapBB(ijbbox, bbox);
            if ov < 0.7
                continue;
            end
            fprintf(fid,'%s 1 %d %d %d %d\n',id,ijbbox);
            iposBBs = [iposBBs; ijbbox];
            bbs = [bbs; {id 1 ijbbox(1) ijbbox(2) ijbbox(3) ijbbox(4)}];
        end
    end
    
    % step 3.2: find the hard negative samples
    % step 3.2.1: get the selective search segmented bounding boxes
    featFile = sprintf(VOCopts.featpath, id);
    if ~exist(featFile,'file')
        continue;
    end
    load(featFile, 'boxes');
    % step 3.2.2: get the negative bounding boxes
    nBoxes= size(boxes,1);
    ovs = zeros(nBoxes,1);
    recallIdx = zeros(size(iposBBs,1), 1);
    for iBox = 1:nBoxes
        ibox = boxes(iBox,:);
        max_ov = 0;
        for iPos = 1:size(iposBBs,1)
            gtBB = iposBBs(iPos,:);
            ov = wl_overlapBB(gtBB,ibox);
            if ov > max_ov
                max_ov = ov;
            end
            if ov > 0.5
                recallIdx(iPos) = 1;
            end
        end
        ovs(iBox) = max_ov;
    end
    [so,si] = sort(ovs);
    boxes = boxes(si,:);
    ovs = ovs(si);
    keepidx = ovs>=0.2 & ovs<0.5;
    inegBBs = boxes(keepidx,:);
    
    % step 4.2.3: find the non-duplicated negative bounding boxes
    selNegBBs = [];
    for iNeg = 1:size(inegBBs,1)
        ineg = inegBBs(iNeg,:);
        max_ov = 0;
        for jNeg = 1:size(selNegBBs,1)
            jneg = selNegBBs(jNeg,:);
            ov = wl_overlapBB(ineg,jneg);
            if ov > max_ov
                max_ov = ov;
            end
        end
        if max_ov < 0.7
            fprintf(fid,'%s -1 %d %d %d %d\n',id,ineg);
            bbs = [bbs; {id -1 ineg(1) ineg(2) ineg(3) ineg(4)}];
            selNegBBs(end+1,:)  = ineg;
        end
        % exception for person because it has too many images
        if strcmp(modelName,'person') && size(selNegBBs,1) >= 40
            break;
        end
    end
    
    % step 4.2.4: accumulate the recall value
    nRecallBBs = nRecallBBs + sum(recallIdx);
end

% step 5: close the file
%bbsInfo = [{bbs(:,1)} cell2mat(bbs(:,2)) cell2mat(bbs(:,3)) cell2mat(bbs(:,4)) cell2mat(bbs(:,5)) cell2mat(bbs(:,6))];
fclose(fid);

% step 5.1: display the recall rate
if ~isempty(bbs)
    labels = cell2mat(bbs(:,2));
    nPosBBs = sum(labels==1);
    recallRate = nRecallBBs/nPosBBs;
    fprintf('nPosBBs: %d, nRecallBBs: %d, recall rate: %f\n', nPosBBs, nRecallBBs, recallRate);
end
