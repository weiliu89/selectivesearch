function [ap,rec,prec,ffpi,tp,fp,ids,boxes,confidences] = wl_evalBoxes(modelName,detAllFile,annoFile,resultFile,overlap,redetectFlag,saveFlag)
% wl_evalBoxes() will evaluate the detection results
% Input:
%   modelName: the name of the model. e.g.: cat, dog
%	detAllFile: the file which contains all the detection info.
%	annoFile: the annotation file
%	resultFile: the file to store the evaluation results
%	overlap: the overlap threshold
%   redetectFlag: indicator about whether to consider duplicated detection
%   saveFlag: indicator about whether to save the results or not
%

%% step 0: preprocessing
% step 0.1: return if resultFile already exist
%if exist(resultFile, 'file')
%    fprintf('%s already exist!\n', resultFile);
%    return;
%end

% step 0.2: default overlap value during evaluation
if nargin < 5
    overlap = 0.5;
    redetectFlag = false;
    saveFlag = true;
elseif nargin < 6
    redetectFlag = false;
    saveFlag = true;
elseif nargin < 7
    saveFlag = true;
else
    % get all the parameters
end

% step 0.3: set up environment
global VOCopts;
wl_setup();

%% step 1: read detection results
if ~exist(detAllFile, 'file')
    fprintf('%s does not exist!\n', detAllFile);
    return;
end
fid = fopen(detAllFile);
if fid == -1
    fprintf('Cannot open %s!\n', detAllFile);
    return;
end
C = textscan(fid,'%s %f %f %f %f %f');
ids = C{1};
boxes = [C{2} C{3} C{4} C{5}]';
confidences = C{6};
fclose(fid);

% step 1.5: sort detections by decreasing confidences
[sc,si]=sort(-confidences);
ids=ids(si);
boxes=boxes(:,si);
confidences = confidences(si);

%% step 2: collect ground truth annoations
% step 2.1: load annotations
if ~exist(annoFile, 'file')
    [gtids, recs] = wl_getAnnotation(dataset, annoFile);
else
    load(annoFile);
end

% step 2.2: hash image ids
hash=VOChash_init(gtids);

% step 2.3: extract ground truth objects
npos=0;
gt(length(gtids))=struct('boxes',[],'diff',[],'det',[]);
for i=1:length(recs)
    % extract objects of class
    if isempty(recs(i).objects)
        continue;
    end
    nobjs = length(recs(i).objects);
    clsinds = [];
    for iobj = 1:nobjs
        if ~isempty(strfind(modelName,recs(i).objects(iobj).class))
            clsinds(end+1) = iobj;
        end
    end
    gt(i).boxes=cat(1,recs(i).objects(clsinds).bbox)';
    gt(i).diff=[recs(i).objects(clsinds).difficult];
    gt(i).det=false(length(clsinds),1);
    npos=npos+sum(~gt(i).diff);
end

%% step 3: evaluate detection results
% assign detections to ground truth objects
nd=length(confidences);
tp=zeros(nd,1);
fp=zeros(nd,1);
tic;
for d=1:nd
    % display progress
    if toc>1
        fprintf('%s: pr: compute: %d/%d\n',modelName,d,nd);
        drawnow;
        tic;
    end
    
    % find ground truth image
    i=VOChash_lookup(hash,ids{d});
    if isempty(i)
        error('unrecognized image "%s"',ids{d});
    elseif length(i)>1
        error('multiple image "%s"',ids{d});
    end
    
    % assign detection to ground truth object if any
    bb=boxes(:,d);
    ovmax=-inf;
    for j=1:size(gt(i).boxes,2)
        bbgt=gt(i).boxes(:,j);
        bi=[max(bb(1),bbgt(1)) ; max(bb(2),bbgt(2)) ; min(bb(3),bbgt(3)) ; min(bb(4),bbgt(4))];
        iw=bi(3)-bi(1)+1;
        ih=bi(4)-bi(2)+1;
        if iw>0 & ih>0
            % compute overlap as area of intersection / area of union
            ua=(bb(3)-bb(1)+1)*(bb(4)-bb(2)+1)+...
                (bbgt(3)-bbgt(1)+1)*(bbgt(4)-bbgt(2)+1)-...
                iw*ih;
            ov=iw*ih/ua;
            if ov>ovmax
                ovmax=ov;
                jmax=j;
            end
        end
    end
    % assign detection as true positive/don't care/false positive
    if ovmax>=overlap
        if ~gt(i).diff(jmax)
            if ~gt(i).det(jmax)
                tp(d)=1;            % true positive
                if ~redetectFlag
                    gt(i).det(jmax)=true;
                end
            else
                fp(d)=1;            % false positive (multiple detection)
            end
        end
    else
        fp(d)=1;                    % false positive
    end
end

% compute precision/recall
fpSum=cumsum(fp);
tpSum=cumsum(tp);
rec=tpSum/npos;
prec=tpSum./(fpSum+tpSum);

ap=VOCap(rec,prec);
fprintf('AP = %f\n', ap);

nFrames = length(gtids);
ffpi=fpSum/nFrames;

if saveFlag
    resultDir = fileparts(resultFile);
    if ~exist(resultDir, 'dir')
	    mkdir(resultDir);
    end
    save(resultFile,'ap','rec','prec','ffpi','tp','fp','ids','boxes','confidences');
end
