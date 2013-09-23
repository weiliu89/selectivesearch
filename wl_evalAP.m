function [ap,rec,prec,ffpi,tp,fp,ids,boxes,confidences] = wl_evalAP(modelName,dets,dataset,overlap,redetectFlag)
% wl_evalBoxes() will evaluate the detection results
% Input:
%	modelName: the name of the model. e.g.: cat, dog
%	dets: the detection results, including ids, boxes, and confidences
%	dataset: the name of the dataset. e.g.: trainval, test
%	overlap: the overlap threshold
%	redetectFlag: indicator about whether to consider duplicated detection
%

%% step 0: preprocessing
% step 0.1: return if resultFile already exist
%if exist(resultFile, 'file')
%    fprintf('%s already exist!\n', resultFile);
%    return;
%end

% step 0.2: default overlap value during evaluation
if nargin < 4
    overlap = 0.5;
    redetectFlag = false;
elseif nargin < 5
    redetectFlag = false;
else
    % get all the parameters
end

% step 0.3: set up environment
global VOCopts;
wl_setup();

%% step 1: get detection results
idsT = dets{1};
boxesT = dets{2};
confidencesT = dets{3};

% step 1.1: do nms on the detection results
hash = VOChash_init(idsT);
imgNames = unique(idsT);
nd = length(imgNames);
ids = []; boxes = []; confidences = [];
for d = 1:nd
	imgName = imgNames{d};
	idx = VOChash_lookup(hash, imgName);
	[pick, iBoxT] = wl_nms([boxesT(idx,:) confidencesT(idx)], 0.5);
	ids = [ids; idsT(idx(pick))];
	boxes = [boxes, iBoxT(:,1:4)'];
	confidences = [confidences; iBoxT(:,end)];
end
clear idsT boxesT confidencesT

% step 1.5: sort detections by decreasing confidences
[sc,si]=sort(-confidences);
ids=ids(si);
boxes=boxes(:,si);
confidences = confidences(si);

%% step 2: collect ground truth annoations
% step 2.1: load annotations
annoFile = [VOCopts.localdir 'anno_' dataset '.mat'];
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
        %fprintf('%s: pr: compute: %d/%d\n',modelName,d,nd);
        %drawnow;
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
    %       if ovmax>=VOCopts.minoverlap
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
