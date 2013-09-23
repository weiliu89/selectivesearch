function wl_evalModel(modelName,iter,reEvalFlag)
% wl_evalModel() will evaluate the model with modelName's performance
%
% Input:
%   modelName: the name of the model. e.g.: cat, dog
%   iter: the iteration number
%   reEvalFlag: re-evaluate flag
%

if nargin < 3
    reEvalFlag = false;
end

%% step 0: set up the environment
global VOCopts;
wl_setup;

%% step 1: evaluate the performance of model
detDir = [VOCopts.resdir 'detections/'];
apFile = [VOCopts.resdir 'aps/' modelName '_iter' num2str(iter) '.mat'];
if ~exist(apFile, 'file') || reEvalFlag
    % step 1.1: get all the detection result for the model
    evalFileList = sprintf(VOCopts.clsimgsetpath,modelName,VOCopts.testset);
    if ~exist(evalFileList, 'file')
        fprintf('%s does not exist!\n', evalFileList);
        return;
    end
    fid = fopen(evalFileList);
    if fid == -1
        fprintf('Cannot open %s!\n', evalFileList);
        return;
    end
    C = textscan(fid, '%s %d');
    ids = C{1};
    labels = C{2};
    nd = length(ids);
    fclose(fid);
    
    % step 1.2: get the ground truth bounding boxes
    gtBBFile = [VOCopts.localdir 'anno_' VOCopts.testset '.mat'];
    if ~exist(gtBBFile, 'file')
        [gtids,recs] = wl_getAnnotation(VOCopts.testset, gtBBFile);
    end
    
    % step 1.3: get the detected bounding boxes for the model
    detAllFile = [VOCopts.resdir 'detections/' modelName '_iter' num2str(iter) '.txt'];
    if ~exist(detAllFile, 'file') || reEvalFlag
        fid = fopen(detAllFile, 'w');
        if fid == -1
            fprintf('Cannot open %s!\n', detAllFile);
            return;
        end
        detMatFile = [VOCopts.resdir 'detections/' modelName '/' modelName '_iter' num2str(iter) '.mat'];
        if ~exist(detMatFile, 'file')
            for d = 1:nd
                id = ids{d};
                % step 1.3.1: get one detection result of a image
                detFile = [detDir modelName '/' id '_iter' num2str(iter) '.mat'];
                if ~exist(detFile, 'file')
                    fprintf('%s does not exist!\n', detFile);
                    continue;
                end
                load(detFile);
                
                % step 1.3.2: save the detection results
                nBBs = size(boxes,1);
                for iBB = 1:nBBs
                    fprintf(fid, '%s %d %d %d %d %f\n', id, boxes(iBB,:));
                end
            end
        else
            load(detMatFile);
            for d = 1:nd
                id = ids{d};
                boxes = boxesAll{d};
                % step 1.3.2: save the detection results
                nBBs = size(boxes,1);
                for iBB = 1:nBBs
                    fprintf(fid, '%s %d %d %d %d %f\n', id, boxes(iBB,:));
                end
            end
        end
        fclose(fid);
    end
    
    %% step 2: compute ap value for current class
    [ap,rec,prec,ffpi,tp,fp,ids,boxes,confidences] = wl_evalBoxes(modelName,detAllFile,gtBBFile,apFile);
end
