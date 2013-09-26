function wl_trainModel(modelName)
% wl_trainModel() will train a model using selective search method
% Reference:
%   "Segmentation As Selective Search for Object Recognition", ICCV 2011
%   by Koen E. A. van de Sande, Jasper R. R. Uijlings, Theo Gevers, Arnold W. M. Smeulders
%
% Input:
%	modelName: the name of the model. e.g.: cat, dog
%

%% step 0: set up the environment
global VOCopts
wl_setup;

%% step 1: train a liblinear model with several iterations
featsAll = []; labelsAll = []; boxesAll = []; idsAll = [];
for iter = 1:2
    modelIterFile = [VOCopts.resdir 'models/' modelName '_iter' num2str(iter) '.mat'];
    modelDir = fileparts(modelIterFile);
    if ~exist(modelDir, 'dir')
        mkdir(modelDir);
    end
    if ~exist(modelIterFile, 'file')
        % step 1.1: get training bounding boxes
        bbFile = [VOCopts.resdir 'trainBB/' modelName '_iter' num2str(iter) '.txt'];
        th = tic;
        if iter == 1
            % step 1.1.1: get initial training bounding boxes
            wl_getSSTrainBB(modelName, bbFile);
            fprintf('Iter%d wl_getSSTrainBB() time: %f\n', iter, toc(th));
            
            % step 1.1.2: read the box information
            fid = fopen(bbFile, 'r');
            % [bbName label x_min y_min x_max ymax]
            bbsInfo = textscan(fid, '%s %d %d %d %d %d');
            fclose(fid);
            
            % step 1.1.3: get the features for given bounding boxes
            th = tic;
            [feats,labels,boxes,ids] = wl_getSSFeat(bbsInfo);
            featsAll = [featsAll, feats];
            labelsAll = [labelsAll; labels];
            boxesAll = [boxesAll; boxes];
            idsAll = [idsAll(:); ids(:)];
            clear bbsInfo feats labels boxes ids
            fprintf('%s Iter%d wl_getSSFeat() time: %f\n', modelName, iter, toc(th));
        else
            % step 1.1.2: harvest the hard negative bounding boxes
            % step 1.1.2.1: get the model from previous iteration
            prevModelFile = [VOCopts.resdir 'models/' modelName '_iter' num2str(iter-1) '.mat'];
            if ~exist(prevModelFile, 'file')
                fprintf('%s does not exist!\n', prevModelFile);
                break;
            end
            load(prevModelFile);
            % get topK detections from each negative image
            topK = 5;
            [feats,labels,boxes,ids] = wl_getSSHardBB(linearModel, bbFile, topK);
            featsAll = [featsAll, feats];
            labelsAll = [labelsAll; labels];
            boxesAll = [boxesAll; boxes];
            idsAll = [idsAll(:); ids(:)];
            clear bbsInfo feats labels boxes ids linearModel
            fprintf('Iter%d wl_getSSHardBB() time: %f\n', iter, toc(th));
        end
        
        % step 1.2: train the model using all the feature data
        th = tic;
        linearModel = wl_linearTrain(featsAll,labelsAll,boxesAll,idsAll,modelName);
        linearModel.iter = iter;
        fprintf('%s Iter%d wl_linearTrain() time: %f\n', modelName, iter, toc(th));
        save(modelIterFile, 'linearModel');
        
        %% step 2: detect objects
        if ~exist([VOCopts.jobdir 'detections/'], 'dir')
            mkdir([VOCopts.jobdir 'detections/']);
        end
        jobFile = [VOCopts.jobdir 'detections/' modelName '_iter' num2str(iter) '.m'];
        % step 2.1: delete the job file
        wl_delete_command = sprintf('rm -rf %s*', jobFile);
        unix(wl_delete_command);
        fid = fopen(jobFile, 'w');
        if fid == -1
            fprintf('Cannot open %s!\n', jobFile);
            continue;
        end
        if VOCopts.year == 2007
            fprintf(fid, 'wl_detectObject(''%s'',''test'', %d, 1, 4952, 1)\n',modelName, iter);
        else
            fprintf(fid, 'wl_detectObject(''%s'',''test'', %d, 1, 10991, 1)\n',modelName, iter);
        end
        fclose(fid);
        if strcmp(location, 'unc')
            wl_config_command = sprintf('$HOME/bin/configArrayKilldevilJob.sh %s.array %s $HOME/projects/pascal/selectivesearch/ 1 1 10 4', jobFile, jobFile);
            unix(wl_config_command);
            wl_qsub_command = sprintf('bsub < %s.array', jobFile);
            unix(wl_qsub_command);
        elseif strcmp(location, 'sbu')
            wl_config_command = sprintf('$HOME/bin/configArrayJob.sh %s.array %s $HOME/projects/pascal/selectivesearch/ 1 1 0 1 1', jobFile, jobFile);
            unix(wl_config_command);
            wl_qsub_command = sprintf('ssh wliu@bigeye.cs.stonybrook.edu ''qsub %s.array''', jobFile);
            unix(wl_qsub_command);
        else
            fprintf('Unknown location: %s!\n', location);
        end
    end
end
