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
            wl_getSSHardBB(linearModel, bbFile, topK);
            clear linearModel
            fprintf('Iter%d wl_getSSHardBB() time: %f\n', iter, toc(th));
        end
        
        % step 1.2: get previous the feature data
        if iter > 1 && isempty(featsAll)
            for jIter = 1:iter-1
                % step 1.2.1: compute the feature if it does not exist!
                % step 1.2.1.1: get the bounding boxes
                prevBBFile = [VOCopts.resdir 'trainBB/' modelName '_iter' num2str(jIter) '.txt'];
                if ~exist(prevBBFile, 'file')
                    fprintf('%s does not exist!\n', prevBBFile);
                    break;
                end
                fid = fopen(prevBBFile, 'r');
                % [bbName label x_min y_min x_max ymax]
                prevBBsInfo = textscan(fid, '%s %d %d %d %d %d');
                fclose(fid);
                
                % step 1.2.1.2: compute the features for the bounding boxes
                th = tic;
                [feats, labels, boxes, ids] = wl_getSSFeat(prevBBsInfo);
                fprintf('%s Iter%d wl_getSSFeat() time: %f\n', modelName, jIter, toc(th));
                % step 1.2.2: aggregate the feature data
                if ~isempty(labels)
                    featsAll = [featsAll, feats];
                    labelsAll = [labelsAll; labels];
                    boxesAll = [boxesAll; boxes];
                    idsAll = [idsAll(:); ids(:)];
                end
                % step 1.2.3: clean up
                clear feats labels boxes ids
            end
        end
        
        % step 1.3: remove correctly classified negative samples
        if iter > 1 && ~isempty(featsAll) && false
            % step 1.3.1: get the previous model
            th = tic;
            prevModelFile = [VOCopts.resdir 'models/' modelName '_iter' num2str(iter-1) '.mat'];
            if ~exist(prevModelFile, 'file')
                fprintf('%s does not exist!\n', prevModelFile);
                continue;
            end
            load(prevModelFile);
            % step 1.3.2: remove some negative samples
            [featsAll, labelsAll, goodIdx] = wl_removeSSFeat(featsAll,labelsAll,linearModel);
            boxesAll = boxesAll(goodIdx, :);
            idsAll = idsAll(goodIdx);
            fprintf('%s Iter%d wl_removeSSFeat() time: %f\n', modelName, iter, toc(th));
        end
        
        % step 1.4: read the box information
        fid = fopen(bbFile, 'r');
        % [bbName label x_min y_min x_max ymax]
        bbsInfo = textscan(fid, '%s %d %d %d %d %d');
        fclose(fid);
        
        % step 1.5: get the features for given bounding boxes
        th = tic;
        [feats,labels,boxes,ids] = wl_getSSFeat(bbsInfo);
        featsAll = [featsAll, feats];
        labelsAll = [labelsAll; labels];
        boxesAll = [boxesAll; boxes];
        idsAll = [idsAll(:); ids(:)];
        clear bbsInfo feats labels boxes ids
        fprintf('%s Iter%d wl_getSSFeat() time: %f\n', modelName, iter, toc(th));
        
        % step 1.6: train the model using all the feature data
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
