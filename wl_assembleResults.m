function wl_assembleResults(dataset,cls,iter)
% wl_assembleResults() will assemble results of detection results of cls
% Input:
%   dataset: the name of the dataset, e.g.: val, test
%   cls: the name of the class, e.g.: cat, dog
%   iter: the number of iteration
%

%% step 0: setup the environment
global VOCopts;
wl_setup();

%% step 1: get all the required information
% step 1.1: get the names for the test images
fileList = sprintf(VOCopts.imgsetpath, dataset);
if ~exist(fileList, 'file')
    fprintf('%s does not exist!\n', fileList);
    return;
end
fid = fopen(fileList);
if fid == -1
    fprintf('Cannot open %s!\n', fileList);
    return;
end
C = textscan(fid, '%s');
ids = C{1};
clear C
fclose(fid);

% step 1.2: get the detection directory
detDir = sprintf('%s/detections/%s/', VOCopts.resdir, cls);
if ~exist(detDir, 'dir')
    fprintf('%s does not exist!\n', detDir);
    return;
end

% step 1.3: get the output file name
outDir = sprintf('%s/submissions/iter%d/results/VOC2012/Main', VOCopts.resdir, iter);
if ~exist(outDir, 'dir')
    mkdir(outDir);
end
outFile = sprintf('%s/comp3_det_%s_%s.txt', outDir, dataset, cls);
if ~exist(outFile, 'file')
    th = tic;
    % step 1.4: open output file to write results
    fid = fopen(outFile, 'w');
    if fid == -1
        fprintf('Cannot open %s!\n', outFile);
        return;
    end
    %% step 2: assemble all the results
    detAllFile = sprintf('%s/%s_iter%d.mat', detDir, cls, iter);
    if ~exist(detAllFile, 'file')
        nd = length(ids);
        for d = 1:nd
            % step 2.1: load the detection file
            detFile = sprintf('%s/%s_iter%d.mat', detDir, ids{d}, iter);
            if ~exist(detFile, 'file')
                fprintf('%s does not exist!\n', detFile);
                continue;
            end
            load(detFile);
            % step 2.2: write the detection file
            nBoxes = size(boxes,1);
            for i = 1:nBoxes
                fprintf(fid, '%s %f %d %d %d %d\n', ids{d}, boxes(i,end), boxes(i,1:4));
            end
        end
    else
        % step 2.1: load all the detection results
        load(detAllFile);
        nd = length(boxesAll);
        for d = 1:nd
            boxes = boxesAll{d};
            % step 2.2: write the detection file
            nBoxes = size(boxes,1);
            for i = 1:nBoxes
                fprintf(fid, '%s %f %d %d %d %d\n', ids{d}, boxes(i,end), boxes(i,1:4));
            end
        end
    end
    % step 2.3: finish writing
    fclose(fid);
    fprintf('Finish output results of %s: %f\n', cls, toc(th));
end