function [gtids, recs] = wl_getAnnotation(dataset, outFile)
% wl_getAnnotation() will get the annotation for dataset
% Input:
%	dataset: the name of the dataset. e.g.: val, test
%	outFile: the file path to store the annotation
% Output:
%	gtids: the id of the ground truth file
%	recs: the annotation of the ground truth
%

%% step 1: get annoations
if ~exist(outFile, 'file')
    % step 1.0: set up the environment
    global VOCopts;
    wl_setup();
    
    % step 1.1: read the annotation file list
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
    ids = textscan(fid, '%s');
    ids = ids{1};
    fclose(fid);
    
    % step 1.2: read annoations
    nd = length(ids);
    gtids = [];
    for d = 1:nd
        id = ids{d};
        annoPath = sprintf(VOCopts.annopath, id);
        if ~exist(annoPath, 'file')
            fprintf('%s does not exist!\n', annoPath);
            continue;
        end
        gtids{d} = id;
        recs(d) = PASreadrecord(annoPath);
    end
    save(outFile, 'gtids', 'recs');
end
