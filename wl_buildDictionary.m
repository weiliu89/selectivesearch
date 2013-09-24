function wl_buildDictionary(trainImgList, trainFeatList, K)
% wl_buildDictionary() will use images from trainImgList to build a
% dictionary of size k using given type of feature
% Input:
%   trainImgList: the training image list
%   trainFeatList: the path to store the feature of the corresponding image
%   featType: the type of the feature
%   K: the number of cluster centers
% Output:
%   dict: the output dictionary
%

%% step 0: check
wl_setup;

if ~exist(trainImgList, 'file')
    fprintf('%s does not exist!\n', trainImgList);
    return;
end
if ~exist(trainFeatList, 'file')
    fprintf('%s does not exist!\n', trainFeatList);
    return;
end
if K <= 0
    fprintf('K should be positive number!\n');
    return;
end

%% step 1: submit jobs to extract features
if false
nd = numel(textread(trainImgList,'%1c%*[^\n]'));
step = ceil(nd/100);
jobFile = sprintf('%s/selectivesearch/jobs/wl_extractFeature_batch.m', VOCopts.datadir);
wl_delete_command = sprintf('rm -rf %s*', jobFile);
unix(wl_delete_command);
fid = fopen(jobFile, 'w');
if fid == -1
    fprintf('Cannot open %s!\n', jobFile);
    return;
end
for d = 1:step:nd
    startIdx = d;
    endIdx = min(nd, d+step-1);
    fprintf(fid, 'wl_extractFeature(''%s'', ''%s'', %d, %d)\n', trainImgList, trainFeatList, startIdx, endIdx);
end
fclose(fid);
jobID = wl_submitJob(jobFile, 1);
fprintf('Submit %d\n', jobID);

%% step 2: check if the job has been finished or not
while 1
    if wl_checkJobFinished(jobID)
        break;
    end
end
end

%% step 3: start collecting the features
% read in the feature files
fid = fopen(trainFeatList);
if fid == -1
    fprintf('Cannot open %s!\n', trainFeatList);
    return;
end
C = textscan(fid, '%s');
featFiles = C{1};
clear C
fclose(fid);
nd = length(featFiles);
% load the feature
maxN = 12000000;
feats = zeros(128, maxN, 'single');
count = 1;
th = tic;
for d = 1:nd
    tic
    featFile = sprintf('%s/%s/%s', VOCopts.datadir, VOCopts.dataset, featFiles{d});
    if ~exist(featFile, 'file')
        fprintf('%s does not exist!\n', featFile);
        continue;
    end
    try
        a = load(featFile);
        % delete all 0 features
        zeroIdx = find(all(a.features.descr==0));
        if ~isempty(zeroIdx)
            a.features.descr(:, zeroIdx) = [];
        end
    catch
        fprintf('Cannot load %s\n', featFile);
        continue;
    end
    nf = size(a.features.descr, 2);
    feats(:,count:count+nf-1) = a.features.descr;
    count = count+nf;
    fprintf('Load %s time: %.2f\n', featFile, toc);
end
if count < maxN
    feats(:, count:maxN) = [];
end
% step 2.1: RootSIFT
feats = sqrt(feats);
% step 2.2: L2 normalization
feats = bsxfun(@times, feats, 1./max(1e-5, sqrt(sum(feats.^2,1))));
%save('data/feature.mat', 'feats', '-v7.3');
fprintf('Collect %d features, with %.2f sec!\n', count, toc(th));

%% step 4: do kmeans
iter = 5;
codebook = FLANN_Kmeans(feats, K, iter, []);
dictFile = 'data/codebook.mat';
save(dictFile, 'codebook', '-v7.3');
