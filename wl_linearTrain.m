function linearModel = wl_linearTrain(feats, labels, boxes, ids, modelName)
% wl_linearTrain() will train a liblinear model using the given features
% and labels
% Input:
%   feats: feature vector Nxp
%   labels: label vector Nx1, either 1 or -1
%   boxes: the correlated boxes
%   ids: the correlated image names
%   modelName: the name of the model. e.g.: cat, dog
% Output:
%   linearModel: the final liblinear model
%

%% step 0: set up the environment
global VOCopts
wl_setup;

% step 1: partition the data into k partitions
k = 3;
cvIds = wl_cvIds(ids, labels, k);

% step 2: do cross validation on a given set of parameters
bestC = 1;
bestAP = 0;
cs = [0.001 0.01 0.1 1 10 100];
%cs = [0.1 0.5 1 2 4 6 8 10 15 20];
%cs = 1:0.5:10
%cs = [1.6 1.8 2 2.2 2.4 2.6 2.8 3 3.2 3.4 3.6 3.8 4 4.2 4.4 4.6 4.8 5 5.2 5.4 5.6 5.8 6 6.2 6.4 6.6 6.8 7 7.2 7.4 7.6 7.8];
nc = length(cs);
for ci = 1:nc
    c = cs(ci);
    fprintf('%s: c %f\n', modelName, c);
    ids2 = []; boxes2 = []; confidences2 = [];
    for i = 1:k
        % step 2.1: get the train and test indices
        trainIds = [];
        for j = 1:k
            if j ~= i
                trainIds = [trainIds; cvIds{j}];
            end
        end
        testIds = cvIds{i};
        % step 2.2: randomly drop out 50% of the features with two copies
        %[trainFeats, trainLabels] = wl_randDropout(feats(trainIds,:),labels(trainIds));
        w1 = sqrt(sum(labels(trainIds)~=1)/sum(labels(trainIds)==1));
        lineartrainopt = sprintf('-s 3 -c %f -w1 %f -w-1 1 -B 1 -q', c, w1);
        linearModel = train(labels(trainIds), feats(:,trainIds), lineartrainopt, 'col');
        %clear trainFeats trainLabels
        % step 2.3: evaluate on the testing elements
        lineartestopt = '-b 0 -q';
        [plabels, acc, pscores] = predict(labels(testIds), feats(:,testIds), linearModel, lineartestopt, 'col');
        % step 2.3.1: adjust the scores
        pscores = pscores*linearModel.Label(1);
        % step 2.4: gather the detection results
        ids2 = [ids2; ids(testIds)];
        boxes2 = [boxes2; boxes(testIds,:)];
        confidences2 = [confidences2; pscores];
    end
    % step 2.5: compute the AP value for the detection results
    dets{1} = ids2; dets{2} = boxes2; dets{3} = confidences2;
    ap = wl_evalAP(modelName, dets, VOCopts.trainset);
    if ap > bestAP
        bestAP = ap;
        bestC = c;
    end
end

% step 3: use the bestC to train the final model
% step 3.1: randomly drop out 50% of the features with two copies
%[trainFeats, trainLabels] = wl_randDropout(feats,labels);
w1 = sqrt(sum(labels~=1)/sum(labels==1));
linearopt = sprintf('-s 3 -c %f -w1 %f -w-1 1 -B 1 -q', bestC, w1);
linearModel = train(labels, feats, linearopt, 'col');
linearModel.modelName = modelName;
linearModel.bestC = bestC;
linearModel.w1 = w1;
fprintf('bestC: %f\n', bestC);
% clear trainFeats trainLabels

function cvIds = wl_cvIds(ids, labels, k)
% wl_cvIds() will partition the labels into k parts with equally
% number of images
% Input:
%	ids: the image name for all the features
%	labels: the label for all the features
%	k: the number of partitions
%

% step 1: hash the image names
hash = VOChash_init(ids);

% step 2: get the unique name of the images
imgNames = unique(ids);

% step 2.1: get the positive image names
posImgNames = unique(ids(labels==1));
nPosImgs = length(posImgNames);

% step 2.2: get the negative image names
negImgNames = setdiff(imgNames, posImgNames);
nNegImgs = length(negImgNames);

% step 3: randomly split the positive image names
if nPosImgs ~= 0
    % step 3.1: randomly permute the postive image names
    posImgNames = posImgNames(randperm(nPosImgs));
    % step 3.2: split the image names into k parts
    n = floor(nPosImgs/k);
    count = 0;
    i = 1;
    cvIds{i} = [];
    for d=1:nPosImgs
        imgName = posImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
        else
            cvIds{i} = [cvIds{i}; idx'];
            i = i+1;
            count = 0;
            cvIds{i} = [];
        end
    end
end

% step 4: randomly split the negative image names
if nNegImgs ~= 0
    % step 4.1: randomly permute the negative image names
    negImgNames = negImgNames(randperm(nNegImgs));
    % step 3.2: split the image names into k parts
    n = floor(nNegImgs/k);
    count = 0;
    i = 1;
    for d=1:nNegImgs
        imgName = negImgNames{d};
        count = count + 1;
        % step 4.1: get the indices for the image name
        idx = VOChash_lookup(hash, imgName);
        if count < n || i==k
            cvIds{i} = [cvIds{i}; idx'];
        else
            cvIds{i} = [cvIds{i}; idx'];
            i = i+1;
            count = 0;
        end
    end
end

function [trainFeats,trainLabels] = wl_randDropout(feats,labels)
% wl_randDropout() will randomly dropout 50% of the feature to 0
% Input:
%	feats: the input feature, Nxp dimension, where N is the
%		number of features, and p is the feature dimension
%   labels: the input label, Nx1 dimension
% Output:
%	trainFeats: every element of feats has 50% to set to zero, two random copy
%   trainLabels: double the training label
%

dropoutMat1 = sprand(feats)>=0.5;
dropoutMat2 = sprand(feats)>=0.5;
trainFeats = [feats.*dropoutMat1; feats.*dropoutMat2];
trainLabels = [labels; labels];
clear dropoutMat1 dropoutMat2
