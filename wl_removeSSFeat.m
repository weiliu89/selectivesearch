function [feats,labels,goodIdx] = wl_removeSSFeat(feats,labels,linearModel)
% wl_removeSSFeat() will remove all the negative samples which can be
% correctly predicted
% Input:
%   feats: the feature vector. pxn
%   labels: the label. nx1
%   linearModel: the model trained using liblinear
% Output:
%   feats: the features after removal
%   labels: the corresponding labels after removal
%   goodIdx: the corresponding index which are reserved
%

%% step 0: set up the environment
wl_setup;

%% step 1: predict on the features
linearopt = '-b 0';
[plabels,acc,pscores] = predict(labels,feats,linearModel,linearopt,'col');

% step 1.1: find the feature to keep
goodIdx = find(labels == 1 | pscores.*labels*linearModel.Label(1)<=1);

% step 1.2: preserve the positive features and negative features which are
% predicted wrong
feats = feats(:,goodIdx);
labels = labels(goodIdx);
