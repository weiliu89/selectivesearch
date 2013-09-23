function [frames, descs] = wl_extractFeatureOne(img)
% wl_extractFeatureOne() will extract feature from one image
% Input:
%   img: the input image
%   opt: the options
%

%% step 0: setup the environment
wl_setup;

%% step 1: extract the feature
binSize = 4;
step = 4;
tic
[frames, descs] = cv.FREAK(img, ...
    'Size', binSize, ...
    'Step', step);
toc