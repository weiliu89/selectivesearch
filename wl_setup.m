% this code will set up the environment of the experiment
%
global year;
year = 2007;

% add the path if needed
addpath(sprintf('VOC%d/VOCdevkit/VOCcode/',year));
addpath('SelectiveSearchPcode/');
addpath('vlfeat-0.9.14/toolbox/');
addpath('vlfeat-0.9.14/toolbox/mex/mexa64/');
addpath('liblinear-1.91/matlab/');
addpath('kmeans/kmeans_yunchao')
addpath('kmeans/flann-1.8.4-src/src/matlab')
addpath('kmeans/flann-1.8.4-src/build/src/matlab')

% set up some global parameters
VOCinit;
