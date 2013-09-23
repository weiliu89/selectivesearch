% this code will set up the environment of the experiment
%
global year;
year = 2007;

% add the path if needed
addpath(sprintf('VOC%d/VOCdevkit/VOCcode/',year));
addpath('SelectiveSearchPcode/');
addpath('vlfeat-0.9.17/toolbox/');
addpath('vlfeat-0.9.17/toolbox/mex/mexa64/');
addpath('liblinear-1.91/matlab/');
addpath('/home/wliu/projects/mexopencv/');

addpath('kmeans/kmeans_yunchao')
addpath('kmeans/flann-1.8.4-src/src/matlab')
addpath('kmeans/flann-1.8.4-src/build/src/matlab')

% set up vlfeat
vl_setup;

% set up some global parameters
VOCinit;
