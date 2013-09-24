% this code will set up the environment of the experiment
%
global location;
location = 'unc';
global year;
year = 2007;

% add the path if needed
addpath(sprintf('VOC%d/VOCdevkit/VOCcode/',year));
addpath('kmeans/kmeans_yunchao')
addpath('3rdparty/SelectiveSearchCodeIJCV/');
addpath('3rdparty/vlfeat-0.9.17/toolbox/');
addpath('3rdparty/vlfeat-0.9.17/toolbox/mex/mexa64/');
addpath('3rdparty/liblinear-1.93/matlab/');
addpath('3rdparty/flann-1.8.4-src/src/matlab')
addpath('3rdparty/flann-1.8.4-src/build/src/matlab')

% setup vlfeat
vl_setup;

% set up some global parameters
global HOMEDIR;
[~, HOMEDIR] = unix('echo $HOME');
HOMEDIR = HOMEDIR(1:end-1);
VOCinit;
