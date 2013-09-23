function [dic] = ConstructKmeansCodebook(K)

% number of sample images
nums = 3500;
iters = 10;

% load filenames
data_dir = 'D:\landmark images\Code\Image_Feature\data\';
fnames = dir(fullfile(data_dir, '*.mat'));
num_files = size(fnames,1);
filenames = cell(num_files,1);
for f = 1:num_files
	filenames{f} = fnames(f).name;
end


% load random data patches
rand = randperm(length(filenames));
X = [];
for i=1:nums
    i
    name = strcat(data_dir, filenames(rand(i)));
    xx = load(name{1});
    X = [X, xx.y];
end
size(X)


% perform approximate kmeans
[dic] = FLANN_Kmeans(X, K, iters);



% save data
save -v7.3 'dictionary' dic;




















