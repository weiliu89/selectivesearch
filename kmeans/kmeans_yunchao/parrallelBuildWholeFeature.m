function parrallelBuildWholeFeature(filenames, dic)

% load the data names
% data_dir = 'F:\High_DIM_Landmark_Data\ImageNet_Image_Feature\SIFT\';
% fnames = dir(fullfile(data_dir, '*.mat'));
% num_files = size(fnames,1);
% filenames = cell(num_files,1);
% for f = 1:num_files
% 	filenames{f} = fnames(f).name;
% end


parfor i=1:10
    buildWholeFeature(filenames, dic, (i-1)*100000+1:i*100000);
end

