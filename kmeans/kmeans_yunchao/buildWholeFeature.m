function buildWholeFeature(filenames, dic, indx)


data_dir = 'F:\High_DIM_Landmark_Data\ImageNet_Image_Feature\SIFT\';

dic = single(dic);
filenames = filenames(indx);


% build kd tree index
tic;
index = flann_build_index(dic, struct('algorithm', 'kdtree', 'trees', 10));
toc


% start the building
for i=1:length(filenames)
    i
    % load each data separately
    name = strcat(data_dir, filenames(i));
    xx = load(name{1});

    % learn visual feature
    [v] = buildLargeVisualCodebook(single(xx.y), dic, index);
    
    % store
    savestr = sprintf('F:\\High_DIM_Landmark_Data\\ImageNet_Image_Feature\\BOW\\%s', filenames{i});
    save(savestr, 'v');
end


























