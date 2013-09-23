% This demo shows how to use the software described in our ICCV paper: 
%   Segmentation as Selective Search for Object Recognition,
%   K.E.A. van de Sande, J.R.R. Uijlings, T. Gevers, A.W.M. Smeulders, ICCV 2011
%%

fprintf('Demo of how to run the code for:\n');
fprintf('   K. van de Sande, J. Uijlings, T. Gevers, A. Smeulders\n');
fprintf('   Segmentation as Selective Search for Object Recognition\n');
fprintf('   ICCV 2011\n\n');

% Compile anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compiling the anisotropic gauss filtering of:\n');
    fprintf('   J. Geusebroek, A. Smeulders, and J. van de Weijer\n');
    fprintf('   Fast anisotropic gauss filtering\n');
    fprintf('   IEEE Transactions on Image Processing, 2003\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://staff.science.uva.nl/~mark/downloads.html#anigauss\n\n');
    mex anigaussm/anigauss_mex.c anigaussm/anigauss.c -output anigauss
end


% Compile the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm of:\n');
    fprintf('   P. Felzenszwalb and D. Huttenlocher\n');
    fprintf('   Efficient Graph-Based Image Segmentation\n');
    fprintf('   International Journal of Computer Vision, 2004\n');
    fprintf('Source code/Project page:\n');
    fprintf('   http://www.cs.brown.edu/~pff/segment/\n');
    fprintf('Note: A small Matlab wrapper was made. See demo.m for usage\n\n');
%     fprintf('   
    mex FelzenSegment/mexFelzenSegmentIndex.cpp -output mexFelzenSegmentIndex;
end

%%
% Parameters. Note that this controls the number of hierarchical
% segmentations which are combined.
colorTypes = {'Rgb', 'Hsv', 'RGI', 'Opp'};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
kThresholds = [100 200]; 
sigma = 0.8;
numHierarchy = length(colorTypes) * length(kThresholds);

% As an example, use a single Pascal VOC image
images = {'000015.jpg'};

%%%
%%% Alternatively, do it on the whole set. (Un)comment line 67/68
%%%
% VOCinit;
% theSet = 'test'
% [images, labs] = textread(sprintf(VOCopts.imgsetpath, theSet), '%s %s');

% For each image do Selective Search
fprintf('Performing selective search: ');
tic;
boxes = cell(1, length(images));
for i=1:length(images)
    if mod(i,100) == 0
        fprintf('%d ', i);
    end
    idx = 1;
    currBox = cell(1, numHierarchy);
    im = imread(images{i});
    %im = imread(sprintf(VOCopts.imgpath, images{i})); % For Pascal Data
    for k = kThresholds
        minSize = k; % We use minSize = k.
        
        for colorTypeI = 1:length(colorTypes)
            colorType = colorTypes{colorTypeI};
            
            currBox{idx} = SelectiveSearch(im, sigma, k, minSize, colorType);
            idx = idx + 1;
        end
    end
    
    boxes{i} = cat(1, currBox{:}); % Concatenate results of all hierarchies
    boxes{i} = unique(boxes{i}, 'rows'); % Remove duplicate boxes
end
fprintf('Elapsed time: %f seconds\n', toc);

%% Show a couple of good boxes in the image
fprintf('Showing examples of good boxes\n');
goodBoxes = boxes{1}([48 1075 808 762 467 445], :);
figure; 
for i=1:6
    subplot(2, 3, i);
    boxIm = im(goodBoxes(i,1):goodBoxes(i,3), goodBoxes(i,2):goodBoxes(i,4), :);
    imshow(boxIm);
end

%%
% Test overlap scores Pascal 2007 test
if exist('SelectiveSearchVOC2007test.mat')
    load GroundTruthVOC2007test.mat; % Load ground truth boxes
    load SelectiveSearchVOC2007test.mat; % Load selective search boxes

    % Remove small boxes
    for i=1:length(boxes)
        [nR nC] = BoxSize(boxes{i});
        keepIdx = min(nR, nC) > 20;% Keep boxes with width/height > 20 pixels
        boxes{i} = boxes{i}(keepIdx,:);
        numberBoxes(i) = size(boxes{i},1);
    end

    % Get for each ground truth box the best Pascal Overlap Score
    maxScores = MaxOverlapScores(gtBoxes, gtImIds, boxes);

    % Get recall per class
    for cI=1:length(maxScores)
        recall(cI) = sum(maxScores{cI} > 0.5) ./ length(maxScores{cI});
        averageBestOverlap(cI) = mean(maxScores{cI});
    end

    recall
    fprintf('Number of boxes per image: %.0f\nMean Average Best Overlap: %f\nMean Recall: %f\n', mean(numberBoxes), mean(averageBestOverlap), mean(recall));
end
    
%% Example of segmentation
% sigma = 0.8
% k = 100
% minSize = 200
segIndIm = mexFelzenSegmentIndex(im, 0.8, 100, 200);

% segIndIm has the same number of rows and columns as im. The range of 
% segIndIm is 1:S, where S is the number of segments: Each number
% in segIndIm corresponds to the segment the pixel belongs to.
