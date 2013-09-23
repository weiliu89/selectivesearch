function scores = MaxOverlapScores(gtBoxes, gtImIds, testBoxes)
% Get best overlap scores for each box in gtBoxes
% gtImIds contains for each ground truth box the index of the corresponding
% image in which it can be found. This index again corresponds
% with the testBoxes cell-array


for cI = 1:length(gtBoxes) % For all classes
    classBoxes = gtBoxes{cI}; 
    for i=1:length(classBoxes)
        testIds = gtImIds{cI}(i); % Get a single GT box
        
        % Calculate Pascal Overlap score and take best
        scores{cI}(i) = max(OverlapScores(classBoxes(i,:), testBoxes{testIds}));
    end
end