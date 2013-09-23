function scores = OverlapScores(gtBox, testBoxes)
% Get pascal overlap scores. gtBox versus all testBoxes


gtBoxes = repmat(gtBox, size(testBoxes,1), 1);
intersectBoxes = BoxIntersection(gtBoxes, testBoxes);
overlapI = intersectBoxes(:,1) ~= -1; % Get which boxes overlap

% Intersection size
[nr nc intersectionSize] = BoxSize(intersectBoxes(overlapI,:));

% Union size
[nr nc testBoxSize] = BoxSize(testBoxes(overlapI,:));
[nr nc gtBoxSize] = BoxSize(gtBox);
unionSize = testBoxSize + gtBoxSize - intersectionSize;

scores = zeros(size(testBoxes,1),1);
scores(overlapI) = intersectionSize ./ unionSize;
