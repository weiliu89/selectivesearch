function jboxes = wl_jitterBB(box, imH, imW, n)
% wl_jitterBB() will jitter the bounding box to create more boxes
% Input:
%	box: [left top right bottom]
%	imH: the height of the image
%	imW: the width of the image
%	n: number of jittered boxes
% Ouput:
%	jboxes: [left top right bottom] nx4 matrices
%

% enlarge the bounding box by 10%
boxW = box(3)-box(1)+1;
boxH = box(4)-box(2)+1;
min_x = max(box(1)-boxW/10, 1);
min_y = max(box(2)-boxH/10, 1);

% specify the possible size of the jittered box
min_jboxW = boxW*0.9;
min_jboxH = boxH*0.9;
max_jboxW = boxW*1.1;
max_jboxH = boxH*1.1;
min_min_jx = min_x;
max_min_jx = min_x+boxW/10;
min_min_jy = min_y;
max_min_jy = min_y+boxH/10;

% random generate several jittered bounding boxes
min_jxx = min_min_jx + (max_min_jx-min_min_jx).*rand(n,1);
min_jyy = min_min_jy + (max_min_jy-min_min_jy).*rand(n,1);
jboxWW = min_jboxW + (max_jboxW-min_jboxW).*rand(n,1);
jboxHH = min_jboxH + (max_jboxH-min_jboxH).*rand(n,1);

%% get the jittered bounding boxes
jboxes = round([min_jxx min_jyy min_jxx+jboxWW-1 min_jyy+jboxHH-1]);
jboxes(:,1) = max([jboxes(:,1), ones(n, 1)], [], 2);
jboxes(:,2) = max([jboxes(:,2), ones(n, 1)], [], 2);
jboxes(:,3) = min([jboxes(:,3), repmat(imW, n, 1)], [], 2);
jboxes(:,4) = min([jboxes(:,4), repmat(imH, n, 1)], [], 2);
