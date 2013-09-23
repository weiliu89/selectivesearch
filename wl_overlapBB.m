function ov = wl_overlapBB(bb1,bb2)
% wl_overlapBB() will compute the overlap between two bounding boxes
% Input:
%   bb1: [x1_min y1_min x1_max y1_max]
%   bb2: [x2_min y2_min x2_max y2_max]
%

bi=[max(bb1(1),bb2(1)), max(bb1(2),bb2(2)), min(bb1(3),bb2(3)), min(bb1(4),bb2(4))];
iw=bi(3)-bi(1)+1;
ih=bi(4)-bi(2)+1;
ov = 0;
if iw>0 && ih>0
    % compute overlap as area of intersection / area of union
    ua=(bb1(3)-bb1(1)+1)*(bb1(4)-bb1(2)+1)+...
        (bb2(3)-bb2(1)+1)*(bb2(4)-bb2(2)+1)-...
        iw*ih;
    ov=iw*ih/ua;
end