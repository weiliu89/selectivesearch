function wl_plotICCV()
% wl_plotICCV() will plot the figures used to illustrate the framework in
% ICCV paper
%

%% step 0: set up the environment
global VOCopts
wl_setup;

gtcolor = 'g';
selcolor = 'b';
hardcolor = 'r';
linewidth = 3;
replot = true;

%% step 1: plot the selective search boxes
typeName = 'bicycle';
% step 1.1: load the imageset ids
imgsetFile = sprintf(VOCopts.imgsetpath, [typeName '_trainval']);
if ~exist(imgsetFile, 'file')
    fprintf('%s does not exist!\n', imgsetFile);
    return;
end
fid = fopen(imgsetFile);
if fid == -1
    fprintf('Cannot open %s!\n', imgsetFile);
    return;
end
C = textscan(fid, '%s %d');
ids = C{1};
labels = C{2};
clear C
fclose(fid);
% step 1.2: plot selective search boxes on positive images
posIdx = find(labels==1);
posIds = ids(posIdx);
selPos = [33 37];
nSelBoxes = 150;
for d = selPos
    id = ids{posIdx(d)};
    figFile = sprintf('%s/iccv/%s_selbox.eps', VOCopts.resdir, id);
    if ~exist(figFile, 'file') || replot
        % step 1.2.1: read the image
        imgFile = sprintf(VOCopts.imgpath, id);
        if ~exist(imgFile, 'file')
            fprintf('%s does not exist!\n', imgFile);
            continue;
        end
        img = imread(imgFile);
        figure(1); clf; imshow(img); hold on;
        origFile = sprintf('%s/iccv/%s.eps', VOCopts.resdir, id);
        print('-depsc2', origFile);
        % step 1.2.2: load the selective search boxes
        featFile = sprintf(VOCopts.featpath, id);
        if ~exist(featFile, 'file')
            fprintf('%s does not exist!\n', featFile);
            continue;
        end
        load(featFile, 'boxes');
        % step 1.2.3: plot some of the selective boxes
        selIdx = randperm(size(boxes,1));
        for i = selIdx(1:nSelBoxes)
            rect = [boxes(i,1:2), boxes(i,3:4)-boxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', selcolor, 'LineWidth', linewidth);
        end
        % step 1.2.4: print the result
        print('-depsc2', figFile);
    end
end
% step 1.3: plot ground truth box and init negative boxes
% step 1.3.1: read in the trainBB file
bbFileInit = sprintf('%s/trainBB/%s_iter1.txt', VOCopts.resdir, typeName);
if ~exist(bbFileInit, 'file')
    fprintf('%s does not exist!\n', bbFileInit);
    return;
end
fid = fopen(bbFileInit);
if fid == -1
    fprintf('Cannot open %s!\n', bbFileInit);
    return;
end
C = textscan(fid, '%s %d %d %d %d %d');
initids = C{1};
initlabels = C{2};
initboxes = [C{3} C{4} C{5} C{6}];
clear C
fclose(fid);
topK = 10;
for d = selPos
    id = ids{posIdx(d)};
    % step 1.3.2: read the image
    imgFile = sprintf(VOCopts.imgpath, id);
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        continue;
    end
    img = imread(imgFile);
    % step 1.3.3: plot image with ground truth box
    figFile = sprintf('%s/iccv/%s_init.eps', VOCopts.resdir, id);
    if ~exist(figFile, 'file') || replot
        figure(1); clf; imshow(img); hold on;
        % step 1.3.3.1: get the top5 negative boxes
        isel = find(strcmp(initids, id) & initlabels==-1);
        isel = isel(1:min(topK,length(isel)));
        % step 1.3.3.2: plot the negative boxes
        for i = isel'
            rect = [initboxes(i,1:2), initboxes(i,3:4)-initboxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', hardcolor, 'LineWidth', linewidth);
        end
        % step 1.3.3.3: get the ground truth box
        isel = find(strcmp(initids, id) & initlabels==1);
        % step 1.3.3.4: plot the ground truth boxes
        for i = isel'
            rect = [initboxes(i,1:2), initboxes(i,3:4)-initboxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', gtcolor, 'LineWidth', 2*linewidth);
        end
        % step 1.3.3.5: print the result
        print('-depsc2', figFile);
        % step 1.3.3.6: get the ground truth box only
        figure(1); clf; imshow(img); hold on;
        isel = find(strcmp(initids, id) & initlabels==1);
        % step 1.3.3.7: plot the ground truth boxes
        for i = isel'
            rect = [initboxes(i,1:2), initboxes(i,3:4)-initboxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', gtcolor, 'LineWidth', 2*linewidth);
        end
        % step 1.3.3.8: print the result
        figFile = sprintf('%s/iccv/%s_gtbox.eps', VOCopts.resdir, id);
        print('-depsc2', figFile);
    end
end
% step 1.4: plot hard negative boxes
% step 1.4.1: read in the trainBB file
bbFileInit = sprintf('%s/trainBB/%s_iter2.txt', VOCopts.resdir, typeName);
if ~exist(bbFileInit, 'file')
    fprintf('%s does not exist!\n', bbFileInit);
    return;
end
fid = fopen(bbFileInit);
if fid == -1
    fprintf('Cannot open %s!\n', bbFileInit);
    return;
end
C = textscan(fid, '%s %d %d %d %d %d');
hardids = C{1};
hardlabels = C{2};
hardboxes = [C{3} C{4} C{5} C{6}];
clear C
fclose(fid);
uhardids = unique(hardids);
selNeg = [1 3 4];
for d = selNeg
    id = uhardids{d};
    % step 1.4.2: read the image
    imgFile = sprintf(VOCopts.imgpath, id);
    if ~exist(imgFile, 'file')
        fprintf('%s does not exist!\n', imgFile);
        continue;
    end
    img = imread(imgFile);
    % step 1.4.3: plot image with ground truth box
    figFile = sprintf('%s/iccv/%s_hardboxAll.eps', VOCopts.resdir, id);
    if ~exist(figFile, 'file') || replot
        figure(1); clf; imshow(img); hold on;
        % step 1.4.3.1: get some selective boxes
        featFile = sprintf(VOCopts.featpath, id);
        if ~exist(featFile, 'file')
            fprintf('%s does not exist!\n', featFile);
            continue;
        end
        load(featFile, 'boxes');
        % step 1.4.3.2: plot some of the selective boxes
        selIdx = randperm(size(boxes,1));
        for i = selIdx(1:nSelBoxes)
            rect = [boxes(i,1:2), boxes(i,3:4)-boxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', selcolor, 'LineWidth', linewidth);
        end
        % step 1.4.3.3: get the hard negative boxes
        isel = find(strcmp(hardids, id));
        % step 1.4.3.4: plot the hard negative boxes
        for i = isel'
            rect = [hardboxes(i,1:2), hardboxes(i,3:4)-hardboxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', hardcolor, 'LineWidth', 2*linewidth);
        end
        % step 1.4.3.5: print the result
        print('-depsc2', figFile);
        % step 1.4.3.6: plot the hard negative boxes only
        figure(1); clf; imshow(img); hold on;
        for i = isel'
            rect = [hardboxes(i,1:2), hardboxes(i,3:4)-hardboxes(i,1:2)+1];
            rectangle('Position', rect, 'EdgeColor', hardcolor, 'LineWidth', 2*linewidth);
        end
        % step 1.4.3.6: print the result
        figFile = sprintf('%s/iccv/%s_hardbox.eps', VOCopts.resdir, id);
        print('-depsc2', figFile);
    end
end