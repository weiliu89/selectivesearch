function lines = wl_getLines(fileList, startIdx, endIdx)
% wl_getLines() will get the lines from fileList given the range
% Input:
%   fileList: the file list
%   startIdx: the start index of the file list
%   endIdx: the end index of the file list
%

if (endIdx < startIdx)
    fprintf('Incorrect range!\n');
    return;
end

%% step 1: seek the correct file
totalLines = 14197087;
if startIdx > totalLines
    lines = cell(0,1);
    return;
end
endIdx = min(endIdx, totalLines);

nLines = 14197;
startFile = ceil(startIdx/nLines);
endFile = min(ceil(endIdx/nLines), 1000);
[~, fileName] = fileparts(fileList);

%% step 2: read the lines
count = 1;
lines = cell(endIdx-startIdx+1, 1);
for iFile = startFile:endFile
    % step 2.1: seek the position
    iFileList = sprintf('%s-split/%s-%d.txt', fileList, fileName, iFile);
    iStartIdx = max(startIdx - (iFile-1)*nLines, 1);
    if iFile < 1000
        iEndIdx = min(endIdx - (iFile-1)*nLines, nLines);
    else
        iEndIdx = endIdx - (iFile-1)*nLines;
    end
    nd = iEndIdx - iStartIdx + 1;
    
    % step 2.2: open the file
    if ~exist(iFileList, 'file')
        fprintf('%s does not exist!\n', iFileList);
        return;
    end
    fid = fopen(iFileList);
    if fid == -1
        fprintf('Cannot open %s!\n', iFileList);
        return;
    end
    
    % step 2.3: read and store the lines
    C = textscan(fid, '%s');
    iLines = C{1};
    lines(count:count+nd-1) = iLines(iStartIdx:iEndIdx);
    count = count+nd;
    fclose(fid);
end