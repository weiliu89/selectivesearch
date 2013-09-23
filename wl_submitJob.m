function jobID = wl_submitJob(jobFile, matlab)
% wl_submitJob() will automatically generate job config file and submit
%
% Input:
%   jobFile: the path of the job file
%   matlab: indicator if the job is matlab job or linux job
% Output:
%   jobID: the id of the submitted job
%

%% step 0: check
if ~exist(jobFile, 'file')
    fprintf('%s does not exist!', jobFile);
    return;
end
if nargin == 1
    matlab = 1;
end

%% step 1: generate the command for submitting job
cmd = sprintf('/home/wliu/bin/configArrayJob.sh %s.array %s %s %d 1 0 1', jobFile, jobFile, pwd, matlab);
unix(cmd);
cmd = sprintf('ssh wliu@bigeye.cs.stonybrook.edu ''qsub %s.array''', jobFile);
[~, results] = unix(cmd);

%% step 2: parse the jobID
cmd = sprintf('echo "%s" | cut -d'' '' -f3 | cut -d''.'' -f1', results);
[~, jobID] = unix(cmd);