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
wl_setup;
if ~exist(jobFile, 'file')
    fprintf('%s does not exist!', jobFile);
    return;
end
if nargin == 1
    matlab = 1;
end

%% step 1: generate the command for submitting job
if strcmp(location, 'unc')
	cmd = sprintf('%s/bin/configArrayKilldevilJob.sh %s.array %s %s %d 1', HOMEDIR, jobFile, jobFile, pwd, matlab);
elseif strcmp(location, 'sbu')
	cmd = sprintf('%s/bin/configArrayJob.sh %s.array %s %s %d 1 0 1', HOMEDIR, jobFile, jobFile, pwd, matlab);
else
	fprintf('Unknown locatoin: %s!\n', location);
	jobID = [];
	return;
end
unix(cmd);
if strcmp(location, 'unc')
	cmd = sprintf('bsub < %s.array', jobFile);
elseif strcmp(location, 'sbu')
	cmd = sprintf('ssh wliu@bigeye.cs.stonybrook.edu ''qsub %s.array''', jobFile);
else
	fprintf('Unknown locatoin: %s!\n', location);
	jobID = [];
	return;
end
[~, results] = unix(cmd);

%% step 2: parse the jobID
if strcmp(location, 'unc')
	cmd = sprintf('echo "%s" | grep Job | cut -d''<'' -f2 | cut -d''>'' -f1', results);
elseif strcmp(location, 'sbu')
	cmd = sprintf('echo "%s" | cut -d'' '' -f3 | cut -d''.'' -f1', results);
else
	fprintf('Unknown locatoin: %s!\n', location);
	jobID = [];
	return;
end
[~, jobID] = unix(cmd);
jobID = str2double(jobID(1:end-1));
