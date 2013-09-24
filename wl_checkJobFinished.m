function finished = wl_checkJobFinished(jobID)
% wl_checkJobFinished() will check if a job has finished or not
%
% Input:
%   jobID: the id of the job
% Output:
%   finished: 1 if the job has been finished

wl_setup;
% wait 2 mins before check
pause(120);

if strcmp(location, 'unc')
	cmd = sprintf('bjobs | grep %d | wc -l', jobID);
elseif strcmp(location, 'sbu')
	cmd = sprintf('ssh wliu@bigeye.cs.stonybrook.edu "qstat | grep %d | wc -l"', jobID);
end
[~, result] = unix(cmd);

if str2double(result) == 0
    finished = 1;
else
    finished = 0;
end
