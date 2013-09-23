function finished = wl_checkJobFinished(jobID)
% wl_checkJobFinished() will check if a job has finished or not
%
% Input:
%   jobID: the id of the job
% Output:
%   finished: 1 if the job has been finished

% wait 2 mins before check
pause(120);

cmd = sprintf('ssh wliu@bigeye.cs.stonybrook.edu "qstat | grep %d | wc -l"', jobID)
[~, result] = unix(cmd);
result

if str2double(result) == 0
    finished = 1;
else
    finished = 0;
end
