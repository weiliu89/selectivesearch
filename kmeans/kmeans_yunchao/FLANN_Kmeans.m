function [dic] = FLANN_Kmeans(X, K, iter, dic, silent)

if nargin == 4
    silent = false;
end

% X is the data 128 * n
X = single(X);

if isempty(dic)
    % initialize centers using random samples
    randindex = randperm(size(X,2));
    dic = single(X(:,randindex(1:K)));
end


% start approximate Kmeans iterations
for i=1:iter
    if ~silent
        disp(['Iteration: ' num2str(i)])
    end
    % build kd-tree index
    if ~silent
        th = tic;
    end
    index = flann_build_index(dic, struct('algorithm', 'kdtree', 'trees', 10));
    if ~silent
        fprintf('flann_build_index time: %.2f\n', toc(th));
    end
    
    
    % search for NN using indexes
    if ~silent
        th = tic;
    end
    [results, mD] = flann_search(index, X, 1, struct('checks', 512, 'cores', 0));
    if ~silent
        fprintf('flann_search time: %.2f\n', toc(th));
    end
    %oldDic = dic;
    
    % caculate the new centers
    if ~silent
        th = tic;
    end
    for j=1:K
        idx = find(results==j);
        if(isempty(idx))
            % do nothing, keep the original centers
        elseif(length(idx)==1)
            dic(:,j) = X(:,idx);
        else
            % update the centers
            subsamp = X(:,idx);
            mm = mean(subsamp, 2);
            dic(:,j) = mm;
            % NOTE: added ./ norm(mm) so that the center is unit length
            %dic(:,j) = mm(:) ./ norm(mm);
        end
    end
    if ~silent
        fprintf('compute mean time: %.2f\n', toc(th));
    end
    
%     if ~silent
%         th = tic;
%     end
%     % compute the mean
%     newdic = oldDic;
%     %newdic = accumarray(results', X, [], @mean);
%     groupID = unique(results);
%     parfor k=1:size(oldDic, 1)
%         newdic(k, :) = accumarray(results', X(k,:), [], @mean);
%     end
%     %newdic(:, groupID) = cell2mat(arrayfun(@(k) mean(X(:,results==k),2), groupID, 'UniformOutput', false));
%     if ~silent
%         fprintf('compute mean fast time: %.2f\n', toc(th));
%     end
%     
%     if ~silent
%         fprintf('Two mean method diff: %.2f\n', sum(sum(dic-newdic)));
%     end
    
    % report min error
    if ~silent
        disp(['Min Error: ' num2str(sum(mD))])
    end
    
    
    % free indexes
    flann_free_index(index);
end
