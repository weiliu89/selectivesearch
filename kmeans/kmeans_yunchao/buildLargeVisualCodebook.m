function [v] = buildLargeVisualCodebook(X, dic, index)



sss = size(dic, 2);

% search for the indexes
results = flann_search(index, X, 1, struct('checks', 512));

% histogram and output the vector
v = hist(results, 1:sss);
v = v(:)';



























