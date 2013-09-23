function ComputeFeatureMultipleMachines(allnum, dic)

% all num = 317857;

% do it in parrallel
seg = 20000;
partition = ceil(allnum/seg)


% first compute top 9 partitions
parfor i=1:(partition-1)
    buildWholeFeature(dic, (i-1)*seg+1:i*seg);
end


% the last one
buildWholeFeature(dic, (partition-1)*seg+1:allnum);





























