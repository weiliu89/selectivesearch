function encoder = wl_trainEncoder(trainFeatList, k)
% wl_trainEncoder() will train FV GMM encoder
%

encodeFile = 'data/encoder.mat';
if ~exist(encodeFile, 'file')
    %% step 0: setup the environment
    wl_setup;
    
    %% step 1: read in the image names
    % read in the feature files
    if ~exist(trainFeatList, 'file')
        fprintf('%s does not exist!\n', trainFeatList);
        return;
    end
    fid = fopen(trainFeatList);
    if fid == -1
        fprintf('Cannot open %s!\n', trainFeatList);
        return;
    end
    C = textscan(fid, '%s');
    featFiles = C{1};
    clear C
    fclose(fid);
    nd = length(featFiles);
    % load the feature
    maxN = 12000000;
    descrs = zeros(128, maxN, 'single');
    count = 1;
    th = tic;
    for d = 1:nd
        tic
        featFile = featFiles{d};
        if ~exist(featFile, 'file')
            fprintf('%s does not exist!\n', featFile);
            continue;
        end
        try
            a = load(featFile);
            % delete all 0 features
            zeroIdx = find(all(a.features.descr==0));
            if ~isempty(zeroIdx)
                a.features.descr(:, zeroIdx) = [];
            end
        catch
            fprintf('Cannot load %s\n', featFile);
            continue;
        end
        nf = size(a.features.descr, 2);
        descrs(:,count:count+nf-1) = a.features.descr;
        count = count+nf;
        %fprintf('Load %s time: %.2f\n', featFile, toc);
    end
    if count < maxN
        descrs(:, count:maxN) = [];
    end
    %save('data/feature.mat', 'descrs', '-v7.3');
    fprintf('Collect %d features, with %.2f sec!\n', count, toc(th));
    
    %% step 2: normalize the data
    % step 2.1: RootSIFT
    descrs = sqrt(descrs);
    % step 2.2: L2 normalization
    descrs = bsxfun(@times, descrs, 1./max(1e-5, sqrt(sum(descrs.^2,1))));
    
    %% Step 3: learn PCA projection
    fprintf('learning PCA rotation/projection\n');
    encoder.projectionCenter = mean(descrs,2);
    x = bsxfun(@minus, descrs, encoder.projectionCenter);
    X = x*x' / size(x,2) ;
    [V,D] = eig(X) ;
    d = diag(D) ;
    [~,perm] = sort(d,'descend') ;
    m = min(64, size(descrs,1)) ;
    V = V(:,perm) ;
    encoder.projection = V(:,1:m)';
    descrs = encoder.projection * x;
    clear x X V D d ;
    % descrs = bsxfun(@times, descrs, 1./max(1e-12, sqrt(sum(descrs.^2)))) ;
    
    %% Step 4: learn a VQ or GMM vocabulary
    fprintf('learning GMM\n');
    dimension = size(descrs,1) ;
    numDescriptors = size(descrs,2) ;
    
    vl_twister('state', 1) ;
    v = var(descrs')' ;
    [encoder.means, encoder.covariances, encoder.priors] = ...
        vl_gmm(descrs, k, 'verbose', ...
        'Initialization', 'kmeans', ...
        'CovarianceBound', double(max(v)*0.0001), ...
        'NumRepetitions', 5);
    save(encodeFile, 'encoder');
end