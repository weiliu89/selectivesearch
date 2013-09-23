function features = wl_getDenseSIFT(im, varargin)
% wl_getDenseSIFT will extract dense SIFT features

opts.scales = logspace(log10(1), log10(.25), 3);
opts.contrastthreshold = 0;
opts.step = 20;
opts.rootSift = false;
opts.normalizeSift = false;
opts.binSize = 4;
opts.geometry = [4 4 8];
opts = vl_argparse(opts, varargin);

%dsiftOpts = {'fast', ...
%dsiftOpts = {'fast', 'floatdescriptors', ...
dsiftOpts = {'step', opts.step, ...
             'size', opts.binSize, ...
             'geometry', opts.geometry} ;

if size(im,3)>1, im = rgb2gray(im) ; end
im = im2single(im) ;

for si = 1:numel(opts.scales)
  im_ = imresize(im, opts.scales(si)) ;

  [frames{si}, descrs{si}] = vl_dsift(im_, dsiftOpts{:}) ;

  % root SIFT
  if opts.rootSift
    descrs{si} = sqrt(descrs{si}) ;
  end
  if opts.normalizeSift
    descrs{si} = snorm(descrs{si}) ;
  end
  
  % store frames
  frames{si}(1:2,:) = (frames{si}(1:2,:)-1) / opts.scales(si) + 1 ;
  scales{si} = frames{si}(1,:);
  scales{si}(1,:) = opts.binSize / opts.scales(si) / 3 ;
end

features.frame = cat(2, frames{:}) ;
features.descr = cat(2, descrs{:}) ;
features.scale = cat(2, scales{:});

function x = snorm(x)
x = bsxfun(@times, x, 1./max(1e-5,sqrt(sum(x.^2,1))));
