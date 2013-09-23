function [img2, resize_factor] = wl_resizeIm(img)
% wl_resizeIm() will resize the image to enforce the larger dimension be
% less than 500 pixels
% Input:
%   img: the input image
%

img2 = img;
resize_factor = 1;
max_dim = 500;
if size(img,1)>max_dim || size(img, 2)>max_dim
    resize_factor = min(max_dim/size(img, 1), max_dim/size(img, 2));
    img2 = imresize(img, factor);
end