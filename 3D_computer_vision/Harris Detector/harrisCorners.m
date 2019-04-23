function [corners,values] = harrisCorners(I,alpha,sigma,treshold,n)
% Inputs:
%   I : 2*2 RGB Image
%   alpha : sensitivity parameter
%   treshold : treshold 
%   n : number of corners to return
% Returns:   
%   corners : set of pixel coordinates of harris corners of I 

%Compute Smooth derivative kernel
dx = [-1/2 0 1/2];
g = gaussian1d(1);
Gx = conv(dx,g);
Gy = Gx';
%
I = double(I);
%Compute image derivatives
Ix = conv2(I,Gx,'same');
Iy = conv2(I,Gy,'same');

Ix2 = Ix.*Ix;
Ixy = Iy.*Ix;
Iy2 = Iy.*Iy;

%Smooth the product images and compute the convolution with the window w

g1 = gaussian1d(sigma);
g2 = gaussian2d(sigma);
Ix2 = conv2(Ix2,g1,'same');

Ixy = conv2(Ixy,g2,'same');

Iy2 = conv2(Iy2,g1','same');

%Compute the corner indicators for each pixel
Mc = Ix2.*Iy2 - Ixy.*Ixy - alpha*(Ix2 + Iy2).*(Ix2 + Iy2);

% Sort the indicator matrix (after stacking its columns)
[ordered, order] = sort(Mc(:),'descend');

% Compare indicators to a treshold
maxval = ordered(1,1);

mask = (ordered >= treshold*maxval);

[y, x] = ind2sub(size(Mc),order);

XY = [x y];

XY = XY(mask,:);

values = ordered(mask,:);

corners = XY(1:min(n,end),:);

values = ordered(1:min(n,end),:);

end