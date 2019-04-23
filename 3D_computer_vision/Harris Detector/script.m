%%
%Load Various images
I1 = rgb2gray(imread('images/lena.png'));
I2 = rgb2gray(imread('images/chess.jpg'));
I3 = rgb2gray(imread('images/notredame.jpg'));
I4 = rgb2gray(imread('images/scene.jpeg'));
I5 = rgb2gray(imread('images/house.jpg'));
%%
%Try harris detector on different images
I = I1;
[corners, ~] = harrisCorners(I,0.04,1.5,0.01,200);
displayCorners(I,corners);
%
I = I2;
[corners, ~] = harrisCorners(I,0.04,1.5,0.01,200);
displayCorners(I,corners);
%
I = I3;
[corners, ~] = harrisCorners(I,0.04,1.5,0.01,200);
displayCorners(I,corners);
%
I = I4;
[corners, ~] = harrisCorners(I,0.04,1.5,0.01,200);
displayCorners(I,corners);
%
I = I5;
[corners, ~] = harrisCorners(I,0.04,1.5,0.01,200);
displayCorners(I,corners);
%%
I = I2;
% Comparison between the different methods (Harris with and without anms, Matlab function)
% Without ANMS
[corners, ~] = harrisCorners(I,0.04,1.5,0.01,200);
displayCorners(I,corners);

% Matlab corner function
corners = corner(I,200);
displayCorners(I,corners);

%With ANMS
corners = harrisANMS(I,0.04,1.5,0.01,200);
displayCorners(I,corners);

%%

%Test different parameters
% Change the separable smoothing parameter
sigmas = 1:10
for sigma = sigmas
    corners = harrisANMS(I,0.04,sigma,0.01,200);
    displayCorners(I,corners);
end
%%
I = I1;
%Change the Quality level parameter
quals = [0.01:0.1:1];
for qual = quals
    corners = harrisANMS(I,0.04,1.5,qual,200);
    displayCorners(I,corners);
end
    
%%
%Test covariance with rotation
thetas = [0:45:180]
for theta = thetas
    I = imrotate(I1,theta);
    corners = harrisANMS(I,0.04,1.5,0.01,200);
    displayCorners(I,corners);
end
%%
% Test effect of noise

I = I3;
corners = harrisANMS(I,0.04,1.5,0.01,200);
displayCorners(I,corners);
J = imnoise(I,'gaussian');
corners = harrisANMS(J,0.04,1.5,0.01,200);
displayCorners(J,corners);

%%
%Test effect of scaling
I = I1;
scales = [0.4:0.1:1];
for s = scales 
    J = imresize(I,s);
    corners = harrisANMS(J,0.04,1.5,0.01,200);
    displayCorners(J,corners);
end

%%
%Test sensitivity to viewpoint change
V1 = rgb2gray(imread('images/img1.png'));
V2 = rgb2gray(imread('images/img3.png'));
V3 = rgb2gray(imread('images/img5.png'));

corners = harrisANMS(V1,0.04,1.5,0.01,200);
displayCorners(V1,corners);
corners = harrisANMS(V2,0.04,1.5,0.01,200);
displayCorners(V2,corners);
corners = harrisANMS(V3,0.04,1.5,0.01,200);
displayCorners(V3,corners);
%%
I = I1;
%%
% Comparison between the different methods (Harris with and without anms, Matlab function)
% Without ANMS
[corners, ~] = harrisCorners(I,0.04,0.75,0.01,200);
displayCorners(I,corners);

% Matlab corner function
corners = corner(I,'harris',200);
displayCorners(I,corners);

%With ANMS
corners = harrisANMS(I,0.04,0.75,0.01,200);
displayCorners(I,corners);

