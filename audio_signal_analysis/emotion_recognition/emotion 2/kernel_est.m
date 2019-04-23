function [ y ] = kernel_est( q , G,hv,ha)
%KERNEL_EST Summary of this function goes here
%   Computes the kernel density estimation of the probability distribution
%   given  the annotations q , the size of the grid G , and the scale
%   factors hv and ha

delta=2/G; % step of the grid
sigma=diag([hv,ha]);
for i=1:G
    for j=1:G
        p=[(i-1/2)*delta-1 (j-1/2)*delta-1];
        y(i,j)=mean(diag(rbf(p,q,sigma)));
    end
end
y=y/sum(y(:));

end

