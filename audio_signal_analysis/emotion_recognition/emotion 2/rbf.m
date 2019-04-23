function [ K] = rbf(z1,z2,sigma)
%RBF Summary of this function goes here
% Computes the value of the rbf kernel in z , with scaling matrix sigma
% Inputs must be of size 2,n (column vectors)
z=z1-z2;
s=(size(z));
m=min(s);
M=max(s);
if m==1
    tmp=reshape(z,[2 1]);
end
if m*M==4
    tmp=z
end

if M>2
    tmp=reshape(z,[2,M]);
end

K = exp(-tmp'*pinv(sigma)*tmp/2)/(2*pi*sqrt(det(sigma)));

end

