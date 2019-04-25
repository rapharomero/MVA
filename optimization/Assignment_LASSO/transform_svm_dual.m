function [Q,p,A,b] = transform_svm_dual(tau,X,y)
% Parameters: 
%   tau : regularization parameter
%   X : data matrix
%   y : labels
% Returns:
% Q,p,A,b encoding the dual of svm as a QP on the dual variable
% lambda
n = size(X,1);
Q = diag(y)*(X*X')*diag(y);
p = -1.*ones(n,1);

A = [eye(n);
     -eye(n)];
b =[1/(tau*n) * ones(n,1);
     zeros(n,1)];
  
end