function [Q,p,A,b] = transform_svm_primal(tau,X,y)
% Parameters: 
%   tau : regularization parameter
%   X : data matrix
%   y : labels
% Returns:
%   Q,p,A,b encoding the primal of svm as a QP on the optimization variable (w z)^T
n = size(X,1);
d = size(X,2);

Q = [eye(d) zeros(d,n);
     zeros(n,d) zeros(n)];
 
p = [zeros(d,1);
     (1/(tau*n))*ones(n,1)];

A = [-diag(y)*X -1.*eye(n);
      zeros(n,d)  -1.*eye(n)];
 
b = [-ones(n,1);
  zeros(n,1)];
  
end
