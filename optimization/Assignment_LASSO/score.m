function s = score(tau,Xtrain,ytrain,Xtest,ytest,mu,tol)
% Inputs:
%   tau : regularization parameter
%   X_, y_ :  data and labels
%  Returns : 
%   s : score = proportion of correctly predicted labels
    n = size(Xtrain,1);
    d = size(Xtrain,2);
    z0 = 2*ones(n,1);
    w0 = zeros(d,1);
    x0 = [w0;
          z0];
    
    %Compute parameters of primal SVM Problem
    [Q, p, A, b] = transform_svm_primal(tau,Xtrain,ytrain);
    
    [x_sol,~] = barr_method(Q,p,A,b,x0,mu,tol);
    
    w_sol = x_sol(1:d);
    
    ypred = sign(Xtest*w_sol);
    s = mean(2*ypred.*ytest -1);
    

end
