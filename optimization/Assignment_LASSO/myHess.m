function h = myHess(x,t,Q,p,A,b)
%Inputs : Same as before
%Returns : hessian of phi evaluated on x

u = 1./(b-A*x);

d = diag((u).^2);

h = t*Q + A'*d*A;


end

