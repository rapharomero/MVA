function f = myPhi(x,t,Q,p,A,b)
%Inputs: All vectors must be column vectors, A and Q must be matrices and t
%must be a single valued(float, double...).
%Returns : value of phi(x)

    f = t*(0.5* x'*Q*x + p'*x) - sum(log(max(b-A*x,0)));

end
