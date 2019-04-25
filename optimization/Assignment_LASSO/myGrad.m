function g = myGrad(x,t,Q,p,A,b)
%Inputs : Same as phi
%Returns :  Gradient of phi evaluated on x

g = t*(Q*x + p) + A'*(1./(b-A*x));

end
