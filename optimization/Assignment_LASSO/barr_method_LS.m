function [x_sol,xhist] = barr_method_LS(Q,p,A,b,x0,mu,tol)
% Parameters: 
%   Q,p,A,b : Parameters of the svm problem
%   x0 : initial point 
%   mu : increasing rate of the barrier parameter t
%   tol : tolerance treshold
% Returns:
%   x_sol : solution of the barrier method
%   x_hist : history of the successive points

    t = 1;
    m = size(b,2);
    x_star = x0;
    xhist = x0;

    while m/t >= tol
        f = @(x) myPhi(x,t,Q,p,A,b);
        g = @(x) myGrad(x,t,Q,p,A,b);
    	h = @(x) myHess(x,t,Q,p,A,b); 
        
        [x_star,xh] = newtonLS(x_star,f,g,h,tol);
        xhist = [xhist, xh];
        t = mu*t;
    end
    x_sol = x_star;
end