function [xstar, xhist] = dampedNewton(x0,f,g,h,tol)
% Parameters: 
%   f,g,h : target, gradient and hessian
%   x0 : initial point 
%   tol : tolerance treshold 
% Returns:
%   xstar : solution of the newton method
%   x_hist : history of the newton steps
    x = x0;
    gap = +inf;
    xhist = x0;
    while gap > tol
        [x, gap] = dampedNewtonStep(x,f,g,h);
        xhist = [xhist, x];
    end
    xstar = x;
end

function [xnew,gap] = dampedNewtonStep(x,f,g,h);
% Parameters: 
%   f,g,h : target, gradient and hessian
%   x : current point
% Returns:
%   xnew : newton step
%   gap : estimated gap between new point and optimal point
    gradient = g(x);
    hessian = h(x);
    
    lambda = sqrt(gradient' * linsolve(hessian, gradient));
    
    xnew = x - (1./(1+lambda)) * linsolve(hessian, gradient);
    
    gap = (lambda^2)/2.0;
end

   