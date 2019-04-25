function [xstar, xhist] = newtonLS(x0,f,g,h,tol)
    [x, gap] = newtonLSStep(x0,f,g,h);
    xhist = [x0 x];
    while gap > tol 
        [x, gap] = newtonLSStep(x,f,g,h);
        xhist = [xhist,x];
    end
    xstar = x;
end

function [xnew,gap] = newtonLSStep(x,f,g,h);
    Grad = g(x);
    Hess = h(x);
%     disp(size(Hess));
%     disp(size(Grad));
    %Backtracking linesearch
    alpha = 0.1;
    beta = 0.7;
    t = 1;
    %Newton's descent direction
    lambda2 =  Grad'*linsolve(Hess,Grad);%square of newton
    while f(x-t*linsolve(Hess,Grad)) > f(x) - alpha*t*Grad'*linsolve(Hess,Grad)
       t = beta * t;
    end
    
    xnew = x  - t *linsolve(Hess,Grad);
   
    gap = (lambda2)/2.;
end



   